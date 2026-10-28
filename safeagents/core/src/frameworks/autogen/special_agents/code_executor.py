import asyncio
import io
import logging
import os
import tarfile
import tempfile
import time
import uuid
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Callable, TypeVar

import docker
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors._common import (
    CommandLineCodeResult,
    get_file_name_from_content,
    lang_to_cmd,
    silence_pip,
)
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from docker.errors import ImageNotFound, NotFound

logger = logging.getLogger(__name__)
T = TypeVar("T")


async def _finish_task(task: asyncio.Task[T]) -> T:
    """Defer cancellation until a Docker operation has finished."""
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except Exception as error:
            if cancelled:
                raise asyncio.CancelledError from error
            raise
    if cancelled:
        raise asyncio.CancelledError
    return result


class TrackedDockerCommandLineCodeExecutor(DockerCommandLineCodeExecutor):
    """Execute tracked code in a credential-free, network-disabled container."""

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        timeout: int = 60,
        working_directory: str | Path | None = None,
        tool_call_tracker: Callable | None = None,
        agent_name: str | None = None,
    ):
        super().__init__(image=image, timeout=timeout)
        self._workspace_parent = Path(working_directory).expanduser().resolve() if working_directory else None
        self._tool_call_tracker = tool_call_tracker
        self._agent_name = agent_name
        self._docker_client: docker.DockerClient | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._creation_attempted = False
        self._ownership_token = ""
        self._command_tasks: set[asyncio.Task] = set()
        self._uid = os.getuid() if hasattr(os, "getuid") else 65534
        self._gid = os.getgid() if hasattr(os, "getgid") else 65534

    def _start_container(self) -> None:
        self._docker_client = docker.from_env(timeout=60)
        try:
            self._docker_client.images.get(self._image)
        except ImageNotFound as error:
            raise RuntimeError(
                f"Docker execution image {self._image!r} is unavailable. "
                "Pull a trusted image before running computer_terminal."
            ) from error

        self._creation_attempted = True
        self._container = self._docker_client.containers.create(
            self._image,
            name=self.container_name,
            entrypoint="/bin/sh",
            command=[],
            tty=True,
            stdin_open=True,
            detach=True,
            auto_remove=True,
            volumes={str(self.work_dir): {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
            network_disabled=True,
            use_config_proxy=False,
            privileged=False,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            user=f"{self._uid}:{self._gid}",
            labels={"safeagents.executor": self._ownership_token},
        )
        self._container.start()
        deadline = time.monotonic() + 60
        while True:
            self._container.reload()
            if self._container.status == "running":
                return
            if self._container.status in {"exited", "dead"} or time.monotonic() >= deadline:
                raise RuntimeError("Docker execution container failed to become ready.")
            time.sleep(0.1)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._running:
                raise RuntimeError("Docker executor is already running.")
            if self._container is not None or self._temp_dir is not None or self._creation_attempted:
                raise RuntimeError("Docker executor cleanup must finish before it can restart.")
            if self._workspace_parent is not None:
                self._workspace_parent.mkdir(parents=True, exist_ok=True)
            self._temp_dir = tempfile.TemporaryDirectory(
                prefix="safeagents-terminal-", dir=self._workspace_parent
            )
            self._work_dir = Path(self._temp_dir.name)
            self._ownership_token = uuid.uuid4().hex
            self.container_name = f"safeagents-code-exec-{self._ownership_token}"
            task = asyncio.create_task(asyncio.to_thread(self._start_container))
            try:
                await _finish_task(task)
                self._running = True
            except BaseException as error:
                try:
                    await self._stop_locked()
                except BaseException as cleanup_error:
                    raise BaseExceptionGroup(
                        "Docker executor startup and cleanup failed", [error, cleanup_error]
                    ) from None
                raise

    def _remove_container(self) -> None:
        if self._container is None and self._creation_attempted:
            try:
                container = self._docker_client.containers.get(self.container_name)
            except NotFound:
                container = None
            if container is not None and container.labels.get("safeagents.executor") == self._ownership_token:
                self._container = container
        if self._container is not None:
            try:
                self._container.remove(force=True)
            except NotFound:
                pass
            self._container = None
        self._creation_attempted = False

    def _release_resources(self) -> None:
        errors = []
        if self._docker_client is not None:
            try:
                self._docker_client.close()
                self._docker_client = None
            except Exception as error:
                errors.append(error)
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
                self._temp_dir = None
                self._work_dir = None
            except OSError as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Docker executor resource cleanup failed", errors)

    async def _stop_resources(self) -> None:
        self._running = False
        await asyncio.to_thread(self._remove_container)
        if self._command_tasks:
            results = await asyncio.gather(*self._command_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    logger.debug("Docker request ended during teardown: %s", result)
        await asyncio.to_thread(self._release_resources)

    async def _stop_locked(self) -> None:
        await _finish_task(asyncio.create_task(self._stop_resources()))

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            await self._stop_locked()

    async def _execute_command(
        self, command: list[str], cancellation_token: CancellationToken
    ) -> tuple[str, int]:
        if self._container is None or not self._running:
            raise RuntimeError("Docker executor must be started before executing code.")
        command_task = asyncio.create_task(asyncio.to_thread(self._container.exec_run, command))
        self._command_tasks.add(command_task)

        async def wait_for_command():
            return await asyncio.shield(command_task)

        waiter = asyncio.create_task(wait_for_command())
        cancellation_token.link_future(waiter)
        try:
            result = await waiter
            if result.exit_code is None:
                raise RuntimeError("Docker did not return a code execution exit status.")
            output = result.output.decode("utf-8")
            if result.exit_code == 124:
                output += "\n Timeout"
            return output, result.exit_code
        except asyncio.CancelledError:
            await self.stop()
            raise
        finally:
            if command_task.done():
                self._command_tasks.discard(command_task)

    async def _stage_code(self, filename: str, code: str) -> None:
        if self._container is None or not self._running:
            raise RuntimeError("Docker executor must be started before staging code.")
        path = PurePosixPath(filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Filename must be relative to the workspace.")
        content = code.encode("utf-8")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as tar:
            entry = tarfile.TarInfo(str(path))
            entry.size = len(content)
            entry.mode = 0o600
            entry.uid, entry.gid = self._uid, self._gid
            tar.addfile(entry, io.BytesIO(content))
        # Stage through Docker so symlinks resolve in the container, not on the host.
        task = asyncio.create_task(
            asyncio.to_thread(self._container.put_archive, "/workspace", archive.getvalue())
        )
        if not await _finish_task(task):
            raise RuntimeError("Docker failed to stage the code block.")

    async def _execute_code_dont_check_setup(
        self, code_blocks: list[CodeBlock], cancellation_token: CancellationToken
    ) -> CommandLineCodeResult:
        if self._container is None or not self._running:
            raise RuntimeError("Docker executor must be started before executing code.")
        if not code_blocks:
            raise ValueError("No code blocks to execute.")
        outputs = []
        files = []
        exit_code = 0
        for block in code_blocks:
            language = block.language.lower()
            if language not in self.SUPPORTED_LANGUAGES:
                return CommandLineCodeResult(exit_code=1, output=f"Unsupported language: {language}")
            code = silence_pip(block.code, language)
            try:
                filename = get_file_name_from_content(code, self.work_dir)
            except ValueError:
                outputs.append("Filename is not in the workspace")
                exit_code = 1
                break
            if not filename:
                filename = f"tmp_code_{sha256(code.encode()).hexdigest()}.{language}"
            await self._stage_code(filename, code)
            files.append(str(self.work_dir / filename))
            output, exit_code = await self._execute_command(
                ["timeout", str(self._timeout), lang_to_cmd(language), filename],
                cancellation_token,
            )
            outputs.append(output)
            if exit_code != 0:
                break
        return CommandLineCodeResult(
            exit_code=exit_code, output="".join(outputs), code_file=files[0] if files else None
        )

    async def execute_code_blocks(
        self, code_blocks: list[CodeBlock], cancellation_token: CancellationToken
    ) -> CommandLineCodeResult:
        code_content = [
            {
                "language": block.language,
                "code": block.code[:200] + "..." if len(block.code) > 200 else block.code,
            }
            for block in code_blocks
        ]
        try:
            result = await super().execute_code_blocks(code_blocks, cancellation_token)
        except (Exception, asyncio.CancelledError) as error:
            if self._tool_call_tracker:
                self._tool_call_tracker(
                    tool_name="execute_code",
                    args={"code_blocks": code_content},
                    result=f"ERROR: {str(error) or 'Code execution was cancelled.'}",
                    agent_name=self._agent_name,
                )
            raise
        if self._tool_call_tracker:
            self._tool_call_tracker(
                tool_name="execute_code",
                args={"code_blocks": code_content},
                result=f"exit_code={result.exit_code}, output={str(result.output)[:200]}",
                agent_name=self._agent_name,
            )
        return result
