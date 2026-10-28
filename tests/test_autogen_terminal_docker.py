import asyncio
import os
import uuid
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock

import docker
import pytest
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from docker.errors import APIError, NotFound

from safeagents import Agent, AgentConfig
from safeagents.core.src.frameworks.autogen.special_agents.agents import get_special_agent
from safeagents.core.src.frameworks.autogen.special_agents.code_executor import (
    TrackedDockerCommandLineCodeExecutor,
)

pytestmark = pytest.mark.docker


@pytest.fixture
def docker_factory(tmp_path, monkeypatch):
    image = os.getenv("SAFEAGENTS_TEST_EXECUTOR_IMAGE")
    if not image:
        pytest.fail("Set SAFEAGENTS_TEST_EXECUTOR_IMAGE to an approved, already-loaded image digest.")
    client = docker.from_env()
    try:
        client.images.get(image)
    except docker.errors.DockerException as error:
        client.close()
        pytest.fail(f"Docker integration prerequisite failed: {error}")

    def forbid_pull(*args, **kwargs):
        raise AssertionError("Integration tests must not pull images.")

    monkeypatch.setattr(docker.models.images.ImageCollection, "pull", forbid_pull)
    executors = []

    def create(**kwargs):
        executor = TrackedDockerCommandLineCodeExecutor(
            image=image, working_directory=tmp_path, **kwargs
        )
        executors.append(executor)
        return executor

    yield create, executors, client, image

    leaked = []
    try:
        for executor in executors:
            try:
                container = client.containers.get(executor.container_name)
            except NotFound:
                continue
            leaked.append(executor.container_name)
            if container.labels.get("safeagents.executor") == executor._ownership_token:
                container.remove(force=True)
        for executor in executors:
            asyncio.run(executor.stop())
    finally:
        client.close()
    assert not leaked, f"Owned containers were not cleaned up: {leaked}"


def test_fixed_messages_execute_without_host_credentials_or_files(docker_factory, tmp_path, monkeypatch):
    create, executors, client, image = docker_factory
    host_file = tmp_path / "host-only.txt"
    host_file.write_text("unchanged")
    monkeypatch.setenv("SAFEAGENTS_TEST_HOST_SECRET", "synthetic-not-a-real-secret")
    monkeypatch.setenv("HTTP_PROXY", "http://synthetic.invalid:9999")
    tracker = Mock()
    agent = Agent(
        AgentConfig(
            name="terminal",
            special_agent="computer_terminal",
            special_agent_kwargs={"image": image, "working_directory": tmp_path},
        )
    )
    terminal = get_special_agent(
        agent, None, None, tool_call_tracker=tracker, register_executor=executors.append
    )
    executor = executors[0]
    code = (
        "import os\nfrom pathlib import Path\n"
        "assert 'SAFEAGENTS_TEST_HOST_SECRET' not in os.environ\n"
        "assert 'HTTP_PROXY' not in os.environ\n"
        f"assert not Path({str(host_file)!r}).exists()\n"
        "assert not Path('/var/run/docker.sock').exists()\n"
        "Path('artifact.txt').write_text('inside')\n"
        "print('python-ok')\n"
    )

    async def run():
        await executor.start()
        workspace = executor.work_dir
        try:
            container = client.containers.get(executor.container_name)
            config = container.attrs
            assert [
                (mount["Source"], mount["Destination"], mount["RW"])
                for mount in config["Mounts"]
            ] == [(str(workspace), "/workspace", True)]
            assert config["Config"]["NetworkDisabled"] is True
            assert config["HostConfig"]["NetworkMode"] != "host"
            assert config["HostConfig"]["Privileged"] is False
            assert config["HostConfig"]["AutoRemove"] is True
            assert config["HostConfig"]["CapDrop"] == ["ALL"]
            assert "no-new-privileges:true" in config["HostConfig"]["SecurityOpt"]
            response = await terminal.on_messages(
                [
                    TextMessage(
                        content=f"```python\n{code}```\n```sh\nprintf 'shell-ok\\n'\n```",
                        source="fixed-test-input",
                    )
                ],
                CancellationToken(),
            )
            assert "python-ok" in response.chat_message.content
            assert "shell-ok" in response.chat_message.content
            assert (workspace / "artifact.txt").read_text() == "inside"
            assert tracker.call_args.kwargs["result"].startswith("exit_code=0")
        finally:
            await executor.stop()
        assert not workspace.exists()

    asyncio.run(run())
    assert host_file.read_text() == "unchanged"


def test_nonzero_exit_prevents_later_blocks(docker_factory):
    create, _, _, _ = docker_factory
    executor = create()

    async def run():
        await executor.start()
        try:
            result = await executor.execute_code_blocks(
                [
                    CodeBlock(code="print('failed')\nraise SystemExit(7)", language="python"),
                    CodeBlock(code="touch should-not-exist", language="sh"),
                ],
                CancellationToken(),
            )
            assert result.exit_code == 7
            assert "failed" in result.output
            assert not (executor.work_dir / "should-not-exist").exists()
        finally:
            await executor.stop()

    asyncio.run(run())


def test_execution_timeout_is_enforced(docker_factory):
    create, _, _, _ = docker_factory
    executor = create(timeout=1)

    async def run():
        await executor.start()
        try:
            result = await asyncio.wait_for(
                executor.execute_code_blocks(
                    [CodeBlock(code="import time\ntime.sleep(30)", language="python")],
                    CancellationToken(),
                ),
                timeout=10,
            )
            assert result.exit_code != 0
        finally:
            await executor.stop()

    asyncio.run(run())


@pytest.mark.parametrize("cancel_token", [True, False])
def test_cancellation_removes_container_and_workspace(docker_factory, cancel_token):
    create, _, _, _ = docker_factory
    tracker = Mock()
    executor = create(tool_call_tracker=tracker)

    async def run():
        await executor.start()
        workspace = executor.work_dir
        token = CancellationToken()
        task = asyncio.create_task(
            executor.execute_code_blocks(
                [
                    CodeBlock(
                        code=(
                            "from pathlib import Path\nimport time\n"
                            "Path('started').write_text('ready')\ntime.sleep(60)"
                        ),
                        language="python",
                    )
                ],
                token,
            )
        )

        async def ready():
            while not (workspace / "started").exists():
                if task.done():
                    await task
                await asyncio.sleep(0.05)

        try:
            await asyncio.wait_for(ready(), timeout=10)
            if cancel_token:
                token.cancel()
            else:
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=10)
            assert not workspace.exists()
            assert not executor._command_tasks
            assert "cancelled" in tracker.call_args.kwargs["result"]
        finally:
            await executor.stop()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_instances_and_repeated_runs_do_not_share_artifacts(docker_factory):
    create, _, _, _ = docker_factory
    first, second = create(), create()

    async def check_empty(executor):
        result = await executor.execute_code_blocks(
            [CodeBlock(code="from pathlib import Path\nassert not Path('private').exists()", language="python")],
            CancellationToken(),
        )
        assert result.exit_code == 0

    async def run():
        await first.start()
        await second.start()
        first_workspace = first.work_dir
        try:
            result = await first.execute_code_blocks(
                [CodeBlock(code="from pathlib import Path\nPath('private').write_text('canary')", language="python")],
                CancellationToken(),
            )
            assert result.exit_code == 0
            await check_empty(second)
            await first.stop()
            await first.start()
            assert first.work_dir != first_workspace
            await check_empty(first)
        finally:
            await first.stop()
            await second.stop()

    asyncio.run(run())


def test_container_symlink_cannot_redirect_host_script_writes(docker_factory, tmp_path):
    create, _, _, _ = docker_factory
    sentinel = tmp_path / "host-only.py"
    sentinel.write_text("unchanged")
    tracker = Mock()
    executor = create(tool_call_tracker=tracker)
    next_code = "print('staged inside the container')"
    filename = f"tmp_code_{sha256(next_code.encode()).hexdigest()}.python"

    async def run():
        await executor.start()
        try:
            setup = await executor.execute_code_blocks(
                [
                    CodeBlock(
                        code=f"from pathlib import Path\nPath({filename!r}).symlink_to({str(sentinel)!r})",
                        language="python",
                    )
                ],
                CancellationToken(),
            )
            assert setup.exit_code == 0
            try:
                await executor.execute_code_blocks(
                    [CodeBlock(code=next_code, language="python")], CancellationToken()
                )
            except APIError:
                assert tracker.call_args.kwargs["result"].startswith("ERROR:")
            assert sentinel.read_text() == "unchanged"
        finally:
            await executor.stop()

    asyncio.run(run())


def test_missing_image_fails_without_pulling_or_creating_container(docker_factory):
    _, executors, _, _ = docker_factory
    executor = TrackedDockerCommandLineCodeExecutor(
        image=f"safeagents-missing-test-image:{uuid.uuid4().hex}"
    )
    executors.append(executor)
    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(executor.start())
    assert executor._container is None
    assert executor._temp_dir is None
