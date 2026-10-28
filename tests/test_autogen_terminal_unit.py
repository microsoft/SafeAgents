import asyncio
import io
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from docker.errors import APIError, DockerException, ImageNotFound, NotFound

from safeagents import Agent, AgentConfig
from safeagents.core.src.frameworks.autogen.autogen import TeamAutogen
from safeagents.core.src.frameworks.autogen.special_agents import code_executor as executor_module
from safeagents.core.src.frameworks.autogen.special_agents.agents import get_special_agent
from safeagents.core.src.frameworks.autogen.special_agents.code_executor import (
    TrackedDockerCommandLineCodeExecutor,
)


@pytest.fixture
def fake_docker(monkeypatch):
    container = Mock(status="running", labels={})
    container.put_archive.return_value = True
    container.exec_run.return_value = SimpleNamespace(exit_code=0, output=b"ok\n")
    client = Mock()

    def create(*args, **kwargs):
        container.labels = kwargs["labels"]
        return container

    client.containers.create.side_effect = create
    client.containers.get.side_effect = NotFound("No such container")
    from_env = Mock(return_value=client)
    monkeypatch.setattr(executor_module.docker, "from_env", from_env)
    return SimpleNamespace(client=client, container=container, from_env=from_env)


def test_factory_uses_docker_and_registers_tracking():
    executors = []
    tracker = Mock()
    agent = Agent(AgentConfig(name="terminal", special_agent="computer_terminal"))
    terminal = get_special_agent(
        agent, None, None, tool_call_tracker=tracker, register_executor=executors.append
    )
    assert terminal._code_executor is executors[0]
    assert isinstance(executors[0], TrackedDockerCommandLineCodeExecutor)
    assert executors[0]._tool_call_tracker is tracker
    assert executors[0]._agent_name == "terminal"


@pytest.mark.parametrize("option", ["extra_volumes", "environment", "privileged", "use_docker"])
def test_factory_rejects_unsafe_or_unknown_options(option):
    agent = Agent(
        AgentConfig(
            name="terminal",
            special_agent="computer_terminal",
            special_agent_kwargs={option: True},
        )
    )
    with pytest.raises(ValueError, match="Unsupported"):
        get_special_agent(agent, None, None, register_executor=lambda executor: None)


def test_factory_requires_lifecycle_owner_and_still_rejects_handoffs():
    agent = Agent(AgentConfig(name="terminal", special_agent="computer_terminal"))
    with pytest.raises(ValueError, match="manage its Docker executor"):
        get_special_agent(agent, None, None)
    with pytest.raises(ValueError, match="does not support handoffs"):
        get_special_agent(agent, None, ["other"], register_executor=lambda executor: None)


def test_container_configuration_is_isolated(fake_docker, tmp_path):
    (tmp_path / ".env").write_text("SYNTHETIC_SECRET=not-a-real-secret")
    executor = TrackedDockerCommandLineCodeExecutor(working_directory=tmp_path)

    async def run():
        await executor.start()
        workspace = executor.work_dir
        try:
            assert workspace.parent == tmp_path
            assert workspace != tmp_path
            assert not (workspace / ".env").exists()
            options = fake_docker.client.containers.create.call_args.kwargs
            assert options["volumes"] == {str(workspace): {"bind": "/workspace", "mode": "rw"}}
            assert options["network_disabled"] is True
            assert options["use_config_proxy"] is False
            assert options["privileged"] is False
            assert options["auto_remove"] is True
            assert options["cap_drop"] == ["ALL"]
            assert options["security_opt"] == ["no-new-privileges:true"]
            assert "environment" not in options
            assert "extra_hosts" not in options
            assert "network_mode" not in options
        finally:
            await executor.stop()
        assert not workspace.exists()

    asyncio.run(run())
    fake_docker.container.remove.assert_called_once_with(force=True)


@pytest.mark.parametrize("exit_code", [0, 7])
def test_execution_preserves_results_tracking_and_container_side_staging(fake_docker, exit_code):
    output = "x" * 300
    fake_docker.container.exec_run.return_value = SimpleNamespace(
        exit_code=exit_code, output=output.encode()
    )
    tracker = Mock()
    executor = TrackedDockerCommandLineCodeExecutor(tool_call_tracker=tracker, agent_name="terminal")
    code = "print('ok')\n" + "#" * 300

    async def run():
        await executor.start()
        try:
            result = await executor.execute_code_blocks(
                [CodeBlock(code=code, language="python")], CancellationToken()
            )
            assert result.exit_code == exit_code
            assert result.output == output
            archive = fake_docker.container.put_archive.call_args.args[1]
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                member = tar.getmembers()[0]
                assert tar.extractfile(member).read().decode() == code
                assert member.uid == executor._uid
            trace = tracker.call_args.kwargs
            assert trace["tool_name"] == "execute_code"
            assert trace["agent_name"] == "terminal"
            assert trace["args"]["code_blocks"][0]["code"] == code[:200] + "..."
            assert trace["result"] == f"exit_code={exit_code}, output={output[:200]}"
        finally:
            await executor.stop()

    asyncio.run(run())


def test_execution_errors_are_tracked_and_propagated(fake_docker):
    fake_docker.container.exec_run.side_effect = RuntimeError("execution failed")
    tracker = Mock()
    executor = TrackedDockerCommandLineCodeExecutor(tool_call_tracker=tracker)

    async def run():
        await executor.start()
        try:
            with pytest.raises(RuntimeError, match="execution failed"):
                await executor.execute_code_blocks(
                    [CodeBlock(code="print(1)", language="python")], CancellationToken()
                )
            assert tracker.call_args.kwargs["result"] == "ERROR: execution failed"
        finally:
            await executor.stop()

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["daemon", "image", "create", "start", "readiness"])
def test_startup_failures_fail_closed_and_rollback(fake_docker, tmp_path, stage):
    if stage == "daemon":
        fake_docker.from_env.side_effect = DockerException("daemon unavailable")
    elif stage == "image":
        fake_docker.client.images.get.side_effect = ImageNotFound("image unavailable")
    elif stage == "create":
        fake_docker.client.containers.create.side_effect = APIError("create failed")
    elif stage == "start":
        fake_docker.container.start.side_effect = APIError("start failed")
    else:
        fake_docker.container.status = "exited"
    executor = TrackedDockerCommandLineCodeExecutor(working_directory=tmp_path)

    async def run():
        with pytest.raises(Exception):
            await executor.start()
        assert not executor._running
        assert not list(tmp_path.iterdir())
        await executor.stop()

    asyncio.run(run())
    fake_docker.client.images.pull.assert_not_called()
    if stage in {"start", "readiness"}:
        fake_docker.container.remove.assert_called_once_with(force=True)


@pytest.mark.parametrize("owned", [True, False])
def test_lost_create_response_only_removes_owned_container(fake_docker, tmp_path, owned):
    executor = TrackedDockerCommandLineCodeExecutor(working_directory=tmp_path)

    def create(*args, **kwargs):
        fake_docker.container.labels = kwargs["labels"] if owned else {"safeagents.executor": "other"}
        fake_docker.client.containers.get.side_effect = None
        fake_docker.client.containers.get.return_value = fake_docker.container
        raise APIError("create response lost")

    fake_docker.client.containers.create.side_effect = create
    with pytest.raises(APIError, match="create response lost"):
        asyncio.run(executor.start())
    assert not list(tmp_path.iterdir())
    assert fake_docker.container.remove.call_count == int(owned)


@pytest.mark.parametrize("stage", ["create", "start", "readiness"])
def test_cancelled_start_waits_for_thread_then_removes_container(fake_docker, tmp_path, stage):
    entered, release = threading.Event(), threading.Event()
    original_create = fake_docker.client.containers.create.side_effect

    def block(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release Docker operation"
        if stage == "create":
            return original_create(*args, **kwargs)

    operation = {
        "create": fake_docker.client.containers.create,
        "start": fake_docker.container.start,
        "readiness": fake_docker.container.reload,
    }[stage]
    operation.side_effect = block
    executor = TrackedDockerCommandLineCodeExecutor(working_directory=tmp_path)

    async def run():
        task = asyncio.create_task(executor.start())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not list(tmp_path.iterdir())
        assert executor._container is None

    asyncio.run(run())
    fake_docker.container.remove.assert_called_once_with(force=True)


def test_cancelled_execution_removes_container_and_drains_command(fake_docker, tmp_path):
    entered, release = threading.Event(), threading.Event()
    tracker = Mock()

    def execute(command):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(exit_code=137, output=b"terminated")

    fake_docker.container.exec_run.side_effect = execute
    fake_docker.container.remove.side_effect = lambda **kwargs: release.set()
    executor = TrackedDockerCommandLineCodeExecutor(
        working_directory=tmp_path, tool_call_tracker=tracker
    )

    async def run():
        await executor.start()
        token = CancellationToken()
        task = asyncio.create_task(
            executor.execute_code_blocks([CodeBlock(code="print(1)", language="python")], token)
        )
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            token.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await executor.stop()
        assert not executor._command_tasks
        assert not list(tmp_path.iterdir())
        assert "cancelled" in tracker.call_args.kwargs["result"]

    asyncio.run(run())


def test_cancelled_cleanup_finishes_before_propagating(fake_docker, tmp_path):
    entered, release = threading.Event(), threading.Event()

    def remove(**kwargs):
        entered.set()
        assert release.wait(5)

    fake_docker.container.remove.side_effect = remove
    executor = TrackedDockerCommandLineCodeExecutor(working_directory=tmp_path)

    async def run():
        await executor.start()
        task = asyncio.create_task(executor.stop())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not list(tmp_path.iterdir())
        await executor.stop()

    asyncio.run(run())


def test_workspace_cleanup_error_does_not_prevent_container_removal(fake_docker, monkeypatch):
    executor = TrackedDockerCommandLineCodeExecutor()

    async def run():
        await executor.start()
        cleanup = executor._temp_dir.cleanup
        monkeypatch.setattr(executor._temp_dir, "cleanup", Mock(side_effect=OSError("cleanup failed")))
        try:
            with pytest.raises(ExceptionGroup, match="resource cleanup failed"):
                await executor.stop()
            fake_docker.container.remove.assert_called_once_with(force=True)
        finally:
            monkeypatch.setattr(executor._temp_dir, "cleanup", cleanup)
            await executor.stop()

    asyncio.run(run())


def _make_team(monkeypatch, fake_docker, count=1, stream=None):
    monkeypatch.setattr(TeamAutogen, "get_client", lambda self: object())

    def instantiate_team(team):
        async def messages(task):
            assert len(team._code_executors) == count
            assert all(executor._running for executor in team._code_executors)
            yield SimpleNamespace(source="terminal", content="ok")

        team.team = SimpleNamespace(run_stream=stream or messages)

    monkeypatch.setattr(TeamAutogen, "instantiate_team", instantiate_team)
    return TeamAutogen(
        [
            Agent(AgentConfig(name=f"terminal_{index}", special_agent="computer_terminal"))
            for index in range(count)
        ],
        architecture="centralized",
    )


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_team_owns_executor_for_all_run_paths(fake_docker, monkeypatch, verbose, streaming):
    async def console(stream):
        async for message in stream:
            pass

    monkeypatch.setattr("autogen_agentchat.ui.Console", console)
    team = _make_team(monkeypatch, fake_docker)
    method = team.run_stream if streaming else team.run
    result = asyncio.run(method("fixed task", verbose=verbose))
    assert result["messages"] == [{"source": "terminal", "content": "ok"}]
    fake_docker.container.remove.assert_called_once_with(force=True)
    assert not team._code_executors[0]._running


@pytest.mark.parametrize("cancel", [False, True])
def test_team_rolls_back_all_executors_if_later_start_fails(fake_docker, monkeypatch, cancel):
    team = _make_team(monkeypatch, fake_docker, count=2)
    events = []
    for index, executor in enumerate(team._code_executors):
        executor.start = AsyncMock(side_effect=lambda index=index: events.append(f"start{index}"))
        executor.stop = AsyncMock(side_effect=lambda index=index: events.append(f"stop{index}"))
    error = asyncio.CancelledError() if cancel else RuntimeError("startup failed")

    async def fail():
        events.append("start1")
        raise error

    team._code_executors[1].start.side_effect = fail
    with pytest.raises(type(error)):
        asyncio.run(team.run("fixed task"))
    assert events == ["start0", "start1", "stop1", "stop0"]


def test_team_attempts_all_cleanup_when_one_stop_fails(fake_docker, monkeypatch):
    team = _make_team(monkeypatch, fake_docker, count=2)
    for executor in team._code_executors:
        executor.start = AsyncMock()
        executor.stop = AsyncMock()
    team._code_executors[1].stop.side_effect = RuntimeError("stop failed")
    team._run = AsyncMock(return_value={})
    with pytest.raises(RuntimeError, match="stop failed"):
        asyncio.run(team.run("fixed task"))
    for executor in team._code_executors:
        executor.stop.assert_awaited_once()


def test_team_cancellation_closes_executor(fake_docker, monkeypatch):
    async def run():
        entered = asyncio.Event()

        async def messages(task):
            entered.set()
            await asyncio.Event().wait()
            yield

        team = _make_team(monkeypatch, fake_docker, stream=messages)
        task = asyncio.create_task(team.run("fixed task"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not team._code_executors[0]._running

    asyncio.run(run())
    fake_docker.container.remove.assert_called_once_with(force=True)


def test_repeated_runs_get_fresh_workspaces(fake_docker, monkeypatch):
    team = _make_team(monkeypatch, fake_docker)

    async def run():
        await team.run("first fixed task")
        await team.run("second fixed task")

    asyncio.run(run())
    calls = fake_docker.client.containers.create.call_args_list
    workspaces = [Path(next(iter(call.kwargs["volumes"]))) for call in calls]
    assert len(set(workspaces)) == 2
    assert all(not path.exists() for path in workspaces)
    assert calls[0].kwargs["name"] != calls[1].kwargs["name"]
