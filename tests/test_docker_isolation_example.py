import asyncio
import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def generate_script(tmp_path):
    example = _load_module(
        Path(__file__).parents[1] / "docs/examples/05_docker_isolation.py", "docker_example"
    )

    def generate(query="Calculate: 123 + 456"):
        path = tmp_path / "generated_task.py"
        example.create_task_script(path, query)
        return _load_module(path, "generated_task")

    return generate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("123 + 456", "579"),
        ("2 * (3 + 4)", "14"),
        (" -3.5 / 2 ", "-1.75"),
        ("7 // 2 + 7 % 2", "4"),
        ("2 ** 10", "1024"),
        ("+4 - -2", "6"),
        ("1e2 + .5", "100.5"),
    ],
)
def test_generated_calculator_accepts_arithmetic(generate_script, expression, expected):
    module = generate_script()
    assert asyncio.run(module.calculate_tool.func(expression)) == f"Result: {expected}"


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",
        "open('/SafeAgents/.env').read()",
        "(1).__class__.__mro__",
        "unknown",
        "[1][0]",
        "(lambda: 1)()",
        "[x for x in range(3)]",
        "'hello'",
        "True + 1",
        "1 / 0",
        "1 +",
        "1e309",
        "(-1) ** .5",
        "2 ** (2 ** 1000)",
        "(9 ** 999) ** 999",
        "-" * 40 + "1",
        "1 + " * 200 + "1",
        "1" * 2048,
    ],
)
def test_generated_calculator_rejects_unsafe_or_excessive_input(generate_script, expression):
    module = generate_script()
    assert asyncio.run(module.calculate_tool.func(expression)).startswith("Error: ")


def test_rejected_expression_has_no_file_side_effect(generate_script, tmp_path):
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("unchanged")
    module = generate_script()
    expression = f"__import__('pathlib').Path({str(sentinel)!r}).write_text('changed')"
    assert asyncio.run(module.calculate_tool.func(expression)).startswith("Error: ")
    assert sentinel.read_text() == "unchanged"


def test_task_text_is_data_and_import_does_not_run_team(generate_script, tmp_path):
    sentinel = tmp_path / "injected"
    query = (
        'Quotes: "\' and triple quotes: """\n'
        f'"); __import__("pathlib").Path({str(sentinel)!r}).write_text("changed"); #'
    )
    module = generate_script(query)
    assert module.TASK_QUERY == query
    assert not sentinel.exists()


def test_generated_main_preserves_task_and_result_files(generate_script, tmp_path, monkeypatch):
    query = 'Calculate "2 + 2"\nand explain.'
    module = generate_script(query)
    result = {"stop_reason": "done", "messages": [{"content": "4"}]}
    fake_team = Mock(run=AsyncMock(return_value=result))
    create = Mock(return_value=fake_team)
    monkeypatch.setattr(module.Team, "create", create)
    monkeypatch.setattr(module, "load_dotenv", lambda: False)
    monkeypatch.setattr(module, "Path", lambda path: tmp_path / "logs")

    asyncio.run(module.main())

    fake_team.run.assert_awaited_once_with(query, verbose=False)
    assert create.call_args.kwargs["framework"] == "openai-agents"
    assert json.loads((tmp_path / "logs/task_result.json").read_text()) == result
    assert (tmp_path / "logs/summary.txt").read_text().startswith(f"Task: {query}\n")
