"""
Docker Isolation Example

Demonstrates running SafeAgents tasks in isolated Docker containers.

Key Concepts:
- DockerRunner for container execution
- Saving results to /SafeAgents/logs/
- Multiple task execution
- Log collection and analysis

Usage:
    python 05_docker_isolation.py              # Run all examples
    python 05_docker_isolation.py --example 1  # Run specific example
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safeagents import Team, Agent, AgentConfig, tool
from safeagents.runtime.docker_runner import DockerRunner


def create_task_script(script_path: Path, task_query: str):
    """
    Create a task script that will run in Docker.

    Args:
        script_path: Where to save the script
        task_query: Task for the agent to perform
    """
    script_content = f'''"""
Task script to run in Docker container.
"""
import ast
import asyncio
import json
import math
import operator
from pathlib import Path
from dotenv import load_dotenv
from safeagents import Team, Agent, AgentConfig, tool

TASK_QUERY = {task_query!r}


def _evaluate_expression(expression: str) -> int | float:
    if len(expression) > 1024:
        raise ValueError("Expression is too long")
    tree = ast.parse(expression.strip(), mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression is too complex")
    operators = {{
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }}

    def evaluate(node, depth=0):
        if depth > 16:
            raise ValueError("Expression is too deeply nested")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = evaluate(node.operand, depth + 1)
            value = operand if isinstance(node.op, ast.UAdd) else -operand
        elif isinstance(node, ast.BinOp) and type(node.op) in operators:
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 1000:
                    raise ValueError("Exponent is too large")
                if isinstance(left, int) and isinstance(right, int) and right >= 0:
                    if left.bit_length() * right > 4096:
                        raise ValueError("Result is too large")
            value = operators[type(node.op)](left, right)
        else:
            raise ValueError("Only numeric arithmetic is allowed")
        if type(value) not in (int, float):
            raise ValueError("Only real numbers are allowed")
        if isinstance(value, int) and value.bit_length() > 4096:
            raise ValueError("Result is too large")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Result must be finite")
        return value

    return evaluate(tree.body)


@tool()
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Search results for: {{query}}"

@tool()
def calculate_tool(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = _evaluate_expression(expression)
        return f"Result: {{result}}"
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as e:
        return f"Error: {{str(e)}}"


async def main():
    load_dotenv()
    search_agent = Agent(config=AgentConfig(
        name="SearchAgent",
        tools=[search_tool],
        system_message="You help search for information.",
        description="Agent that searches for information"
    ))

    calc_agent = Agent(config=AgentConfig(
        name="CalcAgent",
        tools=[calculate_tool],
        system_message="You help with calculations.",
        description="Agent that performs calculations"
    ))

    team = Team.create(
        agents=[search_agent, calc_agent],
        framework="openai-agents",
        architecture="centralized",
        max_turns=10
    )

    print("Running task in Docker...")
    result = await team.run(TASK_QUERY, verbose=False)

    # Save results in the mounted logs directory so they survive container removal.
    output_dir = Path("/SafeAgents/logs")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "task_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Task: {{TASK_QUERY}}\\n")
        f.write(f"Stop Reason: {{result['stop_reason']}}\\n")
        f.write(f"Messages: {{len(result['messages'])}}\\n")

    print(f"✅ Task completed: {{result['stop_reason']}}")
    print(f"📄 Results saved to {{result_file}}")


if __name__ == "__main__":
    asyncio.run(main())
'''

    script_path.write_text(script_content)
    print(f"✅ Created task script: {script_path}")


def example_1_basic_docker():
    """Example 1: Basic Docker execution"""
    print("\n" + "=" * 80)
    print("Example 1: Basic Docker Execution")
    print("=" * 80)

    # Create task script
    script_path = Path("docker_task_basic.py")
    create_task_script(
        script_path,
        "Search for Python programming best practices"
    )

    try:
        # Initialize runner
        runner = DockerRunner(
            framework="openai-agents",
            log_output_dir=Path("./docker_logs")
        )

        # Prepare and build
        print("\n📦 Preparing environment...")
        runner.prepare_environment()

        print("🔨 Building Docker image (this may take a few minutes)...")
        runner.build_image()

        # Run task
        print("🚀 Running task in Docker...")
        result = runner.run_script(
            script_path=script_path,
            timeout=300
        )

        # Show results
        print("\n" + "-" * 80)
        if result['success']:
            print("✅ Task completed successfully!")
            print(f"📁 Logs directory: {result['log_dir']}")

            # Show saved files
            log_dir = Path(result['log_dir'])
            print(f"\n📄 Saved files:")
            for file in log_dir.iterdir():
                print(f"   - {file.name} ({file.stat().st_size} bytes)")

            # Read and display results
            result_file = log_dir / "task_result.json"
            if result_file.exists():
                with open(result_file) as f:
                    task_result = json.load(f)
                print(f"\n📊 Stop Reason: {task_result['stop_reason']}")
                print(f"💬 Messages: {len(task_result['messages'])}")
        else:
            print(f"❌ Task failed: {result.get('error')}")
            print(f"📋 Check logs at: {result['log_dir']}")

    finally:
        # Cleanup
        if script_path.exists():
            script_path.unlink()


def example_2_multiple_tasks():
    """Example 2: Running multiple tasks with same image"""
    print("\n" + "=" * 80)
    print("Example 2: Multiple Tasks with Same Image")
    print("=" * 80)

    # Create multiple task scripts
    tasks = [
        ("docker_task_1.py", "Search for information about AI safety"),
        ("docker_task_2.py", "Calculate: 123 + 456"),
        ("docker_task_3.py", "Search for Python libraries"),
    ]

    scripts = []
    for script_name, query in tasks:
        script_path = Path(script_name)
        create_task_script(script_path, query)
        scripts.append(script_path)

    try:
        # Initialize runner
        runner = DockerRunner(
            framework="openai-agents",
            log_output_dir=Path("./docker_logs")
        )

        # Build once
        print("\n📦 Preparing environment...")
        runner.prepare_environment()

        print("🔨 Building Docker image...")
        runner.build_image()

        # Run all tasks
        results = []
        for script_path in scripts:
            print(f"\n🚀 Running {script_path.name}...")
            result = runner.run_script(
                script_path=script_path,
                timeout=300
            )

            results.append({
                'script': script_path.name,
                'success': result['success'],
                'log_dir': result.get('log_dir')
            })

            status = "✅" if result['success'] else "❌"
            print(f"   {status} Completed")

        # Summary
        print("\n" + "-" * 80)
        print("📊 Summary:")
        successful = sum(1 for r in results if r['success'])
        print(f"   Total tasks: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")

    finally:
        # Cleanup
        for script_path in scripts:
            if script_path.exists():
                script_path.unlink()


def example_3_custom_packages():
    """Example 3: Using additional packages"""
    print("\n" + "=" * 80)
    print("Example 3: Custom Packages in Docker")
    print("=" * 80)

    # Create task that uses pandas
    script_path = Path("docker_task_pandas.py")
    script_content = '''"""
Task using pandas (additional package).
"""
import asyncio
import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from safeagents import Team, Agent, AgentConfig, tool

load_dotenv()

@tool()
def analyze_data() -> str:
    """Analyze some data using pandas."""
    # Create sample data
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    return f"Data shape: {df.shape}, Sum: {df.sum().sum()}"

agent = Agent(config=AgentConfig(
    name="DataAgent",
    tools=[analyze_data],
    system_message="You analyze data."
))

team = Team.create(
    agents=[agent],
    framework="openai-agents",
    architecture="centralized"
)

result = asyncio.run(team.run(
    "Analyze the data",
    verbose=False
))

# Save results
output_dir = Path("/SafeAgents/logs")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "pandas_result.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"✅ Task completed with pandas!")
'''

    script_path.write_text(script_content)
    print(f"✅ Created task script: {script_path}")

    try:
        # Initialize runner with additional packages
        runner = DockerRunner(
            framework="openai-agents",
            log_output_dir=Path("./docker_logs"),
            additional_packages=["pandas"]  # Add pandas
        )

        print("\n📦 Preparing environment with pandas...")
        runner.prepare_environment()

        print("🔨 Building Docker image with custom packages...")
        runner.build_image()

        print("🚀 Running task...")
        result = runner.run_script(
            script_path=script_path,
            timeout=300
        )

        print("\n" + "-" * 80)
        if result['success']:
            print("✅ Task with pandas completed successfully!")
            print(f"📁 Logs: {result['log_dir']}")
        else:
            print(f"❌ Task failed: {result.get('error')}")

    finally:
        if script_path.exists():
            script_path.unlink()


def main():
    """Run examples"""
    examples = {
        "1": ("Basic Docker Execution", example_1_basic_docker),
        "2": ("Multiple Tasks", example_2_multiple_tasks),
        "3": ("Custom Packages", example_3_custom_packages),
    }

    # Check for specific example
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        example_num = sys.argv[1]
        _, example_func = examples[example_num]
        example_func()
    else:
        # Run all examples
        print("Running all Docker isolation examples...")
        print("=" * 80)

        for num, (name, func) in examples.items():
            print(f"\n▶️  Example {num}: {name}")
            func()

        print("\n" + "=" * 80)
        print("All examples completed!")
        print("\n💡 Tip: Run specific example with: python 05_docker_isolation.py --example 1")


if __name__ == "__main__":
    import sys

    # Parse arguments
    if "--example" in sys.argv:
        idx = sys.argv.index("--example")
        if idx + 1 < len(sys.argv):
            sys.argv = [sys.argv[0], sys.argv[idx + 1]]

    main()
