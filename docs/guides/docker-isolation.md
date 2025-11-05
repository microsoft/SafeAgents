# Running SafeAgents in Docker

Run your agent tasks in isolated Docker containers for enhanced safety and reproducibility.

## Why Use Docker?

✅ **Isolation**: Tasks run in complete isolation from your system
✅ **Safety**: Harmful actions are contained within the container
✅ **Reproducibility**: Same environment every time
✅ **Experimentation**: Test risky behaviors safely

## Quick Start

### 1. Create Your Task Script

Create `my_task.py`:

```python
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from safeagents import Team, Agent, AgentConfig, tool

load_dotenv()

@tool()
def search_tool(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

# Create agent
agent = Agent(config=AgentConfig(
    name="SearchAgent",
    tools=[search_tool],
    system_message="You help users search for information."
))

# Create team
team = Team.create(
    agents=[agent],
    framework="openai-agents",
    architecture="centralized"
)

# Run task
result = asyncio.run(team.run(
    "Search for information about Python",
    verbose=False
))

# Save results to /SafeAgents/logs/ so they persist
output_dir = Path("/SafeAgents/logs")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "result.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"✅ Task completed: {result['stop_reason']}")
```

**Important**: Save results to `/SafeAgents/logs/` directory - this is mounted and accessible after the container exits.

### 2. Run in Docker

Create `run_docker.py`:

```python
from pathlib import Path
from safeagents.runtime.docker_runner import DockerRunner

# Initialize runner
runner = DockerRunner(
    framework="openai-agents",
    log_output_dir=Path("./docker_logs")
)

# Prepare and build (only needed once)
print("Preparing environment...")
runner.prepare_environment()

print("Building Docker image...")
runner.build_image()

# Run your task
print("Running task...")
result = runner.run_script(
    script_path=Path("my_task.py"),
    timeout=300
)

# Check results
if result['success']:
    print(f"✅ Success! Logs at: {result['log_dir']}")

    # Access saved results
    result_file = Path(result['log_dir']) / "result.json"
    if result_file.exists():
        print(f"📄 Results saved to: {result_file}")
else:
    print(f"❌ Failed: {result.get('error')}")
    print(f"📋 Check logs at: {result['log_dir']}")
```

### 3. Run It

```bash
python run_docker.py
```

That's it! Your task runs safely in Docker.

## Saving Results

**Critical**: To access results after the container exits, save to `/SafeAgents/logs/`:

```python
from pathlib import Path
import json

# This directory is mounted - files persist
output_dir = Path("/SafeAgents/logs")
output_dir.mkdir(parents=True, exist_ok=True)

# Save your results
with open(output_dir / "my_results.json", 'w') as f:
    json.dump(data, f)

# Also works for any file type
with open(output_dir / "output.txt", 'w') as f:
    f.write("Task completed successfully")
```

After the script runs, find your files at:
```
./docker_logs/run_my_task_xxx/my_results.json
./docker_logs/run_my_task_xxx/output.txt
```

## Configuration Options

### Custom Log Directory

```python
runner = DockerRunner(
    framework="openai-agents",
    log_output_dir=Path("./my_logs")  # Custom location
)
```

### Additional Python Packages

```python
runner = DockerRunner(
    framework="openai-agents",
    additional_packages=["pandas", "numpy", "requests"]
)
```

### Timeout

```python
result = runner.run_script(
    script_path=Path("my_task.py"),
    timeout=600  # 10 minutes
)
```

### Environment Variables

```python
result = runner.run_script(
    script_path=Path("my_task.py"),
    environment_vars={
        "DEBUG": "true",
        "LOG_LEVEL": "INFO"
    }
)
```

### Mount .env File

```python
result = runner.run_script(
    script_path=Path("my_task.py"),
    mount_env_file=True,  # Default
    env_file_path=Path(".env")  # Optional custom path
)
```


## Complete Example

```python
from pathlib import Path
from safeagents.runtime.docker_runner import DockerRunner

# Setup
runner = DockerRunner(
    framework="openai-agents",
    log_output_dir=Path("./docker_logs"),
    image_name="my-safeagents",
    additional_packages=["requests"]
)

# Build once
runner.prepare_environment()
runner.build_image()

# Run multiple tasks
tasks = ["task1.py", "task2.py", "task3.py"]

for task in tasks:
    print(f"\n▶️  Running {task}...")

    result = runner.run_script(
        script_path=Path(task),
        timeout=300,
        environment_vars={"TASK_ID": task}
    )

    if result['success']:
        print(f"   ✅ Completed")
    else:
        print(f"   ❌ Failed: {result.get('error')}")
```

## Viewing Logs

Each run creates a unique log directory:

```
docker_logs/
└── run_my_task_abc123/
    ├── console.log      # Full console output
    ├── stdout.log       # STDOUT only
    ├── stderr.log       # STDERR only
    └── result.json      # Your saved results
```

Access logs:

```bash
# View console output
cat docker_logs/run_my_task_*/console.log

# View your results
cat docker_logs/run_my_task_*/result.json

# List all log directories
ls -lah docker_logs/
```

## Best Practices

### 1. Build Once, Run Many

```python
# Build image once
runner.prepare_environment()
runner.build_image()

# Run many tasks (reuses image)
for task in tasks:
    runner.run_script(script_path=Path(task))
```

### 2. Always Save to /SafeAgents/logs/

```python
# ✅ Good: Files persist
Path("/SafeAgents/logs/result.json").write_text(data)

# ❌ Bad: Files disappear when container exits
Path("/tmp/result.json").write_text(data)
```

### 3. Set Reasonable Timeouts

```python
# ✅ Good: Prevents hanging
result = runner.run_script(script_path=Path("task.py"), timeout=300)

# ❌ Risky: Could hang forever
result = runner.run_script(script_path=Path("task.py"))  # No timeout
```

### 4. Test Locally First

```python
# 1. Test your script locally
python my_task.py

# 2. Then run in Docker
runner.run_script(script_path=Path("my_task.py"))
```

## Troubleshooting

### Build Fails

Try building without cache:

```python
runner.build_image(no_cache=True)
```

### Task Times Out

Increase timeout:

```python
result = runner.run_script(
    script_path=Path("task.py"),
    timeout=1800  # 30 minutes
)
```

### Results Not Found

Ensure you're saving to `/SafeAgents/logs/`:

```python
# Check your script saves to this directory
output_dir = Path("/SafeAgents/logs")
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "results.json", 'w') as f:
    json.dump(data, f)
```

### Environment Variables Missing

Mount .env file:

```python
result = runner.run_script(
    script_path=Path("task.py"),
    mount_env_file=True  # Make sure this is True
)
```

## See Also

- [Example: docker_task.py](../../example_scripts/docker_task.py) - Complete working example
- [Example: docker_examples.py](../../example_scripts/docker_examples.py) - Multiple use cases
- [Getting Started](../getting-started/installation.md) - Installation guide
- [First Agent](../getting-started/first-agent.md) - Creating agents

---

**Need help?** Check [example_scripts/docker_examples.py](../../example_scripts/docker_examples.py) for more examples.
