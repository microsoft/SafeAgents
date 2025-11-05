# SafeAgents Docker Runtime

Run SafeAgents tasks inside isolated Docker containers with automatic log collection.

## Overview

The Docker runtime allows you to:
- Run agent scripts in isolated containers
- Automatically collect logs to a local directory
- Support multiple frameworks (magentic, openai-agents, autogen, langgraph, crewai)
- Mount Azure credentials for authentication
- Control execution with timeouts

## Quick Start

```python
from pathlib import Path
from safeagents.runtime.docker_runner import DockerRunner

# 1. Create a runner
runner = DockerRunner(
    framework="langgraph",
    log_output_dir=Path("./logs")
)

# 2. Prepare and build
runner.prepare_environment()
runner.build_image()

# 3. Run your script
result = runner.run_script(
    script_path=Path("my_agent_script.py"),
    environment_vars={"LOG_LEVEL": "DEBUG"}
)

# 4. Check results
print(f"Success: {result['success']}")
print(f"Logs: {result['log_dir']}")
```

## DockerRunner API

### Constructor

```python
DockerRunner(
    framework: str,                      # Required: framework name
    build_folder: Optional[Path] = None, # Default: ./build_folder
    log_output_dir: Optional[Path] = None, # Default: ./logs
    image_name: str = "safeagents-runtime",
    python_version: str = "3.11",
    additional_packages: Optional[list] = None
)
```

**Supported frameworks:**
- `magentic`
- `openai-agents`
- `autogen`
- `langgraph`
- `crewai`

### Methods

#### `prepare_environment()` → bool
Copies SafeAgents source code and generates a Dockerfile for the specified framework.

```python
runner.prepare_environment()
```

#### `build_image(no_cache: bool = False)` → bool
Builds the Docker image.

```python
runner.build_image(no_cache=True)  # Force rebuild
```

#### `run_script(...)` → dict
Runs a Python script inside a Docker container and collects logs.

```python
result = runner.run_script(
    script_path=Path("script.py"),
    environment_vars={"KEY": "value"},
    timeout=300,  # seconds
    bind_azure_folder=True
)
```

**Returns:**
```python
{
    "success": True/False,
    "exit_code": 0,
    "stdout": "...",
    "stderr": "...",
    "log_dir": "/path/to/logs/run_xxx",
    "run_id": "run_xxx"
}
```

**Log files created:**
- `console.log` - Combined stdout/stderr from script execution
- `stdout.log` - Standard output
- `stderr.log` - Standard error

## Examples

### Custom Log Location

```python
runner = DockerRunner(
    framework="langgraph",
    log_output_dir=Path("/tmp/my_logs")
)
```

### Additional Packages

```python
runner = DockerRunner(
    framework="langgraph",
    additional_packages=["pandas", "numpy", "requests"]
)
```

### Without Azure Folder

```python
result = runner.run_script(
    script_path=Path("script.py"),
    bind_azure_folder=False
)
```

### With Timeout

```python
result = runner.run_script(
    script_path=Path("script.py"),
    timeout=60  # Kill after 60 seconds
)
```

### Multiple Runs

```python
# Build once
runner = DockerRunner(framework="langgraph")
runner.prepare_environment()
runner.build_image()

# Run multiple times
for script in ["task1.py", "task2.py", "task3.py"]:
    result = runner.run_script(script_path=Path(script))
    print(f"{script}: {'✅' if result['success'] else '❌'}")
```

## How It Works

### 1. Environment Preparation
- Copies `safeagents` source code to `build_folder/code/safeagents`
- Generates a Dockerfile with framework-specific dependencies
- Creates log output directory

### 2. Docker Build
- Builds image from generated Dockerfile
- Installs framework dependencies (e.g., langgraph, openai)
- Sets up Python environment and creates non-root user

### 3. Script Execution
- Copies your script to the build folder
- Mounts necessary volumes:
  - Script location → `/safeagents/code`
  - Log directory → `/safeagents/logs`
  - Azure folder → `/root/.azure` (optional)
- Runs script with: `python script.py 2>&1 | tee /safeagents/logs/console.log`
- Collects all output to log files

### 4. Log Collection
Logs are automatically saved to `log_output_dir/run_<id>/`:
```
logs/
└── run_script_abc123/
    ├── console.log   # Combined output
    ├── stdout.log    # Standard output
    └── stderr.log    # Standard error
```

## Volume Mounts

By default, the following are mounted:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `build_folder/code` | `/safeagents/code` | Script and source code |
| `log_output_dir/run_*` | `/safeagents/logs` | Log collection |
| `~/.azure` | `/root/.azure` | Azure credentials (optional) |

## Environment Variables

Pass environment variables to the container:

```python
result = runner.run_script(
    script_path=Path("script.py"),
    environment_vars={
        "LOG_LEVEL": "DEBUG",
        "API_KEY": "secret",
        "CUSTOM_VAR": "value"
    }
)
```

## Troubleshooting

### Script fails to run
- Check `stderr.log` in the log directory
- Ensure all dependencies are installed (use `additional_packages`)
- Verify script path exists

### Timeout issues
- Increase timeout parameter
- Check if script has infinite loops

### Azure authentication fails
- Ensure `~/.azure` folder exists
- Check Azure credentials are valid
- Verify `bind_azure_folder=True`

### Docker build fails
- Check Docker is installed and running
- Verify network connectivity for package downloads
- Try `build_image(no_cache=True)` to force rebuild

## See Also

- `sample-agentharm-code/run_in_docker.py` - Complete example
- `sample-agentharm-code/docker_examples.py` - Multiple usage examples
- `runtime_build.py` - Low-level Docker build utilities
