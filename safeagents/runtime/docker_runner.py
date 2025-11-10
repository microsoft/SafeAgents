import subprocess
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from safeagents.core.logger import logger
from safeagents.runtime.utils.runtime_build import (
    generate_dockerfile,
    build_docker_image,
    run_docker_container,
    prep_build_folder
)


class DockerRunner:
    """
    Run SafeAgents tasks inside Docker containers with log collection.
    """

    def __init__(
        self,
        framework: str,
        build_folder: Optional[Path] = None,
        log_output_dir: Optional[Path] = None,
        image_name: str = "safeagents-runtime",
        python_version: str = "3.11",
        additional_packages: Optional[list] = None
    ):
        """
        Initialize Docker runner.

        Args:
            framework: Framework to use (e.g., 'magentic', 'openai-agents', 'langgraph')
            build_folder: Path to build folder (default: ./build_folder)
            log_output_dir: Directory where logs will be copied (default: ./logs)
            image_name: Name for the Docker image
            python_version: Python version to use
            additional_packages: Additional pip packages to install
        """
        self.framework = framework
        # Ensure paths are absolute for Docker volume mounts
        self.build_folder = (build_folder or Path.cwd() / "build_folder").resolve()
        self.log_output_dir = (log_output_dir or Path.cwd() / "logs").resolve()
        self.image_name = image_name
        self.python_version = python_version
        self.additional_packages = additional_packages or []

        # Create directories if they don't exist
        self.build_folder.mkdir(parents=True, exist_ok=True)
        self.log_output_dir.mkdir(parents=True, exist_ok=True)

        # Container paths - match Dockerfile structure (/SafeAgents/)
        self.container_log_dir = "/SafeAgents/logs"
        self.container_scripts_dir = "/SafeAgents/scripts"

    def prepare_environment(self) -> bool:
        """
        Prepare the Docker environment: copy code and generate Dockerfile.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare build folder with safeagents source code
            logger.info("Preparing build folder with source code...")
            prep_build_folder(self.build_folder)

            # Generate Dockerfile for the specified framework
            logger.info(f"Generating Dockerfile for framework: {self.framework}")
            generate_dockerfile(
                build_folder=self.build_folder,
                framework=self.framework,
                additional_packages=self.additional_packages,
                python_version=self.python_version
            )

            return True

        except Exception as e:
            logger.error(f"Failed to prepare environment: {e}")
            return False

    def build_image(self, no_cache: bool = False) -> bool:
        """
        Build the Docker image.

        Args:
            no_cache: Whether to build without cache

        Returns:
            bool: True if build succeeded, False otherwise
        """
        logger.info(f"Building Docker image: {self.image_name}")
        return build_docker_image(
            build_path=self.build_folder,
            image_name=self.image_name,
            no_cache=no_cache
        )

    def run_script(
        self,
        script_path: Path,
        environment_vars: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        bind_azure_folder: bool = True,
        mount_env_file: bool = True,
        env_file_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run a Python script inside the Docker container and collect logs.

        Args:
            script_path: Path to the Python script to run
            environment_vars: Environment variables to pass to the container
            timeout: Timeout in seconds (None for no timeout)
            bind_azure_folder: Whether to bind mount ~/.azure folder
            mount_env_file: Whether to mount .env file into container (default: True)
            env_file_path: Path to .env file (default: ./.env in current directory)

        Returns:
            dict: Results including stdout, stderr, exit_code, and log_path
        """
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return {
                "success": False,
                "error": f"Script not found: {script_path}"
            }

        # Create a unique run ID for this execution
        run_id = f"run_{script_path.stem}_{uuid.uuid4().hex[:8]}"
        run_log_dir = self.log_output_dir / run_id

        try:
            # Create log directory for this run
            run_log_dir.mkdir(parents=True, exist_ok=True)

            # Create scripts directory in build folder if it doesn't exist
            scripts_dir = self.build_folder / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)

            # Copy the script to build folder's scripts directory
            script_name = script_path.name
            container_script_path = f"{self.container_scripts_dir}/{script_name}"
            host_script_path = scripts_dir / script_name
            shutil.copy(script_path, host_script_path)

            logger.info(f"Running script '{script_name}' in Docker container...")

            # Prepare volumes - mount scripts folder and log directory
            volumes = {
                str(scripts_dir): self.container_scripts_dir,
                str(run_log_dir): self.container_log_dir
            }

            # Mount .env file if requested
            if mount_env_file:
                # Use provided path or default to .env in current directory
                env_path = (env_file_path or Path.cwd() / ".env").resolve()
                if env_path.exists():
                    volumes[str(env_path)] = "/SafeAgents/.env"
                    logger.info(f"Mounting .env file from: {env_path}")
                else:
                    logger.warning(f".env file not found at: {env_path}, skipping mount")

            # Prepare environment variables
            env_vars = environment_vars or {}
            # Set log file path inside container
            env_vars["LOG_FILE"] = f"{self.container_log_dir}/output.log"

            # Build command to run the script and capture output
            command = f"python {container_script_path} 2>&1 | tee {self.container_log_dir}/console.log"

            # Run container with bash to execute the command
            cmd = ["docker", "run", "--rm"]

            # Add Azure folder binding
            if bind_azure_folder:
                azure_folder = Path.home() / ".azure"
                if azure_folder.exists():
                    cmd.extend(["-v", f"{azure_folder}:/home/agentuser/.azure:rw"])

            # Add volume mounts
            for host_path, container_path in volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])

            # Add environment variables
            for key, value in env_vars.items():
                cmd.extend(["-e", f"{key}={value}"])

            cmd.append(self.image_name)
            cmd.extend(["bash", "-c", command])

            logger.debug(f"Running command: {' '.join(cmd)}")

            # Run the container
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Prepare result
            execution_result = {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "log_dir": str(run_log_dir),
                "run_id": run_id
            }

            # Save stdout and stderr to files
            (run_log_dir / "stdout.log").write_text(result.stdout)
            (run_log_dir / "stderr.log").write_text(result.stderr)

            # Log summary
            if result.returncode == 0:
                logger.info(f"Script completed successfully. Logs saved to: {run_log_dir}")
            else:
                logger.error(f"Script failed with exit code {result.returncode}. Logs saved to: {run_log_dir}")

            return execution_result

        except subprocess.TimeoutExpired:
            logger.error(f"Script execution timed out after {timeout} seconds")
            return {
                "success": False,
                "error": f"Timeout after {timeout} seconds",
                "log_dir": str(run_log_dir),
                "run_id": run_id
            }
        except Exception as e:
            logger.error(f"Error running script: {e}")
            return {
                "success": False,
                "error": str(e),
                "log_dir": str(run_log_dir) if run_log_dir.exists() else None,
                "run_id": run_id
            }

    def run_task_interactive(
        self,
        task_query: str,
        environment_vars: Optional[Dict[str, str]] = None,
        mount_env_file: bool = True,
        env_file_path: Optional[Path] = None
    ) -> str:
        """
        Start an interactive container for running tasks.

        Args:
            task_query: The task query to pass to the agent
            environment_vars: Environment variables to pass to the container
            mount_env_file: Whether to mount .env file into container (default: True)
            env_file_path: Path to .env file (default: ./.env in current directory)

        Returns:
            str: Container ID
        """
        # Prepare volumes
        scripts_dir = self.build_folder / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)

        volumes = {
            str(scripts_dir): self.container_scripts_dir,
            str(self.log_output_dir): self.container_log_dir
        }

        # Mount .env file if requested
        if mount_env_file:
            # Use provided path or default to .env in current directory
            env_path = (env_file_path or Path.cwd() / ".env").resolve()
            if env_path.exists():
                volumes[str(env_path)] = "/SafeAgents/.env"
                logger.info(f"Mounting .env file from: {env_path}")
            else:
                logger.warning(f".env file not found at: {env_path}, skipping mount")

        # Prepare environment variables
        env_vars = environment_vars or {}
        env_vars["TASK_QUERY"] = task_query

        container_id = run_docker_container(
            image_name=self.image_name,
            container_name=f"safeagents-{self.framework}",
            detach=True,
            remove=False,
            environment_vars=env_vars,
            volumes=volumes,
            bind_azure_folder=True
        )

        if container_id:
            logger.info(f"Started interactive container: {container_id}")
            logger.info(f"Logs will be available at: {self.log_output_dir}")

        return container_id

    def cleanup(self):
        """Clean up Docker resources."""
        logger.info("Cleaning up Docker resources...")
        # Could add image removal, container cleanup, etc.
        pass
