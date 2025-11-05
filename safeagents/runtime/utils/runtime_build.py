import subprocess
import shutil
import logging
import os
import safeagents
from pathlib import Path
from typing import Optional, List
from safeagents.core.logger import logger

def build_docker_image(
    build_path: str | Path,
    image_name: str,
    dockerfile_name: str = "Dockerfile",
    build_args: Optional[dict] = None,
    no_cache: bool = False
) -> bool:
    """
    Build a Docker image from a given path.

    Args:
        build_path: Path to the directory containing the Dockerfile
        image_name: Name and optionally tag for the built image (e.g., 'myapp:latest')
        dockerfile_name: Name of the Dockerfile (default: 'Dockerfile')
        build_args: Optional dictionary of build arguments
        no_cache: Whether to build without using cache

    Returns:
        bool: True if build succeeded, False otherwise
    """
    build_path = Path(build_path)

    if not _validate_build_path(build_path, dockerfile_name):
        logger.debug(f"Validation failed for build path: {build_path} with Dockerfile: {dockerfile_name}")
        return False

    cmd = ["docker", "build", "-t", image_name]

    if no_cache:
        cmd.append("--no-cache")

    # Prepare build args with host UID/GID for proper file permissions
    final_build_args = {
        "HOST_UID": str(os.getuid()),
        "HOST_GID": str(os.getgid())
    }

    # Merge with user-provided build args (user args take precedence)
    if build_args:
        final_build_args.update(build_args)

    for key, value in final_build_args.items():
        cmd.extend(["--build-arg", f"{key}={value}"])

    cmd.extend(["-f", str(build_path / dockerfile_name), str(build_path)])
    
    try:
        logger.info(f"Building Docker image '{image_name}' from path: {build_path}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Successfully built Docker image: {image_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker build failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during Docker build: {str(e)}")
        return False

def _validate_build_path(build_path: Path, dockerfile_name: str) -> bool:
    """
    Validate that the build path exists and contains a Dockerfile.
    
    Args:
        build_path: Path to validate
        dockerfile_name: Name of the Dockerfile to look for
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not build_path.exists():
        logger.error(f"Build path does not exist: {build_path}")
        return False
    
    if not build_path.is_dir():
        logger.error(f"Build path is not a directory: {build_path}")
        return False
    
    dockerfile_path = build_path / dockerfile_name
    if not dockerfile_path.exists():
        logger.error(f"Dockerfile not found: {dockerfile_path}")
        return False
    
    return True


# Note: We no longer need to manually define packages here.
# All dependencies are managed through requirements.txt which is copied
# to the build folder and used with pip install.


def generate_dockerfile(
    build_folder: Path,
    framework: Optional[str] = None,
    additional_packages: Optional[List[str]] = None,
    python_version: str = "3.12",
    base_image: str = "continuumio/miniconda3"
) -> Path:
    """
    Generate a Dockerfile using conda for Python and pip for packages.

    Uses requirements.txt for dependency management instead of environment.yml.
    This is more reliable and avoids conda dependency resolution issues.
    Layers are ordered from least to most frequently changing for optimal caching.

    Args:
        build_folder: Path to the build folder where Dockerfile will be created
        framework: Framework name (ignored - all frameworks installed via requirements.txt)
                  Kept for backward compatibility
        additional_packages: Additional pip packages to install after requirements.txt
        python_version: Python version to install (default: "3.12")
        base_image: Base Docker image (default: 'continuumio/miniconda3')

    Returns:
        Path: Path to the generated Dockerfile
    """
    dockerfile_path = build_folder / "Dockerfile"

    if framework:
        logger.info(f"Framework '{framework}' will be installed via requirements.txt")

    # Start with base content - system dependencies (rarely change)
    dockerfile_content = f"""FROM {base_image}

# Set working directory
WORKDIR /SafeAgents

# Layer 1: Install system dependencies (rarely changes)
# Install prerequisites for Azure CLI and update package lists
RUN apt-get update && apt-get install -y \\
    ca-certificates \\
    curl \\
    apt-transport-https \\
    lsb-release \\
    gnupg \\
    && rm -rf /var/lib/apt/lists/*

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Layer 2: Create conda environment with Python {python_version}
RUN conda create -n safeagents python={python_version} -y && \\
    conda clean -afy

# Activate conda environment by default
ENV PATH=/opt/conda/envs/safeagents/bin:$PATH
ENV CONDA_DEFAULT_ENV=safeagents

# Layer 3: Copy requirements.txt (changes when dependencies change)
# Copying separately so changes to source code don't invalidate pip install
COPY requirements.txt /SafeAgents/requirements.txt

# Layer 4: Install Python packages from requirements.txt
RUN pip install --no-cache-dir -r /SafeAgents/requirements.txt

"""

    # Layer 5: Install Playwright (must be done as root, before user switch)
    # Separate layer so it's cached unless playwright version changes
    dockerfile_content += """# Layer 5: Install Playwright browsers with system dependencies
# This must be done as root before switching to non-root user
RUN playwright install --with-deps chromium

"""

    # Layer 6: Additional packages (most frequently changes)
    if additional_packages:
        logger.info(f"Adding additional packages: {additional_packages}")
        additional_packages_str = " ".join(additional_packages)
        dockerfile_content += f"""# Layer 6: Install additional packages (user-specified)
RUN pip install --no-cache-dir {additional_packages_str}

"""

    # Layer 7: Copy source code (changes most frequently)
    # This is done AFTER all package installs to maximize cache hits
    dockerfile_content += """# Layer 7: Copy safeagents source code
# Done last so code changes don't invalidate package installation cache
COPY ./code/safeagents /SafeAgents/safeagents

"""

    # Final layer: Environment setup and user creation
    dockerfile_content += """# Set up environment
ENV PYTHONPATH=/SafeAgents:$PYTHONPATH

# Create a non-root user with host UID for proper file permissions
# This ARG can be overridden at build time with --build-arg HOST_UID=$(id -u)
ARG HOST_UID=1000
ARG HOST_GID=1000

# Create group if it doesn't exist, otherwise use existing group
RUN if ! getent group ${HOST_GID} >/dev/null; then \\
        groupadd -g ${HOST_GID} agentuser; \\
    fi && \\
    useradd -u ${HOST_UID} -g ${HOST_GID} -m -s /bin/bash agentuser && \\
    chown -R ${HOST_UID}:${HOST_GID} /SafeAgents

USER agentuser

# Set the default command
CMD ["/bin/bash"]
"""

    # Write the Dockerfile
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)

    logger.info(f"Generated Dockerfile at: {dockerfile_path}")
    logger.debug(f"Dockerfile content:\n{dockerfile_content}")

    return dockerfile_path


def run_docker_container(
    image_name: str,
    container_name: Optional[str] = None,
    detach: bool = False,
    remove: bool = True,
    environment_vars: Optional[dict] = None,
    volumes: Optional[dict] = None,
    bind_azure_folder: bool = False,
    command: Optional[str] = None
) -> Optional[str]:
    """
    Run a Docker container with specified configuration.

    Args:
        image_name: Name of the Docker image to run
        container_name: Optional name for the container
        detach: Run container in background and return container ID
        remove: Automatically remove container when it exits
        environment_vars: Dictionary of environment variables
        volumes: Dictionary mapping host paths to container paths
        bind_azure_folder: Whether to bind mount ~/.azure folder
        command: Optional command to run in the container

    Returns:
        str: Container ID if detached, None otherwise
    """
    cmd = ["docker", "run"]

    if detach:
        cmd.append("-d")

    if remove:
        cmd.append("--rm")

    if container_name:
        cmd.extend(["--name", container_name])

    # Add Azure folder binding
    if bind_azure_folder:
        azure_folder = Path.home() / ".azure"
        if azure_folder.exists():
            cmd.extend(["-v", f"{azure_folder}:/home/agentuser/.azure:rw"])

    # Add volume mounts
    if volumes:
        for host_path, container_path in volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

    # Add environment variables
    if environment_vars:
        for key, value in environment_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

    cmd.append(image_name)

    if command:
        cmd.extend(["bash", "-c", command])

    try:
        logger.info(f"Running Docker container from image: {image_name}")
        logger.debug(f"Docker command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        if detach:
            container_id = result.stdout.strip()
            logger.info(f"Container started with ID: {container_id}")
            return container_id
        else:
            logger.info("Container execution completed")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run Docker container: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running Docker container: {str(e)}")
        return None


def prep_build_folder(
    build_folder: Path,
) -> None:
    """
    Prepare the build folder by copying source code and requirements.txt.

    Args:
        build_folder: Path to the build folder
    """
    # Copy the source code to directory. It will end up in build_folder/code
    # If package is not found, build from source code
    safeagents_source_dir = Path(safeagents.__file__).parent
    project_root = safeagents_source_dir.parent
    logger.debug(f'Building source distribution using project root: {project_root}')

    # check if the source code already exists, if it already exists, remove it
    if (build_folder / 'code' / 'safeagents').exists():
        shutil.rmtree(build_folder / 'code' / 'safeagents')

    # Copy the 'safeagents' directory (Source code)
    shutil.copytree(
        safeagents_source_dir,
        Path(build_folder, 'code', 'safeagents'),
        ignore=shutil.ignore_patterns(
            '.*/',
            '__pycache__/',
            '*.pyc',
            '*.md',
        ),
    )

    # Copy requirements.txt to build folder
    requirements_txt = project_root / 'requirements.txt'
    if requirements_txt.exists():
        shutil.copy(requirements_txt, build_folder / 'requirements.txt')
        logger.info(f'Copied requirements.txt to build folder')
    else:
        logger.warning(f'requirements.txt not found at {requirements_txt}. Docker build may fail.')