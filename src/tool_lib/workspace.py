# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import subprocess
import os
import base64
from pathlib import Path
import shutil
from typing import Optional
from langchain_core.tools import tool, BaseTool

from .base import ToolProvider


def _nvidia_runtime_available() -> bool:
    """Check if NVIDIA Docker runtime is available."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return "nvidia" in result.stdout.lower()
    except Exception:
        return False


########################################################
# Tools
########################################################

class Workspace(ToolProvider):
    """A tool provider for running isolated workspace.

    Each workspace is a separate Docker container, with tools for file management and Python code execution.

    Can be used as a context manager for automatic cleanup:
        with Workspace("my_workspace") as ws:
            ws._run_python_code("print('Hello')")
    """

    # Default timeout values (in seconds)
    DEFAULT_EXEC_TIMEOUT = 60
    DEFAULT_PIP_TIMEOUT = 300
    DEFAULT_SCRIPT_TIMEOUT = 300

    def __init__(self,
                 workspace_id: str,
                 host_workspace_path: str = "workspaces",
                 docker_workspace_path: str = "/workspace",
                 parent_workspace_id: Optional[str] = None,
                 docker_image: str = "agent_container",
                 memory_limit: str = "16g",
                 pids_limit: int = 2048,
                 use_gpu: bool = True):
        """
        Initialize an isolated workspace for running tools in a Docker container.

        Args:
            workspace_id (str): Unique identifier for this workspace.
            host_workspace_path (str): Directory on the host where workspace files are stored (default: "workspaces").
            docker_workspace_path (str): Directory inside the Docker container mapped to the workspace (default: "/workspace").
            parent_workspace_id (str, optional): If provided, the new workspace is initialized by copying files from the parent workspace.
            docker_image (str): Docker image to use for the container (default: "agent_container").
            memory_limit (str): Memory limit for the container (default: "16g").
            pids_limit (int): Maximum number of processes in the container (default: 2048).
            use_gpu (bool): Whether to enable GPU access. Falls back to CPU if NVIDIA runtime unavailable (default: True).

        The isolated workspace ensures all operations are separated per workspace, supporting optional inheritance from a parent workspace.
        """

        self._workspace_id = workspace_id
        self._host_workspace_path = Path(host_workspace_path) / workspace_id
        self._docker_workspace_path = Path(docker_workspace_path)
        self._host_parent_workspace_path = Path(host_workspace_path) / parent_workspace_id if parent_workspace_id is not None else None
        self._docker_image = docker_image
        self._memory_limit = memory_limit
        self._pids_limit = pids_limit
        self._use_gpu = use_gpu and _nvidia_runtime_available()

        self._prepare_workspace()
        self._start_container()
        self._create_tools()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stops and cleans up the workspace."""
        self.stop_workspace(remove_host_workspace=False)
        return False  # Don't suppress exceptions

    def _prepare_workspace(self):
        """Prepare the workspace directory on the host.

        If a parent workspace ID is provided, copies the parent's directory as the initial workspace state.
        If no parent is provided, creates an empty workspace directory.

        Raises:
            ValueError: If the parent workspace does not exist, or if the target workspace already exists when inheriting.
        """

        # If the workspace is inherited from a parent workspace, copy the parent workspace directory to the new workspace path
        if self._host_parent_workspace_path is not None:
            if not self._host_parent_workspace_path.exists():
                raise ValueError(f"Parent workspace {self._host_parent_workspace_path} does not exist")
            if self._host_workspace_path.exists():
                raise ValueError(f"Workspace {self._workspace_id} already exists")

            # Recursively copy the parent workspace directory to the new workspace path
            shutil.copytree(self._host_parent_workspace_path, self._host_workspace_path)
        #
        else:
            os.makedirs(self._host_workspace_path, exist_ok=True)

    def _start_container(self):
        """Start the Docker container for the workspace."""
        self._container_name = f"workspace_{self._workspace_id}"

        # Build docker run command
        cmd = ["docker", "run", "-d"]

        # Add GPU support if enabled and available
        if self._use_gpu:
            cmd.extend(["--runtime=nvidia", "--gpus", "all", "-e", "NVIDIA_VISIBLE_DEVICES=all"])

        cmd.extend([
            "--name", self._container_name,
            "--user", f"{os.getuid()}:{os.getgid()}",  # Run as current user to avoid permission issues
            "-e", f"HOME={self._docker_workspace_path}",  # Set HOME to workspace for pip cache/user installs
            "-v", f"{self._host_workspace_path.resolve()}:{self._docker_workspace_path}:rw",
            "--security-opt=no-new-privileges",
            "--memory", self._memory_limit,
            "--pids-limit", str(self._pids_limit),
            self._docker_image, "sleep", "infinity"
        ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start container '{self._container_name}': {result.stderr.strip()}"
            )

    def _stop_container(self):
        """Stop and remove the Docker container for the workspace.
        """
        result = subprocess.run(
            ["docker", "rm", "-f", self._container_name],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to stop container '{self._container_name}': {result.stderr.strip()}"
            )

    def _exec(self, cmd: str, timeout: Optional[int] = None) -> tuple[bool, str]:
        """Execute a command in the container

        Args:
            cmd: The command to execute.
            timeout: The timeout in seconds (default: DEFAULT_EXEC_TIMEOUT).

        Returns:
            A tuple containing a boolean indicating success and the output of the command.
        """
        if timeout is None:
            timeout = self.DEFAULT_EXEC_TIMEOUT

        try:
            result = subprocess.run(
                ["docker", "exec", "-w", str(self._docker_workspace_path),
                 self._container_name, "bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds"

    def _install_package(self, package: str) -> str:
        """Installs a package using pip in the workspace docker container.
        If the package installation fails, this tools return the error message so you can fix it.

        Args:
            package: The package to install.

        Returns:
            The result of the package installation.

        Examples:
            install("numpy")  # Installs the numpy package
            install("pandas")  # Installs the pandas package
        """
        try:
            result = subprocess.run(
                ["docker", "exec", self._container_name, "pip", "install", "--user", package],
                capture_output=True,
                text=True,
                timeout=self.DEFAULT_PIP_TIMEOUT
            )
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            return f"Package {package} installed successfully"
        except subprocess.TimeoutExpired:
            return f"Error: Package installation timed out after {self.DEFAULT_PIP_TIMEOUT} seconds"
        except Exception as e:
            return f"Error: {str(e)}"

    def _run_python_code(self, code: str, timeout: Optional[int] = None) -> str:
        """Runs Python code in the workspace docker container.

        If the code execution is successful, returns the output of the code execution.
        If the code execution fails, returns the error message.

        Args:
            code: The Python code to run.
            timeout: The timeout in seconds (default: DEFAULT_SCRIPT_TIMEOUT).

        Returns:
            The result of the code execution.

        Examples:
            run_python_code("print('Hello, world!')")  # Runs the code and returns the output
        """
        if timeout is None:
            timeout = self.DEFAULT_SCRIPT_TIMEOUT
        try:
            result = subprocess.run(
                ["docker", "exec", "-w", str(self._docker_workspace_path),
                 self._container_name, "python", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode != 0:
                return f"Error:\n{result.stderr}"
            return result.stdout
        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out after {timeout} seconds"
        except Exception as e:
            return f"Error: {str(e)}"

    def _run_python_script(self, script_path: str, timeout: Optional[int] = None) -> str:
        """Runs a Python script from the workspace in the workspace docker container.

        If the script execution is successful, returns the output of the script execution.
        If the script execution fails, returns the error message.

        Args:
            script_path: The path to the Python script to run (relative to workspace).
            timeout: The timeout in seconds (default: DEFAULT_SCRIPT_TIMEOUT).
        Returns:
            The result of the script execution, or the error message if it fails.
        """
        if timeout is None:
            timeout = self.DEFAULT_SCRIPT_TIMEOUT

        try:
            script_full_path = self._docker_workspace_path / script_path
            result = subprocess.run(
                ["docker", "exec", "-w", str(self._docker_workspace_path),
                 self._container_name, "python", str(script_full_path)],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                return f"Error:\n{result.stderr}"
            return result.stdout

        except subprocess.TimeoutExpired:
            return f"Error: Script execution timed out after {timeout} seconds"
        except Exception as e:
            return f"Error: {str(e)}"

    def _read_file(self, path: str) -> str:
        """Read the contents of a file.

        Args:
            path: Path to the file, relative to the workspace directory.

        Returns:
            The file contents, or an error message if reading fails.

        Examples:
            read_file("main.py")
            read_file("src/utils.py")
        """
        try:
            abs_path = self._docker_workspace_path / path
            success, output = self._exec(f"cat '{abs_path}'")
            if not success:
                return f"Error reading file: {output}"
            return output
        except Exception as e:
            return f"Error: {str(e)}"

    def _write_file(self, path: str, content: str) -> str:
        """Write content to a file. Creates parent directories if needed.

        Args:
            path: Path to the file, relative to the workspace directory.
            content: The content to write to the file.

        Returns:
            Success message or error message if writing fails.

        Examples:
            write_file("main.py", "print('Hello, World!')")
        """
        try:
            abs_path = self._docker_workspace_path / path
            parent = abs_path.parent

            # Create parent directories
            self._exec(f"mkdir -p '{parent}'")

            # Encode content as base64 to handle special characters safely
            encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')

            # Write via docker exec using stdin to avoid command line length limits
            result = subprocess.run(
                ["docker", "exec", "-i", "-w", str(self._docker_workspace_path),
                 self._container_name, "bash", "-c", f"base64 -d > '{abs_path}'"],
                input=encoded,
                capture_output=True,
                text=True,
                timeout=self.DEFAULT_EXEC_TIMEOUT
            )

            if result.returncode != 0:
                return f"Error writing file: {result.stderr}"
            return f"Successfully wrote to {path}"

        except Exception as e:
            return f"Error: {str(e)}"

    def _write_file_binary(self, path: str, content: bytes) -> str:
        """Write binary content to a file. Creates parent directories if needed.

        Args:
            path: Path to the file, relative to the workspace directory.
            content: The binary content to write to the file.

        Returns:
            Success message or error message if writing fails.

        Examples:
            write_file_binary("data.pkl", pickle_bytes)
        """
        try:
            abs_path = self._docker_workspace_path / path
            parent = abs_path.parent

            # Create parent directories
            self._exec(f"mkdir -p '{parent}'")

            # Encode binary content as base64
            encoded = base64.b64encode(content).decode('ascii')

            # Write via docker exec using stdin to avoid command line length limits
            result = subprocess.run(
                ["docker", "exec", "-i", "-w", str(self._docker_workspace_path),
                 self._container_name, "bash", "-c", f"base64 -d > '{abs_path}'"],
                input=encoded,
                capture_output=True,
                text=True,
                timeout=self.DEFAULT_EXEC_TIMEOUT
            )

            if result.returncode != 0:
                return f"Error writing file: {result.stderr}"
            return f"Successfully wrote to {path}"

        except Exception as e:
            return f"Error: {str(e)}"

    def _list_dir(self, path: str = ".") -> str:
        """List contents of a directory.

        Args:
            path: Path to the directory, relative to workspace. Defaults to workspace root.

        Returns:
            List of files and directories, or error message.

        Examples:
            list_dir()
            list_dir("src")
        """
        try:
            abs_path = self._docker_workspace_path / path
            success, output = self._exec(f"ls -la '{abs_path}'")
            if not success:
                return f"Error listing directory: {output}"
            return output
        except Exception as e:
            return f"Error: {str(e)}"

    def _create_dir(self, path: str) -> str:
        """Create a directory (and parent directories if needed).

        Args:
            path: Path to the directory, relative to workspace.

        Returns:
            Success message or error message.

        Examples:
            create_dir("src")
            create_dir("src/utils")
        """
        try:
            abs_path = self._docker_workspace_path / path
            success, output = self._exec(f"mkdir -p '{abs_path}'")
            if not success:
                return f"Error creating directory: {output}"
            return f"Successfully created directory {path}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _edit_file(self, path: str, old_string: str, new_string: str) -> str:
        """Edit a file by replacing a specific string with a new string.

        PREFERRED over write_file when modifying existing files — you only specify the
        changed portion instead of regenerating the entire file content, which is
        significantly more efficient.
        The old_string must appear exactly once in the file to avoid ambiguous edits.

        Args:
            path: Path to the file, relative to the workspace directory.
            old_string: The exact text to find in the file (must be unique).
            new_string: The text to replace it with.

        Returns:
            Success message or error message if the edit fails.

        Examples:
            edit_file("main.py", "x = 1", "x = 2")
            edit_file("src/model.py", "learning_rate = 0.01", "learning_rate = 0.001")
        """
        try:
            content = self._read_file(path)
            if content.startswith("Error"):
                return content

            count = content.count(old_string)
            if count == 0:
                return (
                    f"Error: old_string not found in {path}. "
                    "Make sure the string matches exactly, including whitespace and indentation."
                )
            if count > 1:
                return (
                    f"Error: old_string appears {count} times in {path}. "
                    "Include more surrounding context to make the match unique."
                )

            new_content = content.replace(old_string, new_string, 1)
            return self._write_file(path, new_content)

        except Exception as e:
            return f"Error: {str(e)}"

    def _copy_file(self, source: str, destination: str) -> str:
        """Copy a file within the workspace.

        Args:
            source: Source file path, relative to the workspace directory.
            destination: Destination file path, relative to the workspace directory.

        Returns:
            Success message or error message if copying fails.

        Examples:
            copy_file("draft.py", "solution.py")
            copy_file("config.json", "config_backup.json")
        """
        try:
            src_abs = self._docker_workspace_path / source
            dst_abs = self._docker_workspace_path / destination
            dst_parent = dst_abs.parent

            self._exec(f"mkdir -p '{dst_parent}'")

            success, output = self._exec(f"cp '{src_abs}' '{dst_abs}'")
            if not success:
                return f"Error copying file: {output}"
            return f"Successfully copied {source} to {destination}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _delete(self, path: str) -> str:
        """Delete a file or directory (recursively).

        Args:
            path: Path to the file or directory, relative to workspace.

        Returns:
            Success message or error message.

        Examples:
            delete("temp.py")
            delete("old_src")
        """
        try:
            abs_path = self._docker_workspace_path / path

            success, output = self._exec(f"rm -rf '{abs_path}'")
            if not success:
                return f"Error deleting: {output}"
            return f"Successfully deleted {path}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _create_tools(self):
        """Create tools from bound methods."""
        self.install_package = tool(self._install_package)
        self.run_python_code = tool(self._run_python_code)
        self.run_python_script = tool(self._run_python_script)
        self.read_file = tool(self._read_file)
        self.write_file = tool(self._write_file)
        self.edit_file = tool(self._edit_file)
        self.copy_file = tool(self._copy_file)
        self.list_dir = tool(self._list_dir)
        self.create_dir = tool(self._create_dir)
        self.delete = tool(self._delete)

    def get_tools(self) -> list[BaseTool]:
        """Get the tools for the workspace."""
        return [
            self.install_package,
            self.run_python_code,
            self.run_python_script,
            self.read_file,
            self.write_file,
            self.edit_file,
            self.copy_file,
            self.list_dir,
            self.create_dir,
            self.delete
        ]

    def stop_workspace(self, remove_host_workspace: bool = False):
        """Stop the workspace.

        Args:
            remove_host_workspace: Whether to remove the host workspace directory.
        """
        self._stop_container()
        if remove_host_workspace:
            shutil.rmtree(self._host_workspace_path)
