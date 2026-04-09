# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Integration tests for Workspace class.

These tests run actual Docker containers and perform real filesystem operations.
Requires:
  - Docker daemon running
  - DOCKER_IMAGE available (default: python:3.12-slim; will be pulled if missing)
"""

import pytest
import subprocess
import shutil
import uuid
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tool_lib.workspace import Workspace


# Configuration
HOST_WORKSPACE_BASE = Path("/tmp/test_workspaces")
DOCKER_WORKSPACE_PATH = "/workspace"
# Use a standard image that has python, bash, pip (Docker will pull if missing)
DOCKER_IMAGE = "python:3.12-slim"


def docker_available() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def cleanup_container(container_name: str):
    """Force remove a container if it exists."""
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=30
    )


def cleanup_workspace_dir(path: Path):
    """Remove workspace directory if it exists."""
    if path.exists():
        shutil.rmtree(path)


# Skip all tests if Docker is not available
pytestmark = pytest.mark.skipif(
    not docker_available(),
    reason="Docker daemon not running"
)


@pytest.fixture
def workspace_id():
    """Generate a unique workspace ID for each test."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def workspace(workspace_id):
    """Create a workspace and clean up after the test."""
    # Ensure base directory exists
    HOST_WORKSPACE_BASE.mkdir(parents=True, exist_ok=True)

    ws = None
    try:
        ws = Workspace(
            workspace_id=workspace_id,
            host_workspace_path=str(HOST_WORKSPACE_BASE),
            docker_workspace_path=DOCKER_WORKSPACE_PATH,
            docker_image=DOCKER_IMAGE,
            use_gpu=False,
        )
        yield ws
    finally:
        # Cleanup: stop container and remove workspace directory
        if ws is not None:
            try:
                ws.stop_workspace(remove_host_workspace=True)
            except Exception:
                pass

        # Force cleanup in case stop_workspace failed
        cleanup_container(f"workspace_{workspace_id}")
        cleanup_workspace_dir(HOST_WORKSPACE_BASE / workspace_id)


class TestWorkspaceInitialization:
    """Tests for workspace initialization."""

    def test_workspace_creates_host_directory(self, workspace, workspace_id):
        """Test that workspace creates host directory."""
        host_path = HOST_WORKSPACE_BASE / workspace_id
        assert host_path.exists()
        assert host_path.is_dir()

    def test_workspace_starts_container(self, workspace):
        """Test that workspace starts a Docker container."""
        result = subprocess.run(
            ["docker", "inspect", workspace._container_name],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Container should be running"

    def test_workspace_attributes(self, workspace, workspace_id):
        """Test workspace attributes are set correctly."""
        assert workspace._workspace_id == workspace_id
        assert workspace._host_workspace_path == HOST_WORKSPACE_BASE / workspace_id
        assert workspace._docker_workspace_path == Path(DOCKER_WORKSPACE_PATH)


class TestParentWorkspaceInheritance:
    """Tests for parent workspace inheritance."""

    def test_child_workspace_copies_parent_files(self):
        """Test that child workspace inherits files from parent."""
        parent_id = f"parent_{uuid.uuid4().hex[:8]}"
        child_id = f"child_{uuid.uuid4().hex[:8]}"

        HOST_WORKSPACE_BASE.mkdir(parents=True, exist_ok=True)

        parent_ws = None
        child_ws = None

        try:
            # Create parent workspace
            parent_ws = Workspace(
                workspace_id=parent_id,
                host_workspace_path=str(HOST_WORKSPACE_BASE),
                docker_workspace_path=DOCKER_WORKSPACE_PATH,
                docker_image=DOCKER_IMAGE,
                use_gpu=False,
            )

            # Create a file in parent workspace
            parent_ws._write_file("inherited.txt", "hello from parent")

            # Stop parent but keep files
            parent_ws.stop_workspace(remove_host_workspace=False)
            parent_ws = None

            # Create child workspace from parent
            child_ws = Workspace(
                workspace_id=child_id,
                host_workspace_path=str(HOST_WORKSPACE_BASE),
                docker_workspace_path=DOCKER_WORKSPACE_PATH,
                parent_workspace_id=parent_id,
                docker_image=DOCKER_IMAGE,
                use_gpu=False,
            )

            # Verify child has the inherited file
            content = child_ws._read_file("inherited.txt")
            assert "hello from parent" in content

        finally:
            # Cleanup
            for ws in [parent_ws, child_ws]:
                if ws is not None:
                    try:
                        ws.stop_workspace(remove_host_workspace=True)
                    except Exception:
                        pass

            cleanup_container(f"workspace_{parent_id}")
            cleanup_container(f"workspace_{child_id}")
            cleanup_workspace_dir(HOST_WORKSPACE_BASE / parent_id)
            cleanup_workspace_dir(HOST_WORKSPACE_BASE / child_id)

    def test_parent_not_exists_raises_error(self):
        """Test that creating child from non-existent parent raises error."""
        child_id = f"orphan_{uuid.uuid4().hex[:8]}"

        with pytest.raises(ValueError, match="does not exist"):
            Workspace(
                workspace_id=child_id,
                host_workspace_path=str(HOST_WORKSPACE_BASE),
                docker_workspace_path=DOCKER_WORKSPACE_PATH,
                parent_workspace_id="nonexistent_parent",
                docker_image=DOCKER_IMAGE,
                use_gpu=False,
            )


class TestExecCommand:
    """Tests for _exec method."""

    def test_exec_returns_stdout_on_success(self, workspace):
        """Test _exec returns stdout when command succeeds."""
        success, output = workspace._exec("echo 'hello world'")

        assert success is True
        assert "hello world" in output

    def test_exec_returns_stderr_on_failure(self, workspace):
        """Test _exec returns stderr when command fails."""
        success, output = workspace._exec("ls /nonexistent_path_12345")

        assert success is False
        assert "No such file" in output or "cannot access" in output


class TestRunPythonCode:
    """Tests for Python code execution."""

    def test_run_simple_python_code(self, workspace):
        """Test running simple Python code."""
        result = workspace._run_python_code("print('Hello from Python!')")

        assert "Hello from Python!" in result

    def test_run_python_code_with_computation(self, workspace):
        """Test running Python code with computation."""
        result = workspace._run_python_code("print(sum(range(10)))")

        assert "45" in result

    def test_run_python_code_with_error(self, workspace):
        """Test running Python code that raises an error."""
        result = workspace._run_python_code("print(undefined_variable)")

        assert "Error" in result
        assert "NameError" in result

    def test_run_python_code_multiline(self, workspace):
        """Test running multiline Python code."""
        code = """
x = 10
y = 20
print(f"Sum: {x + y}")
"""
        result = workspace._run_python_code(code)

        assert "Sum: 30" in result


class TestRunPythonScript:
    """Tests for Python script execution."""

    def test_run_python_script(self, workspace):
        """Test running a Python script file."""
        # Create a script file
        workspace._write_file("test_script.py", "print('Script executed!')")

        # Run the script
        result = workspace._run_python_script("test_script.py")

        assert "Script executed!" in result

    def test_run_python_script_not_found(self, workspace):
        """Test running a non-existent script."""
        result = workspace._run_python_script("nonexistent_script.py")

        assert "Error" in result


class TestFileOperations:
    """Tests for file read/write operations."""

    def test_write_and_read_file(self, workspace):
        """Test writing and reading a file."""
        content = "Hello, World!"

        write_result = workspace._write_file("test.txt", content)
        assert "Successfully" in write_result

        read_result = workspace._read_file("test.txt")
        assert read_result.strip() == content

    def test_write_file_with_special_characters(self, workspace):
        """Test writing file with special characters."""
        content = """Line 1
Line 2 with 'quotes' and "double quotes"
Special: $HOME $(echo test) `backticks`
Unicode: café, 日本語"""

        workspace._write_file("special.txt", content)
        read_result = workspace._read_file("special.txt")

        assert "Line 1" in read_result
        assert "'quotes'" in read_result
        assert "café" in read_result

    def test_write_file_creates_parent_directories(self, workspace):
        """Test that write_file creates parent directories."""
        result = workspace._write_file("deeply/nested/dir/file.txt", "content")

        assert "Successfully" in result

        read_result = workspace._read_file("deeply/nested/dir/file.txt")
        assert "content" in read_result

    def test_read_nonexistent_file(self, workspace):
        """Test reading a file that doesn't exist."""
        result = workspace._read_file("does_not_exist.txt")

        assert "Error" in result


class TestDirectoryOperations:
    """Tests for directory operations."""

    def test_create_directory(self, workspace):
        """Test creating a directory."""
        result = workspace._create_dir("new_directory")

        assert "Successfully" in result

        # Verify by listing
        list_result = workspace._list_dir(".")
        assert "new_directory" in list_result

    def test_create_nested_directories(self, workspace):
        """Test creating nested directories."""
        result = workspace._create_dir("a/b/c")

        assert "Successfully" in result

    def test_list_directory(self, workspace):
        """Test listing directory contents."""
        # Create some files
        workspace._write_file("file1.txt", "content1")
        workspace._write_file("file2.txt", "content2")
        workspace._create_dir("subdir")

        result = workspace._list_dir(".")

        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir" in result

    def test_list_nonexistent_directory(self, workspace):
        """Test listing a directory that doesn't exist."""
        result = workspace._list_dir("nonexistent_dir")

        assert "Error" in result


class TestDeleteOperations:
    """Tests for delete operations."""

    def test_delete_file(self, workspace):
        """Test deleting a file."""
        workspace._write_file("to_delete.txt", "content")

        result = workspace._delete("to_delete.txt")
        assert "Successfully" in result

        # Verify file is gone
        read_result = workspace._read_file("to_delete.txt")
        assert "Error" in read_result

    def test_delete_directory(self, workspace):
        """Test deleting a directory recursively."""
        workspace._create_dir("dir_to_delete")
        workspace._write_file("dir_to_delete/file.txt", "content")

        result = workspace._delete("dir_to_delete")
        assert "Successfully" in result

        # Verify directory is gone
        list_result = workspace._list_dir("dir_to_delete")
        assert "Error" in list_result


class TestStopWorkspace:
    """Tests for stopping workspace."""

    def test_stop_workspace_stops_container(self, workspace_id):
        """Test that stop_workspace stops the container."""
        HOST_WORKSPACE_BASE.mkdir(parents=True, exist_ok=True)

        ws = Workspace(
            workspace_id=workspace_id,
            host_workspace_path=str(HOST_WORKSPACE_BASE),
            docker_workspace_path=DOCKER_WORKSPACE_PATH,
            docker_image=DOCKER_IMAGE,
            use_gpu=False,
        )
        container_name = ws._container_name

        # Verify container is running
        result = subprocess.run(
            ["docker", "inspect", container_name],
            capture_output=True
        )
        assert result.returncode == 0, "Container should be running"

        # Stop workspace
        ws.stop_workspace(remove_host_workspace=False)

        # Verify container is stopped (inspect should fail or show stopped state)
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True
        )
        # Container might be stopped or removed
        assert result.returncode != 0 or "false" in result.stdout.lower()

        # Cleanup
        cleanup_container(container_name)
        cleanup_workspace_dir(HOST_WORKSPACE_BASE / workspace_id)

    def test_stop_workspace_removes_host_files(self, workspace_id):
        """Test that stop_workspace can remove host files."""
        HOST_WORKSPACE_BASE.mkdir(parents=True, exist_ok=True)
        host_path = HOST_WORKSPACE_BASE / workspace_id

        ws = Workspace(
            workspace_id=workspace_id,
            host_workspace_path=str(HOST_WORKSPACE_BASE),
            docker_workspace_path=DOCKER_WORKSPACE_PATH,
            docker_image=DOCKER_IMAGE,
            use_gpu=False,
        )

        # Create a file
        ws._write_file("test.txt", "content")
        assert host_path.exists()

        # Stop and remove
        ws.stop_workspace(remove_host_workspace=True)

        # Verify directory is removed
        assert not host_path.exists()

        # Cleanup container if still exists
        cleanup_container(f"workspace_{workspace_id}")


class TestEditFile:
    """Tests for edit_file (search-and-replace) operations."""

    def test_edit_file_basic(self, workspace):
        """Test a simple single-line replacement."""
        workspace._write_file("config.py", "learning_rate = 0.01\nbatch_size = 32\n")

        result = workspace._edit_file("config.py", "learning_rate = 0.01", "learning_rate = 0.001")

        assert "Successfully" in result
        content = workspace._read_file("config.py")
        assert "learning_rate = 0.001" in content
        assert "batch_size = 32" in content

    def test_edit_file_multiline(self, workspace):
        """Test replacing a multi-line block."""
        original = "def greet():\n    return 'hello'\n"
        workspace._write_file("funcs.py", original)

        result = workspace._edit_file(
            "funcs.py",
            "def greet():\n    return 'hello'",
            "def greet(name):\n    return f'hello {name}'",
        )

        assert "Successfully" in result
        content = workspace._read_file("funcs.py")
        assert "def greet(name):" in content
        assert "f'hello {name}'" in content

    def test_edit_file_old_string_not_found(self, workspace):
        """Test error when old_string doesn't exist in the file."""
        workspace._write_file("data.txt", "alpha\nbeta\n")

        result = workspace._edit_file("data.txt", "gamma", "delta")

        assert "Error" in result
        assert "not found" in result

    def test_edit_file_ambiguous_match(self, workspace):
        """Test error when old_string matches more than once."""
        workspace._write_file("dup.txt", "x = 1\nx = 1\n")

        result = workspace._edit_file("dup.txt", "x = 1", "x = 2")

        assert "Error" in result
        assert "2 times" in result
        content = workspace._read_file("dup.txt")
        assert content.count("x = 1") == 2, "File should be left unchanged"

    def test_edit_file_nonexistent_file(self, workspace):
        """Test editing a file that doesn't exist."""
        result = workspace._edit_file("no_such_file.py", "a", "b")

        assert "Error" in result

    def test_edit_file_preserves_rest_of_content(self, workspace):
        """Test that only the matched section is changed; everything else is intact."""
        lines = "line1\nline2\nline3\nline4\nline5\n"
        workspace._write_file("lines.txt", lines)

        workspace._edit_file("lines.txt", "line3", "LINE_THREE")

        content = workspace._read_file("lines.txt")
        assert content.strip() == "line1\nline2\nLINE_THREE\nline4\nline5"

    def test_edit_file_with_special_characters(self, workspace):
        """Test editing content that contains shell-sensitive characters."""
        workspace._write_file("special.py", "msg = 'say $HOME'\n")

        result = workspace._edit_file("special.py", "msg = 'say $HOME'", "msg = 'say $(whoami)'")

        assert "Successfully" in result
        content = workspace._read_file("special.py")
        assert "$(whoami)" in content


class TestCopyFile:
    """Tests for copy_file operations."""

    def test_copy_file_basic(self, workspace):
        """Test copying a file to a new location."""
        workspace._write_file("source.txt", "hello world")

        result = workspace._copy_file("source.txt", "dest.txt")
        assert "Successfully" in result

        content = workspace._read_file("dest.txt")
        assert "hello world" in content

    def test_copy_file_preserves_original(self, workspace):
        """Test that the source file is unchanged after copy."""
        workspace._write_file("original.py", "x = 42")

        workspace._copy_file("original.py", "backup.py")

        original = workspace._read_file("original.py")
        assert "x = 42" in original

    def test_copy_file_overwrites_destination(self, workspace):
        """Test that copying overwrites an existing destination."""
        workspace._write_file("src.txt", "new content")
        workspace._write_file("dst.txt", "old content")

        result = workspace._copy_file("src.txt", "dst.txt")
        assert "Successfully" in result

        content = workspace._read_file("dst.txt")
        assert "new content" in content

    def test_copy_file_creates_parent_directories(self, workspace):
        """Test that copy creates parent directories for destination."""
        workspace._write_file("flat.txt", "data")

        result = workspace._copy_file("flat.txt", "a/b/c/nested.txt")
        assert "Successfully" in result

        content = workspace._read_file("a/b/c/nested.txt")
        assert "data" in content

    def test_copy_file_nonexistent_source(self, workspace):
        """Test copying a source that doesn't exist."""
        result = workspace._copy_file("no_such_file.txt", "dest.txt")
        assert "Error" in result

    def test_copy_file_draft_to_solution(self, workspace):
        """Test the primary use case: saving draft.py as solution.py."""
        code = "def solve():\n    return 42\n"
        workspace._write_file("draft.py", code)

        result = workspace._copy_file("draft.py", "solution.py")
        assert "Successfully" in result

        content = workspace._read_file("solution.py")
        assert "def solve():" in content
        assert "return 42" in content


class TestInstallPackage:
    """Tests for package installation."""

    def test_install_package_success(self, workspace):
        """Test installing a package."""
        # Install a small, quick package
        result = workspace._install_package("pandas")

        assert "successfully" in result.lower() or "already" in result.lower()

    def test_install_nonexistent_package(self, workspace):
        """Test installing a package that doesn't exist."""
        result = workspace._install_package("this_package_definitely_does_not_exist_12345")

        assert "Error" in result
