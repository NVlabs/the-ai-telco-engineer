# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Process-safe printing with role-based headers.

Ensures that output from parallel processes does not interleave.
Each process sets its own header (e.g. "MANAGER", "AGENT-gen00-0001").
A shared multiprocessing lock serializes writes to stdout.

Usage:
    # In the main process:
    import multiprocessing as mp
    import printer

    lock = mp.Lock()
    printer.init(lock, "MANAGER")
    printer.log("Starting optimization")

    # In a worker process:
    printer.init(lock, "WORKER-0")
    printer.log("Processing task")
    printer.set_header("AGENT-gen00-0001")
    printer.log("Rate limit hit")
"""

import multiprocessing as mp
import sys
from typing import Optional

_lock: Optional[mp.Lock] = None
_header: str = ""


def init(lock: mp.Lock, header: str = "") -> None:
    """Initialize the printer in the current process.

    Must be called once per process. In workers, call this with the shared
    lock received from the manager and the initial header for the process.

    Args:
        lock: A multiprocessing.Lock shared across all processes.
        header: Initial header string (e.g. "MANAGER", "WORKER-0").
    """
    global _lock, _header
    _lock = lock
    _header = header


def set_header(header: str) -> None:
    """Update the header for all subsequent prints in this process.

    Args:
        header: New header string (e.g. "AGENT-gen00-0003").
    """
    global _header
    _header = header


def _format_line(line: str) -> str:
    """Format a single line with the current header prefix."""
    if not line:
        return ""
    prefix = f"[{_header}] " if _header else ""
    return f"{prefix}{line}"


def log(*args, sep: str = " ") -> None:
    """Print a single message atomically with the current header prefix.

    Safe to call from any process. Falls back to plain stdout
    if init() has not been called (e.g. during early startup).

    Args:
        *args: Values to print (joined by *sep*).
        sep: Separator between values (default: space).
    """
    message = sep.join(str(a) for a in args)
    output = _format_line(message) + "\n"
    if _lock is not None:
        with _lock:
            sys.stdout.write(output)
            sys.stdout.flush()
    else:
        sys.stdout.write(output)
        sys.stdout.flush()


def section(*lines: str) -> None:
    """Print multiple lines as a single atomic block.

    Each non-empty line is prefixed with the header.
    Empty strings produce blank lines (no prefix).

    Args:
        *lines: One string per output line.
    """
    formatted = []
    for line in lines:
        formatted.append(_format_line(line))
    output = "\n".join(formatted) + "\n"
    if _lock is not None:
        with _lock:
            sys.stdout.write(output)
            sys.stdout.flush()
    else:
        sys.stdout.write(output)
        sys.stdout.flush()
