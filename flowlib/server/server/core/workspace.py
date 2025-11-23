"""Workspace path management."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from server.core.config import settings

logger = logging.getLogger(__name__)


class WorkspaceNotConfiguredError(RuntimeError):
    """Raised when workspace path is not configured."""

    def __init__(self, message: str | None = None) -> None:
        if message is None:
            message = (
                "Workspace path is not configured. "
                "Use the API endpoint /api/v1/workspace/path to configure it, "
                "or set PROJECTS_ROOT environment variable or .env file."
            )
        super().__init__(message)


def get_workspace_path() -> str:
    """Get the current workspace path.

    Returns:
        The workspace path (always absolute and resolved)

    Raises:
        WorkspaceNotConfiguredError: If workspace path is not configured
    """
    workspace = settings.PROJECTS_ROOT

    # If not set, raise error that API endpoints can catch
    if workspace is None or workspace == "":
        raise WorkspaceNotConfiguredError()

    # Always return absolute path
    workspace_path = Path(workspace).expanduser().resolve()
    return str(workspace_path)


def set_workspace_path(path: str, save_to_env: bool = True) -> None:
    """Set the workspace path and optionally save to .env file.

    Args:
        path: Workspace path to set (will be resolved to absolute path)
        save_to_env: If True, save to .env file for persistence

    Raises:
        ValueError: If path is invalid
        OSError: If path cannot be created or .env file cannot be written
    """
    workspace_path = Path(path).expanduser().resolve()

    # Validate path
    if workspace_path.exists() and not workspace_path.is_dir():
        raise ValueError(f"Workspace path exists but is not a directory: {workspace_path}")

    # Create directory if it doesn't exist
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Update settings (this won't persist unless we save to .env)
    settings.PROJECTS_ROOT = str(workspace_path)

    if save_to_env:
        _save_to_env_file(str(workspace_path))


def prompt_workspace_path() -> str:
    """Prompt user interactively for workspace path.

    Returns:
        The absolute workspace path entered by user

    Raises:
        RuntimeError: If not running interactively
        ValueError: If path is invalid
    """
    if not sys.stdin.isatty():
        raise RuntimeError("Cannot prompt for workspace path: not running interactively")

    print("\n" + "=" * 60)
    print("Flowlib Server - Workspace Configuration")
    print("=" * 60)
    print("\nNo workspace path is configured.")
    print("The workspace is where all your Flowlib projects will be stored.")
    print()

    while True:
        default_path = str(Path.home() / "flowlib-workspace")
        prompt = f"Enter workspace path [{default_path}]: "
        user_input = input(prompt).strip()

        if not user_input:
            user_input = default_path

        try:
            workspace_path = Path(user_input).expanduser().resolve()
            # Validate path
            if workspace_path.exists() and not workspace_path.is_dir():
                print(f"Error: Path exists but is not a directory: {workspace_path}")
                continue

            # Confirm creation if doesn't exist
            if not workspace_path.exists():
                confirm = input(
                    f"\nPath does not exist. Create directory at '{workspace_path}'? [Y/n]: "
                ).strip().lower()
                if confirm and confirm not in ("y", "yes"):
                    print("Cancelled. Please try again.")
                    continue

            # Create directory
            workspace_path.mkdir(parents=True, exist_ok=True)

            # Confirm final path
            print(f"\nWorkspace path set to: {workspace_path}")
            confirm = input("Is this correct? [Y/n]: ").strip().lower()
            if confirm and confirm not in ("y", "yes"):
                continue

            return str(workspace_path)

        except (OSError, ValueError) as e:
            print(f"Error: {e}")
            print("Please try again.")


def _save_to_env_file(workspace_path: str) -> None:
    """Save workspace path to .env file.

    Args:
        workspace_path: Workspace path to save

    Raises:
        OSError: If .env file cannot be written
    """
    # Find .env file location
    # Pydantic-settings looks for .env in current working directory (CWD)
    # Server is typically run from flowlib/server/, so .env should be there
    # For safety, we use CWD which should match where run_server.py is
    import os
    env_file = Path(os.getcwd()) / ".env"

    # Read existing .env if it exists
    existing_lines: list[str] = []
    if env_file.exists():
        existing_lines = env_file.read_text(encoding="utf-8").splitlines()

    # Update or add PROJECTS_ROOT line
    updated = False
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if stripped.startswith("PROJECTS_ROOT="):
            new_lines.append(f"PROJECTS_ROOT={workspace_path}")
            updated = True
        elif not stripped.startswith("#") and "PROJECTS_ROOT" in stripped:
            # Handle commented or malformed lines
            new_lines.append(line)
        else:
            new_lines.append(line)

    if not updated:
        # Add new line
        if new_lines and not new_lines[-1].strip() == "":
            new_lines.append("")
        new_lines.append(f"PROJECTS_ROOT={workspace_path}")

    # Write back to .env file
    env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    logger.info("Saved workspace path to %s", env_file)

