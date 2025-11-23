"""Diff utilities for project files."""

from __future__ import annotations

import difflib
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from filelock import FileLock

from flowlib.core.project.validator import ProjectValidator
from server.models.diff import (
    ConfigApplyRequest,
    ConfigApplyResponse,
    ConfigDiffRequest,
    ConfigDiffResponse,
)
from server.models.projects import ProjectValidationIssue

if TYPE_CHECKING:
    from server.services.config_service import ConfigService


class ConfigValidationError(Exception):
    """Raised when project validation fails after applying a change."""

    def __init__(self, issues: list[ProjectValidationIssue]) -> None:
        self.issues = issues
        super().__init__("Project validation failed")


class DiffService:
    """Generate diffs and apply edits for project configuration files."""

    def __init__(
        self, projects_root: str = "./projects", config_service: ConfigService | None = None
    ) -> None:
        self._root = Path(projects_root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._validator = ProjectValidator()
        self._config_service = config_service
        self._locks_dir = self._root / ".locks"
        self._locks_dir.mkdir(exist_ok=True)

    @contextmanager
    def _file_lock(self, file_path: Path) -> Iterator[None]:
        """Context manager for file locking to prevent concurrent modifications.

        Uses hash of full absolute path to create unique lock files,
        preventing collisions between files with the same name in different directories.
        """
        # Create unique lock name from hash of full path to prevent collisions
        path_hash = hashlib.sha256(str(file_path.resolve()).encode("utf-8")).hexdigest()[:16]
        lock_name = f"{file_path.name}.{path_hash}.lock"
        lock_path = self._locks_dir / lock_name

        lock = FileLock(str(lock_path), timeout=10)
        try:
            with lock:
                yield
        finally:
            # Clean up lock file if it exists and is empty
            if lock_path.exists():
                try:
                    lock_path.unlink()
                except (OSError, PermissionError):
                    pass  # Lock file might be in use by another process

    def diff_config(self, payload: ConfigDiffRequest) -> ConfigDiffResponse:
        project_path = self._resolve_project_path(payload.project_id)
        file_path = self._resolve_file_path(project_path, payload.relative_path)

        if file_path.exists():
            current_content = file_path.read_text(encoding="utf-8")
            exists = True
        else:
            current_content = ""
            exists = False

        diff_lines = list(
            difflib.unified_diff(
                current_content.splitlines(keepends=True),
                payload.proposed_content.splitlines(keepends=True),
                fromfile=str(payload.relative_path),
                tofile=f"{payload.relative_path} (proposed)",
            )
        )

        return ConfigDiffResponse(
            project_id=payload.project_id,
            relative_path=payload.relative_path,
            exists=exists,
            diff=diff_lines,
        )

    def apply_config(self, payload: ConfigApplyRequest) -> ConfigApplyResponse:
        project_path = self._resolve_project_path(payload.project_id)
        file_path = self._resolve_file_path(project_path, payload.relative_path)

        # Acquire file lock to prevent concurrent modifications
        with self._file_lock(file_path):
            # Read current content atomically
            if file_path.exists():
                original_content = file_path.read_text(encoding="utf-8")
            else:
                original_content = ""

            # Validate hash hasn't changed - no bypass, fail fast
            current_hash = self._compute_hash(original_content)
            if not payload.sha256_before:
                raise ValueError(
                    "sha256_before is required for safe file modification. "
                    "Cannot apply changes without hash validation."
                )
            if current_hash != payload.sha256_before:
                raise ValueError(
                    "File contents changed since diff was generated. "
                    f"Expected hash {payload.sha256_before}, found {current_hash}."
                )

            # Write new content atomically
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(payload.content, encoding="utf-8")

            # Validate project structure
            validation_result = self._validator.validate(project_path)
            if not validation_result.is_valid:
                # Rollback to original content
                if original_content:
                    file_path.write_text(original_content, encoding="utf-8")
                else:
                    file_path.unlink(missing_ok=True)

                issues = [
                    ProjectValidationIssue(path=issue.path, message=issue.message)
                    for issue in validation_result.issues
                ]
                raise ConfigValidationError(issues)

            new_hash = self._compute_hash(payload.content)

            # Invalidate cached project data after successful modification
            if self._config_service:
                self._config_service.invalidate_cache(payload.project_id)

            return ConfigApplyResponse(
                project_id=payload.project_id,
                relative_path=payload.relative_path,
                sha256_before=current_hash,
                sha256_after=new_hash,
            )

    def rename_config(self, project_id: str, old_relative_path: str, new_relative_path: str) -> None:
        """Rename/move a configuration file and validate project structure, rollback on failure."""
        project_path = self._resolve_project_path(project_id)
        old_path = self._resolve_file_path(project_path, old_relative_path)
        new_path = self._resolve_file_path(project_path, new_relative_path)

        # Use file lock on the old path (the file being moved)
        with self._file_lock(old_path):
            if not old_path.exists():
                raise FileNotFoundError(f"File not found: {old_relative_path}")
            if new_path.exists():
                raise ValueError(f"Destination already exists: {new_relative_path}")

            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.rename(new_path)

            validation_result = self._validator.validate(project_path)
            if not validation_result.is_valid:
                # Rollback
                new_path.rename(old_path)
                issues = [
                    ProjectValidationIssue(path=issue.path, message=issue.message)
                    for issue in validation_result.issues
                ]
                raise ConfigValidationError(issues)

            # Invalidate cached project data after successful modification
            if self._config_service:
                self._config_service.invalidate_cache(project_id)

    def delete_config(self, project_id: str, relative_path: str) -> None:
        """Delete a configuration file and validate project structure, rollback on failure."""
        project_path = self._resolve_project_path(project_id)
        file_path = self._resolve_file_path(project_path, relative_path)

        # Use file lock to prevent concurrent access during deletion
        with self._file_lock(file_path):
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {relative_path}")

            # Backup original content for rollback
            original = file_path.read_text(encoding="utf-8")

            # Delete the file
            file_path.unlink()

            # Validate project structure
            validation_result = self._validator.validate(project_path)
            if not validation_result.is_valid:
                # Rollback - restore the file
                file_path.write_text(original, encoding="utf-8")
                issues = [
                    ProjectValidationIssue(path=issue.path, message=issue.message)
                    for issue in validation_result.issues
                ]
                raise ConfigValidationError(issues)

            # Invalidate cached project data after successful modification
            if self._config_service:
                self._config_service.invalidate_cache(project_id)
    def _resolve_project_path(self, project_id: str) -> Path:
        project_path = (self._root / project_id).resolve()
        if not project_path.exists() or not project_path.is_dir():
            raise FileNotFoundError(f"Project '{project_id}' not found under {self._root}")

        # Validate path is within managed root using relative_to()
        try:
            project_path.relative_to(self._root)
        except ValueError as e:
            raise ValueError(
                f"Project '{project_id}' resolved outside managed root {self._root}"
            ) from e

        # Ensure it's not the root itself
        if project_path == self._root:
            raise ValueError(f"Project path cannot be the root directory {self._root}")

        return project_path

    def _resolve_file_path(self, project_path: Path, relative_path: str) -> Path:
        candidate = (project_path / relative_path).resolve()

        # Validate path is within project directory using relative_to()
        try:
            candidate.relative_to(project_path)
        except ValueError as e:
            raise ValueError(
                f"Relative path '{relative_path}' escapes project directory {project_path}"
            ) from e

        return candidate

    def _compute_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

