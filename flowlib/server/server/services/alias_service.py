"""Alias file management service."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from server.models.configs import AliasEntry
from server.models.diff import ConfigApplyRequest, ConfigApplyResponse
from server.services.diff_service import DiffService

if TYPE_CHECKING:
    from server.services.config_service import ConfigService


class AliasService:
    """Manage `configs/aliases.py` content in a project."""

    def __init__(
        self, projects_root: str = "./projects", config_service: "ConfigService | None" = None
    ) -> None:
        self._root = Path(projects_root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._diff_service = DiffService(projects_root, config_service)
        # SHA256 of empty content, used when creating new files via diff service
        self._EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def _get_current_file_hash(self, project_id: str, relative_path: str) -> str:
        """Get the current hash of a file, or empty content hash if file doesn't exist."""
        project_path = self._diff_service._resolve_project_path(project_id)
        file_path = self._diff_service._resolve_file_path(project_path, relative_path)
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            return self._diff_service._compute_hash(content)
        return self._EMPTY_SHA256

    def apply_aliases(self, project_id: str, aliases: list[AliasEntry]) -> ConfigApplyResponse:
        relative_path = "configs/aliases.py"
        content_lines: list[str] = [
            '"""Auto-generated alias bindings (single source of truth)."""',
            "",
            "from flowlib.config.alias_manager import alias_manager",
            "",
        ]
        for entry in sorted(aliases, key=lambda a: (a.alias, a.canonical)):
            content_lines.append(f'alias_manager.assign_alias("{entry.alias}", "{entry.canonical}")')
        content_lines.append("")

        content = textwrap.dedent("\n".join(content_lines))
        return self._diff_service.apply_config(
            ConfigApplyRequest(
                project_id=project_id,
                relative_path=relative_path,
                content=content,
                sha256_before=self._get_current_file_hash(project_id, relative_path),
            )
        )


