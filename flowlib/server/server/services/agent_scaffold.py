from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from server.models.configs import AgentCreateRequest
from server.models.diff import ConfigApplyRequest, ConfigApplyResponse
from server.services.diff_service import DiffService

if TYPE_CHECKING:
    from server.services.config_service import ConfigService


class AgentScaffoldService:
    """Render and apply agent configuration files using DiffService (fail-fast, validated)."""

    def __init__(
        self, projects_root: str = "./projects", config_service: "ConfigService | None" = None
    ) -> None:
        self._root = Path(projects_root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._diff = DiffService(projects_root, config_service)
        # SHA256 of empty content, used when creating new files via diff service
        self._EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def _get_current_file_hash(self, project_id: str, relative_path: str) -> str:
        """Get the current hash of a file, or empty content hash if file doesn't exist."""
        project_path = self._diff._resolve_project_path(project_id)
        file_path = self._diff._resolve_file_path(project_path, relative_path)
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            return self._diff._compute_hash(content)
        return self._EMPTY_SHA256

    def _slugify(self, name: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in name).strip("-_").lower()

    def _camel_case(self, name: str) -> str:
        parts = [p for p in self._slugify(name).split("-") if p]
        return "".join(p.capitalize() for p in parts)

    def _render_agent_content(
        self,
        name: str,
        persona: str,
        allowed_categories: Iterable[str],
        knowledge_plugins: Iterable[str],
        class_name: str,
        description: str | None,
    ) -> str:
        categories_literal = "[" + ", ".join(f'"{c}"' for c in allowed_categories) + "]"
        plugins_literal = "[" + ", ".join(f'"{p}"' for p in knowledge_plugins) + "]"
        desc = (description or "Agent scaffold.")[:512]
        return textwrap.dedent(
            f'''"""
{desc}
"""
from flowlib.resources.decorators.decorators import agent_config


@agent_config(name="{name}")
class {class_name}:
    persona = {persona!r}
    allowed_tool_categories = {categories_literal}
    knowledge_plugins = {plugins_literal}
    model_name = "default-model"
    llm_name = "default-llm"
    temperature = 0.7
    max_iterations = 10
    enable_learning = False
    verbose = False
'''
        )

    def create_agent(self, payload: AgentCreateRequest) -> ConfigApplyResponse:
        project_path = (self._root / payload.project_id).resolve()
        if not project_path.exists():
            raise FileNotFoundError(f"Project '{payload.project_id}' not found under {self._root}")

        file_name = f"{self._slugify(payload.name)}.py"
        relative_path = f"agents/{file_name}"
        class_name = f"{self._camel_case(payload.name)}AgentConfig"
        content = self._render_agent_content(
            name=self._slugify(payload.name),
            persona=payload.persona,
            allowed_categories=payload.allowed_tool_categories,
            knowledge_plugins=payload.knowledge_plugins,
            class_name=class_name,
            description=payload.description,
        )
        return self._diff.apply_config(
            ConfigApplyRequest(
                project_id=payload.project_id,
                relative_path=relative_path,
                content=content,
                sha256_before=self._EMPTY_SHA256,
            )
        )

    def render_agent_content(
        self,
        name: str,
        persona: str,
        allowed_tool_categories: list[str],
        knowledge_plugins: list[str],
        description: str | None,
    ) -> str:
        class_name = f"{self._camel_case(name)}AgentConfig"
        return self._render_agent_content(
            name=self._slugify(name),
            persona=persona,
            allowed_categories=allowed_tool_categories,
            knowledge_plugins=knowledge_plugins,
            class_name=class_name,
            description=description,
        )

    def apply_agent_structured(
        self,
        project_id: str,
        name: str,
        persona: str,
        allowed_tool_categories: list[str],
        knowledge_plugins: list[str],
        description: str | None,
        model_name: str,
        llm_name: str,
        temperature: float,
        max_iterations: int,
        enable_learning: bool,
        verbose: bool,
    ) -> ConfigApplyResponse:
        # render with all fields (reusing scaffold template but overriding fields directly)
        slug = self._slugify(name)
        class_name = f"{self._camel_case(name)}AgentConfig"
        categories_literal = "[" + ", ".join(f'"{c}"' for c in allowed_tool_categories) + "]"
        plugins_literal = "[" + ", ".join(f'"{p}"' for p in knowledge_plugins) + "]"
        desc = (description or "Agent configuration.")[:512]
        content = textwrap.dedent(
            f'''"""
{desc}
"""
from flowlib.resources.decorators.decorators import agent_config


@agent_config(name="{slug}")
class {class_name}:
    persona = {persona!r}
    allowed_tool_categories = {categories_literal}
    knowledge_plugins = {plugins_literal}
    model_name = {model_name!r}
    llm_name = {llm_name!r}
    temperature = {temperature}
    max_iterations = {max_iterations}
    enable_learning = {enable_learning}
    verbose = {verbose}
'''
        )
        return self._diff.apply_config(
            ConfigApplyRequest(
                project_id=project_id,
                relative_path=f"agents/{slug}.py",
                content=content,
                sha256_before=self._get_current_file_hash(project_id, f"agents/{slug}.py"),
            )
        )


