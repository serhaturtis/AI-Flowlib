"""Project scaffolding utilities for Flowlib."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_README_TEMPLATE = """# {project_name}

{description}

## Structure

- `agents/`: Agent configurations
- `configs/providers/`: Provider configuration modules (LLM, vector DB, cache, etc.)
- `configs/resources/`: Registry resource configs (models, prompts, etc.)
- `configs/aliases.py`: Alias bindings (semantic names → canonical configs)
- `tools/`: Custom tools
- `flows/`: Custom flows (optional)
- `knowledge_plugins/`: Optional knowledge plugins
- `logs/`, `temp/`, `backups/`: Runtime directories
"""

ALIASES_TEMPLATE = '''"""Alias bindings for {project_name}."""

from flowlib.config.alias_manager import alias_manager
from flowlib.config.required_resources import RequiredAlias

# TODO: Create provider and resource configs first, then uncomment and update the aliases below.
#
# Example alias assignments:
# alias_manager.assign_alias(RequiredAlias.DEFAULT_LLM.value, "your-llm-provider-name")
# alias_manager.assign_alias(RequiredAlias.DEFAULT_MODEL.value, "your-model-config-name")
# alias_manager.assign_alias(RequiredAlias.DEFAULT_EMBEDDING.value, "your-embedding-provider-name")
# alias_manager.assign_alias(RequiredAlias.DEFAULT_EMBEDDING_MODEL.value, "your-embedding-model-name")
# alias_manager.assign_alias(RequiredAlias.DEFAULT_VECTOR_DB.value, "your-vector-db-name")
# alias_manager.assign_alias(RequiredAlias.DEFAULT_GRAPH_DB.value, "your-graph-db-name")
'''

AGENT_TEMPLATE = '''"""Agent configuration for {agent_name}."""

from flowlib.config.required_resources import RequiredAlias
from flowlib.resources.decorators.decorators import agent_config


@agent_config("{agent_name}")
class {class_name}:
    """{description}"""

    persona = (
        "{persona}"
    )
    allowed_tool_categories = {allowed_categories}
    model_name = RequiredAlias.DEFAULT_MODEL.value
    llm_name = RequiredAlias.DEFAULT_LLM.value
    temperature = 0.7
    max_iterations = 10
    enable_learning = False
    verbose = False
'''

TOOL_INIT_TEMPLATE = '''"""Package for the {tool_name} tool."""
'''

TOOL_TEMPLATE = '''"""Tool for {description}."""

from flowlib.agent.components.task.core.todo import TodoItem
from flowlib.agent.components.task.execution.decorators import tool
from flowlib.agent.components.task.execution.models import (
    ToolExecutionContext,
    ToolResult,
    ToolStatus,
)

from .models import {class_name}Parameters, {class_name}Result


@tool(
    parameter_type={class_name}Parameters,
    name="{tool_name}",
    description="{description}",
    tool_category="{category}",
)
class {class_name}Tool:
    """{description}."""

    def get_name(self) -> str:
        return "{tool_name}"

    def get_description(self) -> str:
        return "{description}"

    async def execute(
        self, todo: TodoItem, params: {class_name}Parameters, context: ToolExecutionContext
    ) -> ToolResult:
        """Execute tool logic."""
        try:
            # TODO: Implement tool behavior
            return {class_name}Result(
                status=ToolStatus.SUCCESS,
                message="{tool_name} executed successfully",
            )
        except Exception as exc:  # noqa: BLE001
            return {class_name}Result(
                status=ToolStatus.ERROR,
                message=f"Failed to execute {tool_name}: {exc}",
            )
'''

TOOL_MODELS_TEMPLATE = '''"""Models for the {tool_name} tool."""

from pydantic import Field

from flowlib.agent.components.task.execution.models import ToolParameters, ToolResult, ToolStatus


class {class_name}Parameters(ToolParameters):
    """Parameters for {tool_name}."""

    sample_field: str = Field(..., description="Replace with real inputs")


class {class_name}Result(ToolResult):
    """Result from {tool_name} execution."""

    def get_display_content(self) -> str:
        if self.status == ToolStatus.SUCCESS:
            return self.message or "{tool_name} completed."
        return self.message or "Tool execution failed."
'''

TOOL_PROMPTS_TEMPLATE = '''"""Prompts for the {tool_name} tool."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt


@prompt("{tool_name}_generation")
class {class_name}Prompt:
    """Prompt template for {tool_name}."""

    template: str = Field(
        default="""You are executing the {tool_name} tool.

Inputs:
{{inputs}}

Produce the required output based on the inputs above.""",
    )
'''

TOOL_FLOW_TEMPLATE = '''"""Flow helper for the {tool_name} tool."""

from pydantic import Field

from flowlib.core.models import StrictBaseModel
from flowlib.flows.decorators.decorators import flow, pipeline


class {class_name}FlowInput(StrictBaseModel):
    """Input for {tool_name} flow."""

    inputs: str = Field(..., description="Serialized inputs")


class {class_name}FlowOutput(StrictBaseModel):
    """Output for {tool_name} flow."""

    result: str = Field(..., description="Generated result")


@flow(name="{tool_name}-flow", description="{tool_name} helper flow")  # type: ignore[arg-type]
class {class_name}Flow:
    """LLM helper flow for {tool_name}."""

    @pipeline(input_model={class_name}FlowInput, output_model={class_name}FlowOutput)
    async def run_pipeline(self, request: {class_name}FlowInput) -> {class_name}FlowOutput:
        """Convert natural language requests into structured outputs."""
        # TODO: call LLM or other providers here
        return {class_name}FlowOutput(result=request.inputs)
'''


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9-_]+", "-", name.strip())
    cleaned = cleaned.strip("-").lower()
    return cleaned or "project"


def _camel_case(name: str) -> str:
    parts = re.split(r"[^a-zA-Z0-9]", name)
    return "".join(part.capitalize() for part in parts if part)


def _ensure_init(path: Path) -> None:
    if not path.exists():
        path.write_text('"""Package initializer."""\n')


def _write_file(path: Path, content: str) -> None:
    if path.exists():
        return
    path.write_text(content.strip() + "\n")


@dataclass
class ProjectScaffold:
    """Create standardized project structures."""

    root_path: Path = field(default_factory=lambda: Path("projects"))

    def create_project(
        self,
        name: str,
        description: str = "Project scaffold generated by Flowlib.",
        setup_type: str = "empty",
        agent_names: Iterable[str] | None = None,
        tool_categories: Iterable[str] | None = None,
        create_example_tools: bool = False,
    ) -> Path:
        project_slug = _slugify(name)
        project_path = (self.root_path / project_slug).resolve()
        agent_names = list(agent_names or [])
        tool_categories = list(tool_categories or [])

        directories = [
            project_path,
            project_path / "agents",
            project_path / "configs",
            project_path / "configs/providers",
            project_path / "configs/resources",
            project_path / "tools",
            project_path / "flows",
            project_path / "knowledge_plugins",
            project_path / "logs",
            project_path / "temp",
            project_path / "backups",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        _ensure_init(project_path / "__init__.py")
        _ensure_init(project_path / "agents" / "__init__.py")
        _ensure_init(project_path / "configs" / "__init__.py")
        _ensure_init(project_path / "configs/providers" / "__init__.py")
        _ensure_init(project_path / "configs/resources" / "__init__.py")
        _ensure_init(project_path / "tools" / "__init__.py")
        _ensure_init(project_path / "flows" / "__init__.py")

        _write_file(project_path / "README.md", PROJECT_README_TEMPLATE.format(project_name=name, description=description))
        # Note: aliases.py is generated by the calling service based on setup_type

        agent_scaffold = AgentScaffold(project_path)
        for agent in agent_names:
            agent_scaffold.create_agent(agent)

        tool_scaffold = ToolScaffold(project_path)
        if create_example_tools:
            for category in tool_categories:
                tool_scaffold.create_tool(name=f"{category}-helper", category=category, description=f"{category} helper tool")
        else:
            for category in tool_categories:
                category_path = project_path / "tools" / category
                category_path.mkdir(parents=True, exist_ok=True)
                _ensure_init(category_path / "__init__.py")

        return project_path


@dataclass
class AgentScaffold:
    """Create agent configuration stubs."""

    project_path: Path

    def create_agent(
        self,
        name: str,
        persona: str = "I am a helpful Flowlib agent.",
        allowed_categories: Iterable[str] | None = None,
        description: str | None = None,
    ) -> Path:
        allowed_categories = list(allowed_categories or ["generic"])
        class_name = f"{_camel_case(name)}AgentConfig"
        agent_path = self.project_path / "agents"
        agent_path.mkdir(parents=True, exist_ok=True)
        _ensure_init(agent_path / "__init__.py")

        file_path = agent_path / f"{_slugify(name)}.py"
        content = AGENT_TEMPLATE.format(
            agent_name=_slugify(name),
            class_name=class_name,
            description=description or "Agent scaffold.",
            persona=persona,
            allowed_categories=allowed_categories,
        )
        _write_file(file_path, content)
        return file_path


@dataclass
class ToolScaffold:
    """Create tool skeletons."""

    project_path: Path

    def create_tool(
        self,
        name: str,
        category: str,
        description: str,
        include_prompts: bool = False,
        include_flow: bool = False,
    ) -> Path:
        tools_root = self.project_path / "tools"
        tools_root.mkdir(parents=True, exist_ok=True)
        _ensure_init(tools_root / "__init__.py")

        category_slug = _slugify(category)
        tool_slug = _slugify(name)
        tool_class = _camel_case(tool_slug)

        category_path = tools_root / category_slug
        category_path.mkdir(parents=True, exist_ok=True)
        _ensure_init(category_path / "__init__.py")

        tool_path = category_path / tool_slug
        tool_path.mkdir(parents=True, exist_ok=True)
        _write_file(tool_path / "__init__.py", TOOL_INIT_TEMPLATE.format(tool_name=tool_slug))

        _write_file(
            tool_path / "tool.py",
            TOOL_TEMPLATE.format(
                description=description,
                class_name=tool_class,
                tool_name=tool_slug,
                category=category_slug,
            ),
        )
        _write_file(
            tool_path / "models.py",
            TOOL_MODELS_TEMPLATE.format(tool_name=tool_slug, class_name=tool_class),
        )

        if include_prompts:
            _write_file(
                tool_path / "prompts.py",
                TOOL_PROMPTS_TEMPLATE.format(tool_name=tool_slug, class_name=tool_class),
            )

        if include_flow:
            _write_file(
                tool_path / "flow.py",
                TOOL_FLOW_TEMPLATE.format(tool_name=tool_slug, class_name=tool_class),
            )

        return tool_path

