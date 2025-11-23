"""Project structure validation utilities."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class ValidationIssue:
    """Represents a single validation problem."""

    path: str
    message: str


@dataclass
class ValidationResult:
    """Encapsulates validation output."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def add(self, path: Path, message: str) -> None:
        self.issues.append(ValidationIssue(str(path), message))


class ToolValidator:
    """Validate tool directory structure."""

    REQUIRED_FILES = ("tool.py", "models.py")

    def validate(self, tool_path: Path) -> Iterable[ValidationIssue]:
        result: list[ValidationIssue] = []
        for file_name in self.REQUIRED_FILES:
            candidate = tool_path / file_name
            if not candidate.exists():
                result.append(
                    ValidationIssue(str(candidate), f"Missing required tool file '{file_name}'"),
                )
        return result


class AgentValidator:
    """Validate agent configuration files."""

    def validate(self, agents_path: Path) -> Iterable[ValidationIssue]:
        # Empty projects are allowed - agents can be created later
        # Just validate that existing agent files are syntactically correct if needed
        return []


class ProjectValidator:
    """Validate Flowlib project structure before loading."""

    def __init__(self) -> None:
        self._tool_validator = ToolValidator()
        self._agent_validator = AgentValidator()

    def validate(self, project_path: Path) -> ValidationResult:
        result = ValidationResult()

        self._validate_required_directories(project_path, result)
        self._validate_alias_file(project_path, result)
        self._validate_config_layout(project_path, result)
        self._validate_provider_files(project_path, result)
        self._validate_agents(project_path, result)
        self._validate_tools(project_path, result)

        return result

    def _validate_required_directories(self, project_path: Path, result: ValidationResult) -> None:
        required_dirs = [
            project_path / "agents",
            project_path / "configs",
            project_path / "configs/providers",
            project_path / "configs/resources",
        ]

        for directory in required_dirs:
            if not directory.exists():
                result.add(directory, "Required directory is missing")

    def _validate_alias_file(self, project_path: Path, result: ValidationResult) -> None:
        alias_file = project_path / "configs" / "aliases.py"
        if not alias_file.exists():
            result.add(alias_file, "Missing required alias bindings file")

    def _validate_config_layout(self, project_path: Path, result: ValidationResult) -> None:
        configs_root = project_path / "configs"
        for file_path in configs_root.glob("*.py"):
            if file_path.name in {"__init__.py", "aliases.py"}:
                continue
            result.add(
                file_path,
                "Unexpected config file at configs/ root. Place provider configs under "
                "configs/providers/ and resource configs under configs/resources/.",
            )

    def _validate_agents(self, project_path: Path, result: ValidationResult) -> None:
        agents_path = project_path / "agents"
        if not agents_path.exists():
            result.add(agents_path, "Agents directory missing")
            return

        for issue in self._agent_validator.validate(agents_path):
            result.issues.append(issue)

    def _validate_tools(self, project_path: Path, result: ValidationResult) -> None:
        tools_root = project_path / "tools"
        if not tools_root.exists():
            return

        for category_dir in tools_root.iterdir():
            if not category_dir.is_dir():
                continue
            for tool_dir in category_dir.iterdir():
                if not tool_dir.is_dir():
                    continue
                if not (tool_dir / "tool.py").exists():
                    # Skip directories that are not tool packages
                    continue
                for issue in self._tool_validator.validate(tool_dir):
                    result.issues.append(issue)

    def _validate_provider_files(self, project_path: Path, result: ValidationResult) -> None:
        providers_root = project_path / "configs" / "providers"
        resources_root = project_path / "configs" / "resources"

        for provider_file in providers_root.glob("*.py"):
            if provider_file.name == "__init__.py":
                continue
            decorators = self._extract_decorators(provider_file)
            if {"model_config", "embedding_model_config"} & decorators:
                result.add(
                    provider_file,
                    "Resource config detected under configs/providers/. Move to configs/resources/.",
                )

        for resource_file in resources_root.glob("*.py"):
            if resource_file.name == "__init__.py":
                continue
            decorators = self._extract_decorators(resource_file)
            provider_tags = {
                "llm_config",
                "multimodal_llm_config",
                "vector_db_config",
                "database_config",
                "cache_config",
                "storage_config",
                "embedding_config",
                "graph_db_config",
                "message_queue_config",
            }
            if provider_tags & decorators:
                result.add(
                    resource_file,
                    "Provider config detected under configs/resources/. Move to configs/providers/.",
                )

    def _extract_decorators(self, file_path: Path) -> set[str]:
        try:
            tree = ast.parse(file_path.read_text())
        except SyntaxError:
            return set()

        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for decorator in node.decorator_list:
                    target = decorator
                    if isinstance(decorator, ast.Call):
                        target = decorator.func
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        names.add(target.attr)
        return names

