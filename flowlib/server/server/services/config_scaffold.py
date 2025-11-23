"""Config scaffolding utilities."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from flowlib.resources.models.constants import ResourceType

from server.models.configs import (
    ProviderConfigCreateRequest,
    ResourceConfigCreateRequest,
    SchemaField,
    SchemaResponse,
)
from server.models.diff import ConfigApplyRequest, ConfigApplyResponse
from server.services.diff_service import DiffService

if TYPE_CHECKING:
    from server.services.config_service import ConfigService


class ConfigScaffoldService:
    """Create provider/resource configs with validation."""

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

    def create_provider_config(self, request: ProviderConfigCreateRequest) -> ConfigApplyResponse:
        # Coerce resource type string to enum
        try:
            rtype = ResourceType(request.resource_type)
        except ValueError as exc:
            raise ValueError(f"Unsupported provider resource type '{request.resource_type}'") from exc
        decorator = self.provider_decorator(rtype)
        class_name = self.camel_case(request.name)
        file_name = f"{self._slugify(request.name)}.py"
        relative_path = f"configs/providers/{file_name}"
        content = self.render_provider_content(
            decorator=decorator,
            name=request.name,
            provider_type=request.provider_type,
            description=request.description,
            settings=request.settings,
            class_name=class_name,
        )

        return self._diff_service.apply_config(
            ConfigApplyRequest(
                project_id=request.project_id,
                relative_path=relative_path,
                content=content,
                sha256_before=self._EMPTY_SHA256,
            )
        )

    def create_resource_config(self, request: ResourceConfigCreateRequest) -> ConfigApplyResponse:
        # Coerce resource type string to enum
        try:
            rtype = ResourceType(request.resource_type)
        except ValueError as exc:
            raise ValueError(f"Unsupported resource type '{request.resource_type}'") from exc
        if rtype != ResourceType.MODEL_CONFIG:
            raise ValueError("Currently only model_config resources are supported")

        class_name = self.camel_case(request.name)
        file_name = f"{self._slugify(request.name)}.py"
        relative_path = f"configs/resources/{file_name}"
        content = self.render_resource_content(
            name=request.name,
            provider_type=request.provider_type,
            description=request.description,
            config=request.config,
            class_name=class_name,
        )

        return self._diff_service.apply_config(
            ConfigApplyRequest(
                project_id=request.project_id,
                relative_path=relative_path,
                content=content,
                sha256_before=self._EMPTY_SHA256,
            )
        )

    # Structured apply/update (overwrite file with rendered content)
    def apply_provider_structured(
        self,
        project_id: str,
        name: str,
        resource_type: ResourceType,
        provider_type: str,
        description: str,
        settings: dict[str, object],
    ) -> ConfigApplyResponse:
        decorator = self.provider_decorator(resource_type)
        class_name = self.camel_case(name)
        file_name = f"{self._slugify(name)}.py"
        relative_path = f"configs/providers/{file_name}"
        content = self.render_provider_content(
            decorator=decorator,
            name=name,
            provider_type=provider_type,
            description=description,
            settings=settings,
            class_name=class_name,
        )
        return self._diff_service.apply_config(
            ConfigApplyRequest(
                project_id=project_id,
                relative_path=relative_path,
                content=content,
                sha256_before=self._get_current_file_hash(project_id, relative_path),
            )
        )

    def apply_resource_structured(
        self,
        project_id: str,
        name: str,
        resource_type: ResourceType,
        provider_type: str,
        description: str,
        config: dict[str, object],
    ) -> ConfigApplyResponse:
        if resource_type != ResourceType.MODEL_CONFIG:
            raise ValueError("Currently only model_config resources are supported")
        class_name = self.camel_case(name)
        file_name = f"{self._slugify(name)}.py"
        relative_path = f"configs/resources/{file_name}"
        content = self.render_resource_content(
            name=name,
            provider_type=provider_type,
            description=description,
            config=config,
            class_name=class_name,
        )
        return self._diff_service.apply_config(
            ConfigApplyRequest(
                project_id=project_id,
                relative_path=relative_path,
                content=content,
                sha256_before=self._get_current_file_hash(project_id, relative_path),
            )
        )

    # Rendering helpers (single source of truth - public interface for API)
    def render_provider_content(
        self,
        *,
        decorator: str,
        name: str,
        provider_type: str,
        description: str,
        settings: dict[str, object],
        class_name: str,
    ) -> str:
        return textwrap.dedent(
            f'''"""
Auto-generated provider configuration for {name}.
"""
from flowlib.resources.decorators.decorators import {decorator}


@{decorator}(
    name="{name}"
)
class {class_name}ProviderConfig:
    """{description}"""
    # Provider contracts require class-level attributes, not decorator kwargs
    provider_type = "{provider_type}"
    settings = {self._format_dict(settings)}
'''
        )

    def render_resource_content(
        self,
        *,
        name: str,
        provider_type: str,
        description: str,
        config: dict[str, object],
        class_name: str,
    ) -> str:
        return textwrap.dedent(
            f'''"""
Auto-generated resource configuration for {name}.
"""
from flowlib.resources.decorators.decorators import model_config


@model_config(
    name="{name}",
    provider_type="{provider_type}",
    config={self._format_dict(config)}
)
class {class_name}ModelConfig:
    """{description}"""
'''
        )

    def provider_decorator(self, resource_type: ResourceType) -> str:
        mapping: dict[ResourceType, str] = {
            ResourceType.LLM_CONFIG: "llm_config",
            ResourceType.MULTIMODAL_LLM_CONFIG: "multimodal_llm_config",
            ResourceType.VECTOR_DB_CONFIG: "vector_db_config",
            ResourceType.DATABASE_CONFIG: "database_config",
            ResourceType.CACHE_CONFIG: "cache_config",
            ResourceType.STORAGE_CONFIG: "storage_config",
            ResourceType.EMBEDDING_CONFIG: "embedding_config",
            ResourceType.GRAPH_DB_CONFIG: "graph_db_config",
            ResourceType.MESSAGE_QUEUE_CONFIG: "message_queue_config",
        }
        if resource_type not in mapping:
            raise ValueError(f"Unsupported provider resource type '{resource_type}'")
        return mapping[resource_type]

    def _slugify(self, value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
        if not slug:
            raise ValueError("Name must contain alphanumeric characters")
        return slug

    def camel_case(self, value: str) -> str:
        parts = re.split(r"[^a-zA-Z0-9]", value)
        camel = "".join(part.capitalize() for part in parts if part)
        if not camel:
            raise ValueError("Name must contain letters or numbers for class generation")
        return camel

    def _format_dict(self, data: dict[str, object]) -> str:
        return json.dumps(data, indent=4, ensure_ascii=False)

    def get_provider_schema(self, resource_type: ResourceType) -> SchemaResponse:
        decorator = self.provider_decorator(resource_type)
        fields = [
            SchemaField(
                name="name",
                type="string",
                required=True,
                description="Canonical provider configuration name",
                default=None,
                enum=None,
                string_min_length=1,
                pattern=r"^[a-zA-Z0-9._\-]+$",
            ),
            SchemaField(
                name="provider_type",
                type="string",
                required=True,
                description="Provider implementation identifier (e.g., llamacpp)",
                default=None,
                enum=None,
                string_min_length=1,
            ),
            SchemaField(
                name="settings",
                type="object",
                required=True,
                description="Provider-specific settings dict",
                default={
                    "model_path": "/path/to/model.bin",
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                enum=None,
                children=[
                    SchemaField(
                        name="model_path",
                        type="string",
                        required=True,
                        description="Path to model file",
                        default="/path/to/model.bin",
                        enum=None,
                        string_min_length=1,
                    ),
                    SchemaField(
                        name="temperature",
                        type="number",
                        required=False,
                        description="Generation temperature",
                        default=0.7,
                        enum=None,
                        numeric_min=0.0,
                        numeric_max=1.0,
                    ),
                    SchemaField(
                        name="max_tokens",
                        type="integer",
                        required=False,
                        description="Maximum tokens to generate",
                        default=1024,
                        enum=None,
                        numeric_min=1,
                    ),
                ],
            ),
        ]
        return SchemaResponse(title=f"{decorator} schema", fields=fields)

    def get_resource_schema(self, resource_type: ResourceType) -> SchemaResponse:
        if resource_type != ResourceType.MODEL_CONFIG:
            raise ValueError("Currently only model_config schema is available")
        fields = [
            SchemaField(
                name="name",
                type="string",
                required=True,
                description="Canonical resource name",
                default=None,
                enum=None,
                string_min_length=1,
                pattern=r"^[a-zA-Z0-9._\-]+$",
            ),
            SchemaField(
                name="provider_type",
                type="string",
                required=True,
                description="Provider type reference (e.g., llamacpp)",
                default=None,
                enum=None,
                string_min_length=1,
            ),
            SchemaField(
                name="config",
                type="object",
                required=True,
                description="Model configuration data (path, temperature, etc.)",
                default={
                    "model_name": "example-model",
                    "model_path": "/path/to/model.bin",
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                enum=None,
                children=[
                    SchemaField(
                        name="path",
                        type="string",
                        required=False,
                        description="Path to model file (alias of model_path)",
                        default="/path/to/model.bin",
                        enum=None,
                        string_min_length=1,
                    ),
                    SchemaField(
                        name="model_path",
                        type="string",
                        required=False,
                        description="Path to model file",
                        default="/path/to/model.bin",
                        enum=None,
                        string_min_length=1,
                    ),
                    SchemaField(
                        name="model_name",
                        type="string",
                        required=False,
                        description="Model identifier used by the agent",
                        default="example-model",
                        enum=None,
                        string_min_length=1,
                    ),
                    SchemaField(
                        name="chat_format",
                        type="string",
                        required=False,
                        description="Chat formatting style (e.g., chatml)",
                        default="chatml",
                        enum=None,
                    ),
                    SchemaField(
                        name="temperature",
                        type="number",
                        required=False,
                        description="Generation temperature",
                        default=0.7,
                        enum=None,
                        numeric_min=0.0,
                        numeric_max=1.0,
                    ),
                    SchemaField(
                        name="max_tokens",
                        type="integer",
                        required=False,
                        description="Maximum tokens to generate (0 to disable limit)",
                        default=1024,
                        enum=None,
                        numeric_min=0,
                    ),
                    SchemaField(
                        name="top_p",
                        type="number",
                        required=False,
                        description="Top-p nucleus sampling",
                        default=0.95,
                        enum=None,
                        numeric_min=0.0,
                        numeric_max=1.0,
                    ),
                    SchemaField(
                        name="top_k",
                        type="integer",
                        required=False,
                        description="Top-k sampling",
                        default=40,
                        enum=None,
                        numeric_min=0,
                    ),
                    SchemaField(
                        name="repeat_penalty",
                        type="number",
                        required=False,
                        description="Penalty for repeated tokens",
                        default=1.1,
                        enum=None,
                        numeric_min=0.0,
                    ),
                    SchemaField(
                        name="n_ctx",
                        type="integer",
                        required=False,
                        description="Context window size",
                        default=4096,
                        enum=None,
                        numeric_min=1,
                    ),
                    SchemaField(
                        name="use_gpu",
                        type="boolean",
                        required=False,
                        description="Enable GPU acceleration",
                        default=True,
                        enum=None,
                    ),
                    SchemaField(
                        name="n_gpu_layers",
                        type="integer",
                        required=False,
                        description="Number of layers to offload to GPU (-1 for all)",
                        default=-1,
                        enum=None,
                    ),
                    SchemaField(
                        name="n_threads",
                        type="integer",
                        required=False,
                        description="Number of CPU threads",
                        default=4,
                        enum=None,
                        numeric_min=1,
                    ),
                    SchemaField(
                        name="n_batch",
                        type="integer",
                        required=False,
                        description="Batch size",
                        default=512,
                        enum=None,
                        numeric_min=1,
                    ),
                ],
            ),
        ]
        return SchemaResponse(title="model_config schema", fields=fields)

