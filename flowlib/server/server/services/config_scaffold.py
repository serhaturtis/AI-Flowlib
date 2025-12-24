"""Config scaffolding utilities."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from flowlib.core.message_source_config import MessageSourceDefaults
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.models.message_source_resource import MessageSourceType

from server.models.configs import (
    MessageSourceCreateRequest,
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

    # Message Source methods
    def create_message_source(self, request: MessageSourceCreateRequest) -> ConfigApplyResponse:
        """Create a message source configuration file."""
        decorator = self.message_source_decorator(request.source_type)
        class_name = self.camel_case(request.name)
        file_name = f"{self._slugify(request.name)}.py"
        relative_path = f"configs/message_sources/{file_name}"
        content = self.render_message_source_content(
            decorator=decorator,
            name=request.name,
            source_type=request.source_type,
            enabled=request.enabled,
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

    def apply_message_source_structured(
        self,
        project_id: str,
        name: str,
        source_type: str,
        enabled: bool,
        settings: dict[str, object],
    ) -> ConfigApplyResponse:
        """Apply a message source configuration using structured fields."""
        decorator = self.message_source_decorator(source_type)
        class_name = self.camel_case(name)
        file_name = f"{self._slugify(name)}.py"
        relative_path = f"configs/message_sources/{file_name}"
        content = self.render_message_source_content(
            decorator=decorator,
            name=name,
            source_type=source_type,
            enabled=enabled,
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

    def render_message_source_content(
        self,
        *,
        decorator: str,
        name: str,
        source_type: str,
        enabled: bool,
        settings: dict[str, object],
        class_name: str,
    ) -> str:
        """Render message source Python file content."""
        # Build class attributes from settings
        attrs = [f'    enabled = {enabled}']
        for key, value in settings.items():
            # Use repr() for proper escaping of strings (handles quotes, newlines, etc.)
            attrs.append(f'    {key} = {repr(value)}')
        attrs_str = '\n'.join(attrs) if attrs else '    pass'

        return textwrap.dedent(
            f'''"""
Auto-generated message source configuration for {name}.
"""
from flowlib.resources.decorators.message_source import {decorator}


@{decorator}("{name}")
class {class_name}Source:
    """Message source: {source_type}."""
{attrs_str}
'''
        )

    def message_source_decorator(self, source_type: str) -> str:
        """Get decorator name for a message source type."""
        mapping: dict[str, str] = {
            MessageSourceType.TIMER.value: "timer_source",
            MessageSourceType.EMAIL.value: "email_source",
            MessageSourceType.WEBHOOK.value: "webhook_source",
            MessageSourceType.QUEUE.value: "queue_source",
        }
        if source_type not in mapping:
            valid_types = [t.value for t in MessageSourceType]
            raise ValueError(
                f"Unsupported message source type '{source_type}'. "
                f"Supported types: {valid_types}"
            )
        return mapping[source_type]

    def get_message_source_schema(self, source_type: str) -> SchemaResponse:
        """Get schema for a message source type.

        Uses MessageSourceDefaults from flowlib.core.message_source_config
        as the single source of truth for default values.
        """
        base_fields = [
            SchemaField(
                name="name",
                type="string",
                required=True,
                description="Unique message source name",
                default=None,
                enum=None,
                string_min_length=1,
                pattern=r"^[a-zA-Z0-9._\-]+$",
            ),
            SchemaField(
                name="enabled",
                type="boolean",
                required=False,
                description="Whether source is enabled",
                default=MessageSourceDefaults.ENABLED,
                enum=None,
            ),
        ]

        # Add source-type-specific fields (using centralized defaults)
        if source_type == MessageSourceType.TIMER.value:
            base_fields.extend([
                SchemaField(
                    name="interval_seconds",
                    type="number",
                    required=True,
                    description="Interval between triggers in seconds",
                    default=3600,  # Example value, not a default (field is required)
                    enum=None,
                    numeric_min=1,
                ),
                SchemaField(
                    name="run_on_start",
                    type="boolean",
                    required=False,
                    description="Send message immediately on start",
                    default=MessageSourceDefaults.TIMER_RUN_ON_START,
                    enum=None,
                ),
                SchemaField(
                    name="message_content",
                    type="string",
                    required=False,
                    description="Content for timer messages",
                    default=MessageSourceDefaults.TIMER_MESSAGE_CONTENT,
                    enum=None,
                ),
            ])
        elif source_type == MessageSourceType.EMAIL.value:
            base_fields.extend([
                SchemaField(
                    name="email_provider_name",
                    type="string",
                    required=True,
                    description="Reference to email provider config",
                    default=None,
                    enum=None,
                    string_min_length=1,
                ),
                SchemaField(
                    name="check_interval_seconds",
                    type="number",
                    required=False,
                    description="Polling interval in seconds",
                    default=MessageSourceDefaults.EMAIL_CHECK_INTERVAL_SECONDS,
                    enum=None,
                    numeric_min=1,
                ),
                SchemaField(
                    name="folder",
                    type="string",
                    required=False,
                    description="Email folder to monitor",
                    default=MessageSourceDefaults.EMAIL_FOLDER,
                    enum=None,
                ),
                SchemaField(
                    name="only_unread",
                    type="boolean",
                    required=False,
                    description="Only process unread emails",
                    default=MessageSourceDefaults.EMAIL_ONLY_UNREAD,
                    enum=None,
                ),
                SchemaField(
                    name="mark_as_read",
                    type="boolean",
                    required=False,
                    description="Mark emails as read after processing",
                    default=MessageSourceDefaults.EMAIL_MARK_AS_READ,
                    enum=None,
                ),
            ])
        elif source_type == MessageSourceType.WEBHOOK.value:
            base_fields.extend([
                SchemaField(
                    name="path",
                    type="string",
                    required=True,
                    description="Webhook path (e.g., '/webhook/slack')",
                    default="/webhook/default",  # Example value, not a default (field is required)
                    enum=None,
                    string_min_length=1,
                ),
                SchemaField(
                    name="methods",
                    type="array",
                    required=False,
                    description="Allowed HTTP methods",
                    default=list(MessageSourceDefaults.WEBHOOK_METHODS),
                    enum=None,
                    items_type="string",
                ),
                SchemaField(
                    name="secret_header",
                    type="string",
                    required=False,
                    description="Header name for secret validation",
                    default=MessageSourceDefaults.WEBHOOK_SECRET_HEADER,
                    enum=None,
                ),
            ])
        elif source_type == MessageSourceType.QUEUE.value:
            base_fields.extend([
                SchemaField(
                    name="queue_provider_name",
                    type="string",
                    required=True,
                    description="Reference to queue provider (Redis, RabbitMQ)",
                    default=None,
                    enum=None,
                    string_min_length=1,
                ),
                SchemaField(
                    name="queue_name",
                    type="string",
                    required=True,
                    description="Queue name to consume from",
                    default=None,
                    enum=None,
                    string_min_length=1,
                ),
                SchemaField(
                    name="consumer_group",
                    type="string",
                    required=False,
                    description="Consumer group for load balancing",
                    default=MessageSourceDefaults.QUEUE_CONSUMER_GROUP,
                    enum=None,
                ),
            ])
        else:
            valid_types = [t.value for t in MessageSourceType]
            raise ValueError(
                f"Unsupported message source type '{source_type}'. "
                f"Supported types: {valid_types}"
            )

        return SchemaResponse(title=f"{source_type}_source schema", fields=base_fields)

