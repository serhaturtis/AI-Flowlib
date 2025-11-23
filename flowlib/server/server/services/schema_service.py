"""Dynamic schema service that derives field schemas from live project configs."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import importlib
import inspect

from flowlib.resources.models.constants import ResourceType
from flowlib.resources.registry.registry import resource_registry
from flowlib.core.project.project import Project
from flowlib.utils.pydantic.schema import model_to_schema_dict
from pydantic import BaseModel
from flowlib.providers.core.registry import provider_registry

from server.core.registry_lock import registry_lock
from server.models.configs import SchemaField, SchemaResponse


class SchemaService:
    """Generate runtime schemas from actual project configurations (fail-fast, no hardcoding)."""

    def __init__(self, projects_root: str = "./projects") -> None:
        self._root = Path(projects_root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def provider_schema(self, project_id: str, resource_type: ResourceType, provider_type_hint: str | None = None) -> SchemaResponse:
        """Infer provider settings schema from the first available provider config of given type.
        If none exist and provider_type_hint is provided, return a minimal schema requiring settings."""
        # If provider_type is given, prefer strict Pydantic schema from settings_class via provider registry
        if provider_type_hint:
            strict_fields = self._schema_from_provider_settings_class(resource_type, provider_type_hint)
            if strict_fields is not None:
                # Include name and provider_type at the top, then the strict settings object
                base_fields = [
                    SchemaField(
                        name="name",
                        type="string",
                        required=True,
                        description="Canonical provider configuration name",
                        default=None,
                        enum=None,
                    ),
                    SchemaField(
                        name="provider_type",
                        type="string",
                        required=True,
                        description="Provider implementation identifier",
                        default=provider_type_hint,
                        enum=None,
                    ),
                ]
                return SchemaResponse(
                    title=f"{resource_type} schema",
                    fields=base_fields
                    + [
                        SchemaField(
                            name="settings",
                            type="object",
                            required=True,
                            description="Provider-specific settings",
                            default=None,
                            enum=None,
                            children=strict_fields,
                        )
                    ],
                )

        # Load project to populate registry
        with registry_lock:
            try:
                resource_registry.clear()
                self._load_project(project_id)
                entries = resource_registry.get_by_type(resource_type)
            finally:
                resource_registry.clear()

        if not entries:
            # No existing configs: return minimal schema so client can still create one
            minimal_fields: list[SchemaField] = [
                SchemaField(
                    name="name",
                    type="string",
                    required=True,
                    description="Canonical provider configuration name",
                    default=None,
                    enum=None,
                    string_min_length=1,
                ),
                SchemaField(
                    name="provider_type",
                    type="string",
                    required=True,
                    description="Provider implementation identifier",
                    default=provider_type_hint or None,
                    enum=None,
                    string_min_length=1,
                ),
                SchemaField(
                    name="settings",
                    type="object",
                    required=True,
                    description="Provider-specific settings",
                    default={},
                    enum=None,
                    children=[],
                ),
            ]
            return SchemaResponse(title=f"{resource_type} schema", fields=minimal_fields)
        # Use the first instance to infer keys/types
        sample = next(iter(entries.values()))
        provider_type = getattr(sample, "provider_type", "") or ""
        settings = getattr(sample, "settings", {}) or {}
        if not isinstance(settings, dict):
            raise ValueError("Provider settings must be a dict to derive schema")

        fields: list[SchemaField] = [
            SchemaField(
                name="name",
                type="string",
                required=True,
                description="Canonical provider configuration name",
                default=None,
                enum=None,
                string_min_length=1,
            ),
            SchemaField(
                name="provider_type",
                type="string",
                required=True,
                description="Provider implementation identifier",
                default=str(provider_type) if provider_type else None,
                enum=None,
                string_min_length=1,
            ),
            SchemaField(
                name="settings",
                type="object",
                required=True,
                description="Provider-specific settings",
                default=settings,
                enum=None,
                children=self._infer_object_children(settings),
            ),
        ]
        return SchemaResponse(title=f"{resource_type} schema", fields=fields)

    def agent_schema(self, project_id: str) -> SchemaResponse:
        """Return agent config schema derived from Flowlib's AgentConfig Pydantic model."""
        # Import AgentConfig at runtime to avoid circulars
        try:
            from flowlib.agent.models.config import AgentConfig  # type: ignore
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            raise ValueError(f"Failed to import AgentConfig: {exc}") from exc
        schema_dict = model_to_schema_dict(AgentConfig)
        fields = self._schema_properties_to_fields(schema_dict)
        # Ensure ordering and include only editable fields present in scaffold
        wanted = {
            "persona",
            "allowed_tool_categories",
            "model_name",
            "llm_name",
            "temperature",
            "max_iterations",
            "enable_learning",
            "verbose",
        }
        filtered = [f for f in fields if f.name in wanted]
        # allowed_tool_categories should be an array of strings; if not present as object, keep as array primitives
        return SchemaResponse(title="agent_config schema", fields=filtered)

    def resource_schema(self, project_id: str, resource_type: ResourceType, provider_type_hint: str | None = None) -> SchemaResponse:
        """Infer resource config schema for resources like model_config.
        Priority:
        1) If provider_type_hint is provided, attempt strict schema via provider's model config Pydantic class.
        2) Otherwise infer from first available resource instance of given type.
        3) If none exist, return minimal schema requiring provider_type and config."""
        # If provider_type is given, prefer strict Pydantic schema derived from provider's model config class
        if provider_type_hint:
            strict_fields = self._schema_from_model_config_class(resource_type, provider_type_hint)
            if strict_fields is not None:
                base_fields = [
                    SchemaField(
                        name="name",
                        type="string",
                        required=True,
                        description="Canonical resource name",
                        default=None,
                        enum=None,
                    ),
                    SchemaField(
                        name="provider_type",
                        type="string",
                        required=True,
                        description="Provider type reference",
                        default=provider_type_hint,
                        enum=None,
                    ),
                ]
                return SchemaResponse(
                    title=f"{resource_type} schema",
                    fields=base_fields
                    + [
                        SchemaField(
                            name="config",
                            type="object",
                            required=True,
                            description="Model configuration",
                            default=None,
                            enum=None,
                            children=strict_fields,
                        )
                    ],
                )

        # Load project to populate registry
        with registry_lock:
            try:
                resource_registry.clear()
                self._load_project(project_id)
                entries = resource_registry.get_by_type(resource_type)
            finally:
                resource_registry.clear()

        if not entries:
            minimal_resource_fields: list[SchemaField] = [
                SchemaField(
                    name="name",
                    type="string",
                    required=True,
                    description="Canonical resource name",
                    default=None,
                    enum=None,
                    string_min_length=1,
                ),
                SchemaField(
                    name="provider_type",
                    type="string",
                    required=True,
                    description="Provider type reference",
                    default=provider_type_hint or None,
                    enum=None,
                    string_min_length=1,
                ),
                SchemaField(
                    name="config",
                    type="object",
                    required=True,
                    description="Model configuration",
                    default={},
                    enum=None,
                    children=[],
                ),
            ]
            return SchemaResponse(title=f"{resource_type} schema", fields=minimal_resource_fields)
        sample = next(iter(entries.values()))
        metadata: dict[str, Any] = getattr(sample, "metadata", {}) or getattr(sample, "__dict__", {})
        provider_type = metadata.get("provider_type")
        config = metadata.get("config") or {}
        if not isinstance(config, dict):
            raise ValueError("Resource config must be a dict to derive schema")

        fields: list[SchemaField] = [
            SchemaField(
                name="name",
                type="string",
                required=True,
                description="Canonical resource name",
                default=None,
                enum=None,
                string_min_length=1,
            ),
            SchemaField(
                name="provider_type",
                type="string",
                required=True,
                description="Provider type reference",
                default=str(provider_type) if provider_type else None,
                enum=None,
                string_min_length=1,
            ),
            SchemaField(
                name="config",
                type="object",
                required=True,
                description="Model configuration",
                default=config,
                enum=None,
                children=self._infer_object_children(config),
            ),
        ]
        return SchemaResponse(title=f"{resource_type} schema", fields=fields)

    def _infer_object_children(self, obj: dict[str, Any]) -> list[SchemaField]:
        children: list[SchemaField] = []
        for key, val in obj.items():
            ftype, default, enum = self._infer_type(val)
            children.append(
                SchemaField(
                    name=key,
                    type=ftype,
                    required=False,
                    description=None,
                    default=default,
                    enum=enum,
                )
            )
        return children

    def _schema_from_provider_settings_class(
        self, resource_type: ResourceType, provider_type: str
    ) -> list[SchemaField] | None:
        """Get provider settings schema from the registry (no project configs needed)."""
        # provider_registry keeps factories keyed by (category, provider_type). We need the category.
        category = self._infer_provider_category(resource_type)

        # Get settings class directly from registry (single source of truth)
        settings_class = provider_registry.get_settings_class(category, provider_type)

        if settings_class is None:
            return None

        # Generate schema from settings class - let errors propagate (fail-fast)
        schema_dict = model_to_schema_dict(settings_class)
        return self._schema_properties_to_fields(schema_dict)

    def _infer_provider_category(self, resource_type: ResourceType) -> str:
        mapping: dict[ResourceType, str] = {
            ResourceType.LLM_CONFIG: "llm",
            ResourceType.MULTIMODAL_LLM_CONFIG: "multimodal_llm",
            ResourceType.VECTOR_DB_CONFIG: "vector_db",
            ResourceType.DATABASE_CONFIG: "db",
            ResourceType.CACHE_CONFIG: "cache",
            ResourceType.STORAGE_CONFIG: "storage",
            ResourceType.EMBEDDING_CONFIG: "embedding",
            ResourceType.GRAPH_DB_CONFIG: "graph_db",
            ResourceType.MESSAGE_QUEUE_CONFIG: "mq",
        }
        if resource_type not in mapping:
            raise ValueError(f"Unsupported provider resource type '{resource_type}'")
        return mapping[resource_type]

    def _schema_properties_to_fields(self, schema: dict[str, Any]) -> list[SchemaField]:
        """Convert JSON schema dict (from Pydantic) to SchemaField list (for object properties)."""
        props = (schema or {}).get("properties", {})
        required_list = set((schema or {}).get("required", []) or [])
        fields: list[SchemaField] = []
        for name, prop in props.items():
            ftype, allowed_types = self._jsonschema_type_to_field_type_with_union(prop)
            description = prop.get("description")
            default = prop.get("default", None)
            enum = prop.get("enum", None)
            # constraints
            string_min_length = prop.get("minLength")
            string_max_length = prop.get("maxLength")
            numeric_min = prop.get("minimum")
            numeric_max = prop.get("maximum")
            pattern = prop.get("pattern")
            children = None
            items_type_hint = None
            item_union = None
            if ftype == "object":
                children = self._schema_properties_to_fields(prop)
            # Arrays: expose object item schemas via children
            if ftype == "array":
                items = prop.get("items", {}) or {}
                item_type, item_union = self._jsonschema_type_to_field_type_with_union(items) if items else (None, None)
                items_type_hint = item_type
                if item_type == "object":
                    # Children represent the fields of the object items
                    children = self._schema_properties_to_fields(items)
            fields.append(
                SchemaField(
                    name=name,
                    type=ftype,
                    required=name in required_list,
                    description=description,
                    default=default,
                    enum=enum,
                    string_min_length=string_min_length,
                    string_max_length=string_max_length,
                    numeric_min=numeric_min,
                    numeric_max=numeric_max,
                    pattern=pattern,
                    children=children,
                    items_type=items_type_hint,
                    items_allowed_types=item_union,
                    allowed_types=allowed_types,
                )
            )
        return fields

    def _jsonschema_type_to_field_type(self, prop: dict[str, Any]) -> str:
        t = prop.get("type")
        if isinstance(t, list):
            # Choose first non-nullable type
            non_null_types = [x for x in t if x != "null"]
            t = non_null_types[0] if non_null_types else "string"
        # Ensure t is a string for dict lookup
        type_str = str(t) if t is not None else "string"
        return {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "object": "object",
            "array": "array",
        }.get(type_str, "string")

    def _jsonschema_type_to_field_type_with_union(self, prop: dict[str, Any]) -> tuple[str, list[str] | None]:
        t = prop.get("type")
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            mapped = [self._jsonschema_type_to_field_type({"type": x}) for x in non_null]
            return (mapped[0] if mapped else "string", mapped or None)
        # Handle anyOf/oneOf primitive unions
        for key in ("anyOf", "oneOf"):
            if key in prop and isinstance(prop[key], list):
                variants = prop[key]
                primitive_types = []
                for v in variants:
                    vt = v.get("type")
                    if isinstance(vt, str):
                        primitive_types.append(self._jsonschema_type_to_field_type({"type": vt}))
                if primitive_types:
                    return (primitive_types[0], primitive_types)
        return (self._jsonschema_type_to_field_type(prop), None)

    def _infer_type(self, value: Any) -> tuple[str, Any, list[str] | None]:
        if isinstance(value, bool):
            return "boolean", value, None
        if isinstance(value, int) and not isinstance(value, bool):
            return "integer", value, None
        if isinstance(value, float):
            return "number", value, None
        if isinstance(value, str):
            return "string", value, None
        if isinstance(value, dict):
            # Nested object: recurse
            return "object", value, None
        if isinstance(value, list):
            # Represent arrays as string for first cut; future: item schemas
            return "array", value, None
        # Fallback: string representation
        return "string", str(value), None

    def _load_project(self, project_id: str) -> Project:
        project_path = (self._root / project_id).resolve()
        if not project_path.exists() or not project_path.is_dir():
            raise FileNotFoundError(f"Project '{project_id}' not found under {self._root}")
        project = Project(str(project_path))
        project.initialize()
        project.load_configurations()
        return project

    def _schema_from_model_config_class(
        self, resource_type: ResourceType, provider_type: str
    ) -> list[SchemaField] | None:
        """Get model config schema from category-level models module (e.g., flowlib.providers.llm.models).
        Model config classes are shared across providers of the same category, not provider-specific."""
        # Only applies to model-like resource types
        model_like_types = {
            ResourceType.MODEL_CONFIG,
            ResourceType.MULTIMODAL_LLM_CONFIG,
            ResourceType.EMBEDDING_CONFIG,
        }
        if resource_type not in model_like_types:
            return None
        category = self._infer_provider_category(
            ResourceType.LLM_CONFIG
            if resource_type in {ResourceType.MODEL_CONFIG, ResourceType.MULTIMODAL_LLM_CONFIG}
            else ResourceType.EMBEDDING_CONFIG
        )
        # Model config classes are at category level, NOT provider level
        # e.g., flowlib.providers.llm.models (not flowlib.providers.llm.llama_cpp.models)
        module_path = f"flowlib.providers.{category}.models"
        try:
            mod = importlib.import_module(module_path)
        except (ImportError, ModuleNotFoundError):
            # Module doesn't exist for this category - this is expected for some types
            return None
        # Find Pydantic BaseModel subclass ending with 'ModelConfig'
        candidates: list[type[BaseModel]] = []
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            try:
                if issubclass(obj, BaseModel) and obj.__name__.endswith("ModelConfig"):
                    candidates.append(obj)
            except TypeError:
                # issubclass raises TypeError for non-class types - skip them
                continue
        if not candidates:
            return None
        # Match provider_type to model config class name
        # e.g., "llamacpp" -> LlamaModelConfig, "google_ai" -> GoogleAIModelConfig
        # Use explicit mapping for known providers, then fuzzy match
        provider_to_class_map = {
            "llamacpp": "LlamaModelConfig",
            "google_ai": "GoogleAIModelConfig",
        }

        chosen = None
        # First try exact mapping
        if provider_type in provider_to_class_map:
            target_class_name = provider_to_class_map[provider_type]
            for cls in candidates:
                if cls.__name__ == target_class_name:
                    chosen = cls
                    break

        # Fallback: fuzzy match by checking if provider_type substring is in class name
        if chosen is None:
            normalized_provider = provider_type.replace("-", "").replace("_", "").lower()
            for cls in candidates:
                class_name_lower = cls.__name__.lower()
                # Check if any significant part of provider matches class name
                if normalized_provider in class_name_lower or class_name_lower.replace("modelconfig", "") in normalized_provider:
                    chosen = cls
                    break

        # Last resort: use LlamaModelConfig as default if available (most common), otherwise shortest
        if chosen is None:
            for cls in candidates:
                if cls.__name__ == "LlamaModelConfig":
                    chosen = cls
                    break
            if chosen is None:
                chosen = sorted(candidates, key=lambda c: len(c.__name__))[0]
        schema_dict = model_to_schema_dict(chosen)
        return self._schema_properties_to_fields(schema_dict)


