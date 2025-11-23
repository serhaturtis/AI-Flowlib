from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from flowlib.resources.models.constants import ResourceType

if TYPE_CHECKING:
    from flowlib.resources.registry.registry import ResourceRegistry


def _get_resource_registry() -> "ResourceRegistry":
    """Lazy import of resource registry to avoid circular dependencies."""
    from flowlib.resources.registry.registry import resource_registry

    return resource_registry


def _block_direct_instantiation(cls: type, name: str, resource_type: str) -> None:
    """Prevent direct instantiation of resource classes.

    Args:
        cls: The class to protect
        name: The resource name for the error message
        resource_type: The type of resource for the error message
    """

    def __init_blocked(self: Any, *args: Any, **kwargs: Any) -> None:
        """Prevent direct instantiation of resource classes."""
        raise RuntimeError(
            f"Cannot instantiate {resource_type} class '{cls.__name__}' directly.\n"
            f"Resources must be retrieved from the resource registry:\n"
            f"  resource = resource_registry.get('{name}')\n\n"
            f"This ensures single source of truth and prevents duplicate instances."
        )

    cls.__init__ = __init_blocked  # type: ignore[method-assign]


def resource(
    name: str, resource_type: str = ResourceType.MODEL_CONFIG, **metadata: Any
) -> Callable[[type], type]:
    """Register a class or function as a resource.

    This decorator wraps plain classes in ConfigResource.
    ResourceBase is internal to Flowlib and should not be used externally.
    """

    def decorator(obj: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Attach metadata to class first
        obj.__resource_name__ = name  # type: ignore[attr-defined]
        obj.__resource_type__ = resource_type  # type: ignore[attr-defined]
        obj.__resource_metadata__ = {"name": name, "type": resource_type, **metadata}  # type: ignore[attr-defined]

        # Always wrap in ConfigResource
        from flowlib.resources.models.config_resource import ConfigResource

        instance = ConfigResource(name=name, type=resource_type)

        registry.register(name=name, obj=instance, resource_type=resource_type, **metadata)
        _block_direct_instantiation(obj, name, "resource")
        return obj

    return decorator


def model_config(
    name: str,
    provider_type: str,
    config: dict[str, Any] | None = None,
    **metadata: Any,
) -> Callable[[type], type]:
    """Register a class as a model configuration resource.
    Wraps plain classes in ModelResource. ResourceBase is internal only.

    Args:
        name: Unique name for the model config
        provider_type: Provider implementation to use (e.g., "llamacpp", "google_ai", "openai")
        config: Additional model configuration
        **metadata: Additional metadata
    """

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Explicit config required - no fallbacks
        if config is None:
            raise ValueError(f"model_config '{name}' requires explicit config parameter")

        # Attach metadata to class - store all config data as class attributes (single source of truth)
        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.MODEL_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {
            "name": name,
            "type": ResourceType.MODEL_CONFIG,
            "provider_type": provider_type,
            "config": config,
            **metadata,
        }  # type: ignore[attr-defined]

        # Store provider_type and config as class attributes for single source of truth
        cls.__provider_type__ = provider_type  # type: ignore[attr-defined]
        cls.__config__ = config  # type: ignore[attr-defined]

        # Add properties to expose config data from class attributes
        def config_prop(self: Any) -> dict[str, Any]:
            """Access config from class attributes (single source of truth)."""
            return getattr(self.__class__, "__config__", {})

        def provider_type_prop(self: Any) -> str | None:
            """Access provider_type from class attributes (single source of truth)."""
            return getattr(self.__class__, "__provider_type__", None)

        cls.config = property(config_prop)  # type: ignore[attr-defined]
        cls.provider_type = property(provider_type_prop)  # type: ignore[attr-defined]

        # Always wrap in ModelResource
        from flowlib.resources.models.model_resource import ModelResource

        instance = ModelResource(
            name=name,
            provider_type=provider_type,
            config=config,
            type="model_config",
            model_path=config.get("model_path"),
            model_name=config.get("model_name", name),
            chat_format=config.get("chat_format"),
        )
        registry.register(
            name=name, obj=instance, resource_type=ResourceType.MODEL_CONFIG, **metadata
        )

        # Prevent direct instantiation
        _block_direct_instantiation(cls, name, "model config")

        return cls

    return decorator


def prompt(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a prompt resource.

    This decorator automatically wraps non-ResourceBase classes in PromptResource,
    ensuring all prompts conform to the PromptTemplate protocol.

    Args:
        name: Unique name for the prompt
        **metadata: Additional metadata about the prompt

    Returns:
        Decorator function that returns a class conforming to PromptTemplate

    Raises:
        ValueError: If the decorated object does not have a 'template' attribute
    """

    def decorator(obj: type) -> type:
        registry = _get_resource_registry()

        # Check if template exists before registration using safe attribute access
        # Support both Pydantic models and plain classes with Field() annotations
        template_exists = False
        template_value = None
        try:
            # FIRST: Check for template in Pydantic model fields (handles Pydantic inheritance)
            if getattr(obj, "model_fields", None) and "template" in obj.model_fields:  # type: ignore[attr-defined]
                template_exists = True
                field_info = obj.model_fields["template"]  # type: ignore[attr-defined]
                if hasattr(field_info, "default"):
                    template_value = field_info.default
            # SECOND: Check in class __dict__ for Field() definitions (plain classes)
            elif "template" in obj.__dict__:
                template_exists = True
                attr_value = obj.__dict__["template"]
                if hasattr(attr_value, "default"):
                    template_value = attr_value.default
                else:
                    template_value = attr_value
            # THIRD: Check for inherited template attribute (plain class inheritance)
            elif getattr(obj, "template", None) is not None:
                template_exists = True
                template_value = obj.template
                # For plain class inheritance, getattr returns the parent's FieldInfo
                if hasattr(template_value, "default"):
                    template_value = template_value.default
        except (AttributeError, TypeError):
            pass

        if not template_exists:
            raise ValueError(f"Prompt '{name}' must have a 'template' attribute")

        # Attach metadata to class
        obj.__resource_name__ = name  # type: ignore[attr-defined]
        obj.__resource_type__ = ResourceType.PROMPT_CONFIG  # type: ignore[attr-defined]
        obj.__resource_metadata__ = {"name": name, "type": ResourceType.PROMPT, **metadata}  # type: ignore[attr-defined]

        # Get config if it exists (support both Pydantic and plain class inheritance)
        config_value = None
        try:
            # FIRST: Check in Pydantic model fields (handles Pydantic inheritance)
            if getattr(obj, "model_fields", None) and "config" in obj.model_fields:  # type: ignore[attr-defined]
                field_info = obj.model_fields["config"]  # type: ignore[attr-defined]
                if hasattr(field_info, "default"):
                    config_value = field_info.default
            # SECOND: Check in class __dict__ (plain classes)
            elif "config" in obj.__dict__:
                attr_value = obj.__dict__["config"]
                if hasattr(attr_value, "default"):
                    config_value = attr_value.default
                else:
                    config_value = attr_value
            # THIRD: Check for inherited attribute (plain class inheritance)
            elif getattr(obj, "config", None) is not None:
                attr_config = obj.config
                # For plain class inheritance, getattr returns the parent's FieldInfo
                if hasattr(attr_config, "default"):
                    config_value = attr_config.default
                else:
                    config_value = attr_config
        except (AttributeError, TypeError):
            pass

        # Register with global registry if available
        if registry is not None:
            # Check if already registered (handle duplicate imports gracefully)
            try:
                existing = registry.get(name)
                if existing is not None:
                    # Already registered, skip
                    return obj
            except Exception:
                # Not registered, continue
                pass

            # Always wrap in PromptResource
            from flowlib.resources.models.prompt_resource import PromptResource

            instance = PromptResource(
                name=name,
                type="prompt_config",
                template=template_value or "",
                config=config_value
                if config_value is not None
                else {
                    "max_tokens": 2048,
                    "temperature": 0.5,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                },
            )

            registry.register(
                name=name, obj=instance, resource_type=ResourceType.PROMPT_CONFIG, **metadata
            )

        # Prevent direct instantiation
        _block_direct_instantiation(obj, name, "prompt")

        # This object now conforms to PromptTemplate protocol
        return obj

    return decorator


def config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a configuration resource.

    This decorator automatically wraps non-ResourceBase classes in ConfigResource.

    Args:
        name: Unique name for the config
        **metadata: Additional metadata about the config

    Returns:
        Decorator function
    """

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Attach metadata to class
        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {"name": name, "type": ResourceType.CONFIG, **metadata}  # type: ignore[attr-defined]

        # Always wrap in ConfigResource
        from flowlib.resources.models.config_resource import ConfigResource

        instance = ConfigResource(name=name, type=ResourceType.CONFIG)

        registry.register(name=name, obj=instance, resource_type=ResourceType.CONFIG, **metadata)

        _block_direct_instantiation(cls, name, "config")
        return cls

    return decorator


def llm_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as an LLM configuration resource.

    Args:
        name: Unique name for the LLM config (e.g., RequiredAlias.DEFAULT_LLM.value, 'fast-chat')
        **metadata: Additional metadata about the config
    """

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Extract provider_type and settings from class attributes
        provider_type = getattr(cls, "provider_type", "")
        settings = getattr(cls, "settings", {})

        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.LLM_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {"name": name, "type": ResourceType.LLM_CONFIG, **metadata}  # type: ignore[attr-defined]

        # Always wrap in LLMConfigResource
        from flowlib.resources.models.config_resource import LLMConfigResource

        instance = LLMConfigResource(
            name=name, type=ResourceType.LLM_CONFIG, provider_type=provider_type, settings=settings
        )

        registry.register(
            name=name, obj=instance, resource_type=ResourceType.LLM_CONFIG, **metadata
        )
        _block_direct_instantiation(cls, name, "LLM config")
        return cls

    return decorator


def database_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a database configuration resource."""

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Extract provider_type and settings from class attributes
        provider_type = getattr(cls, "provider_type", "")
        settings = getattr(cls, "settings", {})

        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.DATABASE_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {"name": name, "type": ResourceType.DATABASE_CONFIG, **metadata}  # type: ignore[attr-defined]

        # Always wrap in DatabaseConfigResource
        from flowlib.resources.models.config_resource import DatabaseConfigResource

        instance = DatabaseConfigResource(
            name=name,
            type=ResourceType.DATABASE_CONFIG,
            provider_type=provider_type,
            settings=settings,
        )

        registry.register(
            name=name, obj=instance, resource_type=ResourceType.DATABASE_CONFIG, **metadata
        )
        _block_direct_instantiation(cls, name, "database config")
        return cls

    return decorator


def vector_db_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a vector database configuration resource."""

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Extract provider_type and settings from class attributes
        provider_type = getattr(cls, "provider_type", "")
        settings = getattr(cls, "settings", {})

        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.VECTOR_DB_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {
            "name": name,
            "type": ResourceType.VECTOR_DB_CONFIG,
            **metadata,
        }  # type: ignore[attr-defined]

        # Always wrap in VectorDBConfigResource
        from flowlib.resources.models.config_resource import VectorDBConfigResource

        instance = VectorDBConfigResource(
            name=name,
            type=ResourceType.VECTOR_DB_CONFIG,
            provider_type=provider_type,
            settings=settings,
        )

        registry.register(
            name=name, obj=instance, resource_type=ResourceType.VECTOR_DB_CONFIG, **metadata
        )
        _block_direct_instantiation(cls, name, "vector DB config")
        return cls

    return decorator


def cache_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a cache configuration resource."""

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Extract provider_type and settings from class attributes
        provider_type = getattr(cls, "provider_type", "")
        settings = getattr(cls, "settings", {})

        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.CACHE_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {"name": name, "type": ResourceType.CACHE_CONFIG, **metadata}  # type: ignore[attr-defined]

        # Always wrap in CacheConfigResource
        from flowlib.resources.models.config_resource import CacheConfigResource

        instance = CacheConfigResource(
            name=name,
            type=ResourceType.CACHE_CONFIG,
            provider_type=provider_type,
            settings=settings,
        )

        registry.register(
            name=name, obj=instance, resource_type=ResourceType.CACHE_CONFIG, **metadata
        )
        _block_direct_instantiation(cls, name, "cache config")
        return cls

    return decorator


def storage_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a storage configuration resource."""

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Extract provider_type and settings from class attributes
        provider_type = getattr(cls, "provider_type", "")
        settings = getattr(cls, "settings", {})

        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.STORAGE_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {"name": name, "type": ResourceType.STORAGE_CONFIG, **metadata}  # type: ignore[attr-defined]

        # Always wrap in StorageConfigResource
        from flowlib.resources.models.config_resource import StorageConfigResource

        instance = StorageConfigResource(
            name=name,
            type=ResourceType.STORAGE_CONFIG,
            provider_type=provider_type,
            settings=settings,
        )

        registry.register(
            name=name, obj=instance, resource_type=ResourceType.STORAGE_CONFIG, **metadata
        )
        _block_direct_instantiation(cls, name, "storage config")
        return cls

    return decorator


def embedding_provider_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as an embedding provider configuration resource.

    This decorator is for embedding PROVIDER infrastructure configs (e.g., llamacpp_embedding provider).
    For embedding MODEL configs, use @embedding_config instead.

    Args:
        name: Unique name for the embedding provider config (e.g., RequiredAlias.DEFAULT_EMBEDDING.value, 'fast-embedding')
        **metadata: Additional metadata about the config
    """

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Extract provider_type and settings from class attributes
        provider_type = getattr(cls, "provider_type", "")
        settings = getattr(cls, "settings", {})

        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.EMBEDDING_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {
            "name": name,
            "type": ResourceType.EMBEDDING_CONFIG,
            **metadata,
        }  # type: ignore[attr-defined]

        # Always wrap in EmbeddingConfigResource
        from flowlib.resources.models.config_resource import EmbeddingConfigResource

        instance = EmbeddingConfigResource(
            name=name,
            type=ResourceType.EMBEDDING_CONFIG,
            provider_type=provider_type,
            settings=settings,
        )

        registry.register(
            name=name, obj=instance, resource_type=ResourceType.EMBEDDING_CONFIG, **metadata
        )
        _block_direct_instantiation(cls, name, "embedding provider config")
        return cls

    return decorator


def embedding_config(
    name: str,
    provider_type: str,
    config: dict[str, Any] | None = None,
    **metadata: Any,
) -> Callable[[type], type]:
    """Register a class as an embedding configuration resource.
    If the decorated class is not a ResourceBase subclass, wrap it in EmbeddingResource.

    Args:
        name: Unique name for the embedding config
        provider_type: Provider implementation to use (e.g., "llamacpp_embedding", "openai_embedding", "huggingface")
        config: Additional embedding configuration
        **metadata: Additional metadata
    """

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Explicit config required - no fallbacks
        if config is None:
            raise ValueError(f"embedding_config '{name}' requires explicit config parameter")

        # Attach metadata to class - store all config data as class attributes (single source of truth)
        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.EMBEDDING_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {
            "name": name,
            "type": ResourceType.EMBEDDING_CONFIG,
            "provider_type": provider_type,
            "config": config,
            **metadata,
        }  # type: ignore[attr-defined]

        # Store provider_type and config as class attributes for single source of truth
        cls.__provider_type__ = provider_type  # type: ignore[attr-defined]
        cls.__config__ = config  # type: ignore[attr-defined]

        # Add properties to expose config data from class attributes
        def config_prop(self: Any) -> dict[str, Any]:
            """Access config from class attributes (single source of truth)."""
            return getattr(self.__class__, "__config__", {})

        def provider_type_prop(self: Any) -> str | None:
            """Access provider_type from class attributes (single source of truth)."""
            return getattr(self.__class__, "__provider_type__", None)

        cls.config = property(config_prop)  # type: ignore[attr-defined]
        cls.provider_type = property(provider_type_prop)  # type: ignore[attr-defined]

        # Register with global registry if available
        if registry is not None:
            # Always wrap embedding configs in EmbeddingResource for consistent structure
            # regardless of base class, to ensure provider compatibility
            from flowlib.resources.models.embedding_resource import EmbeddingResource

            instance = EmbeddingResource(
                name=name,
                provider_type=provider_type,
                config=config,
                type="embedding_config",
                model_path=config.get("model_path") or config.get("path"),
                model_name=config.get("model_name", name),
                dimensions=config.get("dimensions", 768),
                max_length=config.get("max_length", 512),
                normalize=config.get("normalize", True),
            )
            registry.register(
                name=name, obj=instance, resource_type=ResourceType.EMBEDDING_CONFIG, **metadata
            )

        # Prevent direct instantiation
        _block_direct_instantiation(cls, name, "embedding config")

        return cls

    return decorator


def multimodal_llm_provider_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a multimodal LLM provider configuration resource.

    This decorator is for multimodal LLM PROVIDER infrastructure configs (e.g., llamacpp_multimodal provider).
    For multimodal MODEL configs, use @multimodal_llm_config instead.

    Args:
        name: Unique name for the multimodal LLM provider config (e.g., 'default-multimodal-llm')
        **metadata: Additional metadata about the config
    """

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Extract provider_type and settings from class attributes
        provider_type = getattr(cls, "provider_type", "")
        settings = getattr(cls, "settings", {})

        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.MULTIMODAL_LLM_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {
            "name": name,
            "type": ResourceType.MULTIMODAL_LLM_CONFIG,
            **metadata,
        }  # type: ignore[attr-defined]

        # Always wrap in MultimodalLLMConfigResource
        from flowlib.resources.models.config_resource import MultimodalLLMConfigResource

        instance = MultimodalLLMConfigResource(
            name=name,
            type=ResourceType.MULTIMODAL_LLM_CONFIG,
            provider_type=provider_type,
            settings=settings,
        )

        registry.register(
            name=name, obj=instance, resource_type=ResourceType.MULTIMODAL_LLM_CONFIG, **metadata
        )
        _block_direct_instantiation(cls, name, "multimodal LLM provider config")
        return cls

    return decorator


def multimodal_llm_config(
    name: str,
    provider_type: str | None = None,
    provider: str | None = None,
    config: dict[str, Any] | None = None,
    **metadata: Any,
) -> Callable[[type], type]:
    """Register a class as a multimodal LLM model configuration resource.

    This decorator creates a model configuration that can use multimodal LLM providers.
    Similar to @model_config but specifically for multimodal vision-language models.

    Args:
        name: Unique name for the multimodal LLM model config
        provider_type: Provider implementation to use (e.g., "llamacpp_multimodal")
        config: Model configuration (path, clip_model_path, n_ctx, etc.)
        **metadata: Additional metadata
    """

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.MULTIMODAL_LLM_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {
            "name": name,
            "type": ResourceType.MULTIMODAL_LLM_CONFIG,
            "provider_type": provider_type,
            **metadata,
        }  # type: ignore[attr-defined]

        # Always wrap in ModelResource with multimodal provider type
        from flowlib.resources.models.model_resource import ModelResource

        instance = ModelResource(
            name=name,
            type=ResourceType.MULTIMODAL_LLM_CONFIG,
            provider_type=provider_type,
            config=config or {},
        )

        registry.register(
            name=name, obj=instance, resource_type=ResourceType.MULTIMODAL_LLM_CONFIG, **metadata
        )

        # Prevent direct instantiation
        _block_direct_instantiation(cls, name, "multimodal LLM model config")

        return cls

    return decorator


def graph_db_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a graph database configuration resource."""

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Extract provider_type and settings from class attributes
        provider_type = getattr(cls, "provider_type", "")
        settings = getattr(cls, "settings", {})

        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.GRAPH_DB_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {"name": name, "type": ResourceType.GRAPH_DB_CONFIG, **metadata}  # type: ignore[attr-defined]

        # Always wrap in GraphDBConfigResource
        from flowlib.resources.models.config_resource import GraphDBConfigResource

        instance = GraphDBConfigResource(
            name=name,
            type=ResourceType.GRAPH_DB_CONFIG,
            provider_type=provider_type,
            settings=settings,
        )

        registry.register(
            name=name, obj=instance, resource_type=ResourceType.GRAPH_DB_CONFIG, **metadata
        )
        _block_direct_instantiation(cls, name, "graph DB config")
        return cls

    return decorator


def message_queue_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a message queue configuration resource."""

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Extract provider_type and settings from class attributes
        provider_type = getattr(cls, "provider_type", "")
        settings = getattr(cls, "settings", {})

        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.MESSAGE_QUEUE_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {
            "name": name,
            "type": ResourceType.MESSAGE_QUEUE_CONFIG,
            **metadata,
        }  # type: ignore[attr-defined]

        # Always wrap in MessageQueueConfigResource
        from flowlib.resources.models.config_resource import MessageQueueConfigResource

        instance = MessageQueueConfigResource(
            name=name,
            type=ResourceType.MESSAGE_QUEUE_CONFIG,
            provider_type=provider_type,
            settings=settings,
        )

        registry.register(
            name=name, obj=instance, resource_type=ResourceType.MESSAGE_QUEUE_CONFIG, **metadata
        )
        _block_direct_instantiation(cls, name, "message queue config")
        return cls

    return decorator


def agent_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as an agent configuration resource.

    Agent configs define complete agent setups including:
    - Persona and behavior
    - Profile for tool access
    - Model and LLM selection
    - Temperature and other settings

    Args:
        name: Unique name for the agent config
        **metadata: Additional metadata about the config

    Returns:
        Decorator function
    """

    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Import RequiredAlias here to avoid circular import
        from flowlib.config.required_resources import RequiredAlias

        # Extract all agent config attributes from class attributes
        persona = getattr(cls, "persona", "")
        allowed_tool_categories = getattr(cls, "allowed_tool_categories", [])
        model_name = getattr(cls, "model_name", RequiredAlias.DEFAULT_MODEL.value)
        llm_name = getattr(cls, "llm_name", RequiredAlias.DEFAULT_LLM.value)
        temperature = getattr(cls, "temperature", 0.7)
        max_iterations = getattr(cls, "max_iterations", 10)
        enable_learning = getattr(cls, "enable_learning", True)
        verbose = getattr(cls, "verbose", False)

        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.AGENT_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {"name": name, "type": ResourceType.AGENT_CONFIG, **metadata}  # type: ignore[attr-defined]

        # Always wrap in AgentConfigResource
        from flowlib.resources.models.agent_config_resource import AgentConfigResource

        instance = AgentConfigResource(
            name=name,
            type=ResourceType.AGENT_CONFIG,
            persona=persona,
            allowed_tool_categories=allowed_tool_categories,
            model_name=model_name,
            llm_name=llm_name,
            temperature=temperature,
            max_iterations=max_iterations,
            enable_learning=enable_learning,
            verbose=verbose,
        )

        registry.register(
            name=name, obj=instance, resource_type=ResourceType.AGENT_CONFIG, **metadata
        )
        _block_direct_instantiation(cls, name, "agent config")
        return cls

    return decorator


# Template registration removed - templates now use direct registry.register() calls
