from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from flowlib.resources.models.base import ResourceBase
from flowlib.resources.models.constants import ResourceType

if TYPE_CHECKING:
    from flowlib.resources.registry.registry import ResourceRegistry


def _get_resource_registry() -> 'ResourceRegistry':
    """Lazy import of resource registry to avoid circular dependencies."""
    from flowlib.resources.registry.registry import resource_registry
    return resource_registry




def resource(name: str, resource_type: str = ResourceType.MODEL_CONFIG, **metadata: Any) -> Callable[[type], type]:
    """Register a class or function as a resource.
    Enforces contract: only ResourceBase subclasses can be registered.
    If not, raises TypeError immediately.
    """
    def decorator(obj: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")
        # If already a ResourceBase subclass, register directly
        if not isinstance(obj, type) or not issubclass(obj, ResourceBase):
            raise TypeError(f"Resource '{name}' must be a ResourceBase subclass (pydantic v2), got {type(obj)}")

        # Attach metadata to class first
        obj.__resource_name__ = name  # type: ignore[attr-defined]
        obj.__resource_type__ = resource_type  # type: ignore[attr-defined]
        obj.__resource_metadata__ = {'name': name, 'type': resource_type, **metadata}  # type: ignore[attr-defined]

        # Instantiate the resource for registration with minimal required fields
        instance = obj(name=name, type=resource_type)
        registry.register(
            name=name,
            obj=instance,
            resource_type=resource_type,
            **metadata
        )
        return obj
    return decorator

def model_config(name: str, provider_type: Optional[str] = None, provider: Optional[str] = None, config: Optional[dict[str, Any]] = None, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a model configuration resource.
    If the decorated class is not a ResourceBase subclass, wrap it in ModelResource.
    
    Args:
        name: Unique name for the model config
        provider_type: Provider implementation to use (e.g., "llamacpp", "google_ai", "openai")
        provider: DEPRECATED - use provider_type instead
        config: Additional model configuration
        **metadata: Additional metadata
    """
    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Determine provider type - fail-fast validation, no fallbacks
        if provider and provider_type:
            raise ValueError("Cannot specify both 'provider' and 'provider_type'. Use 'provider_type'.")

        final_provider_type = provider_type or provider
        if not final_provider_type:
            raise ValueError(f"model_config '{name}' requires explicit provider_type parameter (e.g., 'llamacpp', 'openai')")

        # Explicit config required - no fallbacks
        if config is None:
            raise ValueError(f"model_config '{name}' requires explicit config parameter")

        # Attach metadata to class - store all config data as class attributes (single source of truth)
        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.MODEL_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {'name': name, 'type': ResourceType.MODEL_CONFIG, 'provider_type': final_provider_type, 'config': config, **metadata}  # type: ignore[attr-defined]

        # Store provider_type and config as class attributes for single source of truth
        cls.__provider_type__ = final_provider_type  # type: ignore[attr-defined]
        cls.__config__ = config  # type: ignore[attr-defined]

        # Add properties to expose config data from class attributes
        def config_prop(self: Any) -> Dict[str, Any]:
            """Access config from class attributes (single source of truth)."""
            return getattr(self.__class__, '__config__', {})

        def provider_type_prop(self: Any) -> Optional[str]:
            """Access provider_type from class attributes (single source of truth)."""
            return getattr(self.__class__, '__provider_type__', None)

        cls.config = property(config_prop)  # type: ignore[attr-defined]
        cls.provider_type = property(provider_type_prop)  # type: ignore[attr-defined]

        # Register with global registry if available
        if registry is not None:
            # If already a ResourceBase subclass, register directly
            if isinstance(cls, type) and issubclass(cls, ResourceBase):
                # ResourceBase accepts only name and type parameters
                instance = cls(name=name, type="model_config")
            else:
                # Wrap non-contract class in ModelResource
                from flowlib.resources.models.model_resource import ModelResource
                instance = ModelResource(
                    name=name,
                    provider_type=final_provider_type,
                    config=config,
                    type="model_config",
                    model_path=config.get("model_path"),
                    model_name=config.get("model_name", name),
                    chat_format=config.get("chat_format")
                )
            registry.register(
                name=name,
                obj=instance,
                resource_type=ResourceType.MODEL_CONFIG,
                **metadata
            )

        return cls
    return decorator


def prompt(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a prompt resource.
    
    This decorator ensures the decorated class adheres to the PromptTemplate protocol
    by requiring a 'template' attribute and adding a default 'config' attribute if missing.
    
    Args:
        name: Unique name for the prompt
        **metadata: Additional metadata about the prompt
        
    Returns:
        Decorator function that returns a class conforming to PromptTemplate
        
    Raises:
        ValueError: If the decorated object does not have a 'template' attribute
                   or if 'config' is not present after decoration
    """
    def decorator(obj: type) -> type:
        registry = _get_resource_registry()

        # Check if template exists before registration using safe attribute access
        template_exists = False
        try:
            # Check for template in annotations
            if getattr(obj, '__annotations__', None) and 'template' in obj.__annotations__:
                template_exists = True
            # Check for template in Pydantic model fields
            elif getattr(obj, 'model_fields', None) and 'template' in obj.model_fields:  # type: ignore[attr-defined]
                template_exists = True
            # Check for direct template attribute
            elif getattr(obj, 'template', None) is not None:
                template_exists = True
        except (AttributeError, TypeError):
            pass

        if not template_exists:
            raise ValueError(f"Prompt '{name}' must have a 'template' attribute")

        # Attach metadata to class
        obj.__resource_name__ = name  # type: ignore[attr-defined]
        obj.__resource_type__ = ResourceType.PROMPT_CONFIG  # type: ignore[attr-defined]
        obj.__resource_metadata__ = {'name': name, 'type': ResourceType.PROMPT, **metadata}  # type: ignore[attr-defined]

        # Add default config only if it doesn't already exist using safe attribute access
        config_exists = False
        try:
            # Check for config in annotations
            if getattr(obj, '__annotations__', None) and 'config' in obj.__annotations__:
                config_exists = True
            # Check for config in Pydantic model fields
            elif getattr(obj, 'model_fields', None) and 'config' in obj.model_fields:  # type: ignore[attr-defined]
                config_exists = True
            # Check for direct config attribute
            elif getattr(obj, 'config', None) is not None:
                config_exists = True
        except (AttributeError, TypeError):
            pass
        if not config_exists:
            obj.config = {  # type: ignore[attr-defined]
                "max_tokens": 2048,
                "temperature": 0.5,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }

        # Register with global registry if available
        if registry is not None:
            # If already a ResourceBase subclass, register directly
            if isinstance(obj, type) and issubclass(obj, ResourceBase):
                instance = obj(name=name, type="prompt_config")
            else:
                # For non-ResourceBase classes, call the resource decorator
                decorated_obj = resource(name, ResourceType.PROMPT_CONFIG, **metadata)(obj)
                return decorated_obj

            registry.register(
                name=name,
                obj=instance,
                resource_type=ResourceType.PROMPT_CONFIG,
                **metadata
            )

        # This object now conforms to PromptTemplate protocol
        return obj

    return decorator

def config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a configuration resource.

    This decorator is a specialized version of @resource for configs.

    Args:
        name: Unique name for the config
        **metadata: Additional metadata about the config

    Returns:
        Decorator function
    """
    return resource(name, ResourceType.CONFIG, **metadata)


def llm_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as an LLM configuration resource.

    Args:
        name: Unique name for the LLM config (e.g., 'default-llm', 'fast-chat')
        **metadata: Additional metadata about the config
    """
    return resource(name, ResourceType.LLM_CONFIG, **metadata)

def database_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a database configuration resource."""
    return resource(name, ResourceType.DATABASE_CONFIG, **metadata)

def vector_db_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a vector database configuration resource."""
    return resource(name, ResourceType.VECTOR_DB_CONFIG, **metadata)

def cache_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a cache configuration resource."""
    return resource(name, ResourceType.CACHE_CONFIG, **metadata)

def storage_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a storage configuration resource."""
    return resource(name, ResourceType.STORAGE_CONFIG, **metadata)

def embedding_config(name: str, provider_type: Optional[str] = None, provider: Optional[str] = None, config: Optional[dict[str, Any]] = None, **metadata: Any) -> Callable[[type], type]:
    """Register a class as an embedding configuration resource.
    If the decorated class is not a ResourceBase subclass, wrap it in EmbeddingResource.

    Args:
        name: Unique name for the embedding config
        provider_type: Provider implementation to use (e.g., "llamacpp_embedding", "openai_embedding", "huggingface")
        provider: DEPRECATED - use provider_type instead
        config: Additional embedding configuration
        **metadata: Additional metadata
    """
    def decorator(cls: type) -> type:
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")

        # Determine provider type - fail-fast validation, no fallbacks
        if provider and provider_type:
            raise ValueError("Cannot specify both 'provider' and 'provider_type'. Use 'provider_type'.")

        final_provider_type = provider_type or provider
        if not final_provider_type:
            raise ValueError(f"embedding_config '{name}' requires explicit provider_type parameter (e.g., 'llamacpp_embedding', 'openai_embedding')")

        # Explicit config required - no fallbacks
        if config is None:
            raise ValueError(f"embedding_config '{name}' requires explicit config parameter")

        # Attach metadata to class - store all config data as class attributes (single source of truth)
        cls.__resource_name__ = name  # type: ignore[attr-defined]
        cls.__resource_type__ = ResourceType.EMBEDDING_CONFIG  # type: ignore[attr-defined]
        cls.__resource_metadata__ = {'name': name, 'type': ResourceType.EMBEDDING_CONFIG, 'provider_type': final_provider_type, 'config': config, **metadata}  # type: ignore[attr-defined]

        # Store provider_type and config as class attributes for single source of truth
        cls.__provider_type__ = final_provider_type  # type: ignore[attr-defined]
        cls.__config__ = config  # type: ignore[attr-defined]

        # Add properties to expose config data from class attributes
        def config_prop(self: Any) -> Dict[str, Any]:
            """Access config from class attributes (single source of truth)."""
            return getattr(self.__class__, '__config__', {})

        def provider_type_prop(self: Any) -> Optional[str]:
            """Access provider_type from class attributes (single source of truth)."""
            return getattr(self.__class__, '__provider_type__', None)

        cls.config = property(config_prop)  # type: ignore[attr-defined]
        cls.provider_type = property(provider_type_prop)  # type: ignore[attr-defined]

        # Register with global registry if available
        if registry is not None:
            # Always wrap embedding configs in EmbeddingResource for consistent structure
            # regardless of base class, to ensure provider compatibility
            from flowlib.resources.models.embedding_resource import EmbeddingResource
            instance = EmbeddingResource(
                name=name,
                provider_type=final_provider_type,
                config=config,
                type="embedding_config",
                model_path=config.get("model_path") or config.get("path"),
                model_name=config.get("model_name", name),
                dimensions=config.get("dimensions", 768),
                max_length=config.get("max_length", 512),
                normalize=config.get("normalize", True)
            )
            registry.register(
                name=name,
                obj=instance,
                resource_type=ResourceType.EMBEDDING_CONFIG,
                **metadata
            )

        return cls
    return decorator

def graph_db_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a graph database configuration resource."""
    return resource(name, ResourceType.GRAPH_DB_CONFIG, **metadata)

def message_queue_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a message queue configuration resource."""
    return resource(name, ResourceType.MESSAGE_QUEUE_CONFIG, **metadata)

def agent_profile_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as an agent profile configuration resource."""
    return resource(name, ResourceType.AGENT_PROFILE_CONFIG, **metadata)

def role_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a role configuration resource.

    Args:
        name: Unique name for the role (e.g., 'composer', 'software_engineer')
        **metadata: Additional metadata about the role
    """
    return resource(name, ResourceType.ROLE_CONFIG, **metadata)

def tool_category_config(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register a class as a tool category configuration resource.

    Args:
        name: Unique name for the tool category (e.g., 'music', 'software')
        **metadata: Additional metadata about the tool category
    """
    return resource(name, ResourceType.TOOL_CATEGORY_CONFIG, **metadata)


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
    return resource(name, ResourceType.AGENT_CONFIG, **metadata)

# Template registration removed - templates now use direct registry.register() calls


