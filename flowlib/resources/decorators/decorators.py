from typing import Dict, Any, Protocol, runtime_checkable
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.models.model_resource import ModelResource
from flowlib.core.interfaces import PromptTemplate


def _get_resource_registry():
    """Lazy import of resource registry to avoid circular dependencies."""
    from flowlib.resources.registry.registry import resource_registry
    return resource_registry




def resource(name: str, resource_type: str = ResourceType.MODEL_CONFIG, **metadata):
    """Register a class or function as a resource.
    Enforces contract: only ResourceBase subclasses can be registered.
    If not, raises TypeError immediately.
    """
    def decorator(obj):
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")
        # If already a ResourceBase subclass, register directly
        if not isinstance(obj, type) or not issubclass(obj, ResourceBase):
            raise TypeError(f"Resource '{name}' must be a ResourceBase subclass (pydantic v2), got {type(obj)}")
        
        # Attach metadata to class first
        obj.__resource_name__ = name
        obj.__resource_type__ = resource_type
        obj.__resource_metadata__ = {'name': name, 'type': resource_type, **metadata}
        
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

def model_config(name: str, provider_type: str = None, provider: str = None, config: dict = None, **metadata):
    """Register a class as a model configuration resource.
    If the decorated class is not a ResourceBase subclass, wrap it in ModelResource.
    
    Args:
        name: Unique name for the model config
        provider_type: Provider implementation to use (e.g., "llamacpp", "google_ai", "openai")
        provider: DEPRECATED - use provider_type instead
        config: Additional model configuration
        **metadata: Additional metadata
    """
    def decorator(cls):
        registry = _get_resource_registry()
        if registry is None:
            raise RuntimeError("Resource registry not initialized")
            
        # Attach metadata to class
        cls.__resource_name__ = name
        cls.__resource_type__ = ResourceType.MODEL_CONFIG
        cls.__resource_metadata__ = {'name': name, 'type': ResourceType.MODEL, **metadata}
        
        # Register with global registry if available
        if registry is not None:
            # Determine provider type - fail-fast validation, no fallbacks
            if provider and provider_type:
                raise ValueError("Cannot specify both 'provider' and 'provider_type'. Use 'provider_type'.")
            
            final_provider_type = provider_type or provider
            if not final_provider_type:
                raise ValueError(f"model_config '{name}' requires explicit provider_type parameter (e.g., 'llamacpp', 'openai')")
            
            # If already a ResourceBase subclass, register directly
            if isinstance(cls, type) and issubclass(cls, ResourceBase):
                # Check if it's a ModelResource subclass (has provider_type and config fields)
                if hasattr(cls, 'model_fields') and 'provider_type' in cls.model_fields and 'config' in cls.model_fields:
                    # Explicit config required - no fallbacks
                    if config is None:
                        raise ValueError(f"model_config '{name}' requires explicit config parameter")
                    instance = cls(name=name, type="model_config", provider_type=final_provider_type, config=config)
                else:
                    # Basic ResourceBase - just pass name and type
                    instance = cls(name=name, type="model_config")
            else:
                # Wrap non-contract class in ModelResource
                from flowlib.resources.models.model_resource import ModelResource
                # Explicit config required - no fallbacks
                if config is None:
                    raise ValueError(f"model_config '{name}' requires explicit config parameter")
                instance = ModelResource(name=name, provider_type=final_provider_type, config=config, type="model_config")
            registry.register(
                name=name,
                obj=instance,
                resource_type=ResourceType.MODEL_CONFIG,
                **metadata
            )
        
        return cls
    return decorator


def prompt(name: str, **metadata):
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
    def decorator(obj):
        registry = _get_resource_registry()
        
        # Check if template exists before registration
        # For Pydantic models, check both class annotations and model_fields
        has_template = (hasattr(obj, '__annotations__') and 'template' in obj.__annotations__) or \
                      (hasattr(obj, 'model_fields') and 'template' in obj.model_fields) or \
                      hasattr(obj, 'template')
        if not has_template:
            raise ValueError(f"Prompt '{name}' must have a 'template' attribute")
        
        # Attach metadata to class
        obj.__resource_name__ = name
        obj.__resource_type__ = ResourceType.PROMPT_CONFIG
        obj.__resource_metadata__ = {'name': name, 'type': ResourceType.PROMPT, **metadata}
        
        # Add default config only if it doesn't already exist
        has_existing_config = (hasattr(obj, '__annotations__') and 'config' in obj.__annotations__) or \
                             (hasattr(obj, 'model_fields') and 'config' in obj.model_fields) or \
                             hasattr(obj, 'config')
        if not has_existing_config:
            obj.config = {
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

def config(name: str, **metadata):
    """Register a class as a configuration resource.
    
    This decorator is a specialized version of @resource for configs.
    
    Args:
        name: Unique name for the config
        **metadata: Additional metadata about the config
        
    Returns:
        Decorator function
    """
    return resource(name, ResourceType.CONFIG, **metadata)


def llm_config(name: str, **metadata):
    """Register a class as an LLM configuration resource.
    
    Args:
        name: Unique name for the LLM config (e.g., 'default-llm', 'fast-chat')
        **metadata: Additional metadata about the config
    """
    return resource(name, ResourceType.LLM_CONFIG, **metadata)

def database_config(name: str, **metadata):
    """Register a class as a database configuration resource."""
    return resource(name, ResourceType.DATABASE_CONFIG, **metadata)

def vector_db_config(name: str, **metadata):
    """Register a class as a vector database configuration resource."""
    return resource(name, ResourceType.VECTOR_DB_CONFIG, **metadata)

def cache_config(name: str, **metadata):
    """Register a class as a cache configuration resource."""
    return resource(name, ResourceType.CACHE_CONFIG, **metadata)

def storage_config(name: str, **metadata):
    """Register a class as a storage configuration resource."""
    return resource(name, ResourceType.STORAGE_CONFIG, **metadata)

def embedding_config(name: str, **metadata):
    """Register a class as an embedding configuration resource."""
    return resource(name, ResourceType.EMBEDDING_CONFIG, **metadata)

def graph_db_config(name: str, **metadata):
    """Register a class as a graph database configuration resource."""
    return resource(name, ResourceType.GRAPH_DB_CONFIG, **metadata)

def message_queue_config(name: str, **metadata):
    """Register a class as a message queue configuration resource."""
    return resource(name, ResourceType.MESSAGE_QUEUE_CONFIG, **metadata)

# Template registration removed - templates now use direct registry.register() calls


