"""Clean decorators eliminating circular dependencies.

This module provides decorators that register components through events
rather than direct registry imports, completely eliminating circular
dependencies. All imports are at the top of the file.
"""

from typing import Any, Callable, Dict, Optional, Type
from functools import wraps

from flowlib.core.events.events import emit_registration, emit_configuration, RegistrationEvent, ConfigurationEvent

# All imports at top - no circular dependencies possible


def provider(provider_type: str, name: str, **metadata):
    """Provider decorator using event-based registration.
    
    Args:
        provider_type: Type of provider (llm, database, vector_db, etc.)
        name: Unique name for the provider
        **metadata: Additional metadata
        
    Returns:
        Decorated class
        
    Example:
        @provider('llm', 'llamacpp')
        class LlamaCppProvider:
            pass
    """
    def decorator(cls: Type) -> Type:
        def factory():
            return cls()
        
        event = RegistrationEvent(
            item_type='provider',
            name=name,
            factory=factory,
            metadata={
                'provider_type': provider_type,
                'class_name': cls.__name__,
                'module': cls.__module__,
                **metadata
            }
        )
        
        emit_registration(event)
        return cls
    
    return decorator


def resource(name: str, resource_type: str, **metadata):
    """Resource decorator using event-based registration.
    
    Args:
        name: Unique name for the resource
        resource_type: Type of resource (prompt, model, config, etc.)
        **metadata: Additional metadata
        
    Returns:
        Decorated class
        
    Example:
        @resource('my-prompt', 'prompt')
        class MyPrompt:
            pass
    """
    def decorator(cls: Type) -> Type:
        def factory():
            return cls()
        
        event = RegistrationEvent(
            item_type='resource',
            name=name,
            factory=factory,
            metadata={
                'resource_type': resource_type,
                'class_name': cls.__name__,
                'module': cls.__module__,
                **metadata
            }
        )
        
        emit_registration(event)
        return cls
    
    return decorator


def flow(name: str, description: Optional[str] = None, **metadata):
    """Flow decorator using event-based registration.
    
    Args:
        name: Unique name for the flow
        description: Optional description
        **metadata: Additional metadata
        
    Returns:
        Decorated class
        
    Example:
        @flow('entity-extraction', description='Extract entities from text')
        class EntityExtractionFlow:
            pass
    """
    def decorator(cls: Type) -> Type:
        def factory():
            return cls()
        
        event = RegistrationEvent(
            item_type='flow',
            name=name,
            factory=factory,
            metadata={
                'description': description,
                'class_name': cls.__name__,
                'module': cls.__module__,
                **metadata
            }
        )
        
        emit_registration(event)
        return cls
    
    return decorator


def config(name: str, provider_type: str, **settings):
    """Configuration decorator using event-based registration.
    
    Args:
        name: Configuration name
        provider_type: Provider type this config is for
        **settings: Configuration settings
        
    Returns:
        Decorated class
        
    Example:
        @config('default-llm', 'llamacpp', model_path='/path/to/model.gguf')
        class DefaultLLMConfig:
            pass
    """
    def decorator(cls: Type) -> Type:
        event = ConfigurationEvent(
            config_name=name,
            config_class=cls,
            provider_type=provider_type,
            settings=settings
        )
        
        emit_configuration(event)
        return cls
    
    return decorator


# Specific configuration decorators for different provider types
def llm_config(name: str, **settings):
    """LLM configuration decorator.
    
    Args:
        name: Configuration name
        **settings: LLM settings
        
    Example:
        @llm_config('default-llm', provider_type='llamacpp', model_path='...')
        class DefaultLLM:
            pass
    """
    provider_type = settings.pop('provider_type', 'llamacpp')
    return config(name, provider_type, **settings)


def database_config(name: str, **settings):
    """Database configuration decorator.
    
    Args:
        name: Configuration name
        **settings: Database settings
    """
    provider_type = settings.pop('provider_type', 'sqlite')
    return config(name, provider_type, **settings)


def vector_db_config(name: str, **settings):
    """Vector database configuration decorator.
    
    Args:
        name: Configuration name
        **settings: Vector DB settings
    """
    provider_type = settings.pop('provider_type', 'chroma')
    return config(name, provider_type, **settings)


def graph_db_config(name: str, **settings):
    """Graph database configuration decorator.
    
    Args:
        name: Configuration name
        **settings: Graph DB settings
    """
    provider_type = settings.pop('provider_type', 'neo4j')
    return config(name, provider_type, **settings)


def cache_config(name: str, **settings):
    """Cache configuration decorator.
    
    Args:
        name: Configuration name
        **settings: Cache settings
    """
    provider_type = settings.pop('provider_type', 'redis')
    return config(name, provider_type, **settings)


def embedding_config(name: str, **settings):
    """Embedding configuration decorator.
    
    Args:
        name: Configuration name
        **settings: Embedding settings
    """
    provider_type = settings.pop('provider_type', 'llamacpp_embedding')
    return config(name, provider_type, **settings)


def tool(name: str, description: str, **metadata):
    """Register a class as a tool provider with automatic registration.
    
    This decorator follows flowlib's event-driven registration pattern,
    automatically creating a tool provider wrapper and registering it
    via the event system without circular dependencies.
    
    Args:
        name: Unique tool name
        description: Tool description for LLM
        **metadata: Additional tool metadata
        
    Returns:
        Decorator function
        
    Example:
        @tool(name="read", description="Read file contents")
        class ReadTool:
            class Parameters(StrictBaseModel):
                file_path: str
                
            async def execute(self, params: Parameters, context) -> Dict[str, Any]:
                # Implementation here
                pass
    """
    def decorator(cls):
        raise DeprecationWarning(
            "The @tool decorator in flowlib.core.decorators is deprecated. "
            "Use @tool from flowlib.agent.components.task_execution.decorators instead."
        )
    
    return decorator


# Legacy decorator compatibility (will be phased out)
def model(name: str, **metadata):
    """Model decorator (legacy compatibility).
    
    This is a compatibility decorator that maps to the new resource system.
    """
    return resource(name, 'model', **metadata)


def prompt(name: str, **metadata):
    """Prompt decorator using resource system.
    
    Args:
        name: Prompt name
        **metadata: Additional metadata
    """
    return resource(name, 'prompt', **metadata)


def template(name: str, **metadata):
    """Template decorator using resource system.
    
    Args:
        name: Template name
        **metadata: Additional metadata
    """
    return resource(name, 'template', **metadata)


# Utility decorators
def singleton(cls: Type) -> Type:
    """Singleton decorator for classes that should have only one instance.
    
    Args:
        cls: Class to make singleton
        
    Returns:
        Singleton class
    """
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def lazy_init(cls: Type) -> Type:
    """Lazy initialization decorator for expensive-to-create objects.
    
    Args:
        cls: Class to make lazy
        
    Returns:
        Lazy class
    """
    class LazyWrapper:
        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs
            self._instance = None
        
        def __getattr__(self, name):
            if self._instance is None:
                self._instance = cls(*self._args, **self._kwargs)
            return getattr(self._instance, name)
    
    return LazyWrapper


# Dependency injection decorator
def inject(**dependencies):
    """Dependency injection decorator.
    
    Args:
        **dependencies: Dependencies to inject
        
    Example:
        @inject(llm='default-llm', vector_db='default-vector-db')
        async def my_function(input_data, llm, vector_db):
            # llm and vector_db are automatically injected
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Import here to avoid circular dependency
            from flowlib.core.container.container import get_container
            
            container = get_container()
            
            # Inject dependencies
            for param_name, config_name in dependencies.items():
                if param_name not in kwargs:
                    if config_name.startswith('provider:'):
                        # Provider injection
                        config_name = config_name[9:]  # Remove 'provider:' prefix
                        kwargs[param_name] = await container.get_provider(config_name)
                    elif config_name.startswith('resource:'):
                        # Resource injection
                        config_name = config_name[9:]  # Remove 'resource:' prefix
                        kwargs[param_name] = container.get_resource(config_name)
                    elif config_name.startswith('flow:'):
                        # Flow injection
                        config_name = config_name[5:]  # Remove 'flow:' prefix
                        kwargs[param_name] = container.get_flow(config_name)
                    else:
                        # Default to provider injection
                        kwargs[param_name] = await container.get_provider(config_name)
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator