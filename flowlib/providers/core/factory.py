"""Provider factory for creating different types of providers.

This module provides a factory for creating different types of providers
based on the provider_type specified in the configuration.
"""

import logging
from typing import Any, Dict, Optional, Type, cast

from flowlib.core.errors.errors import ErrorContext, ProviderError
from flowlib.core.errors.models import ProviderErrorContext
from flowlib.providers.db.base import DBProvider
from flowlib.providers.graph.base import GraphDBProvider

from .base import Provider

# Removed ProviderType import - using config-driven provider access
from .registry import provider_registry

logger = logging.getLogger(__name__)

# Registry of provider specific implementations
# This is separate from the provider_registry and helps with factory creation
PROVIDER_IMPLEMENTATIONS: Dict[str, Dict[str, Optional[Type[Provider[Any]]]]] = {
    "llm": {
        "llamacpp": None,  # Will be populated on import
        "llama": None,     # Will be populated on import
        "googleai": None,  # Will be populated on import
    },
    "database": {
        "postgres": None,
        "postgresql": None,
        "mongodb": None,
        "mongo": None,
        "sqlite": None,
        "sqlite3": None,
    },
    "message_queue": {
        "rabbitmq": None,
        "rabbit": None,
        "kafka": None,
    },
    "cache": {
        "redis": None,
        "memory": None,
        "inmemory": None,
    },
    "vector_db": {
        "chroma": None,
        "chromadb": None,
        "pinecone": None,
        "qdrant": None,
    },
    "storage": {
        "s3": None,
        "aws": None,
        "local": None,
        "localfile": None,
        "file": None,
    },
    "embedding": {
        "llamacpp": None,
        "llamacpp_embedding": None,
    },
    "graph_db": {
        "neo4j": None,
    },
    "state_persister": {
        "redis": None,
        "mongodb": None,
        "postgres": None,
        "file": None,
    },
}

def create_provider(
    provider_type: str,
    name: str,
    implementation: Optional[str] = None,
    register: bool = True,
    **kwargs: Any
) -> Provider[Any]:
    """Create a provider based on the provider_type and optional implementation.
    
    Args:
        provider_type: Type of provider (e.g., "llm", "database")
        name: Unique name for the provider instance
        implementation: Optional specific implementation (e.g., "postgres" for db)
        register: Whether to register the provider in the registry
        **kwargs: Additional arguments to pass to the provider constructor
        
    Returns:
        Provider instance of the appropriate type
        
    Raises:
        ProviderError: If the specified provider_type or implementation is not supported
    """
    # NOTE: This factory function is primarily for testing and legacy support.
    # Modern code should use config-driven provider access:
    # provider = await provider_registry.get_by_config("config-name")

    # No registry existence checks - factory creates new instances per CLAUDE.md principles

    # If implementation is specified, try to get specific provider class
    provider_class = None

    if implementation and provider_type in PROVIDER_IMPLEMENTATIONS:
        impl_registry = PROVIDER_IMPLEMENTATIONS[provider_type]
        if implementation.lower() in impl_registry:
            provider_class = impl_registry[implementation.lower()]

            # Lazy import provider classes to avoid import issues
            if provider_class is None:
                provider_class = _import_provider_class(provider_type, implementation.lower())
                impl_registry[implementation.lower()] = provider_class

        if provider_class is None:
            list(impl_registry.keys())
            raise ProviderError(
                message=f"Unsupported implementation {implementation} for provider type {provider_type}",
                context=ErrorContext.create(
                    flow_name="provider_factory",
                    error_type="UnsupportedImplementationError",
                    error_location="create_provider",
                    component="provider_factory",
                    operation="validate_implementation"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=name,
                    provider_type=provider_type,
                    operation="create_provider",
                    retry_count=0
                )
            )

    if provider_class is None:
        raise ProviderError(
            message=f"No provider found for type {provider_type}",
            context=ErrorContext.create(
                flow_name="provider_factory",
                error_type="UnsupportedProviderTypeError",
                error_location="create_provider",
                component="provider_factory",
                operation="validate_provider_type"
            ),
            provider_context=ProviderErrorContext(
                provider_name=name,
                provider_type=provider_type,
                operation="create_provider",
                retry_count=0
            )
        )

    try:
        # If 'settings' is a dict, convert to the provider's settings_class
        if 'settings' in kwargs and isinstance(kwargs['settings'], dict):
            if hasattr(provider_class, 'settings_class') and provider_class.settings_class:
                settings_obj = provider_class.settings_class(**kwargs['settings'])
                kwargs['settings'] = settings_obj
            else:
                raise ProviderError(
                    message=f"Provider class {provider_class.__name__} is missing a valid settings_class for conversion.",
                    context=ErrorContext.create(
                        flow_name="provider_factory",
                        error_type="MissingSettingsClassError",
                        error_location="create_provider",
                        component="provider_factory",
                        operation="convert_settings"
                    ),
                    provider_context=ProviderErrorContext(
                        provider_name=name,
                        provider_type=provider_type,
                        operation="create_provider",
                        retry_count=0
                    )
                )
        # Pop settings from kwargs to avoid passing it twice.
        # kwargs['settings'] would have been converted to a Pydantic object by the block above if it was a dict.
        # If 'settings' was not in kwargs initially, or was not a dict, pop will return None (due to default) or the original value.
        final_settings_arg = kwargs.pop('settings', None) # Get the processed settings, remove from kwargs

        # Some providers handle provider_type internally
        if issubclass(provider_class, (GraphDBProvider, DBProvider)):
            provider = cast(Provider[Any], provider_class(name=name, settings=final_settings_arg, **kwargs))
        else:
            provider = provider_class(name=name, settings=final_settings_arg, provider_type=provider_type, **kwargs)
        # Register if requested
        if register:
            # Use register_provider for provider objects
            provider_registry.register_provider(provider)
        return provider
    except Exception as e:
        raise ProviderError(
            message=f"Failed to create provider '{name}' of type '{provider_type}': {str(e)}",
            context=ErrorContext.create(
                flow_name="provider_factory",
                error_type="ProviderCreationError",
                error_location="create_provider",
                component="provider_factory",
                operation="instantiate_provider"
            ),
            provider_context=ProviderErrorContext(
                provider_name=name,
                provider_type=provider_type,
                operation="create_provider",
                retry_count=0
            ),
            cause=e
        )

async def create_and_initialize_provider(
    provider_type: str,
    name: str,
    implementation: Optional[str] = None,
    register: bool = True,
    **kwargs: Any
) -> Provider[Any]:
    """Create and initialize a provider.
    
    This is a convenience function that combines create_provider with initialization.
    
    Args:
        provider_type: Type of provider
        name: Unique name for the provider
        implementation: Optional specific implementation
        register: Whether to register the provider
        **kwargs: Additional arguments for the provider
        
    Returns:
        Initialized provider instance
        
    Raises:
        ProviderError: If provider creation or initialization fails
    """
    # No registry existence checks - factory creates new instances per CLAUDE.md principles

    # Create the provider
    provider = create_provider(
        provider_type=provider_type,
        name=name,
        implementation=implementation,
        register=register,
        **kwargs
    )

    # Initialize and return
    try:
        await provider.initialize()
        return provider
    except Exception as e:
        raise ProviderError(
            message=f"Failed to initialize provider '{name}' of type '{provider_type}': {str(e)}",
            context=ErrorContext.create(
                flow_name="provider_factory",
                error_type="ProviderInitializationError",
                error_location="create_and_initialize_provider",
                component="provider_factory",
                operation="initialize_provider"
            ),
            provider_context=ProviderErrorContext(
                provider_name=name,
                provider_type=provider_type,
                operation="initialize_provider",
                retry_count=0
            ),
            cause=e
        )

def _import_provider_class(provider_type: str, implementation: str) -> Type[Provider[Any]]:
    """Import a provider class by type and implementation.
    
    Args:
        provider_type: Type of provider
        implementation: Specific implementation
        
    Returns:
        Provider class
        
    Raises:
        ImportError: If provider class cannot be imported
    """
    # Map of provider types and implementations to provider classes
    provider_map = {
        "llm": {
            "llamacpp": "LlamaCppProvider",
        },
        "database": {
            "postgresql": "PostgreSQLProvider",
            "mongodb": "MongoDBProvider",
            "sqlite": "SQLiteProvider",
        },
        "message_queue": {
            "rabbitmq": "RabbitMQProvider",
        },
        "vector_db": {
            "chromadb": "ChromaDBProvider",
            "pinecone": "PineconeProvider",
            "qdrant": "QdrantProvider",
        },
        "embedding": {
            "llamacpp": "LlamaCppEmbeddingProvider",
            "llamacpp_embedding": "LlamaCppEmbeddingProvider",
        },
        "graph_db": {
            "neo4j": "Neo4jProvider",
        },
        "state_persister": {
            "redis": "RedisStatePersister",
            "mongodb": "MongoStatePersister",
            "postgres": "PostgresStatePersister",
            "file": "FileStatePersister",
        },
    }

    # Map of provider types to modules
    module_map = {
        "llm": "flowlib.providers.llm",
        "database": "flowlib.providers.db",
        "vector_db": "flowlib.providers.vector",
        "message_queue": "flowlib.providers.mq",
        "cache": "flowlib.providers.cache",
        "storage": "flowlib.providers.storage",
        "embedding": "flowlib.providers.embedding",
        "graph_db": "flowlib.providers.graph",
        "state_persister": "flowlib.agent.persistence.adapters",
    }

    # Check if we have a mapping for this provider type and implementation
    if provider_type not in provider_map or implementation not in provider_map[provider_type]:
        # Special handling for file persister as it might be in a different file
        if provider_type == "state_persister" and implementation == "file":
            try:
                from flowlib.agent.components.persistence.file import FileStatePersister
                return cast(Type[Provider[Any]], FileStatePersister)
            except ImportError as e:
                logger.error(f"Failed to import provider class for {provider_type}/{implementation}: {str(e)}")
                raise
        else:
            raise ImportError(f"No provider mapping for {provider_type}/{implementation}")

    # Get the module path and class name
    # Handle potential special case for file persister path
    if provider_type == "state_persister" and implementation == "file":
         pass
    else:
         module_map[provider_type]
         provider_map[provider_type][implementation]

    try:
        # Use direct imports for each provider type/implementation
        if provider_type == "llm":
            if implementation == "llamacpp":
                from ..llm.llama_cpp.provider import LlamaCppProvider
                return cast(Type[Provider[Any]], LlamaCppProvider)
        elif provider_type == "database":
            if implementation == "postgresql":
                from ..db.postgres.provider import PostgreSQLProvider
                return cast(Type[Provider[Any]], PostgreSQLProvider)
            elif implementation == "mongodb":
                from ..db.mongodb.provider import MongoDBProvider
                return cast(Type[Provider[Any]], MongoDBProvider)
            elif implementation == "sqlite":
                from ..db.sqlite.provider import SQLiteDBProvider
                return cast(Type[Provider[Any]], SQLiteDBProvider)
        elif provider_type == "vector_db":
            if implementation == "chromadb":
                from ..vector.chroma.provider import ChromaDBProvider
                return cast(Type[Provider[Any]], ChromaDBProvider)
            elif implementation == "pinecone":
                from ..vector.pinecone.provider import PineconeProvider
                return cast(Type[Provider[Any]], PineconeProvider)
            elif implementation == "qdrant":
                from ..vector.qdrant.provider import QdrantProvider
                return cast(Type[Provider[Any]], QdrantProvider)
        elif provider_type == "embedding":
            if implementation == "llamacpp":
                from ..embedding.llama_cpp.provider import LlamaCppEmbeddingProvider
                return cast(Type[Provider[Any]], LlamaCppEmbeddingProvider)
            elif implementation == "llamacpp_embedding":
                from ..embedding.llama_cpp.provider import LlamaCppEmbeddingProvider
                return cast(Type[Provider[Any]], LlamaCppEmbeddingProvider)
        elif provider_type == "graph_db":
            if implementation == "neo4j":
                from ..graph.neo4j.provider import Neo4jProvider
                return cast(Type[Provider[Any]], Neo4jProvider)
        elif provider_type == "message_queue":
            if implementation == "rabbitmq":
                from ..mq.rabbitmq.provider import RabbitMQProvider
                return RabbitMQProvider
        elif provider_type == "state_persister":
             if implementation == "redis":
                 from flowlib.agent.components.persistence.adapters import (
                     RedisStatePersister,
                 )
                 return cast(Type[Provider[Any]], RedisStatePersister)
             elif implementation == "mongodb":
                 from flowlib.agent.components.persistence.adapters import (
                     MongoStatePersister,
                 )
                 return cast(Type[Provider[Any]], MongoStatePersister)
             elif implementation == "postgres":
                 from flowlib.agent.components.persistence.adapters import (
                     PostgresStatePersister,
                 )
                 return cast(Type[Provider[Any]], PostgresStatePersister)
             elif implementation == "file": # Import handled earlier
                 from flowlib.agent.components.persistence.file import (
                     FileStatePersister,
                 )
                 return cast(Type[Provider[Any]], FileStatePersister)

        # If we get here, something went wrong with our mappings
        raise ImportError(f"Provider import failed for {provider_type}/{implementation}")
    except ImportError as e:
        logger.error(f"Failed to import provider class for {provider_type}/{implementation}: {str(e)}")
        raise

def _import_provider_type(provider_type: str) -> Type[Provider[Any]]:
    """Dynamically import a base provider class by type.
    
    Args:
        provider_type: Type of provider
        
    Returns:
        Provider class
        
    Raises:
        ImportError: If provider class cannot be imported
    """
    # Map of provider types to import paths
    import_map = {
        "llm": "from ..llm.base import LLMProvider; return LLMProvider",
        "database": "from ..db.base import DBProvider; return DBProvider",
        "message_queue": "from ..mq.base import MQProvider; return MQProvider",
        "cache": "from ..cache.base import CacheProvider; return CacheProvider",
        "vector_db": "from ..vector.base import VectorDBProvider; return VectorDBProvider",
        "storage": "from ..storage.base import StorageProvider; return StorageProvider",
        "embedding": "from ..embedding.base import EmbeddingProvider; return EmbeddingProvider",
        "graph_db": "from ..graph.base import GraphDBProvider; return GraphDBProvider",
    }

    # Get import statement
    if provider_type in import_map:
        import_statement = import_map[provider_type]
        try:
            # Execute import statement
            local_vars: Dict[str, Any] = {}
            exec(import_statement, globals(), local_vars)
            return cast(Type[Provider[Any]], local_vars["return"])
        except ImportError as e:
            logger.error(f"Failed to import provider class for {provider_type}: {str(e)}")
            raise

    raise ImportError(f"No import mapping for {provider_type}")
