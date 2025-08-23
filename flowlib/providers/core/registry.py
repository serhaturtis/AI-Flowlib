"""Clean Provider Registry Implementation.

This module provides a clean, config-driven provider registry with no legacy methods.
All provider access is done through get_by_config() with centralized configurations.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, Type

from flowlib.core.registry.registry import BaseRegistry
from flowlib.core.errors.errors import ExecutionError
from .base import Provider
from flowlib.resources.registry.registry import resource_registry
from flowlib.resources.models.constants import ResourceType

logger = logging.getLogger(__name__)


class CleanProviderRegistry(BaseRegistry[Any]):
    """Clean provider registry with config-driven access only.
    
    This registry enforces the config-driven pattern for all provider access:
    - No legacy type+name access methods
    - Only get_by_config() for provider retrieval  
    - Centralized provider configuration management
    - Automatic provider initialization and lifecycle management
    """
    
    def __init__(self):
        """Initialize clean provider registry."""
        # Core storage for providers: (provider_type, name) -> provider
        self._providers: Dict[Tuple[str, str], Provider] = {}
        
        # Factory storage for dynamic provider creation
        self._factories: Dict[Tuple[str, str], Callable[[], Provider]] = {}
        self._factory_metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        # Async initialization tracking
        self._initialized_providers: Dict[Tuple[str, str], Provider] = {}
        self._initialization_locks: Dict[Tuple[str, str], asyncio.Lock] = {}

    def register_provider(self, provider: Provider, **metadata: Any) -> None:
        """Register a provider instance.
        
        Args:
            provider: Provider to register (must be a ProviderBase subclass)
            **metadata: Additional metadata about the provider
            
        Raises:
            TypeError: If the provider is not a ProviderBase subclass
            ValueError: If provider lacks required attributes
        """
        from .provider_base import ProviderBase
        
        if not isinstance(provider, ProviderBase):
            raise TypeError(f"Provider must be a ProviderBase subclass, got {type(provider)}")
        
        # Strict validation - provider must have required attributes defined in ProviderBase
        try:
            if not provider.name:
                raise ValueError("Provider must have a non-empty name")
        except AttributeError:
            raise ValueError("Provider must have a 'name' attribute")
            
        try:
            if not provider.provider_type:
                raise ValueError("Provider must have a non-empty provider_type")
        except AttributeError:
            raise ValueError("Provider must have a 'provider_type' attribute")
            
        key = (provider.provider_type, provider.name)
        self._providers[key] = provider
        
        logger.info(f"Registered provider: {provider.name} (type: {provider.provider_type})")
    
    def register(self, name: str, obj: Any, **metadata: Any) -> None:
        """Register an object by name (BaseRegistry interface).
        
        Args:
            name: Name for the object (for providers, this should be config name)
            obj: Object to register (must be a Provider)
            **metadata: Additional metadata
        """
        if hasattr(obj, 'name') and hasattr(obj, 'provider_type'):
            # This is a provider object - use the provider-specific method
            self.register_provider(obj, **metadata)
        else:
            raise TypeError(
                "CleanProviderRegistry only accepts Provider objects. "
                f"Got {type(obj).__name__}."
            )

    def register_factory(self, provider_type: str, name: str, factory: Callable[[], Provider], **metadata: Any) -> None:
        """Register a factory for creating providers.
        
        Args:
            provider_type: Type of provider (e.g., llm, database)
            name: Unique name for this provider
            factory: Factory function that creates the provider
            **metadata: Additional metadata about the provider
        """
        key = (provider_type, name)
        self._factories[key] = factory
        self._factory_metadata[key] = {"provider_type": provider_type, **metadata}
        
        logger.info(f"Registered provider factory: {name} (type: {provider_type})")

    async def get_by_config(self, config_name: str) -> Provider:
        """Get a provider instance using a configuration resource.
        
        This is the ONLY way to access providers in the clean registry.
        
        Args:
            config_name: Name of the configuration resource
            
        Returns:
            Initialized and configured provider instance
            
        Raises:
            KeyError: If config not found
            ExecutionError: If provider initialization fails
            
        Example:
            llm = await provider_registry.get_by_config("default-llm")
            vector_db = await provider_registry.get_by_config("default-vector-db")
        """
        try:
            # Get the configuration resource
            from flowlib.resources.models.config_resource import ProviderConfigResource
            
            config = resource_registry.get(config_name)
            
            if not isinstance(config, ProviderConfigResource):
                raise TypeError(
                    f"Resource '{config_name}' is not a ProviderConfigResource, "
                    f"got {type(config).__name__}"
                )
            
            # Extract provider information from config
            provider_type = config.get_provider_type()
            if not provider_type:
                raise ValueError(f"Config '{config_name}' does not specify a provider_type")
            
            # Determine provider category from config type
            provider_category = self._infer_provider_category_from_config(provider_type, config)
            
            # Get or create the provider with settings from config
            settings = config.get_settings()
            provider = await self._get_or_create_provider(provider_category, provider_type, settings)
            
            logger.info(
                f"Successfully resolved provider '{provider_type}' "
                f"for config '{config_name}' with category '{provider_category}'"
            )
            
            return provider
            
        except Exception as e:
            if isinstance(e, (KeyError, TypeError, ValueError, ExecutionError)):
                raise
            
            raise ExecutionError(
                message=f"Failed to get provider by config '{config_name}': {str(e)}",
                context={
                    "config_name": config_name,
                    "available_configs": self._list_available_configs()
                },
                cause=e
            ) from e

    async def _get_or_create_provider(self, provider_category: str, provider_type: str, settings: Optional[Dict[str, Any]] = None) -> Provider:
        """Get or create a provider with automatic initialization.
        
        Args:
            provider_category: Category of provider (e.g., 'llm', 'database')
            provider_type: Type of provider (e.g., 'googleai', 'postgres')
            settings: Optional settings dict to pass to factory
            
        Returns:
            Initialized provider instance
        """
        key = (provider_category, provider_type)
        
        # Check if already initialized
        if key in self._initialized_providers:
            return self._initialized_providers[key]
        
        # Get or create lock for thread-safe initialization
        if key not in self._initialization_locks:
            self._initialization_locks[key] = asyncio.Lock()
        
        async with self._initialization_locks[key]:
            # Double-checked locking
            if key in self._initialized_providers:
                return self._initialized_providers[key]
            
            try:
                # Try direct provider first
                if key in self._providers:
                    provider = self._providers[key]
                    if hasattr(provider, 'initialized') and not provider.initialized:
                        await provider.initialize()
                    self._initialized_providers[key] = provider
                    return provider
                
                # Try factory if direct provider not found
                elif key in self._factories:
                    factory = self._factories[key]
                    # Pass settings to factory - this is the critical fix
                    provider = factory(settings)
                    
                    if hasattr(provider, 'initialize'):
                        await provider.initialize()
                    
                    self.register_provider(provider)
                    self._initialized_providers[key] = provider
                    return provider
                
                else:
                    raise KeyError(
                        f"Provider '{provider_type}' of category '{provider_category}' not found. "
                        f"Available providers: {self.list_providers()}"
                    )
                    
            except Exception as e:
                raise ExecutionError(
                    message=f"Failed to initialize provider '{provider_type}' of category '{provider_category}': {str(e)}",
                    context={
                        "provider_type": provider_type,
                        "provider_category": provider_category,
                        "available_providers": self.list_providers()
                    },
                    cause=e
                )

    # REMOVED: _configure_provider_with_config - Settings now passed at creation time
    # Settings are now passed to the factory at provider creation time via
    # _get_or_create_provider(provider_category, provider_type, settings)
    # This follows our principle of single source of truth - no double configuration

    def _infer_provider_category_from_config(self, provider_type: str, config: 'ProviderConfigResource') -> str:
        """Infer provider category from config type and provider type."""
        from flowlib.resources.models.config_resource import (
            LLMConfigResource, DatabaseConfigResource, VectorDBConfigResource,
            CacheConfigResource, StorageConfigResource, EmbeddingConfigResource,
            GraphDBConfigResource, MessageQueueConfigResource
        )
        
        # Infer from config type first
        if isinstance(config, LLMConfigResource):
            return "llm"
        elif isinstance(config, DatabaseConfigResource):
            return "database"
        elif isinstance(config, VectorDBConfigResource):
            return "vector_db"
        elif isinstance(config, CacheConfigResource):
            return "cache"
        elif isinstance(config, StorageConfigResource):
            return "storage"
        elif isinstance(config, EmbeddingConfigResource):
            return "embedding"
        elif isinstance(config, GraphDBConfigResource):
            return "graph_db"
        elif isinstance(config, MessageQueueConfigResource):
            return "message_queue"
        
        # Fallback to provider type mapping
        provider_category_map = {
            "llamacpp": "llm",
            "google_ai": "llm",
            "openai": "llm",
            "postgres": "database",
            "mongodb": "database",
            "sqlite": "database",
            "chroma": "vector_db",
            "pinecone": "vector_db",
            "qdrant": "vector_db",
            "redis": "cache",
            "s3": "storage",
            "local": "storage",
            "llamacpp_embedding": "embedding",
            "openai_embedding": "embedding",
            "neo4j": "graph_db",
            "arango": "graph_db",
            "rabbitmq": "message_queue",
            "kafka": "message_queue"
        }
        
        if provider_type not in provider_category_map:
            # If not in map, raise error - no fallbacks
            raise ValueError(f"Unknown provider type: {provider_type}")
        return provider_category_map[provider_type]

    def _list_available_configs(self) -> List[str]:
        """List available provider configurations for debugging."""
        try:
            configs = []
            config_types = [
                ResourceType.MODEL_CONFIG, ResourceType.LLM_CONFIG,
                ResourceType.DATABASE_CONFIG, ResourceType.VECTOR_DB_CONFIG,
                ResourceType.CACHE_CONFIG, ResourceType.STORAGE_CONFIG,
                ResourceType.EMBEDDING_CONFIG, ResourceType.GRAPH_DB_CONFIG,
                ResourceType.MESSAGE_QUEUE_CONFIG
            ]
            
            for config_type in config_types:
                try:
                    all_resources = resource_registry.get_by_type(config_type)
                    configs.extend(all_resources.keys())
                except KeyError:
                    # Config type not found - expected, continue
                    continue
                except Exception as e:
                    # Unexpected error - log and fail fast
                    logger.error(f"Failed to list resources for type {config_type}: {e}")
                    raise
            
            return configs
        except Exception as e:
            logger.error(f"Failed to list available configs: {e}")
            raise

    def list_providers(self) -> List[str]:
        """List all registered providers."""
        providers = []
        for (provider_type, name) in self._providers.keys():
            providers.append(f"{provider_type}:{name}")
        for (provider_type, name) in self._factories.keys():
            providers.append(f"{provider_type}:{name} (factory)")
        return providers

    def contains(self, name: str) -> bool:
        """Check if a provider config exists by name (BaseRegistry interface).
        
        Args:
            name: Name of the configuration to check
            
        Returns:
            True if the configuration exists, False otherwise
        """
        try:
            # Check if config exists in resource registry
            config = resource_registry.get(name)
            from flowlib.resources.models.config_resource import ProviderConfigResource
            return isinstance(config, ProviderConfigResource)
        except KeyError:
            return False
    
    # Abstract method implementations required by BaseRegistry
    def get(self, name: str, expected_type: Optional[Type] = None) -> Any:
        """Get provider by config name (BaseRegistry interface).
        
        This method is not supported in async-first architecture.
        Use 'await get_by_config(name)' instead.
        """
        raise NotImplementedError(
            "Synchronous get() method not supported in async-first architecture. "
            "Use 'await get_by_config(name)' instead."
        )
    
    def list(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """List provider configs (BaseRegistry interface)."""
        try:
            configs = self._list_available_configs()
            if filter_criteria:
                # Simple filtering by provider_type if specified
                if 'provider_type' in filter_criteria:
                    provider_type = filter_criteria['provider_type']
                    configs = [c for c in configs if provider_type in c.lower()]
            return configs
        except Exception:
            return []
    
    def clear(self) -> None:
        """Clear all registrations from the registry.
        
        This method removes all registered providers, factories, and initialized providers.
        It also performs proper shutdown of any active provider instances.
        """
        # Shutdown all initialized providers first
        if self._initialized_providers:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a task to shutdown providers
                    asyncio.create_task(self.shutdown_all())
                else:
                    loop.run_until_complete(self.shutdown_all())
            except Exception as e:
                logger.warning(f"Error during provider shutdown in clear(): {e}")
        
        # Clear all storage
        self._providers.clear()
        self._factories.clear()
        self._factory_metadata.clear()
        self._initialized_providers.clear()
        self._initialization_locks.clear()
        
        logger.debug("Cleared all providers from registry")
    
    def remove(self, name: str) -> bool:
        """Remove a specific provider registration from the registry.
        
        Args:
            name: Name of the provider configuration to remove
            
        Returns:
            True if the provider was found and removed, False if not found
        """
        removed = False
        
        # Find providers matching the config name
        keys_to_remove = []
        for key in self._providers.keys():
            provider_category, provider_type = key
            # Check if this provider matches the config name
            try:
                config = resource_registry.get(name)
                from flowlib.resources.models.config_resource import ProviderConfigResource
                if isinstance(config, ProviderConfigResource):
                    config_provider_type = config.get_provider_type()
                    if config_provider_type == provider_type:
                        keys_to_remove.append(key)
            except KeyError:
                continue
        
        # Remove found providers
        for key in keys_to_remove:
            # Shutdown if initialized
            if key in self._initialized_providers:
                provider = self._initialized_providers[key]
                if hasattr(provider, 'shutdown'):
                    try:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(provider.shutdown())
                        else:
                            loop.run_until_complete(provider.shutdown())
                    except Exception as e:
                        logger.warning(f"Error shutting down provider {key} during removal: {e}")
                
                del self._initialized_providers[key]
            
            # Remove from main storage
            if key in self._providers:
                del self._providers[key]
                removed = True
            
            # Remove from factories
            if key in self._factories:
                del self._factories[key]
                self._factory_metadata.pop(key, None)
                removed = True
            
            # Clean up initialization locks
            self._initialization_locks.pop(key, None)
        
        if removed:
            logger.debug(f"Removed provider configuration '{name}' from registry")
        
        return removed
    
    def update(self, name: str, obj: Any, **metadata) -> bool:
        """Update or replace an existing provider registration.
        
        Args:
            name: Name of the provider configuration to update
            obj: New provider object to register
            **metadata: Additional metadata about the provider
            
        Returns:
            True if an existing provider was updated, False if this was a new registration
        """
        # Check if provider exists
        existing_found = self.contains(name)
        
        if existing_found:
            # Remove existing
            self.remove(name)
            
            # Re-register
            self.register(name, obj, **metadata)
            logger.debug(f"Updated existing provider configuration '{name}' in registry")
            return True
        else:
            # New registration
            self.register(name, obj, **metadata)
            logger.debug(f"Registered new provider configuration '{name}' in registry")
            return False
    

    async def initialize_all(self) -> None:
        """Initialize all registered providers."""
        for key in list(self._providers.keys()) + list(self._factories.keys()):
            if key not in self._initialized_providers:
                try:
                    await self._get_or_create_provider(key[0], key[1])
                except Exception as e:
                    logger.warning(f"Failed to initialize provider {key}: {e}")

    async def shutdown_all(self) -> None:
        """Shutdown all initialized providers."""
        for key, provider in list(self._initialized_providers.items()):
            if hasattr(provider, 'shutdown'):
                try:
                    await provider.shutdown()
                    logger.info(f"Shut down provider {key}")
                except Exception as e:
                    logger.warning(f"Error shutting down provider {key}: {e}")
            
            self._initialized_providers.pop(key, None)
            self._initialization_locks.pop(key, None)


# Global provider registry instance
provider_registry = CleanProviderRegistry()