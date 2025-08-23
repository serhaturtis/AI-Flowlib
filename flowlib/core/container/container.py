"""Central dependency container eliminating circular dependencies.

This module provides a unified dependency container that replaces all
individual registries, completely eliminating circular dependencies
through centralized dependency management and resolution.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from dataclasses import dataclass

from flowlib.core.interfaces.interfaces import Provider, Resource, Flow, Configuration, Container
from flowlib.core.loader.loader import DynamicLoader
from .models import ProviderMetadata, ResourceMetadata, FlowMetadata, ConfigMetadata, RegistryEntryData

logger = logging.getLogger(__name__)


@dataclass
class RegistryEntry:
    """Entry in the dependency registry."""
    name: str
    item_type: str
    factory: Callable[[], Any]
    metadata: Dict[str, Any]
    instance: Optional[Any] = None
    initialized: bool = False


class DependencyContainer:
    """Central dependency container eliminating circular dependencies.
    
    This container replaces all individual registries (provider, resource, flow)
    with a single, unified dependency management system. No circular imports
    are possible as all dependencies are resolved through this central container.
    """
    
    def __init__(self):
        # Central registry eliminating multiple registries
        self._entries: Dict[Tuple[str, str], RegistryEntry] = {}  # (type, name) -> entry
        self._aliases: Dict[str, Tuple[str, str]] = {}  # alias -> (type, name)
        
        # Type tracking
        self._provider_types: Set[str] = set()
        self._resource_types: Set[str] = set()
        self._flow_names: Set[str] = set()
        
        # Async initialization tracking
        self._initialization_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        
        # Configuration cache
        self._config_cache: Dict[str, Configuration] = {}
        
        # Dynamic loader for creating instances
        self._loader = DynamicLoader()
    
    def register(self, item_type: str, name: str, factory: Callable[[], Any], metadata: Dict[str, Any]) -> None:
        """Register an item in the central container.
        
        Args:
            item_type: Type of item ('provider', 'resource', 'flow', 'config')
            name: Unique name for the item
            factory: Factory function to create the item
            metadata: Additional metadata about the item
        """
        key = (item_type, name)
        
        entry = RegistryEntry(
            name=name,
            item_type=item_type,
            factory=factory,
            metadata=metadata
        )
        
        self._entries[key] = entry
        
        # Update type tracking with strict parsing
        if item_type == 'provider':
            # Parse as provider metadata
            try:
                provider_meta = ProviderMetadata(**metadata)
                self._provider_types.add(provider_meta.provider_type)
            except Exception as e:
                raise ValueError(f"Invalid provider metadata for {name}: {e}")
        elif item_type in ('resource', 'config'):
            # Parse as resource metadata
            try:
                resource_meta = ResourceMetadata(**metadata)
                self._resource_types.add(resource_meta.resource_type)
            except Exception as e:
                raise ValueError(f"Invalid resource metadata for {name}: {e}")
        elif item_type == 'flow':
            # Parse as flow metadata
            try:
                flow_meta = FlowMetadata(**metadata)
                self._flow_names.add(name)
            except Exception as e:
                raise ValueError(f"Invalid flow metadata for {name}: {e}")
        
        logger.debug(f"Registered {item_type}: {name}")
    
    def register_alias(self, alias: str, item_type: str, name: str) -> None:
        """Register an alias for an item.
        
        Args:
            alias: Alias name
            item_type: Type of the target item
            name: Name of the target item
        """
        self._aliases[alias] = (item_type, name)
        logger.debug(f"Registered alias '{alias}' -> {item_type}:{name}")
    
    async def get_provider(self, config_name: str) -> Provider:
        """Get provider by configuration name.
        
        Args:
            config_name: Name of the provider configuration
            
        Returns:
            Initialized provider instance
            
        Raises:
            KeyError: If configuration not found
            ValueError: If configuration is invalid
        """
        # Get configuration - try config type first, then any resource type
        try:
            config = self.get_resource(config_name, 'config')
        except KeyError:
            # Try to find it as any resource type
            try:
                config = self.get_resource(config_name)
            except KeyError:
                raise KeyError(f"Configuration '{config_name}' not found")
        
        if not isinstance(config, Configuration):
            raise ValueError(f"Resource '{config_name}' is not a configuration")
        
        provider_type = config.get_provider_type()
        settings = config.get_settings()
        
        # Create or get cached provider
        provider_key = f"{provider_type}:{config_name}"
        
        if provider_key not in self._config_cache:
            # Create provider using dynamic loader
            provider = self._loader.create_provider(provider_type, settings)
            
            # Initialize if needed - no fallbacks, be explicit
            if hasattr(provider, 'initialize'):
                # Check if already initialized (assume False if attribute doesn't exist)
                is_initialized = hasattr(provider, 'initialized') and provider.initialized
                if not is_initialized:
                    await provider.initialize()
            
            self._config_cache[provider_key] = provider
            logger.debug(f"Created and cached provider: {provider_key}")
        
        return self._config_cache[provider_key]
    
    def get_resource(self, name: str, resource_type: Optional[str] = None) -> Resource:
        """Get resource by name and optional type.
        
        Args:
            name: Resource name
            resource_type: Optional resource type for disambiguation
            
        Returns:
            Resource instance
            
        Raises:
            KeyError: If resource not found
        """
        # Try alias first
        if name in self._aliases:
            alias_type, alias_name = self._aliases[name]
            key = (alias_type, alias_name)
        else:
            # Direct lookup
            if resource_type:
                key = (resource_type, name)
            else:
                # Search all resource types
                key = self._find_resource_key(name)
        
        if key not in self._entries:
            raise KeyError(f"Resource '{name}' not found")
        
        entry = self._entries[key]
        
        # Create instance if not cached
        if entry.instance is None:
            entry.instance = entry.factory()
            logger.debug(f"Created resource instance: {name}")
        
        return entry.instance
    
    def get_flow(self, name: str) -> Flow:
        """Get flow by name.
        
        Args:
            name: Flow name
            
        Returns:
            Flow instance
            
        Raises:
            KeyError: If flow not found
        """
        # Try alias first
        if name in self._aliases:
            alias_type, alias_name = self._aliases[name]
            key = (alias_type, alias_name)
        else:
            key = ('flow', name)
        
        if key not in self._entries:
            raise KeyError(f"Flow '{name}' not found")
        
        entry = self._entries[key]
        
        # Create instance if not cached
        if entry.instance is None:
            entry.instance = entry.factory()
            logger.debug(f"Created flow instance: {name}")
        
        return entry.instance
    
    async def get_provider_by_type(self, provider_type: str, name: Optional[str] = None) -> Provider:
        """Get provider by provider type and optional name.
        
        Args:
            provider_type: Provider type identifier
            name: Optional provider name
            
        Returns:
            Provider instance
        """
        if name:
            # Look for specific provider
            key = ('provider', name)
            if key in self._entries:
                entry = self._entries[key]
                if entry.instance is None:
                    entry.instance = entry.factory()
                    if hasattr(entry.instance, 'initialize'):
                        await entry.instance.initialize()
                return entry.instance
        
        # Create provider using dynamic loader
        provider = self._loader.create_provider(provider_type, {})
        if hasattr(provider, 'initialize'):
            await provider.initialize()
        
        return provider
    
    def _find_resource_key(self, name: str) -> Tuple[str, str]:
        """Find resource key by searching all resource types.
        
        Args:
            name: Resource name to find
            
        Returns:
            Resource key tuple
            
        Raises:
            KeyError: If resource not found
        """
        for (item_type, item_name), entry in self._entries.items():
            if item_type in ('resource', 'config') and item_name == name:
                return (item_type, item_name)
        
        raise KeyError(f"Resource '{name}' not found in any type")
    
    def list_providers(self) -> List[str]:
        """List all registered provider names.
        
        Returns:
            List of provider names
        """
        return [name for (item_type, name) in self._entries.keys() if item_type == 'provider']
    
    def list_resources(self) -> List[str]:
        """List all registered resource names.
        
        Returns:
            List of resource names
        """
        return [name for (item_type, name) in self._entries.keys() if item_type in ('resource', 'config')]
    
    def list_flows(self) -> List[str]:
        """List all registered flow names.
        
        Returns:
            List of flow names
        """
        return [name for (item_type, name) in self._entries.keys() if item_type == 'flow']
    
    def get_provider_types(self) -> Set[str]:
        """Get all registered provider types.
        
        Returns:
            Set of provider types
        """
        return self._provider_types.copy()
    
    def get_resource_types(self) -> Set[str]:
        """Get all registered resource types.
        
        Returns:
            Set of resource types
        """
        return self._resource_types.copy()
    
    def contains(self, item_type: str, name: str) -> bool:
        """Check if item exists in container.
        
        Args:
            item_type: Type of item
            name: Name of item
            
        Returns:
            True if item exists
        """
        return (item_type, name) in self._entries
    
    def get_metadata(self, item_type: str, name: str) -> Dict[str, Any]:
        """Get metadata for an item.
        
        Args:
            item_type: Type of item
            name: Name of item
            
        Returns:
            Item metadata
            
        Raises:
            KeyError: If item not found
        """
        key = (item_type, name)
        if key not in self._entries:
            raise KeyError(f"{item_type} '{name}' not found")
        
        return self._entries[key].metadata.copy()
    
    async def initialize_all_providers(self) -> None:
        """Initialize all registered providers."""
        for (item_type, name), entry in self._entries.items():
            if item_type == 'provider':
                try:
                    if entry.instance is None:
                        entry.instance = entry.factory()
                    
                    if hasattr(entry.instance, 'initialize') and not entry.initialized:
                        await entry.instance.initialize()
                        entry.initialized = True
                        
                    logger.debug(f"Initialized provider: {name}")
                except Exception as e:
                    logger.error(f"Failed to initialize provider '{name}': {e}")
    
    async def shutdown_all_providers(self) -> None:
        """Shutdown all initialized providers."""
        for (item_type, name), entry in self._entries.items():
            if item_type == 'provider' and entry.instance and entry.initialized:
                try:
                    if hasattr(entry.instance, 'shutdown'):
                        await entry.instance.shutdown()
                    logger.debug(f"Shutdown provider: {name}")
                except Exception as e:
                    logger.error(f"Failed to shutdown provider '{name}': {e}")
                finally:
                    # Always mark as not initialized, even if shutdown failed
                    entry.initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get container statistics.
        
        Returns:
            Container statistics
        """
        type_counts = {}
        for (item_type, name) in self._entries.keys():
            # No fallbacks - use explicit counting
            if item_type not in type_counts:
                type_counts[item_type] = 0
            type_counts[item_type] += 1
        
        return {
            'total_entries': len(self._entries),
            'type_counts': type_counts,
            'aliases': len(self._aliases),
            'provider_types': len(self._provider_types),
            'resource_types': len(self._resource_types),
            'cached_configs': len(self._config_cache)
        }


# Global container instance
_global_container = DependencyContainer()


def get_container() -> DependencyContainer:
    """Get the global dependency container.
    
    Returns:
        Global dependency container instance
    """
    return _global_container


def set_global_container(container: DependencyContainer) -> None:
    """Set the global dependency container.
    
    Args:
        container: Container instance to use globally
    """
    global _global_container
    _global_container = container