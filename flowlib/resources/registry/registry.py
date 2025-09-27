"""Resource registry implementation for non-provider resources.

This module provides a concrete implementation of the BaseRegistry for
managing non-provider resources like models, prompts, and configurations.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from flowlib.core.registry.registry import BaseRegistry
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.models.base import ResourceBase
# Auto-discovery removed - projects handle configuration loading explicitly

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ResourceRegistry(BaseRegistry[ResourceBase]):
    """Registry for non-provider resources.
    
    This class implements the BaseRegistry interface for managing resources
    like models, prompts, and configurations with type safety and validation.
    """
    
    def __init__(self) -> None:
        """Initialize resource registry."""
        # Main storage for resources: (resource_type, name) -> resource
        self._resources: Dict[tuple[str, str], ResourceBase] = {}
        # Storage for resource metadata
        self._metadata: Dict[tuple[str, str], Dict[str, Any]] = {}
        # Collection of resource types in use
        self._resource_types: set[str] = set()
        # Alias mapping: alias_name -> (canonical_resource_type, canonical_name)
        self._aliases: Dict[str, tuple[str, str]] = {}
        # Reverse mapping: (canonical_resource_type, canonical_name) -> [alias_names]
        self._canonical_to_aliases: Dict[tuple[str, str], List[str]] = {}
        
        # Registry is now stateless - no auto-discovery needed
    
    # Auto-discovery method removed - projects handle configuration loading explicitly
        
    def register(self, name: str, obj: ResourceBase, resource_type: str = ResourceType.MODEL_CONFIG, **metadata: Union[str, int, bool]) -> None:
        """Register a resource with the registry.
        
        Args:
            name: Unique name for the resource
            obj: The resource to register (must be a ResourceBase subclass)
            resource_type: Type of the resource
            **metadata: Additional metadata about the resource
        
        Raises:
            TypeError: If the resource is not a ResourceBase subclass
            ValueError: If resource with same name/type already exists or invalid resource type
        """
        # Validate resource_type against ResourceType enum (allow test types)
        known_types = [rt.value for rt in ResourceType]
        if resource_type not in known_types and not resource_type.startswith(('type', 'test')):
            raise ValueError(f"Invalid resource type '{resource_type}'. Must be one of: {known_types}")
            
        key = (resource_type, name)
        if not isinstance(obj, ResourceBase):
            raise TypeError(f"Resource '{name}' must be a ResourceBase subclass (pydantic v2), got {type(obj)}")
        if key in self._resources:
            raise ValueError(f"Resource '{name}' of type '{resource_type}' already exists")
        self._resources[key] = obj
        self._metadata[key] = metadata
        self._resource_types.add(resource_type)
        logger.debug(f"Registered {resource_type} '{name}'")
        
    def get(self, name: str, expected_type: Optional[Type[Any]] = None) -> ResourceBase:
        """Get a resource by name or alias (type determined by decorator).
        
        This is the ONLY way to access resources in the clean registry.
        The resource type is automatically determined by the decorator used
        when the resource was registered (@prompt, @model_config, etc.).
        
        Args:
            name: Name or alias of the resource
            expected_type: Optional Python type for validation
            
        Returns:
            The requested resource
            
        Raises:
            KeyError: If resource doesn't exist
            TypeError: If resource doesn't match expected type
            
        Example:
            prompt = resource_registry.get("my-prompt-name")
            config = resource_registry.get("my-config")
            # Works with aliases too:
            llm = resource_registry.get("default-model")  # alias -> canonical
        """
        # Trigger auto-discovery on first access
        # Configuration loading now handled explicitly by projects
        # First, check if it's an alias
        if name in self._aliases:
            canonical_key = self._aliases[name]
            resource = self._resources[canonical_key]
            if expected_type and not isinstance(resource, expected_type):
                raise TypeError(
                    f"Resource '{name}' has type {type(resource).__name__}, "
                    f"expected {expected_type.__name__}"
                )
            return resource
        
        # Search across all resource types for the canonical name
        for (resource_type, resource_name), resource in self._resources.items():
            if resource_name == name:
                if expected_type and not isinstance(resource, expected_type):
                    raise TypeError(
                        f"Resource '{name}' has type {type(resource).__name__}, "
                        f"expected {expected_type.__name__}"
                    )
                return resource
        
        raise KeyError(f"Resource '{name}' not found")
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a resource by name only.
        
        Args:
            name: Name of the resource
            
        Returns:
            Metadata dictionary for the resource
            
        Raises:
            KeyError: If resource doesn't exist
        """
        # Search across all resource types for the name
        for (resource_type, resource_name), resource in self._resources.items():
            if resource_name == name:
                metadata_key = (resource_type, resource_name)
                if metadata_key not in self._metadata:
                    raise KeyError(f"Resource '{name}' found but metadata missing")
                return self._metadata[metadata_key]
        
        raise KeyError(f"Resource '{name}' not found")
        
    
    def contains(self, name: str) -> bool:
        """Check if a resource exists by name only.
        
        Args:
            name: Name to check
            
        Returns:
            True if the resource exists, False otherwise
        """
        # Search across all resource types for the name
        for (resource_type, resource_name), resource in self._resources.items():
            if resource_name == name:
                return True
        return False
    
    
    def list(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """List registered resources matching criteria.
        
        Args:
            filter_criteria: Optional criteria to filter results
                - resource_type: Filter by resource type
                
        Returns:
            List of resource names matching the criteria
        """
        filter_type = None
        if filter_criteria and 'resource_type' in filter_criteria:
            filter_type = filter_criteria['resource_type']
        
        result = []
        for (rt, name) in self._resources.keys():
            if filter_type is None or rt == filter_type:
                result.append(name)
                
        return result
    
    def list_types(self) -> List[str]:
        """List all resource types in the registry.
        
        Returns:
            List of resource types
        """
        return list(self._resource_types)
    
    def get_by_type(self, resource_type: str) -> Dict[str, Any]:
        """Get all resources of a specific type.
        
        Args:
            resource_type: Type of resources to retrieve
            
        Returns:
            Dictionary of resource names to resources
        """
        result = {}
        for (rt, name), resource in self._resources.items():
            if rt == resource_type:
                result[name] = resource
        return result
    
    def remove(self, name: str) -> bool:
        """Remove a specific registration from the registry.
        
        Args:
            name: Name of the resource to remove
            
        Returns:
            True if the resource was found and removed, False if not found
        """
        # Search across all resource types for the name
        keys_to_remove = []
        for key in self._resources.keys():
            resource_type, resource_name = key
            if resource_name == name:
                keys_to_remove.append(key)
        
        if keys_to_remove:
            for key in keys_to_remove:
                del self._resources[key]
                self._metadata.pop(key, None)
            logger.debug(f"Removed resource '{name}' from registry")
            return True
        
        return False
    
    def update(self, name: str, obj: ResourceBase, resource_type: Optional[str] = None, **metadata: Union[str, int, bool]) -> bool:
        """Update or replace an existing registration.
        
        Args:
            name: Name of the resource to update
            obj: New resource object to register
            resource_type: Type of the resource (required for new registrations)
            **metadata: Additional metadata about the resource
            
        Returns:
            True if an existing resource was updated, False if this was a new registration
        """
        # Check if resource exists
        existing_found = False
        for key in self._resources.keys():
            rt, resource_name = key
            if resource_name == name:
                existing_found = True
                break
        
        if existing_found:
            # Remove existing
            self.remove(name)
            
            # Re-register with same or new type
            final_resource_type = resource_type or rt
            self.register(name, obj, final_resource_type, **metadata)
            logger.debug(f"Updated existing resource '{name}' in registry")
            return True
        else:
            # New registration
            if resource_type is None:
                raise ValueError(f"resource_type is required for new registration of '{name}'")
            self.register(name, obj, resource_type, **metadata)
            logger.debug(f"Registered new resource '{name}' in registry")
            return False
    
    # Enhanced alias support methods
    def register_with_aliases(self, canonical_name: str, obj: ResourceBase, aliases: Optional[List[str]] = None, resource_type: str = ResourceType.MODEL_CONFIG, **metadata: Union[str, int, bool]) -> None:
        """Register a resource with canonical name and optional aliases.
        
        Args:
            canonical_name: Primary/canonical name for the resource
            obj: The resource to register
            aliases: Optional list of alias names that point to the canonical name
            resource_type: Type of the resource
            **metadata: Additional metadata about the resource
        """
        canonical_key = (resource_type, canonical_name)
        
        # Only register canonical name if it doesn't exist yet
        if not self.contains(canonical_name):
            self.register(canonical_name, obj, resource_type, **metadata)
        
        # Register aliases pointing to canonical resource
        if aliases:
            for alias in aliases:
                # Skip if alias already exists
                if alias in self._aliases:
                    logger.debug(f"Alias '{alias}' already exists, skipping")
                    continue
                    
                self._aliases[alias] = canonical_key
                if canonical_key not in self._canonical_to_aliases:
                    self._canonical_to_aliases[canonical_key] = []
                self._canonical_to_aliases[canonical_key].append(alias)
                logger.debug(f"Created alias '{alias}' -> '{canonical_name}' ({resource_type})")
    
    def create_alias(self, alias_name: str, canonical_name: str) -> bool:
        """Create an alias that points to an existing canonical name.
        
        Args:
            alias_name: New alias name to create
            canonical_name: Existing canonical name to point to
            
        Returns:
            True if alias was created, False if canonical name doesn't exist
        """
        # Find the canonical resource key
        canonical_key = None
        for (resource_type, resource_name) in self._resources.keys():
            if resource_name == canonical_name:
                canonical_key = (resource_type, resource_name)
                break
        
        if not canonical_key:
            return False
        
        # Create alias
        self._aliases[alias_name] = canonical_key
        if canonical_key not in self._canonical_to_aliases:
            self._canonical_to_aliases[canonical_key] = []
        self._canonical_to_aliases[canonical_key].append(alias_name)
        
        logger.debug(f"Created alias '{alias_name}' -> '{canonical_name}' ({canonical_key[0]})")
        return True
    
    def remove_alias(self, alias_name: str) -> bool:
        """Remove an alias (but not the canonical object).
        
        Args:
            alias_name: Alias name to remove
            
        Returns:
            True if alias was removed, False if not found
        """
        if alias_name not in self._aliases:
            return False
        
        canonical_key = self._aliases[alias_name]
        del self._aliases[alias_name]
        
        # Update reverse mapping
        if canonical_key in self._canonical_to_aliases:
            self._canonical_to_aliases[canonical_key].remove(alias_name)
            if not self._canonical_to_aliases[canonical_key]:
                del self._canonical_to_aliases[canonical_key]
        
        logger.debug(f"Removed alias '{alias_name}'")
        return True
    
    def list_aliases(self, canonical_name: str) -> List[str]:
        """List all aliases that point to a canonical name.
        
        Args:
            canonical_name: Canonical name to find aliases for
            
        Returns:
            List of alias names pointing to the canonical name
        """
        # Find the canonical resource key
        for (resource_type, resource_name) in self._resources.keys():
            if resource_name == canonical_name:
                canonical_key = (resource_type, resource_name)
                if canonical_key not in self._canonical_to_aliases:
                    # No aliases exist for this canonical name
                    return []
                return self._canonical_to_aliases[canonical_key]
        
        return []
    
    def clear(self) -> None:
        """Clear all registrations and aliases from the registry."""
        self._resources.clear()
        self._metadata.clear()
        self._resource_types.clear()
        self._aliases.clear()
        self._canonical_to_aliases.clear()
        logger.debug("Cleared all resources and aliases from registry")
        
resource_registry: ResourceRegistry = ResourceRegistry()