"""Base registry interface for the flowlib registry system.

This module defines the abstract base class for all registry types in the 
flowlib system, providing a common interface for registration and retrieval.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic

T = TypeVar('T')

class BaseRegistry(ABC, Generic[T]):
    """Abstract base class for all registry types.
    
    This class defines the complete interface that all registries must implement,
    establishing a consistent pattern for registration, retrieval, and management operations.
    """
    
    @abstractmethod
    def register(self, name: str, obj: T, **metadata) -> None:
        """Register an object with the registry.
        
        Args:
            name: Unique name for the object
            obj: The object to register
            **metadata: Additional metadata about the object
        """
        pass
        
    @abstractmethod
    def get(self, name: str, expected_type: Optional[Type] = None) -> T:
        """Get an object by name with optional type checking.
        
        Args:
            name: Name of the object to retrieve
            expected_type: Optional type for type checking
            
        Returns:
            The registered object
            
        Raises:
            KeyError: If the object doesn't exist
            TypeError: If the object doesn't match the expected type
        """
        pass
        
    @abstractmethod
    def contains(self, name: str) -> bool:
        """Check if an object exists in the registry.
        
        Args:
            name: Name to check
            
        Returns:
            True if the object exists, False otherwise
        """
        pass
        
    @abstractmethod
    def list(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """List registered objects matching criteria.
        
        Args:
            filter_criteria: Optional criteria to filter results
            
        Returns:
            List of object names matching the criteria
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all registrations from the registry.
        
        This method removes all registered objects and associated metadata.
        """
        pass
    
    @abstractmethod
    def remove(self, name: str) -> bool:
        """Remove a specific registration from the registry.
        
        Args:
            name: Name of the object to remove
            
        Returns:
            True if the object was found and removed, False if not found
        """
        pass
    
    @abstractmethod
    def update(self, name: str, obj: T, **metadata) -> bool:
        """Update or replace an existing registration.
        
        Args:
            name: Name of the object to update
            obj: New object to register
            **metadata: Additional metadata about the object
            
        Returns:
            True if an existing object was updated, False if this was a new registration
        """
        pass
    
    # Alias support methods (optional implementation)
    def register_with_aliases(self, canonical_name: str, obj: T, aliases: Optional[List[str]] = None, **metadata) -> None:
        """Register an object with canonical name and optional aliases.
        
        Args:
            canonical_name: Primary/canonical name for the object
            obj: The object to register
            aliases: Optional list of alias names that point to the canonical name
            **metadata: Additional metadata about the object
            
        Default implementation registers canonical name and each alias separately.
        Subclasses can override for more efficient alias handling.
        """
        # Register the canonical name
        self.register(canonical_name, obj, **metadata)
        
        # Register each alias
        if aliases:
            for alias in aliases:
                self.register(alias, obj, **metadata)
    
    def create_alias(self, alias_name: str, canonical_name: str) -> bool:
        """Create an alias that points to an existing canonical name.
        
        Args:
            alias_name: New alias name to create
            canonical_name: Existing canonical name to point to
            
        Returns:
            True if alias was created, False if canonical name doesn't exist
            
        Default implementation gets the canonical object and registers alias.
        Subclasses can override for more efficient alias handling.
        """
        if not self.contains(canonical_name):
            return False
            
        obj = self.get(canonical_name)
        self.register(alias_name, obj)
        return True
    
    def remove_alias(self, alias_name: str) -> bool:
        """Remove an alias (but not the canonical object).
        
        Args:
            alias_name: Alias name to remove
            
        Returns:
            True if alias was removed, False if not found
            
        Default implementation uses standard remove.
        Subclasses can override to preserve canonical objects.
        """
        return self.remove(alias_name)
    
    def list_aliases(self, canonical_name: str) -> List[str]:
        """List all aliases that point to a canonical name.
        
        Args:
            canonical_name: Canonical name to find aliases for
            
        Returns:
            List of alias names pointing to the canonical name
            
        Default implementation returns empty list.
        Subclasses should override for proper alias tracking.
        """
        return []