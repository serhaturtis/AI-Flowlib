"""Tests for core registry base class."""

import pytest
from typing import Any, Dict, List, Optional, Type
from unittest.mock import Mock

from flowlib.core.registry.registry import BaseRegistry


class MockRegistryItem:
    """Mock item class for registry testing."""
    
    def __init__(self, name: str, value: str, category: str = "default"):
        self.name = name
        self.value = value
        self.category = category
    
    def __eq__(self, other):
        return (isinstance(other, MockRegistryItem) and 
                self.name == other.name and 
                self.value == other.value and 
                self.category == other.category)
    
    def __repr__(self):
        return f"MockRegistryItem(name='{self.name}', value='{self.value}', category='{self.category}')"


class ConcreteRegistry(BaseRegistry[MockRegistryItem]):
    """Concrete implementation of BaseRegistry for testing."""
    
    def __init__(self):
        self._items: Dict[str, MockRegistryItem] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, obj: MockRegistryItem, **metadata) -> None:
        """Register an object with the registry."""
        self._items[name] = obj
        self._metadata[name] = metadata
    
    def get(self, name: str, expected_type: Optional[Type] = None) -> MockRegistryItem:
        """Get an object by name with optional type checking."""
        if name not in self._items:
            raise KeyError(f"Object '{name}' not found in registry")
        
        obj = self._items[name]
        
        if expected_type is not None and not isinstance(obj, expected_type):
            raise TypeError(f"Object '{name}' is of type {type(obj).__name__}, expected {expected_type.__name__}")
        
        return obj
    
    def contains(self, name: str) -> bool:
        """Check if an object exists in the registry."""
        return name in self._items
    
    def list(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[str]:
        """List registered objects matching criteria."""
        if filter_criteria is None:
            return list(self._items.keys())
        
        filtered_names = []
        for name, obj in self._items.items():
            metadata = self._metadata[name]
            
            # Check if object matches filter criteria
            matches = True
            for key, value in filter_criteria.items():
                # Check in object attributes first
                if hasattr(obj, key):
                    if getattr(obj, key) != value:
                        matches = False
                        break
                # Then check in metadata
                elif key in metadata:
                    if metadata[key] != value:
                        matches = False
                        break
                else:
                    # Criteria key not found in object or metadata
                    matches = False
                    break
            
            if matches:
                filtered_names.append(name)
        
        return filtered_names
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for an object (helper method for testing)."""
        if name not in self._metadata:
            raise KeyError(f"No metadata for '{name}'")
        return self._metadata[name].copy()
    
    def clear(self) -> None:
        """Clear all registrations from the registry."""
        self._items.clear()
        self._metadata.clear()
    
    def remove(self, name: str) -> bool:
        """Remove a specific registration from the registry."""
        if name in self._items:
            del self._items[name]
            self._metadata.pop(name, None)
            return True
        return False
    
    def update(self, name: str, obj: MockRegistryItem, **metadata) -> bool:
        """Update or replace an existing registration."""
        existing_found = self.contains(name)
        
        if existing_found:
            # Remove existing
            self.remove(name)
            
            # Re-register
            self.register(name, obj, **metadata)
            return True
        else:
            # New registration
            self.register(name, obj, **metadata)
            return False


class TestBaseRegistry:
    """Test BaseRegistry abstract base class."""
    
    def test_registry_initialization(self):
        """Test that concrete registry can be instantiated."""
        registry = ConcreteRegistry()
        assert registry is not None
        assert isinstance(registry, BaseRegistry)
    
    def test_register_and_get_object(self):
        """Test basic object registration and retrieval."""
        registry = ConcreteRegistry()
        
        item = MockRegistryItem("test_item", "test_value")
        registry.register("test_key", item)
        
        retrieved_item = registry.get("test_key")
        assert retrieved_item == item
        assert retrieved_item.name == "test_item"
        assert retrieved_item.value == "test_value"
    
    def test_register_with_metadata(self):
        """Test object registration with metadata."""
        registry = ConcreteRegistry()
        
        item = MockRegistryItem("test_item", "test_value")
        metadata = {"version": "1.0", "author": "test", "priority": 5}
        
        registry.register("test_key", item, **metadata)
        
        retrieved_item = registry.get("test_key")
        assert retrieved_item == item
        
        # Check metadata was stored
        stored_metadata = registry.get_metadata("test_key")
        assert stored_metadata == metadata
    
    def test_get_nonexistent_object(self):
        """Test retrieving non-existent object raises KeyError."""
        registry = ConcreteRegistry()
        
        with pytest.raises(KeyError, match="Object 'nonexistent' not found in registry"):
            registry.get("nonexistent")
    
    def test_get_with_type_checking_success(self):
        """Test successful type checking during retrieval."""
        registry = ConcreteRegistry()
        
        item = MockRegistryItem("test_item", "test_value")
        registry.register("test_key", item)
        
        # Should succeed with correct type
        retrieved_item = registry.get("test_key", expected_type=MockRegistryItem)
        assert retrieved_item == item
    
    def test_get_with_type_checking_failure(self):
        """Test type checking failure during retrieval."""
        registry = ConcreteRegistry()
        
        item = MockRegistryItem("test_item", "test_value")
        registry.register("test_key", item)
        
        # Should fail with wrong expected type
        with pytest.raises(TypeError, match="Object 'test_key' is of type MockRegistryItem, expected str"):
            registry.get("test_key", expected_type=str)
    
    def test_contains_existing_object(self):
        """Test contains method with existing object."""
        registry = ConcreteRegistry()
        
        item = MockRegistryItem("test_item", "test_value")
        registry.register("test_key", item)
        
        assert registry.contains("test_key") is True
    
    def test_contains_nonexistent_object(self):
        """Test contains method with non-existent object."""
        registry = ConcreteRegistry()
        
        assert registry.contains("nonexistent") is False
    
    def test_list_all_objects(self):
        """Test listing all registered objects."""
        registry = ConcreteRegistry()
        
        # Register multiple items
        items = [
            ("item1", MockRegistryItem("item1", "value1", "cat1")),
            ("item2", MockRegistryItem("item2", "value2", "cat2")),
            ("item3", MockRegistryItem("item3", "value3", "cat1"))
        ]
        
        for name, item in items:
            registry.register(name, item)
        
        all_names = registry.list()
        assert set(all_names) == {"item1", "item2", "item3"}
    
    def test_list_empty_registry(self):
        """Test listing objects in empty registry."""
        registry = ConcreteRegistry()
        
        all_names = registry.list()
        assert all_names == []
    
    def test_list_with_filter_criteria_attribute(self):
        """Test listing objects with filter criteria based on object attributes."""
        registry = ConcreteRegistry()
        
        # Register items with different categories
        items = [
            ("item1", MockRegistryItem("item1", "value1", "cat1")),
            ("item2", MockRegistryItem("item2", "value2", "cat2")),
            ("item3", MockRegistryItem("item3", "value3", "cat1")),
            ("item4", MockRegistryItem("item4", "value4", "cat3"))
        ]
        
        for name, item in items:
            registry.register(name, item)
        
        # Filter by category
        filtered_names = registry.list(filter_criteria={"category": "cat1"})
        assert set(filtered_names) == {"item1", "item3"}
        
        # Filter by value
        filtered_names = registry.list(filter_criteria={"value": "value2"})
        assert filtered_names == ["item2"]
    
    def test_list_with_filter_criteria_metadata(self):
        """Test listing objects with filter criteria based on metadata."""
        registry = ConcreteRegistry()
        
        # Register items with different metadata
        registry.register("item1", MockRegistryItem("item1", "value1"), version="1.0", author="alice")
        registry.register("item2", MockRegistryItem("item2", "value2"), version="2.0", author="bob")
        registry.register("item3", MockRegistryItem("item3", "value3"), version="1.0", author="alice")
        
        # Filter by metadata
        filtered_names = registry.list(filter_criteria={"version": "1.0"})
        assert set(filtered_names) == {"item1", "item3"}
        
        filtered_names = registry.list(filter_criteria={"author": "bob"})
        assert filtered_names == ["item2"]
    
    def test_list_with_filter_criteria_combined(self):
        """Test listing objects with multiple filter criteria."""
        registry = ConcreteRegistry()
        
        # Register items
        registry.register("item1", MockRegistryItem("item1", "value1", "cat1"), version="1.0")
        registry.register("item2", MockRegistryItem("item2", "value2", "cat1"), version="2.0")
        registry.register("item3", MockRegistryItem("item3", "value3", "cat2"), version="1.0")
        
        # Filter by both attribute and metadata
        filtered_names = registry.list(filter_criteria={"category": "cat1", "version": "1.0"})
        assert filtered_names == ["item1"]
    
    def test_list_with_filter_criteria_no_matches(self):
        """Test listing objects with filter criteria that match nothing."""
        registry = ConcreteRegistry()
        
        registry.register("item1", MockRegistryItem("item1", "value1", "cat1"))
        registry.register("item2", MockRegistryItem("item2", "value2", "cat2"))
        
        # Filter that matches nothing
        filtered_names = registry.list(filter_criteria={"category": "nonexistent"})
        assert filtered_names == []
    
    def test_list_with_filter_criteria_missing_attribute(self):
        """Test listing objects with filter criteria for missing attributes."""
        registry = ConcreteRegistry()
        
        registry.register("item1", MockRegistryItem("item1", "value1"))
        
        # Filter by non-existent attribute
        filtered_names = registry.list(filter_criteria={"nonexistent_attr": "value"})
        assert filtered_names == []
    
    def test_register_overwrite_existing(self):
        """Test that registering with same name overwrites existing object."""
        registry = ConcreteRegistry()
        
        # Register initial item
        item1 = MockRegistryItem("item1", "value1")
        registry.register("test_key", item1)
        
        # Register new item with same key
        item2 = MockRegistryItem("item2", "value2")
        registry.register("test_key", item2, new_metadata="test")
        
        # Should get the new item
        retrieved_item = registry.get("test_key")
        assert retrieved_item == item2
        assert retrieved_item.name == "item2"
        
        # Metadata should also be updated
        metadata = registry.get_metadata("test_key")
        assert metadata == {"new_metadata": "test"}
    
    def test_multiple_registrations_and_retrievals(self):
        """Test multiple registrations and retrievals."""
        registry = ConcreteRegistry()
        
        # Register multiple items with various metadata
        test_data = [
            ("key1", MockRegistryItem("name1", "value1", "cat1"), {"meta1": "data1"}),
            ("key2", MockRegistryItem("name2", "value2", "cat2"), {"meta2": "data2"}),
            ("key3", MockRegistryItem("name3", "value3", "cat1"), {"meta3": "data3"})
        ]
        
        # Register all items
        for key, item, metadata in test_data:
            registry.register(key, item, **metadata)
        
        # Verify all can be retrieved correctly
        for key, expected_item, expected_metadata in test_data:
            retrieved_item = registry.get(key)
            assert retrieved_item == expected_item
            
            metadata = registry.get_metadata(key)
            assert metadata == expected_metadata
        
        # Verify all are listed
        all_names = registry.list()
        assert set(all_names) == {"key1", "key2", "key3"}


class AbstractRegistry(BaseRegistry[str]):
    """Abstract registry implementation to test abstract methods."""
    pass


class TestAbstractMethods:
    """Test that abstract methods raise NotImplementedError."""
    
    def test_abstract_register(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            # Abstract class cannot be instantiated
            AbstractRegistry()
    
    def test_cannot_instantiate_abstract_registry(self):
        """Test that BaseRegistry cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRegistry()


class TestGenericTypeSupport:
    """Test that registry supports generic types properly."""
    
    def test_generic_type_parameter(self):
        """Test that registry preserves generic type parameter."""
        registry = ConcreteRegistry()
        
        # The registry should work with MockRegistryItem instances
        item = MockRegistryItem("test", "value")
        registry.register("key", item)
        
        retrieved = registry.get("key")
        assert isinstance(retrieved, MockRegistryItem)
        assert retrieved == item
    
    def test_type_checking_with_inheritance(self):
        """Test type checking with inheritance."""
        class SpecialItem(MockRegistryItem):
            def __init__(self, name: str, value: str, special: str):
                super().__init__(name, value)
                self.special = special
        
        registry = ConcreteRegistry()
        
        special_item = SpecialItem("special", "value", "extra")
        registry.register("special_key", special_item)
        
        # Should work with base type
        retrieved = registry.get("special_key", expected_type=MockRegistryItem)
        assert isinstance(retrieved, SpecialItem)
        assert retrieved.special == "extra"
        
        # Should also work with specific type
        retrieved = registry.get("special_key", expected_type=SpecialItem)
        assert isinstance(retrieved, SpecialItem)