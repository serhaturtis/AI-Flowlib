"""Comprehensive tests for agent memory models module."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch
from pydantic import ValidationError

from flowlib.agent.components.memory.models import (
    MemoryItem,
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySearchRequest,
    MemorySearchResult,
    MemoryContext
)


class TestMemoryItem:
    """Test MemoryItem model."""
    
    def test_memory_item_creation_minimal(self):
        """Test creating MemoryItem with minimal required fields."""
        item = MemoryItem(
            key="test_key",
            value="test_value"
        )
        assert item.key == "test_key"
        assert item.value == "test_value"
        assert item.context == "default"
        assert isinstance(item.created_at, datetime)
        assert item.updated_at is None
        assert item.metadata == {}
    
    def test_memory_item_creation_full(self):
        """Test creating MemoryItem with all fields."""
        created_at = datetime.now() - timedelta(hours=1)
        updated_at = datetime.now()
        metadata = {"type": "test", "category": "example"}
        
        item = MemoryItem(
            key="full_key",
            value={"complex": "data"},
            context="test_context",
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata
        )
        assert item.key == "full_key"
        assert item.value == {"complex": "data"}
        assert item.context == "test_context"
        assert item.created_at == created_at
        assert item.updated_at == updated_at
        assert item.metadata == metadata
    
    def test_memory_item_validation_key_required(self):
        """Test that key is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryItem(value="test_value")
        
        assert "key" in str(exc_info.value)
    
    def test_memory_item_validation_value_required(self):
        """Test that value is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryItem(key="test_key")
        
        assert "value" in str(exc_info.value)
    
    def test_memory_item_validation_key_non_empty(self):
        """Test that key cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryItem(key="", value="test_value")
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_memory_item_update_value(self):
        """Test updating value and timestamp."""
        item = MemoryItem(key="test_key", value="original_value")
        original_created_at = item.created_at
        
        # Wait a moment to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        updated_item = item.update_value("new_value")
        
        assert updated_item.value == "new_value"
        assert updated_item.updated_at is not None
        assert updated_item.updated_at > original_created_at
        assert updated_item.created_at == original_created_at  # Created time unchanged
        
        # Original item should be unchanged
        assert item.value == "original_value"
        assert item.updated_at is None
    
    def test_memory_item_value_types(self):
        """Test different value types."""
        test_values = [
            "string_value",
            42,
            3.14,
            True,
            [1, 2, 3],
            {"nested": "dict"},
            None
        ]
        
        for value in test_values:
            item = MemoryItem(key="test_key", value=value)
            assert item.value == value
    
    def test_memory_item_default_created_at(self):
        """Test that created_at defaults to current time."""
        before = datetime.now()
        item = MemoryItem(key="test_key", value="test_value")
        after = datetime.now()
        
        assert before <= item.created_at <= after
    
    def test_memory_item_context_default(self):
        """Test that context defaults to 'default'."""
        item = MemoryItem(key="test_key", value="test_value")
        assert item.context == "default"
    
    def test_memory_item_metadata_types(self):
        """Test different metadata types."""
        metadata = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "null": None
        }
        
        item = MemoryItem(
            key="test_key",
            value="test_value",
            metadata=metadata
        )
        assert item.metadata == metadata


class TestMemoryStoreRequest:
    """Test MemoryStoreRequest model."""
    
    def test_memory_store_request_creation_minimal(self):
        """Test creating MemoryStoreRequest with minimal required fields."""
        request = MemoryStoreRequest(
            key="test_key",
            value="test_value"
        )
        assert request.key == "test_key"
        assert request.value == "test_value"
        assert request.context is None
        assert request.ttl is None
        assert request.metadata is None
        assert request.importance == 0.5
    
    def test_memory_store_request_creation_full(self):
        """Test creating MemoryStoreRequest with all fields."""
        metadata = {"category": "important", "source": "user"}
        
        request = MemoryStoreRequest(
            key="full_key",
            value={"complex": "data"},
            context="test_context",
            ttl=3600,
            metadata=metadata,
            importance=0.8
        )
        assert request.key == "full_key"
        assert request.value == {"complex": "data"}
        assert request.context == "test_context"
        assert request.ttl == 3600
        assert request.metadata == metadata
        assert request.importance == 0.8
    
    def test_memory_store_request_validation_key_required(self):
        """Test that key is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryStoreRequest(value="test_value")
        
        assert "key" in str(exc_info.value)
    
    def test_memory_store_request_validation_value_required(self):
        """Test that value is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryStoreRequest(key="test_key")
        
        assert "value" in str(exc_info.value)
    
    def test_memory_store_request_validation_key_non_empty(self):
        """Test that key cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryStoreRequest(key="", value="test_value")
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_memory_store_request_importance_validation_clamping(self):
        """Test that importance is clamped to valid range."""
        # Test value above 1.0
        request = MemoryStoreRequest(
            key="test_key",
            value="test_value",
            importance=1.5
        )
        assert request.importance == 1.0
        
        # Test value below 0.0
        request = MemoryStoreRequest(
            key="test_key",
            value="test_value",
            importance=-0.5
        )
        assert request.importance == 0.0
        
        # Test valid values
        valid_values = [0.0, 0.3, 0.5, 0.7, 1.0]
        for value in valid_values:
            request = MemoryStoreRequest(
                key="test_key",
                value="test_value",
                importance=value
            )
            assert request.importance == value
    
    def test_memory_store_request_ttl_positive(self):
        """Test TTL validation."""
        # Valid TTL values
        valid_ttls = [None, 60, 3600, 86400]
        for ttl in valid_ttls:
            request = MemoryStoreRequest(
                key="test_key",
                value="test_value",
                ttl=ttl
            )
            assert request.ttl == ttl
        
        # Test zero TTL (should be valid)
        request = MemoryStoreRequest(
            key="test_key",
            value="test_value",
            ttl=0
        )
        assert request.ttl == 0
    
    def test_memory_store_request_original_importance_tracking(self):
        """Test that original importance is tracked."""
        request = MemoryStoreRequest(
            key="test_key",
            value="test_value",
            importance=1.5  # Will be clamped to 1.0
        )
        
        assert request.importance == 1.0  # Clamped value
        assert hasattr(request, 'original_importance')
        assert request.original_importance == 1.5  # Original value preserved
    
    def test_memory_store_request_value_types(self):
        """Test different value types."""
        test_values = [
            "string_value",
            42,
            3.14,
            True,
            [1, 2, 3],
            {"nested": "dict"},
            None
        ]
        
        for value in test_values:
            request = MemoryStoreRequest(key="test_key", value=value)
            assert request.value == value


class TestMemoryRetrieveRequest:
    """Test MemoryRetrieveRequest model."""
    
    def test_memory_retrieve_request_creation_minimal(self):
        """Test creating MemoryRetrieveRequest with minimal required fields."""
        request = MemoryRetrieveRequest(key="test_key")
        
        assert request.key == "test_key"
        assert request.context is None
        assert request.default is None
        assert request.metadata_only is False
    
    def test_memory_retrieve_request_creation_full(self):
        """Test creating MemoryRetrieveRequest with all fields."""
        request = MemoryRetrieveRequest(
            key="full_key",
            context="test_context",
            default="default_value",
            metadata_only=True
        )
        assert request.key == "full_key"
        assert request.context == "test_context"
        assert request.default == "default_value"
        assert request.metadata_only is True
    
    def test_memory_retrieve_request_validation_key_required(self):
        """Test that key is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryRetrieveRequest()
        
        assert "key" in str(exc_info.value)
    
    def test_memory_retrieve_request_validation_key_non_empty(self):
        """Test that key cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryRetrieveRequest(key="")
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_memory_retrieve_request_default_types(self):
        """Test different default value types."""
        test_defaults = [
            None,
            "default_string",
            42,
            {"default": "object"},
            [1, 2, 3]
        ]
        
        for default in test_defaults:
            request = MemoryRetrieveRequest(
                key="test_key",
                default=default
            )
            assert request.default == default


class TestMemorySearchRequest:
    """Test MemorySearchRequest model."""
    
    def test_memory_search_request_creation_minimal(self):
        """Test creating MemorySearchRequest with minimal required fields."""
        request = MemorySearchRequest(query="test query")
        
        assert request.query == "test query"
        assert request.context is None
        assert request.limit == 10
        assert request.threshold is None
        assert request.sort_by is None
        assert request.search_type == "hybrid"
        assert request.metadata_filter is None
    
    def test_memory_search_request_creation_full(self):
        """Test creating MemorySearchRequest with all fields."""
        metadata_filter = {"category": "important", "type": "document"}
        
        request = MemorySearchRequest(
            query="comprehensive search",
            context="test_context",
            limit=20,
            threshold=0.8,
            sort_by="relevance",
            search_type="semantic",
            metadata_filter=metadata_filter
        )
        assert request.query == "comprehensive search"
        assert request.context == "test_context"
        assert request.limit == 20
        assert request.threshold == 0.8
        assert request.sort_by == "relevance"
        assert request.search_type == "semantic"
        assert request.metadata_filter == metadata_filter
    
    def test_memory_search_request_validation_query_required(self):
        """Test that query is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemorySearchRequest()
        
        assert "query" in str(exc_info.value)
    
    def test_memory_search_request_validation_query_non_empty(self):
        """Test that query cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            MemorySearchRequest(query="")
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_memory_search_request_validation_limit_positive(self):
        """Test that limit must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            MemorySearchRequest(query="test", limit=0)
        
        assert "Input should be greater than 0" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            MemorySearchRequest(query="test", limit=-1)
        
        assert "Input should be greater than 0" in str(exc_info.value)
    
    def test_memory_search_request_threshold_range(self):
        """Test threshold validation."""
        # Valid threshold values
        valid_thresholds = [None, 0.0, 0.5, 1.0]
        for threshold in valid_thresholds:
            request = MemorySearchRequest(
                query="test",
                threshold=threshold
            )
            assert request.threshold == threshold
        
        # Invalid threshold values
        with pytest.raises(ValidationError) as exc_info:
            MemorySearchRequest(query="test", threshold=1.5)
        
        assert "Input should be less than or equal to 1" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            MemorySearchRequest(query="test", threshold=-0.1)
        
        assert "Input should be greater than or equal to 0" in str(exc_info.value)
    
    def test_memory_search_request_search_type_values(self):
        """Test search type validation."""
        valid_types = ["semantic", "keyword", "hybrid"]
        
        for search_type in valid_types:
            request = MemorySearchRequest(
                query="test",
                search_type=search_type
            )
            assert request.search_type == search_type
        
        # Invalid search type
        with pytest.raises(ValidationError) as exc_info:
            MemorySearchRequest(query="test", search_type="invalid")
        
        assert "Input should be 'semantic', 'keyword' or 'hybrid'" in str(exc_info.value)
    
    def test_memory_search_request_sort_by_values(self):
        """Test sort_by validation."""
        valid_sort_values = [None, "relevance", "created_at", "updated_at"]
        
        for sort_by in valid_sort_values:
            request = MemorySearchRequest(
                query="test",
                sort_by=sort_by
            )
            assert request.sort_by == sort_by
        
        # Invalid sort_by
        with pytest.raises(ValidationError) as exc_info:
            MemorySearchRequest(query="test", sort_by="invalid")
        
        assert "Input should be 'relevance', 'created_at' or 'updated_at'" in str(exc_info.value)


class TestMemorySearchResult:
    """Test MemorySearchResult model."""
    
    def test_memory_search_result_creation_minimal(self):
        """Test creating MemorySearchResult with default values."""
        result = MemorySearchResult()
        
        assert result.items == []
        assert result.count == 0
        assert result.query == ""
        assert result.context is None
    
    def test_memory_search_result_creation_full(self):
        """Test creating MemorySearchResult with all fields."""
        items = [
            MemoryItem(key="item1", value="value1"),
            MemoryItem(key="item2", value="value2")
        ]
        
        result = MemorySearchResult(
            items=items,
            count=2,
            query="test search",
            context="test_context"
        )
        assert result.items == items
        assert result.count == 2
        assert result.query == "test search"
        assert result.context == "test_context"
    
    def test_memory_search_result_validation_count_non_negative(self):
        """Test that count must be non-negative."""
        with pytest.raises(ValidationError) as exc_info:
            MemorySearchResult(count=-1)
        
        assert "Input should be greater than or equal to 0" in str(exc_info.value)
    
    def test_memory_search_result_items_validation(self):
        """Test that items must be valid MemoryItem instances."""
        valid_items = [
            MemoryItem(key="item1", value="value1"),
            MemoryItem(key="item2", value="value2")
        ]
        
        result = MemorySearchResult(items=valid_items)
        assert len(result.items) == 2
        assert all(isinstance(item, MemoryItem) for item in result.items)
    
    def test_memory_search_result_empty_items(self):
        """Test with empty items list."""
        result = MemorySearchResult(items=[])
        assert result.items == []
        assert len(result.items) == 0


class TestMemoryContext:
    """Test MemoryContext model."""
    
    def test_memory_context_creation_minimal(self):
        """Test creating MemoryContext with minimal required fields."""
        context = MemoryContext(
            name="test_context",
            path="/test/context"
        )
        assert context.name == "test_context"
        assert context.path == "/test/context"
        assert context.parent is None
        assert context.metadata == {}
        assert isinstance(context.created_at, datetime)
    
    def test_memory_context_creation_full(self):
        """Test creating MemoryContext with all fields."""
        created_at = datetime.now()
        metadata = {"type": "persistent", "priority": "high"}
        
        context = MemoryContext(
            name="full_context",
            path="/full/test/context",
            parent="/full",
            metadata=metadata,
            created_at=created_at
        )
        assert context.name == "full_context"
        assert context.path == "/full/test/context"
        assert context.parent == "/full"
        assert context.metadata == metadata
        assert context.created_at == created_at
    
    def test_memory_context_validation_name_required(self):
        """Test that name is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryContext(path="/test/context")
        
        assert "name" in str(exc_info.value)
    
    def test_memory_context_validation_path_required(self):
        """Test that path is required."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryContext(name="test_context")
        
        assert "path" in str(exc_info.value)
    
    def test_memory_context_validation_name_non_empty(self):
        """Test that name cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryContext(name="", path="/test/context")
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_memory_context_validation_path_non_empty(self):
        """Test that path cannot be empty."""
        with pytest.raises(ValidationError) as exc_info:
            MemoryContext(name="test_context", path="")
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_memory_context_default_created_at(self):
        """Test that created_at defaults to current time."""
        before = datetime.now()
        context = MemoryContext(
            name="test_context",
            path="/test/context"
        )
        after = datetime.now()
        
        assert before <= context.created_at <= after
    
    def test_memory_context_metadata_types(self):
        """Test different metadata types."""
        metadata = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "null": None
        }
        
        context = MemoryContext(
            name="test_context",
            path="/test/context",
            metadata=metadata
        )
        assert context.metadata == metadata


class TestMemoryModelsIntegration:
    """Test integration between memory models."""
    
    def test_memory_store_and_search_integration(self):
        """Test integration between store request and search request."""
        # Create a store request
        store_request = MemoryStoreRequest(
            key="integration_test",
            value="test data for integration",
            context="integration_context",
            metadata={"category": "test"},
            importance=0.8
        )
        
        # Create a search request for the same context
        search_request = MemorySearchRequest(
            query="integration test",
            context="integration_context",
            limit=5,
            metadata_filter={"category": "test"}
        )
        
        assert store_request.context == search_request.context
        assert store_request.metadata["category"] == search_request.metadata_filter["category"]
    
    def test_memory_search_result_with_items(self):
        """Test search result containing memory items."""
        # Create memory items
        items = [
            MemoryItem(
                key="result1",
                value="first result",
                context="search_context",
                metadata={"score": 0.9}
            ),
            MemoryItem(
                key="result2",
                value="second result",
                context="search_context",
                metadata={"score": 0.7}
            )
        ]
        
        # Create search result
        search_result = MemorySearchResult(
            items=items,
            count=2,
            query="test search",
            context="search_context"
        )
        
        assert len(search_result.items) == 2
        assert search_result.count == len(search_result.items)
        assert all(item.context == search_result.context for item in search_result.items)
    
    def test_memory_context_hierarchical_structure(self):
        """Test hierarchical memory context structure."""
        # Create parent context
        parent_context = MemoryContext(
            name="parent",
            path="/parent"
        )
        
        # Create child context
        child_context = MemoryContext(
            name="child",
            path="/parent/child",
            parent="/parent"
        )
        
        assert child_context.parent == parent_context.path
        assert child_context.path.startswith(parent_context.path)
    
    def test_memory_models_serialization(self):
        """Test that all models can be serialized/deserialized."""
        # Create instances of all models
        memory_item = MemoryItem(
            key="test_item",
            value={"data": "test"},
            metadata={"test": True}
        )
        
        store_request = MemoryStoreRequest(
            key="store_key",
            value="store_value",
            importance=0.7
        )
        
        retrieve_request = MemoryRetrieveRequest(
            key="retrieve_key",
            default="default_value"
        )
        
        search_request = MemorySearchRequest(
            query="search query",
            search_type="semantic"
        )
        
        search_result = MemorySearchResult(
            items=[memory_item],
            count=1,
            query="search query"
        )
        
        memory_context = MemoryContext(
            name="test_context",
            path="/test"
        )
        
        # Test serialization to dict
        models = [
            memory_item,
            store_request,
            retrieve_request,
            search_request,
            search_result,
            memory_context
        ]
        
        for model in models:
            model_dict = model.model_dump()
            assert isinstance(model_dict, dict)
            
            # Test deserialization from dict
            model_class = type(model)
            restored_model = model_class(**model_dict)
            
            # Verify key fields match
            if hasattr(model, 'key'):
                assert restored_model.key == model.key
            if hasattr(model, 'name'):
                assert restored_model.name == model.name
            if hasattr(model, 'query'):
                assert restored_model.query == model.query
    
    def test_memory_item_update_workflow(self):
        """Test updating memory item workflow."""
        # Create initial memory item
        item = MemoryItem(
            key="update_test",
            value="original_value",
            context="update_context"
        )
        
        original_created_at = item.created_at
        assert item.updated_at is None
        
        # Wait a moment to ensure timestamp difference
        import time
        time.sleep(0.01)
        
        # Update the value
        updated_item = item.update_value("updated_value")
        
        # Verify update
        assert updated_item.value == "updated_value"
        assert updated_item.updated_at is not None
        assert updated_item.updated_at > original_created_at
        
        # Original item should be unchanged
        assert item.value == "original_value"
        assert item.updated_at is None
        assert item.created_at == original_created_at  # Should remain unchanged
    
    def test_memory_request_chain(self):
        """Test a complete memory operation chain."""
        context_name = "chain_test"
        
        # 1. Create context
        context = MemoryContext(
            name=context_name,
            path=f"/{context_name}"
        )
        
        # 2. Create store request
        store_request = MemoryStoreRequest(
            key="chain_item",
            value="chain test data",
            context=context.path,
            importance=0.9,
            metadata={"chain": True}
        )
        
        # 3. Create retrieve request
        retrieve_request = MemoryRetrieveRequest(
            key="chain_item",
            context=context.path
        )
        
        # 4. Create search request
        search_request = MemorySearchRequest(
            query="chain test",
            context=context.path,
            metadata_filter={"chain": True}
        )
        
        # Verify consistency across requests
        assert store_request.context == context.path
        assert retrieve_request.context == context.path
        assert search_request.context == context.path
        assert store_request.key == retrieve_request.key
        assert store_request.metadata["chain"] == search_request.metadata_filter["chain"]