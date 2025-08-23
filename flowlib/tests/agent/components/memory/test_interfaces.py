"""Tests for agent memory interfaces."""

import pytest
from typing import Any, Dict, List, Optional, Protocol, Union
from unittest.mock import AsyncMock, Mock

from flowlib.agent.components.memory.interfaces import MemoryInterface
from flowlib.agent.components.memory.models import (
    MemoryItem,
    MemoryStoreRequest,
    MemoryRetrieveRequest, 
    MemorySearchRequest,
    MemorySearchResult,
    MemoryContext
)


class MockMemoryInterface:
    """Mock implementation of MemoryInterface for testing."""
    
    def __init__(self):
        self.stored_items = {}
        self.contexts = {}
        self.store_calls = []
        self.retrieve_calls = []
        self.search_calls = []
    
    async def store_with_model(self, request: MemoryStoreRequest) -> None:
        """Store a value in memory using a structured request."""
        self.store_calls.append(request)
        key = f"{request.context}.{request.key}" if request.context else request.key
        self.stored_items[key] = request.value
    
    async def retrieve_with_model(self, request: MemoryRetrieveRequest) -> Any:
        """Retrieve a value from memory using a structured request."""
        self.retrieve_calls.append(request)
        key = f"{request.context}.{request.key}" if request.context else request.key
        return self.stored_items.get(key, request.default)
    
    async def search_with_model(self, request: MemorySearchRequest) -> MemorySearchResult:
        """Search memory using a structured request."""
        self.search_calls.append(request)
        
        # Simple search implementation for testing
        matching_items = []
        for key, value in self.stored_items.items():
            if request.query.lower() in str(value).lower():
                item = MemoryItem(
                    key=key.split('.')[-1],
                    value=value,
                    context=key.split('.')[0] if '.' in key else "default",
                    metadata={}
                )
                matching_items.append(item)
        
        return MemorySearchResult(
            items=matching_items[:request.limit] if request.limit else matching_items,
            count=len(matching_items),
            query=request.query,
            context=request.context
        )
    
    def create_context(
        self,
        context_name: str,
        parent: Optional[Union[str, MemoryContext]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new memory context."""
        parent_path = ""
        if parent:
            if isinstance(parent, str):
                parent_path = parent
            else:
                parent_path = parent.path
        
        context_path = f"{parent_path}.{context_name}" if parent_path else context_name
        
        context = MemoryContext(
            name=context_name,
            path=context_path,
            parent=parent_path or None,
            metadata=metadata or {}
        )
        
        self.contexts[context_path] = context
        return context_path
    
    def get_context_model(self, context_path: str) -> Optional[MemoryContext]:
        """Get a memory context model by path."""
        return self.contexts.get(context_path)
    
    async def wipe(self, context: Optional[str] = None) -> None:
        """Wipe memory contents."""
        if context is None:
            self.stored_items.clear()
            self.contexts.clear()
        else:
            # Remove items in specific context
            keys_to_remove = [k for k in self.stored_items.keys() if k.startswith(f"{context}.")]
            for key in keys_to_remove:
                del self.stored_items[key]
            
            # Remove context and sub-contexts
            contexts_to_remove = [k for k in self.contexts.keys() if k.startswith(context)]
            for ctx in contexts_to_remove:
                del self.contexts[ctx]


class IncompleteMemoryInterface:
    """Incomplete implementation missing some methods."""
    
    async def store_with_model(self, request: MemoryStoreRequest) -> None:
        pass
    
    # Missing other required methods


class TestMemoryInterfaceProtocol:
    """Test MemoryInterface protocol definition."""
    
    def test_protocol_exists(self):
        """Test that MemoryInterface protocol is properly defined."""
        assert MemoryInterface is not None
        assert isinstance(MemoryInterface, type(Protocol))
    
    def test_protocol_has_required_methods(self):
        """Test that protocol defines all required methods."""
        protocol_attrs = dir(MemoryInterface)
        
        assert 'store_with_model' in protocol_attrs
        assert 'retrieve_with_model' in protocol_attrs
        assert 'search_with_model' in protocol_attrs
        assert 'create_context' in protocol_attrs
        assert 'get_context_model' in protocol_attrs
        assert 'wipe' in protocol_attrs
    
    def test_complete_implementation_satisfies_protocol(self):
        """Test that complete implementation satisfies the protocol."""
        mock_memory = MockMemoryInterface()
        
        # Should satisfy the protocol
        assert isinstance(mock_memory, MemoryInterface)
    
    def test_incomplete_implementation_fails_protocol(self):
        """Test that incomplete implementation fails the protocol."""
        incomplete_memory = IncompleteMemoryInterface()
        
        # Should not satisfy the protocol
        assert not isinstance(incomplete_memory, MemoryInterface)


class TestMemoryInterfaceImplementation:
    """Test the mock implementation of MemoryInterface."""
    
    @pytest.mark.asyncio
    async def test_store_with_model(self):
        """Test storing items with model-based request."""
        memory = MockMemoryInterface()
        
        request = MemoryStoreRequest(
            key="test_key",
            value="test_value",
            context="test_context"
        )
        
        await memory.store_with_model(request)
        
        assert len(memory.store_calls) == 1
        assert memory.store_calls[0] == request
        assert memory.stored_items["test_context.test_key"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_store_without_context(self):
        """Test storing items without context."""
        memory = MockMemoryInterface()
        
        request = MemoryStoreRequest(
            key="global_key",
            value="global_value"
        )
        
        await memory.store_with_model(request)
        
        assert memory.stored_items["global_key"] == "global_value"
    
    @pytest.mark.asyncio
    async def test_retrieve_with_model(self):
        """Test retrieving items with model-based request."""
        memory = MockMemoryInterface()
        
        # Store item first
        store_request = MemoryStoreRequest(
            key="retrieve_key",
            value="retrieve_value",
            context="retrieve_context"
        )
        await memory.store_with_model(store_request)
        
        # Retrieve item
        retrieve_request = MemoryRetrieveRequest(
            key="retrieve_key",
            context="retrieve_context"
        )
        
        result = await memory.retrieve_with_model(retrieve_request)
        
        assert result == "retrieve_value"
        assert len(memory.retrieve_calls) == 1
        assert memory.retrieve_calls[0] == retrieve_request
    
    @pytest.mark.asyncio
    async def test_retrieve_with_default(self):
        """Test retrieving non-existent items returns default."""
        memory = MockMemoryInterface()
        
        retrieve_request = MemoryRetrieveRequest(
            key="nonexistent_key",
            context="nonexistent_context",
            default="default_value"
        )
        
        result = await memory.retrieve_with_model(retrieve_request)
        
        assert result == "default_value"
    
    @pytest.mark.asyncio
    async def test_search_with_model(self):
        """Test searching items with model-based request."""
        memory = MockMemoryInterface()
        
        # Store multiple items
        items = [
            ("key1", "hello world", "context1"),
            ("key2", "hello universe", "context1"),
            ("key3", "goodbye world", "context2")
        ]
        
        for key, value, context in items:
            request = MemoryStoreRequest(key=key, value=value, context=context)
            await memory.store_with_model(request)
        
        # Search for items containing "hello"
        search_request = MemorySearchRequest(
            query="hello",
            limit=10
        )
        
        result = await memory.search_with_model(search_request)
        
        assert len(result.items) == 2
        assert result.count == 2
        assert len(memory.search_calls) == 1
        
        # Check that correct items were found
        values = [item.value for item in result.items]
        assert "hello world" in values
        assert "hello universe" in values
    
    @pytest.mark.asyncio
    async def test_search_with_max_results(self):
        """Test search respects max_results parameter."""
        memory = MockMemoryInterface()
        
        # Store multiple matching items
        for i in range(5):
            request = MemoryStoreRequest(
                key=f"key{i}",
                value=f"test value {i}",
                context="search_context"
            )
            await memory.store_with_model(request)
        
        # Search with limited results
        search_request = MemorySearchRequest(
            query="test",
            limit=3
        )
        
        result = await memory.search_with_model(search_request)
        
        assert len(result.items) == 3
        assert result.count == 5  # Total matches before limiting
    
    def test_create_context_root(self):
        """Test creating root context."""
        memory = MockMemoryInterface()
        
        context_path = memory.create_context("root_context")
        
        assert context_path == "root_context"
        assert "root_context" in memory.contexts
        
        context = memory.contexts["root_context"]
        assert context.name == "root_context"
        assert context.path == "root_context"
        assert context.parent is None
    
    def test_create_context_with_parent_string(self):
        """Test creating context with parent as string."""
        memory = MockMemoryInterface()
        
        # Create parent first
        parent_path = memory.create_context("parent")
        
        # Create child context
        child_path = memory.create_context("child", parent=parent_path)
        
        assert child_path == "parent.child"
        assert "parent.child" in memory.contexts
        
        child_context = memory.contexts["parent.child"]
        assert child_context.name == "child"
        assert child_context.path == "parent.child"
        assert child_context.parent == "parent"
    
    def test_create_context_with_parent_model(self):
        """Test creating context with parent as MemoryContext model."""
        memory = MockMemoryInterface()
        
        # Create parent context
        parent_path = memory.create_context("parent")
        parent_context = memory.get_context_model(parent_path)
        
        # Create child using parent model
        child_path = memory.create_context("child", parent=parent_context)
        
        assert child_path == "parent.child"
        
        child_context = memory.contexts["parent.child"]
        assert child_context.parent == "parent"
    
    def test_create_context_with_metadata(self):
        """Test creating context with metadata."""
        memory = MockMemoryInterface()
        
        metadata = {"description": "Test context", "priority": "high"}
        context_path = memory.create_context("meta_context", metadata=metadata)
        
        context = memory.get_context_model(context_path)
        assert context.metadata == metadata
    
    def test_get_context_model_existing(self):
        """Test getting existing context model."""
        memory = MockMemoryInterface()
        
        context_path = memory.create_context("existing_context")
        context = memory.get_context_model(context_path)
        
        assert context is not None
        assert context.name == "existing_context"
        assert context.path == "existing_context"
    
    def test_get_context_model_nonexistent(self):
        """Test getting non-existent context model returns None."""
        memory = MockMemoryInterface()
        
        context = memory.get_context_model("nonexistent_context")
        
        assert context is None
    
    @pytest.mark.asyncio
    async def test_wipe_all(self):
        """Test wiping all memory contents."""
        memory = MockMemoryInterface()
        
        # Store some items and contexts
        await memory.store_with_model(MemoryStoreRequest(key="key1", value="value1"))
        await memory.store_with_model(MemoryStoreRequest(key="key2", value="value2", context="ctx1"))
        memory.create_context("context1")
        memory.create_context("context2")
        
        # Wipe all
        await memory.wipe()
        
        assert len(memory.stored_items) == 0
        assert len(memory.contexts) == 0
    
    @pytest.mark.asyncio
    async def test_wipe_specific_context(self):
        """Test wiping specific context."""
        memory = MockMemoryInterface()
        
        # Store items in different contexts
        await memory.store_with_model(MemoryStoreRequest(key="key1", value="value1", context="ctx1"))
        await memory.store_with_model(MemoryStoreRequest(key="key2", value="value2", context="ctx2"))
        await memory.store_with_model(MemoryStoreRequest(key="key3", value="value3"))  # No context
        
        memory.create_context("ctx1")
        memory.create_context("ctx2")
        memory.create_context("ctx1.sub")  # Sub-context
        
        # Wipe specific context
        await memory.wipe("ctx1")
        
        # ctx1 items and contexts should be removed
        assert "ctx1.key1" not in memory.stored_items
        assert "ctx1" not in memory.contexts
        assert "ctx1.sub" not in memory.contexts
        
        # Other items should remain
        assert "ctx2.key2" in memory.stored_items
        assert "key3" in memory.stored_items
        assert "ctx2" in memory.contexts


class TestInterfaceTyping:
    """Test interface type annotations and usage."""
    
    def test_interface_as_type_annotation(self):
        """Test using MemoryInterface as type annotation."""
        
        def process_memory(memory: MemoryInterface) -> str:
            """Function that accepts any MemoryInterface."""
            return f"Processing memory interface"
        
        mock_memory = MockMemoryInterface()
        result = process_memory(mock_memory)
        assert result == "Processing memory interface"
    
    @pytest.mark.asyncio
    async def test_interface_async_operations(self):
        """Test async operations through interface."""
        memory: MemoryInterface = MockMemoryInterface()
        
        # Store operation
        store_req = MemoryStoreRequest(key="async_key", value="async_value")
        await memory.store_with_model(store_req)
        
        # Retrieve operation
        retrieve_req = MemoryRetrieveRequest(key="async_key")
        result = await memory.retrieve_with_model(retrieve_req)
        assert result == "async_value"
        
        # Search operation
        search_req = MemorySearchRequest(query="async")
        search_result = await memory.search_with_model(search_req)
        assert len(search_result.items) == 1