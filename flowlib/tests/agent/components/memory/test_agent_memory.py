"""Tests for agent memory agent_memory system."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import asyncio

from flowlib.agent.components.memory.agent_memory import (
    AgentMemory,
    AgentMemoryConfig
)
from flowlib.agent.components.memory.models import (
    MemoryStoreRequest,
    MemoryRetrieveRequest,
    MemorySearchRequest,
)
from flowlib.agent.models.memory import MemoryItem, MemorySearchResult
from flowlib.agent.core.errors import MemoryError


class MockMemoryComponent:
    """Mock memory component for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.stored_items = {}
        self.contexts = set()
        
    async def initialize(self):
        """Initialize the component."""
        self.initialized = True
        
    async def shutdown(self):
        """Shutdown the component.""" 
        self.initialized = False
        
    async def create_context(self, context_name: str, metadata=None):
        """Create a context."""
        self.contexts.add(context_name)
        return context_name
        
    async def store(self, request: MemoryStoreRequest):
        """Store an item."""
        key = f"{request.context}:{request.key}" if request.context else request.key
        self.stored_items[key] = request.value
        return request.key
        
    async def retrieve(self, request: MemoryRetrieveRequest):
        """Retrieve an item."""
        key = f"{request.context}:{request.key}" if request.context else request.key
        value = self.stored_items.get(key)
        if value is not None:
            return MemoryItem(
                key=request.key,
                value=value,
                context=request.context,
                metadata={}
            )
        return None
        
    async def search(self, request: MemorySearchRequest):
        """Search for items."""
        results = []
        for key, value in self.stored_items.items():
            if request.query.lower() in str(value).lower():
                item = MemoryItem(
                    key=key.split(':', 1)[-1],
                    value=value,
                    context=request.context,
                    metadata={}
                )
                result = MemorySearchResult(
                    item=item,
                    score=0.8,
                    metadata={}
                )
                results.append(result)
                if len(results) >= (request.limit or 10):
                    break
        return results
        
    async def wipe_context(self, context: str):
        """Wipe a context."""
        keys_to_remove = [k for k in self.stored_items.keys() if k.startswith(f"{context}:")]
        for key in keys_to_remove:
            del self.stored_items[key]
        self.contexts.discard(context)
        
    def get_stats(self):
        """Get statistics."""
        return {
            "name": self.name,
            "initialized": self.initialized,
            "item_count": len(self.stored_items),
            "context_count": len(self.contexts)
        }


class TestAgentMemoryConfig:
    """Test AgentMemoryConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AgentMemoryConfig()
        
        assert config.fusion_llm_config == "default-llm"
        assert config.store_execution_history is True
        assert config.working_memory is not None
        assert config.vector_memory is not None
        assert config.knowledge_memory is not None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AgentMemoryConfig(
            fusion_llm_config="custom-llm",
            store_execution_history=False
        )
        
        assert config.fusion_llm_config == "custom-llm"
        assert config.store_execution_history is False


class TestAgentMemoryInitialization:
    """Test AgentMemory initialization and lifecycle."""
    
    def test_initialization_defaults(self):
        """Test default initialization."""
        memory = AgentMemory()
        
        assert not memory.initialized
        assert memory._config is not None
        assert memory._vector_memory is None
        assert memory._knowledge_memory is None
        assert memory._working_memory is None
        assert memory._fusion_llm is None
        assert len(memory._contexts) == 0
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = AgentMemoryConfig(fusion_llm_config="test-llm")
        memory = AgentMemory(config=config)
        
        assert memory._config.fusion_llm_config == "test-llm"
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        memory = AgentMemory()
        
        # Mock the memory components
        with patch('flowlib.agent.components.memory.agent_memory.WorkingMemory') as mock_working, \
             patch('flowlib.agent.components.memory.agent_memory.VectorMemory') as mock_vector, \
             patch('flowlib.agent.components.memory.agent_memory.KnowledgeMemory') as mock_knowledge, \
             patch('flowlib.agent.components.memory.agent_memory.provider_registry') as mock_registry:
            
            # Setup mock instances
            mock_working_instance = MockMemoryComponent("working")
            mock_vector_instance = MockMemoryComponent("vector") 
            mock_knowledge_instance = MockMemoryComponent("knowledge")
            mock_llm = Mock()
            
            mock_working.return_value = mock_working_instance
            mock_vector.return_value = mock_vector_instance
            mock_knowledge.return_value = mock_knowledge_instance
            mock_registry.get_by_config = AsyncMock(return_value=mock_llm)
            
            # Initialize
            await memory.initialize()
            
            # Verify initialization
            assert memory.initialized
            assert memory._working_memory is mock_working_instance
            assert memory._vector_memory is mock_vector_instance
            assert memory._knowledge_memory is mock_knowledge_instance
            assert memory._fusion_llm is mock_llm
            
            # Verify components were initialized
            assert mock_working_instance.initialized
            assert mock_vector_instance.initialized
            assert mock_knowledge_instance.initialized
    
    @pytest.mark.asyncio
    async def test_initialize_llm_provider_not_found(self):
        """Test initialization failure when LLM provider not found."""
        memory = AgentMemory()
        
        with patch('flowlib.agent.components.memory.agent_memory.WorkingMemory') as mock_working, \
             patch('flowlib.agent.components.memory.agent_memory.VectorMemory') as mock_vector, \
             patch('flowlib.agent.components.memory.agent_memory.KnowledgeMemory') as mock_knowledge, \
             patch('flowlib.agent.components.memory.agent_memory.provider_registry') as mock_registry:
            
            # Setup mock instances with async initialize methods
            mock_working_instance = Mock()
            mock_working_instance.initialize = AsyncMock()
            mock_working.return_value = mock_working_instance
            
            mock_vector_instance = Mock()
            mock_vector_instance.initialize = AsyncMock()
            mock_vector.return_value = mock_vector_instance
            
            mock_knowledge_instance = Mock()
            mock_knowledge_instance.initialize = AsyncMock()
            mock_knowledge.return_value = mock_knowledge_instance
            
            mock_registry.get_by_config = AsyncMock(return_value=None)
            
            with pytest.raises(MemoryError, match="Fusion LLM provider not found"):
                await memory.initialize()
            
            assert not memory.initialized
    
    @pytest.mark.asyncio
    async def test_shutdown_success(self):
        """Test successful shutdown."""
        memory = AgentMemory()
        
        # Mock initialized components
        memory._working_memory = MockMemoryComponent("working")
        memory._vector_memory = MockMemoryComponent("vector")
        memory._knowledge_memory = MockMemoryComponent("knowledge")
        memory._fusion_llm = Mock()
        memory._contexts.add("test_context")
        memory._initialized = True
        
        # Initialize components
        await memory._working_memory.initialize()
        await memory._vector_memory.initialize()
        await memory._knowledge_memory.initialize()
        
        # Shutdown
        await memory.shutdown()
        
        # Verify shutdown
        assert not memory.initialized
        assert memory._working_memory is None
        assert memory._vector_memory is None
        assert memory._knowledge_memory is None
        assert memory._fusion_llm is None
        assert len(memory._contexts) == 0


class TestAgentMemoryOperations:
    """Test AgentMemory operations."""
    
    @pytest_asyncio.fixture
    async def initialized_memory(self):
        """Create an initialized agent_memory memory for testing."""
        memory = AgentMemory()
        
        # Mock components
        memory._working_memory = MockMemoryComponent("working")
        memory._vector_memory = MockMemoryComponent("vector")
        memory._knowledge_memory = MockMemoryComponent("knowledge")
        memory._fusion_llm = Mock()
        memory._initialized = True
        
        # Initialize components
        await memory._working_memory.initialize()
        await memory._vector_memory.initialize()
        await memory._knowledge_memory.initialize()
        
        return memory
    
    @pytest.mark.asyncio
    async def test_create_context_success(self, initialized_memory):
        """Test successful context creation."""
        context_name = "test_context"
        
        result = await initialized_memory.create_context(context_name)
        
        assert result == context_name
        assert context_name in initialized_memory._contexts
        assert context_name in initialized_memory._working_memory.contexts
        assert context_name in initialized_memory._vector_memory.contexts
        assert context_name in initialized_memory._knowledge_memory.contexts
    
    @pytest.mark.asyncio
    async def test_create_context_not_initialized(self):
        """Test context creation when not initialized."""
        memory = AgentMemory()
        
        with pytest.raises(MemoryError, match="AgentMemory not initialized"):
            await memory.create_context("test")
    
    @pytest.mark.asyncio
    async def test_store_basic(self, initialized_memory):
        """Test basic store operation."""
        request = MemoryStoreRequest(
            key="test_key",
            value="test_value",
            context="test_context"
        )
        
        result = await initialized_memory.store(request)
        
        assert result == "test_key"
        
        # Verify stored in working and vector memory
        assert "test_context:test_key" in initialized_memory._working_memory.stored_items
        assert "test_context:test_key" in initialized_memory._vector_memory.stored_items
        
        # Knowledge memory should not have it (not an entity)
        assert "test_context:test_key" not in initialized_memory._knowledge_memory.stored_items
    
    @pytest.mark.asyncio
    async def test_store_entity(self, initialized_memory):
        """Test storing entity data."""
        # Create a mock entity-like object
        entity_value = Mock()
        entity_value.model_dump.return_value = {"type": "person", "name": "John"}
        
        request = MemoryStoreRequest(
            key="entity_key",
            value=entity_value,
            context="test_context"
        )
        
        result = await initialized_memory.store(request)
        
        assert result == "entity_key"
        
        # Verify stored in all components including knowledge memory
        assert "test_context:entity_key" in initialized_memory._working_memory.stored_items
        assert "test_context:entity_key" in initialized_memory._vector_memory.stored_items
        assert "test_context:entity_key" in initialized_memory._knowledge_memory.stored_items
    
    @pytest.mark.asyncio
    async def test_store_not_initialized(self):
        """Test store when not initialized."""
        memory = AgentMemory()
        request = MemoryStoreRequest(key="test", value="test")
        
        with pytest.raises(MemoryError, match="AgentMemory not initialized"):
            await memory.store(request)
    
    @pytest.mark.asyncio
    async def test_retrieve_from_working_memory(self, initialized_memory):
        """Test retrieving item from working memory."""
        # Store item in working memory
        request = MemoryStoreRequest(
            key="work_key",
            value="work_value",
            context="test_context"
        )
        await initialized_memory.store(request)
        
        # Retrieve item
        retrieve_request = MemoryRetrieveRequest(
            key="work_key",
            context="test_context"
        )
        
        result = await initialized_memory.retrieve(retrieve_request)
        
        assert result is not None
        assert result.key == "work_key"
        assert result.value == "work_value"
    
    @pytest.mark.asyncio
    async def test_retrieve_not_found(self, initialized_memory):
        """Test retrieving non-existent item."""
        retrieve_request = MemoryRetrieveRequest(
            key="nonexistent",
            context="test_context"
        )
        
        result = await initialized_memory.retrieve(retrieve_request)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_search_success(self, initialized_memory):
        """Test successful search across components."""
        # Store items in different components
        initialized_memory._working_memory.stored_items["ctx:key1"] = "hello world"
        initialized_memory._vector_memory.stored_items["ctx:key2"] = "hello universe"
        initialized_memory._knowledge_memory.stored_items["ctx:key3"] = "goodbye world"
        
        search_request = MemorySearchRequest(
            query="hello",
            context="ctx",
            limit=10
        )
        
        results = await initialized_memory.search(search_request)
        
        assert len(results) == 2  # Two items contain "hello"
        values = [r.item.value for r in results]
        assert "hello world" in values
        assert "hello universe" in values
    
    @pytest.mark.asyncio
    async def test_search_no_results(self, initialized_memory):
        """Test search with query that returns no results."""
        search_request = MemorySearchRequest(
            query="nonexistent",
            context="test"
        )
        
        results = await initialized_memory.search(search_request)
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_wipe_context_success(self, initialized_memory):
        """Test successful context wiping."""
        context = "test_context"
        
        # Add context and items
        await initialized_memory.create_context(context)
        initialized_memory._working_memory.stored_items[f"{context}:key1"] = "value1"
        initialized_memory._vector_memory.stored_items[f"{context}:key2"] = "value2"
        initialized_memory._knowledge_memory.stored_items[f"{context}:key3"] = "value3"
        
        # Wipe context
        await initialized_memory.wipe_context(context)
        
        # Verify context is removed
        assert context not in initialized_memory._contexts
        assert context not in initialized_memory._working_memory.contexts
        assert context not in initialized_memory._vector_memory.contexts
        assert context not in initialized_memory._knowledge_memory.contexts
        
        # Verify items are removed
        assert f"{context}:key1" not in initialized_memory._working_memory.stored_items
        assert f"{context}:key2" not in initialized_memory._vector_memory.stored_items
        assert f"{context}:key3" not in initialized_memory._knowledge_memory.stored_items
    
    def test_get_stats_not_initialized(self):
        """Test get_stats when not initialized."""
        memory = AgentMemory()
        
        stats = memory.get_stats()
        
        assert stats["initialized"] is False
        assert stats["context_count"] == 0
        assert stats["contexts"] == []
        assert "config" in stats
        assert "working_memory" not in stats
        assert "vector_memory" not in stats
        assert "knowledge_memory" not in stats
    
    def test_get_stats_initialized(self, initialized_memory):
        """Test get_stats when initialized."""
        # Add some contexts
        initialized_memory._contexts.add("context1")
        initialized_memory._contexts.add("context2")
        
        stats = initialized_memory.get_stats()
        
        assert stats["initialized"] is True
        assert stats["context_count"] == 2
        assert "context1" in stats["contexts"]
        assert "context2" in stats["contexts"]
        assert "config" in stats
        assert "working_memory" in stats
        assert "vector_memory" in stats
        assert "knowledge_memory" in stats