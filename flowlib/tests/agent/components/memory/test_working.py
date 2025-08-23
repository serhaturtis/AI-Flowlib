"""
Tests for the modernized WorkingMemory component.

These tests verify:
1. Global state bug fixes
2. TTL management and background cleanup
3. Resource limits and monitoring
4. Config-driven initialization
5. Error handling with ErrorContext
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from flowlib.agent.components.memory.working import WorkingMemory, WorkingMemoryConfig
from flowlib.agent.components.memory.models import MemoryItem, MemoryStoreRequest, MemoryRetrieveRequest, MemorySearchRequest
from flowlib.agent.core.errors import MemoryError


class TestWorkingMemoryConfig:
    """Test WorkingMemoryConfig validation and defaults."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WorkingMemoryConfig()
        assert config.default_ttl_seconds == 3600
        assert config.max_items == 10000
        assert config.cleanup_interval_seconds == 300
        assert config.max_memory_mb == 100
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WorkingMemoryConfig(
            default_ttl_seconds=1800,
            max_items=5000,
            cleanup_interval_seconds=600,
            max_memory_mb=200
        )
        assert config.default_ttl_seconds == 1800
        assert config.max_items == 5000
        assert config.cleanup_interval_seconds == 600
        assert config.max_memory_mb == 200
    
    def test_config_validation(self):
        """Test configuration validation constraints."""
        with pytest.raises(ValueError):
            WorkingMemoryConfig(default_ttl_seconds=30)  # Below minimum
        
        with pytest.raises(ValueError):
            WorkingMemoryConfig(max_items=50)  # Below minimum


class TestWorkingMemoryGlobalStateBugFix:
    """Test that the global state bug is fixed."""
    
    @pytest.mark.asyncio
    async def test_instance_isolation(self):
        """Test that different WorkingMemory instances don't share state."""
        # Create two separate instances
        memory1 = WorkingMemory(WorkingMemoryConfig())
        memory2 = WorkingMemory(WorkingMemoryConfig())
        
        await memory1.initialize()
        await memory2.initialize()
        
        try:
            # Store data in first instance
            request1 = MemoryStoreRequest(key="test1", value="data1", context="context1", metadata={"source": "test1"})
            await memory1.store(request1)
            
            # Store different data in second instance
            request2 = MemoryStoreRequest(key="test2", value="data2", context="context2", metadata={"source": "test2"})
            await memory2.store(request2)
            
            # Verify isolation - each instance should only have its own data
            retrieve1 = MemoryRetrieveRequest(key="test1", context="context1")
            retrieve2 = MemoryRetrieveRequest(key="test2", context="context2")
            
            result1_from_1 = await memory1.retrieve(retrieve1)
            result2_from_1 = await memory1.retrieve(retrieve2)
            
            result1_from_2 = await memory2.retrieve(retrieve1)
            result2_from_2 = await memory2.retrieve(retrieve2)
            
            # Instance 1 should have test1 but not test2
            assert result1_from_1 is not None
            assert result1_from_1.value == "data1"
            assert result2_from_1 is None
            
            # Instance 2 should have test2 but not test1
            assert result2_from_2 is not None
            assert result2_from_2.value == "data2"
            assert result1_from_2 is None
            
        finally:
            await memory1.shutdown()
            await memory2.shutdown()
    
    @pytest.mark.asyncio
    async def test_context_isolation(self):
        """Test that contexts within same instance are isolated."""
        memory = WorkingMemory(WorkingMemoryConfig())
        await memory.initialize()
        
        try:
            # Create contexts
            await memory.create_context("context1")
            await memory.create_context("context2")
            
            # Store same key in different contexts
            request1 = MemoryStoreRequest(key="same_key", value="data1", context="context1", metadata={"source": "ctx1"})
            request2 = MemoryStoreRequest(key="same_key", value="data2", context="context2", metadata={"source": "ctx2"})
            
            await memory.store(request1)
            await memory.store(request2)
            
            # Retrieve from each context
            retrieve1 = MemoryRetrieveRequest(key="same_key", context="context1")
            retrieve2 = MemoryRetrieveRequest(key="same_key", context="context2")
            
            result1 = await memory.retrieve(retrieve1)
            result2 = await memory.retrieve(retrieve2)
            
            # Should get different data from each context
            assert result1.value == "data1"
            assert result2.value == "data2"
            assert result1.metadata["source"] == "ctx1"
            assert result2.metadata["source"] == "ctx2"
            
        finally:
            await memory.shutdown()


class TestWorkingMemoryTTL:
    """Test TTL (Time-To-Live) functionality."""
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that items expire after TTL."""
        config = WorkingMemoryConfig(
            default_ttl_seconds=60,  # 60 second TTL (minimum allowed)
            cleanup_interval_seconds=60  # 60 second cleanup (minimum allowed)
        )
        memory = WorkingMemory(config)
        await memory.initialize()
        
        try:
            # Store item with custom short TTL (override config minimum for testing)
            request = MemoryStoreRequest(key="ttl_test", value="test_data", context="test_context", ttl=1)
            await memory.store(request)
            
            # Should be retrievable immediately
            retrieve_request = MemoryRetrieveRequest(key="ttl_test", context="test_context")
            result = await memory.retrieve(retrieve_request)
            assert result is not None
            assert result.value == "test_data"
            
            # Manually trigger cleanup to test TTL expiration
            # Instead of waiting, we'll manually expire the item by manipulating time
            # Set the TTL entry to be in the past
            if hasattr(memory, '_ttl_map') and "test_context:ttl_test" in memory._ttl_map:
                from datetime import datetime, timedelta, timezone
                memory._ttl_map["test_context:ttl_test"] = datetime.now(timezone.utc) - timedelta(seconds=10)
                
            # Trigger cleanup
            await memory._cleanup_expired()
            
            # Should be expired and cleaned up
            result = await memory.retrieve(retrieve_request)
            assert result is None
            
        finally:
            await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_custom_ttl(self):
        """Test storing items with custom TTL."""
        config = WorkingMemoryConfig(default_ttl_seconds=3600)
        memory = WorkingMemory(config)
        await memory.initialize()
        
        try:
            # Store with custom short TTL
            request = MemoryStoreRequest(
                key="custom_ttl", 
                value="custom_ttl_data", 
                context="test_context",
                ttl=1  # Override default
            )
            await memory.store(request)
            
            # Should be there initially
            retrieve_request = MemoryRetrieveRequest(key="custom_ttl", context="test_context")
            result = await memory.retrieve(retrieve_request)
            assert result is not None
            
            # Wait for custom TTL to expire
            await asyncio.sleep(1.5)
            
            # Trigger cleanup manually
            await memory._cleanup_expired()
            
            # Should be expired
            result = await memory.retrieve(retrieve_request)
            assert result is None
            
        finally:
            await memory.shutdown()


class TestWorkingMemoryResourceLimits:
    """Test resource limit enforcement."""
    
    @pytest.mark.asyncio
    async def test_max_items_limit(self):
        """Test that max_items limit is enforced."""
        config = WorkingMemoryConfig(max_items=100)  # Use minimum allowed value
        memory = WorkingMemory(config)
        await memory.initialize()
        
        try:
            # Store up to the limit
            for i in range(100):
                request = MemoryStoreRequest(key=f"item_{i}", value=f"data_{i}", context="test")
                await memory.store(request)
            
            # Check stats
            stats = memory.get_stats()
            assert stats["item_count"] == 100
            
            # Try to store beyond limit
            request = MemoryStoreRequest(key="overflow", value="overflow", context="test")
            
            with pytest.raises(MemoryError) as exc_info:
                await memory.store(request)
            
            assert "Memory limit reached" in str(exc_info.value)
            
        finally:
            await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_memory_usage_tracking(self):
        """Test that memory usage is tracked."""
        memory = WorkingMemory(WorkingMemoryConfig())
        await memory.initialize()
        
        try:
            # Store some data
            large_content = "x" * 1000  # 1KB of data
            request = MemoryStoreRequest(key="large_item", value=large_content, context="test", metadata={"size": "large"})
            await memory.store(request)
            
            # Check memory usage tracking
            stats = memory.get_stats()
            assert stats["memory_usage_bytes"] > 0
            assert stats["memory_usage_mb"] > 0
            assert stats["item_count"] == 1
            
        finally:
            await memory.shutdown()


class TestWorkingMemorySearch:
    """Test search functionality."""
    
    @pytest.mark.asyncio
    async def test_search_by_key(self):
        """Test searching by key patterns."""
        memory = WorkingMemory(WorkingMemoryConfig())
        await memory.initialize()
        
        try:
            # Store test data
            items = [
                ("user_123", "User data for 123"),
                ("user_456", "User data for 456"),
                ("task_789", "Task data for 789")
            ]
            
            for key, content in items:
                request = MemoryStoreRequest(key=key, value=content, context="test", metadata={"type": "test"})
                await memory.store(request)
            
            # Search for user items
            search_request = MemorySearchRequest(query="user", context="test", limit=10)
            results = await memory.search(search_request)
            
            # Should find both user items
            assert len(results) == 2
            user_keys = {result.item.key for result in results}
            assert "user_123" in user_keys
            assert "user_456" in user_keys
            
        finally:
            await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_search_by_content(self):
        """Test searching by content patterns."""
        memory = WorkingMemory(WorkingMemoryConfig())
        await memory.initialize()
        
        try:
            # Store items with searchable content
            requests = [
                MemoryStoreRequest(key="item1", value="Important project milestone", context="test"),
                MemoryStoreRequest(key="item2", value="Regular task completion", context="test"),
                MemoryStoreRequest(key="item3", value="Critical project deadline", context="test")
            ]
            
            for request in requests:
                await memory.store(request)
            
            # Search for "project" in content
            search_request = MemorySearchRequest(query="project", context="test", limit=10)
            results = await memory.search(search_request)
            
            # Should find items with "project" in content
            assert len(results) == 2
            project_keys = {result.item.key for result in results}
            assert "item1" in project_keys  # "Important project milestone"
            assert "item3" in project_keys  # "Critical project deadline"
            
        finally:
            await memory.shutdown()


class TestWorkingMemoryErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_uninitialized_operations(self):
        """Test that operations fail on uninitialized memory."""
        memory = WorkingMemory(WorkingMemoryConfig())
        
        # All operations should fail before initialization
        store_request = MemoryStoreRequest(key="test", value="test", context="test")
        retrieve_request = MemoryRetrieveRequest(key="test", context="test")
        search_request = MemorySearchRequest(query="test", context="test")
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.store(store_request)
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.retrieve(retrieve_request)
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.search(search_request)
        
        with pytest.raises(MemoryError, match="not initialized"):
            await memory.wipe_context("test")
    
    @pytest.mark.asyncio
    async def test_context_operations(self):
        """Test context creation and management."""
        memory = WorkingMemory(WorkingMemoryConfig())
        await memory.initialize()
        
        try:
            # Create context
            context_name = await memory.create_context("test_context", {"meta": "data"})
            assert context_name == "test_context"
            
            # Check stats
            stats = memory.get_stats()
            assert "test_context" in stats["contexts"]
            assert stats["context_count"] == 1
            
            # Wipe context
            await memory.wipe_context("test_context")
            
            # Check stats after wipe
            stats = memory.get_stats()
            assert "test_context" not in stats["contexts"]
            assert stats["context_count"] == 0
            
        finally:
            await memory.shutdown()
    
    @pytest.mark.asyncio
    async def test_retrieve_relevant(self):
        """Test retrieve_relevant functionality."""
        memory = WorkingMemory(WorkingMemoryConfig())
        await memory.initialize()
        
        try:
            # Store some relevant data
            items = [
                ("doc1", "Python programming tutorial"),
                ("doc2", "JavaScript web development"),
                ("doc3", "Python data science guide")
            ]
            
            for key, content in items:
                request = MemoryStoreRequest(key=key, value=content, context="docs")
                await memory.store(request)
            
            # Retrieve relevant items for "Python"
            relevant = await memory.retrieve_relevant(
                query="Python", 
                context="docs", 
                limit=5
            )
            
            # Should find Python-related items
            assert len(relevant) == 2
            assert any("Python programming tutorial" in item for item in relevant)
            assert any("Python data science guide" in item for item in relevant)
            
        finally:
            await memory.shutdown()


class TestWorkingMemoryStats:
    """Test statistics and monitoring."""
    
    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test statistics reporting."""
        config = WorkingMemoryConfig(
            default_ttl_seconds=1800,
            max_items=5000
        )
        memory = WorkingMemory(config)
        
        # Stats before initialization
        stats = memory.get_stats()
        assert stats["initialized"] is False
        assert stats["item_count"] == 0
        assert stats["context_count"] == 0
        
        await memory.initialize()
        
        try:
            # Stats after initialization
            stats = memory.get_stats()
            assert stats["initialized"] is True
            assert stats["config"]["default_ttl_seconds"] == 1800
            assert stats["config"]["max_items"] == 5000
            
            # Add some data and check updated stats
            request = MemoryStoreRequest(key="test", value="test data", context="test_ctx")
            await memory.store(request)
            
            stats = memory.get_stats()
            assert stats["item_count"] == 1
            assert stats["context_count"] == 1
            assert "test_ctx" in stats["contexts"]
            
        finally:
            await memory.shutdown()


if __name__ == "__main__":
    pytest.main([__file__])