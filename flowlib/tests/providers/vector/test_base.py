"""Tests for vector database provider base class."""
import pytest
import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from pydantic import BaseModel, ValidationError

from flowlib.providers.vector.base import (
    VectorDBProvider,
    VectorDBProviderSettings,
)
from flowlib.providers.core.base import ProviderSettings
from flowlib.core.errors.errors import ProviderError


class MockDocument(BaseModel):
    """Mock document model."""
    id: str
    text: str
    metadata: Dict[str, Any] = {}


class ConcreteVectorDBProvider(VectorDBProvider):
    """Concrete implementation for testing."""
    
    def __init__(self, name: str = "test_vector", settings: Optional[VectorDBProviderSettings] = None):
        super().__init__(name=name, provider_type="vector", settings=settings)
        # Store operation calls in private attribute to avoid Pydantic validation
        object.__setattr__(self, '_operation_calls', [])
    
    async def create_index(self, index_name: str, vector_dimension: int, metric: str = "cosine", **kwargs) -> bool:
        """Mock create index implementation."""
        self._operation_calls.append({
            "operation": "create_index",
            "index_name": index_name,
            "vector_dimension": vector_dimension,
            "metric": metric,
            "kwargs": kwargs
        })
        return True
    
    async def delete_index(self, index_name: str) -> bool:
        """Mock delete index implementation."""
        self._operation_calls.append({
            "operation": "delete_index",
            "index_name": index_name
        })
        return True
    
    async def index_exists(self, index_name: str) -> bool:
        """Mock index exists implementation."""
        self._operation_calls.append({
            "operation": "index_exists",
            "index_name": index_name
        })
        return True
    
    async def insert_vectors(self, index_name: str, vectors: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> bool:
        """Mock insert vectors implementation."""
        self._operation_calls.append({
            "operation": "insert_vectors",
            "index_name": index_name,
            "vectors": vectors,
            "metadata": metadata,
            "ids": ids
        })
        return True
    
    async def search_vectors(self, index_name: str, query_vector: List[float], top_k: int = 10, filter_conditions: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Mock search vectors implementation."""
        self._operation_calls.append({
            "operation": "search_vectors",
            "index_name": index_name,
            "query_vector": query_vector,
            "top_k": top_k,
            "filter_conditions": filter_conditions
        })
        return [
            {"id": "1", "score": 0.95, "metadata": {"text": "Similar document"}},
            {"id": "2", "score": 0.85, "metadata": {"text": "Another similar document"}}
        ]
    
    async def delete_vectors(self, index_name: str, ids: List[str]) -> bool:
        """Mock delete vectors implementation."""
        self._operation_calls.append({
            "operation": "delete_vectors",
            "index_name": index_name,
            "ids": ids
        })
        return True
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Mock get index stats implementation."""
        self._operation_calls.append({
            "operation": "get_index_stats",
            "index_name": index_name
        })
        return {
            "total_vectors": 1000,
            "dimension": 1536,
            "metric": "cosine"
        }


class TestVectorDBProviderSettings:
    """Test vector database provider settings."""
    
    def test_default_settings(self):
        """Test default vector DB provider settings."""
        settings = VectorDBProviderSettings()
        
        # Test connection settings
        assert settings.host is None
        assert settings.port is None
        assert settings.api_key is None
        assert settings.username is None
        assert settings.password is None
        
        # Test vector settings
        assert settings.index_name == "default"
        assert settings.vector_dimension == 1536
        assert settings.metric == "cosine"
        
        # Test performance settings
        assert settings.batch_size == 100
        assert settings.timeout == 30.0
    
    def test_custom_settings(self):
        """Test custom vector DB provider settings."""
        settings = VectorDBProviderSettings(
            host="vector-db.example.com",
            port=6333,
            api_key="test_api_key",
            username="vector_user",
            password="vector_pass",
            index_name="custom_index",
            vector_dimension=768,
            metric="euclidean",
            batch_size=500,
            timeout=60.0
        )
        
        assert settings.host == "vector-db.example.com"
        assert settings.port == 6333
        assert settings.api_key == "test_api_key"
        assert settings.username == "vector_user"
        assert settings.password == "vector_pass"
        assert settings.index_name == "custom_index"
        assert settings.vector_dimension == 768
        assert settings.metric == "euclidean"
        assert settings.batch_size == 500
        assert settings.timeout == 60.0
    
    def test_settings_inheritance(self):
        """Test that VectorDBProviderSettings inherits from ProviderSettings."""
        settings = VectorDBProviderSettings()
        assert isinstance(settings, ProviderSettings)
        
        # Should have base provider settings (these are attributes, not fields)
        assert hasattr(settings, 'api_key')
        assert hasattr(settings, 'timeout_seconds')
    
    def test_metric_options(self):
        """Test different metric options."""
        metrics = ["cosine", "euclidean", "dot", "manhattan"]
        
        for metric in metrics:
            settings = VectorDBProviderSettings(metric=metric)
            assert settings.metric == metric


class TestVectorDBProvider:
    """Test vector database provider base class."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return VectorDBProviderSettings(
            host="localhost",
            port=6333,
            index_name="test_index",
            vector_dimension=384,
            metric="cosine",
            batch_size=50
        )
    
    @pytest.fixture
    def provider(self, settings):
        """Create test provider."""
        return ConcreteVectorDBProvider(name="test_provider", settings=settings)
    
    def test_provider_initialization(self, settings):
        """Test provider initialization."""
        provider = ConcreteVectorDBProvider(name="test_vector", settings=settings)
        
        assert provider.name == "test_vector"
        assert provider.provider_type == "vector"
        assert provider.settings == settings
        assert provider._operation_calls == []
    
    def test_provider_inheritance(self, provider):
        """Test that VectorDBProvider inherits from Provider."""
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, Provider)
    
    async def test_create_index(self, provider):
        """Test create index operation."""
        result = await provider.create_index("new_index", 768, "euclidean", shards=2)
        
        assert result is True
        assert len(provider._operation_calls) == 1
        
        call = provider._operation_calls[0]
        assert call["operation"] == "create_index"
        assert call["index_name"] == "new_index"
        assert call["vector_dimension"] == 768
        assert call["metric"] == "euclidean"
        assert call["kwargs"]["shards"] == 2
    
    async def test_delete_index(self, provider):
        """Test delete index operation."""
        result = await provider.delete_index("old_index")
        
        assert result is True
        assert len(provider._operation_calls) == 1
        
        call = provider._operation_calls[0]
        assert call["operation"] == "delete_index"
        assert call["index_name"] == "old_index"
    
    async def test_index_exists(self, provider):
        """Test index exists check."""
        result = await provider.index_exists("test_index")
        
        assert result is True
        assert len(provider._operation_calls) == 1
        
        call = provider._operation_calls[0]
        assert call["operation"] == "index_exists"
        assert call["index_name"] == "test_index"
    
    async def test_insert_vectors(self, provider):
        """Test vector insertion."""
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadata = [{"text": "doc1"}, {"text": "doc2"}]
        ids = ["id1", "id2"]
        
        result = await provider.insert_vectors("test_index", vectors, metadata, ids)
        
        assert result is True
        assert len(provider._operation_calls) == 1
        
        call = provider._operation_calls[0]
        assert call["operation"] == "insert_vectors"
        assert call["index_name"] == "test_index"
        assert call["vectors"] == vectors
        assert call["metadata"] == metadata
        assert call["ids"] == ids
    
    async def test_insert_vectors_minimal(self, provider):
        """Test vector insertion with minimal parameters."""
        vectors = [[0.1, 0.2, 0.3]]
        
        result = await provider.insert_vectors("test_index", vectors)
        
        assert result is True
        assert len(provider._operation_calls) == 1
        
        call = provider._operation_calls[0]
        assert call["metadata"] is None
        assert call["ids"] is None
    
    async def test_search_vectors(self, provider):
        """Test vector search."""
        query_vector = [0.1, 0.2, 0.3]
        filter_conditions = {"category": "documents"}
        
        results = await provider.search_vectors("test_index", query_vector, top_k=5, filter_conditions=filter_conditions)
        
        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[0]["score"] == 0.95
        assert len(provider._operation_calls) == 1
        
        call = provider._operation_calls[0]
        assert call["operation"] == "search_vectors"
        assert call["query_vector"] == query_vector
        assert call["top_k"] == 5
        assert call["filter_conditions"] == filter_conditions
    
    async def test_search_vectors_defaults(self, provider):
        """Test vector search with default parameters."""
        query_vector = [0.1, 0.2, 0.3]
        
        results = await provider.search_vectors("test_index", query_vector)
        
        assert len(results) == 2
        call = provider._operation_calls[0]
        assert call["top_k"] == 10  # default
        assert call["filter_conditions"] is None
    
    async def test_delete_vectors(self, provider):
        """Test vector deletion."""
        ids = ["id1", "id2", "id3"]
        
        result = await provider.delete_vectors("test_index", ids)
        
        assert result is True
        assert len(provider._operation_calls) == 1
        
        call = provider._operation_calls[0]
        assert call["operation"] == "delete_vectors"
        assert call["ids"] == ids
    
    async def test_get_index_stats(self, provider):
        """Test getting index statistics."""
        stats = await provider.get_index_stats("test_index")
        
        assert stats["total_vectors"] == 1000
        assert stats["dimension"] == 1536
        assert stats["metric"] == "cosine"
        assert len(provider._operation_calls) == 1
        
        call = provider._operation_calls[0]
        assert call["operation"] == "get_index_stats"
        assert call["index_name"] == "test_index"
    
    async def test_abstract_methods_in_base_class(self):
        """Test that abstract methods raise NotImplementedError in base class."""
        from flowlib.providers.vector.base import VectorDBProvider, VectorDBProviderSettings
        
        # Create a minimal concrete provider to test abstract methods
        class AbstractTestProvider(VectorDBProvider):
            pass
        
        # This should fail during instantiation due to missing settings type - but VectorDBProvider now has proper generic
        # So let's create it with explicit settings
        provider = VectorDBProvider(name="abstract", provider_type="vector", settings=VectorDBProviderSettings())
        
        with pytest.raises(NotImplementedError):
            await provider.create_index("test", 768)
        
        with pytest.raises(NotImplementedError):
            await provider.delete_index("test")
        
        with pytest.raises(NotImplementedError):
            await provider.index_exists("test")
        
        with pytest.raises(NotImplementedError):
            await provider.insert_vectors("test", [[0.1, 0.2]])
        
        with pytest.raises(NotImplementedError):
            await provider.search_vectors("test", [0.1, 0.2])
        
        with pytest.raises(NotImplementedError):
            await provider.delete_vectors("test", ["id1"])
        
        with pytest.raises(NotImplementedError):
            await provider.get_index_stats("test")


class TestVectorDBProviderIntegration:
    """Integration tests for vector database provider."""
    
    @pytest.fixture
    def provider(self):
        """Create provider for integration tests."""
        settings = VectorDBProviderSettings(
            host="localhost",
            port=6333,
            index_name="integration_test",
            vector_dimension=128,
            metric="cosine"
        )
        return ConcreteVectorDBProvider(name="integration_provider", settings=settings)
    
    async def test_full_vector_lifecycle(self, provider):
        """Test complete vector operations lifecycle."""
        # Create index
        await provider.create_index("lifecycle_test", 128, "cosine")
        
        # Check index exists
        exists = await provider.index_exists("lifecycle_test")
        assert exists is True
        
        # Insert vectors
        vectors = [
            [0.1, 0.2] + [0.0] * 126,  # 128-dimensional
            [0.3, 0.4] + [0.0] * 126,
            [0.5, 0.6] + [0.0] * 126
        ]
        metadata = [
            {"text": "First document", "category": "A"},
            {"text": "Second document", "category": "B"},
            {"text": "Third document", "category": "A"}
        ]
        ids = ["doc1", "doc2", "doc3"]
        
        await provider.insert_vectors("lifecycle_test", vectors, metadata, ids)
        
        # Search vectors
        query_vector = [0.15, 0.25] + [0.0] * 126
        results = await provider.search_vectors("lifecycle_test", query_vector, top_k=2)
        assert len(results) == 2
        
        # Search with filter
        filtered_results = await provider.search_vectors(
            "lifecycle_test", 
            query_vector, 
            top_k=5, 
            filter_conditions={"category": "A"}
        )
        assert len(filtered_results) == 2
        
        # Get stats
        stats = await provider.get_index_stats("lifecycle_test")
        assert stats["dimension"] == 1536  # Mock returns this
        
        # Delete some vectors
        await provider.delete_vectors("lifecycle_test", ["doc2"])
        
        # Delete index
        await provider.delete_index("lifecycle_test")
        
        # Verify all operations were called
        assert len(provider._operation_calls) == 8
        
        operations = [call["operation"] for call in provider._operation_calls]
        expected_operations = [
            "create_index", "index_exists", "insert_vectors", 
            "search_vectors", "search_vectors", "get_index_stats",
            "delete_vectors", "delete_index"
        ]
        assert operations == expected_operations
    
    async def test_batch_operations(self, provider):
        """Test batch vector operations."""
        # Large batch of vectors
        batch_size = 250
        vectors = [[0.1, 0.2] + [0.0] * 126 for _ in range(batch_size)]
        metadata = [{"batch": "test", "index": i} for i in range(batch_size)]
        ids = [f"batch_doc_{i}" for i in range(batch_size)]
        
        # Insert in batch
        await provider.insert_vectors("batch_test", vectors, metadata, ids)
        
        # Delete in batch
        delete_ids = [f"batch_doc_{i}" for i in range(0, batch_size, 2)]  # Delete every other
        await provider.delete_vectors("batch_test", delete_ids)
        
        # Verify operations
        assert len(provider._operation_calls) == 2
        
        insert_call = provider._operation_calls[0]
        assert len(insert_call["vectors"]) == batch_size
        assert len(insert_call["metadata"]) == batch_size
        assert len(insert_call["ids"]) == batch_size
        
        delete_call = provider._operation_calls[1]
        assert len(delete_call["ids"]) == batch_size // 2
    
    async def test_error_handling_scenarios(self, provider):
        """Test various error handling scenarios."""
        # These would test error conditions in a real implementation
        # For mock implementation, we just verify the calls are made
        
        scenarios = [
            ("empty_vectors", []),
            ("single_vector", [[0.1] * 128]),
            ("large_batch", [[0.1] * 128] * 1000)
        ]
        
        for index_name, vectors in scenarios:
            try:
                await provider.insert_vectors(index_name, vectors)
            except Exception:
                # In a real implementation, this might raise errors
                # for edge cases like empty vectors
                pass
        
        # Verify attempts were made
        assert len(provider._operation_calls) >= len(scenarios)
    
    def test_vector_dimension_validation(self, provider):
        """Test vector dimension consistency."""
        # In a real implementation, this would validate dimensions
        settings = provider.settings
        
        # Test different dimension scenarios
        dimensions = [128, 384, 768, 1536]
        
        for dim in dimensions:
            # Create settings with different dimensions
            test_settings = VectorDBProviderSettings(vector_dimension=dim)
            assert test_settings.vector_dimension == dim
            
            # Verify dimension is positive
            assert dim > 0
    
    def test_metric_compatibility(self, provider):
        """Test different distance metrics."""
        metrics = ["cosine", "euclidean", "dot", "manhattan"]
        
        for metric in metrics:
            # Each metric should be valid
            test_settings = VectorDBProviderSettings(metric=metric)
            assert test_settings.metric == metric