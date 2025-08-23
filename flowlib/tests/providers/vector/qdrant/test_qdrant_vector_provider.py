"""Tests for Qdrant vector database provider."""
import pytest
import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import uuid4

# Test both with and without qdrant-client installed
try:
    from qdrant_client import QdrantClient, models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    # Mock qdrant for testing
    QdrantClient = None
    models = None

from flowlib.providers.vector.qdrant.provider import (
    QdrantProvider,
    QdrantProviderSettings,
)
from flowlib.core.errors.errors import ProviderError
from flowlib.providers.core.registry import provider_registry


class TestQdrantProviderSettings:
    """Test Qdrant provider settings."""
    
    def test_default_settings(self):
        """Test default Qdrant provider settings."""
        settings = QdrantProviderSettings()
        
        # Test Qdrant-specific defaults (after architecture change to direct inheritance)
        assert settings.url == "http://localhost:6333"
        assert settings.api_key is None
        assert settings.collection_name == "default"  # Has default value, not None
        assert settings.vector_size == 1536
        assert settings.distance == "Cosine"
        assert settings.timeout == 30.0
        assert settings.prefer_grpc is False
        assert settings.host is None
        assert settings.port is None
        assert settings.grpc_port == 6334
        assert settings.https is False
        assert settings.prefix is None
        assert settings.prefer_local is False
        assert settings.path is None
        assert settings.embedding_provider_name == "default_embedding"
        
        # Test inherited provider settings
        assert settings.timeout_seconds == 60.0
        assert settings.max_retries == 3
    
    def test_custom_settings(self):
        """Test custom Qdrant provider settings."""
        settings = QdrantProviderSettings(
            url="https://qdrant.example.com",
            api_key="test_api_key",
            collection_name="custom_collection",
            vector_size=768,
            distance="Euclid",
            timeout=60.0,
            prefer_grpc=True,
            host="custom_host",
            port=9333,
            grpc_port=443,
            https=True,
            prefix="/v1",
            prefer_local=True,
            path="/custom/path",
            embedding_provider_name="custom_embedding",
            timeout_seconds=120.0,
            max_retries=5
        )
        
        # Test Qdrant-specific fields
        assert settings.url == "https://qdrant.example.com"
        assert settings.api_key == "test_api_key"
        assert settings.collection_name == "custom_collection"
        assert settings.vector_size == 768
        assert settings.distance == "Euclid"
        assert settings.timeout == 60.0
        assert settings.prefer_grpc is True
        assert settings.host == "custom_host"
        assert settings.port == 9333
        assert settings.grpc_port == 443
        assert settings.https is True
        assert settings.prefix == "/v1"
        assert settings.prefer_local is True
        assert settings.path == "/custom/path"
        assert settings.embedding_provider_name == "custom_embedding"
        
        # Test inherited provider settings
        assert settings.timeout_seconds == 120.0
        assert settings.max_retries == 5
    
    def test_settings_inheritance(self):
        """Test that QdrantProviderSettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        
        settings = QdrantProviderSettings()
        assert isinstance(settings, ProviderSettings)
    
    def test_url_validation(self):
        """Test different URL formats."""
        urls = [
            "http://localhost:6333",
            "https://cloud.qdrant.io",
            "grpc://localhost:6334",
            "http://192.168.1.100:6333"
        ]
        
        for url in urls:
            settings = QdrantProviderSettings(url=url)
            assert settings.url == url


class TestQdrantProvider:
    """Test Qdrant provider."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return QdrantProviderSettings(
            url="http://localhost:6333",
            collection_name="test_collection",
            vector_size=384,
            distance="Cosine"
        )
    
    @pytest.fixture
    def provider(self, settings):
        """Create test provider."""
        return QdrantProvider(name="test_qdrant", provider_type="vector", settings=settings)
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        mock = Mock()
        mock.get_collections = Mock()
        mock.create_collection = Mock()
        mock.delete_collection = Mock()
        mock.collection_exists = Mock()
        mock.upsert = Mock()
        mock.search = Mock()
        mock.delete = Mock()
        mock.count = Mock()
        mock.get_collection = Mock()
        return mock
    
    def test_provider_initialization(self, settings):
        """Test provider initialization."""
        provider = QdrantProvider(name="test_provider", provider_type="vector", settings=settings)
        
        assert provider.name == "test_provider"
        assert provider.provider_type == "vector"
        assert provider._settings == settings
        assert provider._client is None
    
    def test_provider_inheritance(self, provider):
        """Test that QdrantProvider inherits from VectorDBProvider."""
        from flowlib.providers.vector.base import VectorDBProvider
        assert isinstance(provider, VectorDBProvider)
    
    def test_invalid_settings_type(self):
        """Test provider with invalid settings type."""
        from flowlib.providers.vector.base import VectorDBProviderSettings
        
        invalid_settings = VectorDBProviderSettings()
        
        with pytest.raises(TypeError) as exc_info:
            QdrantProvider(name="test", provider_type="vector", settings=invalid_settings)
        
        assert "must be a QdrantProviderSettings instance" in str(exc_info.value)
    
    @patch('flowlib.providers.vector.qdrant.provider.QdrantClient', None)
    async def test_initialize_without_qdrant(self, provider):
        """Test initialization without qdrant-client package."""
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "qdrant-client package not installed" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_qdrant"
    
    @patch('flowlib.providers.vector.qdrant.provider.QdrantClient')
    async def test_initialize_success(self, mock_qdrant_client_class, provider, mock_qdrant_client):
        """Test successful initialization."""
        mock_qdrant_client_class.return_value = mock_qdrant_client
        
        await provider.initialize()
        
        mock_qdrant_client_class.assert_called_once_with(
            url="http://localhost:6333",
            api_key=None,
            prefer_grpc=False,
            timeout=30.0
        )
        assert provider._client == mock_qdrant_client
        assert provider.initialized is True
    
    @patch('flowlib.providers.vector.qdrant.provider.QdrantClient')
    async def test_initialize_with_auth(self, mock_qdrant_client_class, settings, mock_qdrant_client):
        """Test initialization with authentication."""
        # Create new settings with api_key
        auth_settings = QdrantProviderSettings(
            host=settings.host,
            port=settings.port,
            api_key="test_key"
        )
        provider = QdrantProvider(name="test_qdrant", provider_type="vector", settings=auth_settings)
        mock_qdrant_client_class.return_value = mock_qdrant_client
        
        await provider.initialize()
        
        call_args = mock_qdrant_client_class.call_args[1]
        assert call_args["api_key"] == "test_key"
    
    @patch('flowlib.providers.vector.qdrant.provider.QdrantClient')
    async def test_initialize_qdrant_error(self, mock_qdrant_client_class, provider):
        """Test initialization with Qdrant error."""
        mock_qdrant_client_class.side_effect = Exception("Qdrant initialization failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "Failed to initialize Qdrant provider" in str(exc_info.value)
    
    async def test_shutdown(self, provider, mock_qdrant_client):
        """Test provider shutdown."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        
        await provider.shutdown()
        
        assert provider._client is None
        assert provider.initialized is False
    
    async def test_create_index_success(self, provider, mock_qdrant_client):
        """Test successful index (collection) creation."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        
        with patch('flowlib.providers.vector.qdrant.provider.models') as mock_models:
            mock_models.Distance = Mock()
            mock_models.Distance.COSINE = "cosine"
            mock_models.VectorParams = Mock()
            mock_models.CollectionConfig = Mock()
            
            result = await provider.create_index("test_collection", 384, "cosine")
        
        assert result is True
        mock_qdrant_client.create_collection.assert_called_once()
    
    async def test_create_index_error(self, provider, mock_qdrant_client):
        """Test index creation with error."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        mock_qdrant_client.create_collection.side_effect = Exception("Collection creation failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.create_index("test_collection", 384)
        
        assert "Failed to create index" in str(exc_info.value)
    
    async def test_delete_index_success(self, provider, mock_qdrant_client):
        """Test successful index deletion."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        
        result = await provider.delete_index("test_collection")
        
        assert result is True
        mock_qdrant_client.delete_collection.assert_called_once_with("test_collection")
    
    async def test_delete_index_error(self, provider, mock_qdrant_client):
        """Test index deletion with error."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        mock_qdrant_client.delete_collection.side_effect = Exception("Collection deletion failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.delete_index("test_collection")
        
        assert "Failed to delete index" in str(exc_info.value)
    
    async def test_index_exists_true(self, provider, mock_qdrant_client):
        """Test index exists check for existing collection."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        mock_qdrant_client.collection_exists.return_value = True
        
        result = await provider.index_exists("existing_collection")
        
        assert result is True
        mock_qdrant_client.collection_exists.assert_called_once_with("existing_collection")
    
    async def test_index_exists_false(self, provider, mock_qdrant_client):
        """Test index exists check for non-existent collection."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        mock_qdrant_client.collection_exists.return_value = False
        
        result = await provider.index_exists("nonexistent_collection")
        
        assert result is False
    
    async def test_insert_vectors_success(self, provider, mock_qdrant_client):
        """Test successful vector insertion."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadata = [{"text": "doc1"}, {"text": "doc2"}]
        ids = ["id1", "id2"]
        
        with patch('flowlib.providers.vector.qdrant.provider.models') as mock_models:
            mock_models.PointStruct = Mock()
            
            result = await provider.insert_vectors("test_collection", vectors, metadata, ids)
        
        assert result is True
        mock_qdrant_client.upsert.assert_called_once()
        
        call_args = mock_qdrant_client.upsert.call_args[1]
        assert call_args["collection_name"] == "test_collection"
    
    async def test_insert_vectors_auto_ids(self, provider, mock_qdrant_client):
        """Test vector insertion with auto-generated IDs."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        
        vectors = [[0.1, 0.2, 0.3]]
        metadata = [{"text": "doc1"}]
        
        with patch('flowlib.providers.vector.qdrant.provider.models') as mock_models:
            mock_models.PointStruct = Mock()
            with patch('uuid.uuid4', return_value=Mock(hex="auto_id")):
                result = await provider.insert_vectors("test_collection", vectors, metadata)
        
        assert result is True
    
    async def test_insert_vectors_error(self, provider, mock_qdrant_client):
        """Test vector insertion with error."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        mock_qdrant_client.upsert.side_effect = Exception("Upsert failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.insert_vectors("test_collection", [[0.1, 0.2]], [{"text": "doc"}])
        
        assert "Failed to insert vectors" in str(exc_info.value)
    
    async def test_search_vectors_success(self, provider, mock_qdrant_client):
        """Test successful vector search."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        
        # Mock Qdrant search response
        mock_response = [
            Mock(
                id="id1",
                score=0.95,
                payload={"text": "doc1"},
                vector=[0.1, 0.2]
            ),
            Mock(
                id="id2",
                score=0.85,
                payload={"text": "doc2"},
                vector=[0.4, 0.5]
            )
        ]
        mock_qdrant_client.search.return_value = mock_response
        
        query_vector = [0.1, 0.2]
        
        results = await provider.search_vectors("test_collection", query_vector, top_k=2)
        
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"] == {"text": "doc1"}
        
        mock_qdrant_client.search.assert_called_once()
        call_args = mock_qdrant_client.search.call_args[1]
        assert call_args["collection_name"] == "test_collection"
        assert call_args["query_vector"] == query_vector
        assert call_args["limit"] == 2
    
    async def test_search_vectors_with_filter(self, provider, mock_qdrant_client):
        """Test vector search with filter conditions."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        mock_qdrant_client.search.return_value = []
        
        query_vector = [0.1, 0.2]
        filter_conditions = {"category": "A"}
        
        with patch('flowlib.providers.vector.qdrant.provider.models') as mock_models:
            mock_models.Filter = Mock()
            mock_models.FieldCondition = Mock()
            mock_models.MatchValue = Mock()
            
            results = await provider.search_vectors("test_collection", query_vector, top_k=5, filter_conditions=filter_conditions)
        
        call_args = mock_qdrant_client.search.call_args[1]
        assert "query_filter" in call_args
    
    async def test_search_vectors_error(self, provider, mock_qdrant_client):
        """Test vector search with error."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        mock_qdrant_client.search.side_effect = Exception("Search failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.search_vectors("test_collection", [0.1, 0.2])
        
        assert "Failed to search vectors" in str(exc_info.value)
    
    async def test_delete_vectors_success(self, provider, mock_qdrant_client):
        """Test successful vector deletion."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        
        ids = ["id1", "id2"]
        
        with patch('flowlib.providers.vector.qdrant.provider.models') as mock_models:
            mock_models.PointIdsList = Mock()
            
            result = await provider.delete_vectors("test_collection", ids)
        
        assert result is True
        mock_qdrant_client.delete.assert_called_once()
        call_args = mock_qdrant_client.delete.call_args[1]
        assert call_args["collection_name"] == "test_collection"
    
    async def test_delete_vectors_error(self, provider, mock_qdrant_client):
        """Test vector deletion with error."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        mock_qdrant_client.delete.side_effect = Exception("Delete failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.delete_vectors("test_collection", ["id1"])
        
        assert "Failed to delete vectors" in str(exc_info.value)
    
    async def test_get_index_stats_success(self, provider, mock_qdrant_client):
        """Test successful index stats retrieval."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        
        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.points_count = 1000
        mock_collection_info.config = Mock()
        mock_collection_info.config.params = Mock()
        mock_collection_info.config.params.vectors = Mock()
        mock_collection_info.config.params.vectors.size = 384
        
        mock_qdrant_client.get_collection.return_value = mock_collection_info
        
        stats = await provider.get_index_stats("test_collection")
        
        assert stats["total_vectors"] == 1000
        assert stats["dimension"] == 384
        assert stats["metric"] == "cosine"
    
    async def test_get_index_stats_error(self, provider, mock_qdrant_client):
        """Test index stats retrieval with error."""
        object.__setattr__(provider, '_client', mock_qdrant_client)
        object.__setattr__(provider, '_initialized', True)
        mock_qdrant_client.get_collection.side_effect = Exception("Stats failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.get_index_stats("test_collection")
        
        assert "Failed to get index stats" in str(exc_info.value)
    
    def test_provider_registration(self):
        """Test that provider is properly registered."""
        # This would be tested in integration tests
        # Here we just verify the decorator is applied
        assert hasattr(QdrantProvider, '__provider_type__')
        assert hasattr(QdrantProvider, '__provider_name__')
    
    def test_map_distance_metric(self, provider):
        """Test distance metric mapping to Qdrant format."""
        mappings = {
            "cosine": "Cosine",
            "euclidean": "Euclid", 
            "dot": "Dot",
            "l2": "Euclid"
        }
        
        for input_metric, expected_qdrant in mappings.items():
            result = provider._map_distance_metric(input_metric)
            assert result == expected_qdrant
    
    def test_convert_filter_conditions(self, provider):
        """Test filter condition conversion to Qdrant format."""
        # Simple equality filter
        filter_input = {"category": "A"}
        
        with patch('flowlib.providers.vector.qdrant.provider.models') as mock_models:
            mock_models.Filter = Mock()
            mock_models.FieldCondition = Mock()
            mock_models.MatchValue = Mock()
            
            result = provider._convert_filter_conditions(filter_input)
            
            # Should create appropriate Qdrant filter objects
            mock_models.Filter.assert_called()
            mock_models.FieldCondition.assert_called()
    
    def test_create_point_structs(self, provider):
        """Test creation of Qdrant point structures."""
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        metadata = [{"text": "doc1"}, {"text": "doc2"}]
        ids = ["id1", "id2"]
        
        with patch('flowlib.providers.vector.qdrant.provider.models') as mock_models:
            mock_models.PointStruct = Mock()
            
            points = provider._create_point_structs(vectors, metadata, ids)
            
            # Should create PointStruct for each vector
            assert mock_models.PointStruct.call_count == 2


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client package not available")
@pytest.mark.integration
class TestQdrantProviderIntegration:
    """Integration tests for Qdrant provider.
    
    These tests require a running Qdrant instance.
    """
    
    @pytest.fixture
    def settings(self, qdrant_settings):
        """Create integration test settings from global config."""
        # Override collection name for integration tests
        settings_dict = qdrant_settings.model_dump()
        settings_dict.update({
            "collection_name": "integration_test",
            # Remove invalid fields: vector_dimension, metric
        })
        return QdrantProviderSettings(**settings_dict)
    
    @pytest.fixture
    async def provider(self, settings):
        """Create and initialize provider for integration tests."""
        provider = QdrantProvider(name="integration_qdrant", provider_type="vector", settings=settings)
        
        try:
            await provider.initialize()
            yield provider
        finally:
            # Cleanup
            try:
                await provider.shutdown()
            except:
                pass
    
    async def test_full_vector_lifecycle_integration(self, provider):
        """Test complete vector operations lifecycle with real Qdrant."""
        # Create collection
        await provider.create_index("integration_test", 128, "cosine")
        
        # Check collection exists
        exists = await provider.index_exists("integration_test")
        assert exists is True
        
        # Insert vectors
        vectors = [
            [0.1] * 128,
            [0.2] * 128,
            [0.3] * 128
        ]
        metadata = [
            {"text": "First document", "category": "A"},
            {"text": "Second document", "category": "B"},
            {"text": "Third document", "category": "A"}
        ]
        ids = ["doc1", "doc2", "doc3"]
        
        await provider.insert_vectors("integration_test", vectors, metadata, ids)
        
        # Search vectors
        query_vector = [0.15] * 128
        results = await provider.search_vectors("integration_test", query_vector, top_k=2)
        assert len(results) <= 2
        assert all("id" in result for result in results)
        assert all("score" in result for result in results)
        
        # Search with filter
        filtered_results = await provider.search_vectors(
            "integration_test", 
            query_vector, 
            top_k=5, 
            filter_conditions={"category": "A"}
        )
        # Should return only documents with category A
        for result in filtered_results:
            if "metadata" in result and result["metadata"]:
                assert result["metadata"].get("category") == "A"
        
        # Get stats
        stats = await provider.get_index_stats("integration_test")
        assert stats["total_vectors"] >= 3
        assert stats["dimension"] == 128
        
        # Delete some vectors
        await provider.delete_vectors("integration_test", ["doc2"])
        
        # Verify deletion
        post_delete_stats = await provider.get_index_stats("integration_test")
        assert post_delete_stats["total_vectors"] == stats["total_vectors"] - 1
        
        # Delete collection
        await provider.delete_index("integration_test")
        
        # Verify collection deleted
        exists_after_delete = await provider.index_exists("integration_test")
        assert exists_after_delete is False
    
    def test_settings_integration(self, settings):
        """Test settings integration with real values."""
        assert settings.url == "http://localhost:6333"
        # Removed invalid field assertions: vector_dimension, metric
        assert settings.prefer_grpc is False
    
    def test_provider_creation_integration(self, settings):
        """Test provider creation with real settings."""
        provider = QdrantProvider(name="integration_test", provider_type="vector", settings=settings)
        
        assert provider.name == "integration_test"
        assert provider._settings.url == "http://localhost:6333"
        assert not provider.initialized