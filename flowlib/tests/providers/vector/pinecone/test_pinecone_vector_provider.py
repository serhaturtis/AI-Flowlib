"""Tests for Pinecone vector database provider."""
import pytest
import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import uuid4

# Test both with and without pinecone installed
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    # Mock pinecone for testing
    pinecone = None

from flowlib.providers.vector.pinecone.provider import (
    PineconeProvider,
    PineconeProviderSettings,
)
from flowlib.core.errors.errors import ProviderError
from flowlib.providers.core.registry import provider_registry


class TestPineconeProviderSettings:
    """Test Pinecone provider settings."""
    
    def test_default_settings(self):
        """Test default Pinecone provider settings."""
        settings = PineconeProviderSettings(
            api_key="test_key",
            environment="us-east1-gcp",
            index_name="test_index"
        )
        
        # Test Pinecone-specific defaults
        assert settings.api_key == "test_key"
        assert settings.environment == "us-east1-gcp"
        assert settings.index_name == "test_index"
        assert settings.namespace == ""
        assert settings.dimension == 1536
        assert settings.metric == "cosine"
        assert settings.pod_type == "p1.x1"
        assert settings.api_timeout == 30.0
        
        # Test inherited provider settings
        assert settings.timeout_seconds == 60.0
        assert settings.max_retries == 3
    
    def test_custom_settings(self):
        """Test custom Pinecone provider settings."""
        
        settings = PineconeProviderSettings(
            api_key="custom_api_key",
            environment="us-west1-gcp",
            index_name="custom_index",
            namespace="custom_namespace",
            dimension=768,
            metric="dotproduct",
            pod_type="p2",
            api_timeout=60.0,
            # Use valid base provider settings
            timeout_seconds=120.0,
            max_retries=5
        )
        
        assert settings.api_key == "custom_api_key"
        assert settings.environment == "us-west1-gcp"
        assert settings.index_name == "custom_index"
        assert settings.namespace == "custom_namespace"
        assert settings.dimension == 768
        assert settings.metric == "dotproduct"
        assert settings.pod_type == "p2"
        assert settings.api_timeout == 60.0
        # Test valid base provider settings
        assert settings.timeout_seconds == 120.0
        assert settings.max_retries == 5
    
    def test_settings_inheritance(self):
        """Test that PineconeProviderSettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        
        settings = PineconeProviderSettings(api_key="test", environment="us-east1-gcp", index_name="test_index")
        assert isinstance(settings, ProviderSettings)
    
    def test_metric_validation(self):
        """Test metric options."""
        metrics = ["cosine", "dotproduct", "euclidean"]
        
        for metric in metrics:
            settings = PineconeProviderSettings(
                api_key="test", 
                environment="us-east1-gcp", 
                index_name="test_index", 
                metric=metric
            )
            assert settings.metric == metric
    
    def test_pod_type_validation(self):
        """Test pod type options."""
        pod_types = ["p1", "p2", "s1"]
        
        for pod_type in pod_types:
            settings = PineconeProviderSettings(
                api_key="test", 
                environment="us-east1-gcp", 
                index_name="test_index", 
                pod_type=pod_type
            )
            assert settings.pod_type == pod_type


class TestPineconeProvider:
    """Test Pinecone provider."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return PineconeProviderSettings(
            api_key="test_api_key",
            environment="us-west1-gcp",
            index_name="test_index",
            namespace="test_namespace",
            dimension=384,
            metric="cosine"
        )
    
    @pytest.fixture
    def provider(self, settings):
        """Create test provider."""
        return PineconeProvider(name="test_pinecone", provider_type="vector", settings=settings)
    
    @pytest.fixture
    def mock_pinecone_client(self):
        """Create mock Pinecone client."""
        mock = Mock()
        mock.list_indexes = Mock()
        mock.create_index = Mock()
        mock.delete_index = Mock()
        mock.describe_index = Mock()
        mock.Index = Mock()
        return mock
    
    @pytest.fixture
    def mock_pinecone_index(self):
        """Create mock Pinecone index."""
        mock = Mock()
        mock.upsert = Mock()
        mock.query = Mock()
        mock.delete = Mock()
        mock.fetch = Mock()
        mock.describe_index_stats = Mock()
        return mock
    
    def test_provider_initialization(self, settings):
        """Test provider initialization."""
        provider = PineconeProvider(name="test_provider", provider_type="vector", settings=settings)
        
        assert provider.name == "test_provider"
        assert provider.provider_type == "vector"
        assert provider._settings == settings
        assert provider._client is None
        assert provider._indexes == {}
    
    def test_provider_inheritance(self, provider):
        """Test that PineconeProvider inherits from VectorDBProvider."""
        from flowlib.providers.vector.base import VectorDBProvider
        assert isinstance(provider, VectorDBProvider)
    
    def test_invalid_settings_type(self):
        """Test provider with invalid settings type."""
        from flowlib.providers.vector.base import VectorDBProviderSettings
        
        invalid_settings = VectorDBProviderSettings()
        
        with pytest.raises(TypeError) as exc_info:
            PineconeProvider(name="test", provider_type="vector", settings=invalid_settings)
        
        assert "must be a PineconeProviderSettings instance" in str(exc_info.value)
    
    @patch('flowlib.providers.vector.pinecone.provider.pinecone', None)
    @patch('flowlib.providers.vector.pinecone.provider.PineconeClient')
    async def test_initialize_without_pinecone(self, mock_pinecone_client, provider):
        """Test initialization without pinecone package."""
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "pinecone-client package not installed" in str(exc_info.value)
        assert exc_info.value.context.data.component == "test_pinecone"
    
    @patch('flowlib.providers.vector.pinecone.provider.pinecone', "mocked_pinecone")  # Mock that pinecone is available
    @patch('flowlib.providers.vector.pinecone.provider.PineconeClient')
    async def test_initialize_success(self, mock_pinecone_client_class, provider, mock_pinecone_client):
        """Test successful initialization."""
        mock_pinecone_client_class.return_value = mock_pinecone_client
        
        await provider.initialize()
        
        mock_pinecone_client_class.assert_called_once_with(api_key="test_api_key")
        assert provider._client == mock_pinecone_client
        assert provider.initialized is True
    
    @patch('flowlib.providers.vector.pinecone.provider.pinecone', "mocked_pinecone")  # Mock that pinecone is available
    @patch('flowlib.providers.vector.pinecone.provider.PineconeClient')
    async def test_initialize_pinecone_error(self, mock_pinecone_client_class, provider):
        """Test initialization with Pinecone error."""
        mock_pinecone_client_class.side_effect = Exception("Pinecone initialization failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "Failed to initialize Pinecone provider" in str(exc_info.value)
    
    async def test_shutdown(self, provider, mock_pinecone_client):
        """Test provider shutdown."""
        provider._client = mock_pinecone_client
        provider._indexes = {"test": Mock()}
        provider._initialized = True
        
        await provider.shutdown()
        
        assert provider._client is None
        assert provider._indexes == {}
        assert provider.initialized is False
    
    async def test_get_index_existing(self, provider, mock_pinecone_client, mock_pinecone_index):
        """Test getting existing index."""
        provider._client = mock_pinecone_client
        provider._initialized = True
        provider._indexes["existing_index"] = mock_pinecone_index
        
        index = await provider._get_index("existing_index")
        
        assert index == mock_pinecone_index
    
    async def test_get_index_new(self, provider, mock_pinecone_client, mock_pinecone_index):
        """Test getting new index."""
        provider._client = mock_pinecone_client
        provider._initialized = True
        mock_pinecone_client.Index.return_value = mock_pinecone_index
        
        index = await provider._get_index("new_index")
        
        assert index == mock_pinecone_index
        assert "new_index" in provider._indexes
        mock_pinecone_client.Index.assert_called_once_with("new_index")
    
    async def test_get_index_not_initialized(self, provider):
        """Test getting index when not initialized."""
        provider._initialized = False
        
        with pytest.raises(ProviderError) as exc_info:
            await provider._get_index("test_index")
        
        assert "Provider not initialized" in str(exc_info.value)
    
    async def test_create_index_success(self, provider, mock_pinecone_client):
        """Test successful index creation."""
        provider._client = mock_pinecone_client
        provider._initialized = True
        
        result = await provider.create_index("test_index", 384, "cosine")
        
        assert result is True
        mock_pinecone_client.create_index.assert_called_once()
        call_args = mock_pinecone_client.create_index.call_args
        assert call_args[0][0] == "test_index"  # name
        assert call_args[1]["dimension"] == 384
        assert call_args[1]["metric"] == "cosine"
    
    async def test_create_index_with_options(self, provider, mock_pinecone_client):
        """Test index creation with additional options."""
        provider._client = mock_pinecone_client
        provider._initialized = True
        
        result = await provider.create_index(
            "test_index", 
            768, 
            "dotproduct",
            pod_type="p2",
            replicas=2
        )
        
        assert result is True
        call_args = mock_pinecone_client.create_index.call_args[1]
        assert call_args["pod_type"] == "p2"
        assert call_args["replicas"] == 2
    
    async def test_create_index_error(self, provider, mock_pinecone_client):
        """Test index creation with error."""
        provider._client = mock_pinecone_client
        provider._initialized = True
        mock_pinecone_client.create_index.side_effect = Exception("Index creation failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.create_index("test_index", 384)
        
        assert "Failed to create index" in str(exc_info.value)
    
    async def test_delete_index_success(self, provider, mock_pinecone_client):
        """Test successful index deletion."""
        provider._client = mock_pinecone_client
        provider._initialized = True
        provider._indexes["test_index"] = Mock()
        
        result = await provider.delete_index("test_index")
        
        assert result is True
        mock_pinecone_client.delete_index.assert_called_once_with("test_index")
        assert "test_index" not in provider._indexes
    
    async def test_delete_index_error(self, provider, mock_pinecone_client):
        """Test index deletion with error."""
        provider._client = mock_pinecone_client
        provider._initialized = True
        mock_pinecone_client.delete_index.side_effect = Exception("Index deletion failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.delete_index("test_index")
        
        assert "Failed to delete index" in str(exc_info.value)
    
    async def test_index_exists_true(self, provider, mock_pinecone_client):
        """Test index exists check for existing index."""
        provider._client = mock_pinecone_client
        provider._initialized = True
        mock_pinecone_client.list_indexes.return_value = ["existing_index", "another_index"]
        
        result = await provider.index_exists("existing_index")
        
        assert result is True
    
    async def test_index_exists_false(self, provider, mock_pinecone_client):
        """Test index exists check for non-existent index."""
        provider._client = mock_pinecone_client
        provider._initialized = True
        mock_pinecone_client.list_indexes.return_value = ["other_index"]
        
        result = await provider.index_exists("nonexistent_index")
        
        assert result is False
    
    async def test_insert_vectors_success(self, provider, mock_pinecone_index):
        """Test successful vector insertion."""
        provider._initialized = True
        
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadata = [{"text": "doc1"}, {"text": "doc2"}]
        ids = ["id1", "id2"]
        
        with patch.object(PineconeProvider, '_get_index', return_value=mock_pinecone_index):
            result = await provider.insert_vectors("test_index", vectors, metadata, ids)
        
        assert result is True
        mock_pinecone_index.upsert.assert_called_once()
        
        call_args = mock_pinecone_index.upsert.call_args[1]
        assert call_args["namespace"] == "test_namespace"
        assert len(call_args["vectors"]) == 2
    
    async def test_insert_vectors_auto_ids(self, provider, mock_pinecone_index):
        """Test vector insertion with auto-generated IDs."""
        provider._initialized = True
        
        vectors = [[0.1, 0.2, 0.3]]
        metadata = [{"text": "doc1"}]
        
        with patch.object(PineconeProvider, '_get_index', return_value=mock_pinecone_index):
            with patch('uuid.uuid4', return_value=Mock(hex="auto_id")):
                result = await provider.insert_vectors("test_index", vectors, metadata)
        
        assert result is True
        call_args = mock_pinecone_index.upsert.call_args[1]
        assert call_args["vectors"][0]["id"] == "auto_id"
    
    async def test_insert_vectors_error(self, provider, mock_pinecone_index):
        """Test vector insertion with error."""
        provider._initialized = True
        mock_pinecone_index.upsert.side_effect = Exception("Upsert failed")
        
        with patch.object(PineconeProvider, '_get_index', return_value=mock_pinecone_index):
            with pytest.raises(ProviderError) as exc_info:
                await provider.insert_vectors("test_index", [[0.1, 0.2]], [{"text": "doc"}])
        
        assert "Failed to insert vectors" in str(exc_info.value)
    
    async def test_search_vectors_success(self, provider, mock_pinecone_index):
        """Test successful vector search."""
        provider._initialized = True
        
        # Mock Pinecone query response
        mock_response = {
            "matches": [
                {
                    "id": "id1",
                    "score": 0.95,
                    "metadata": {"text": "doc1"},
                    "values": [0.1, 0.2]
                },
                {
                    "id": "id2", 
                    "score": 0.85,
                    "metadata": {"text": "doc2"},
                    "values": [0.4, 0.5]
                }
            ]
        }
        mock_pinecone_index.query.return_value = mock_response
        
        query_vector = [0.1, 0.2]
        
        with patch.object(PineconeProvider, '_get_index', return_value=mock_pinecone_index):
            results = await provider.search_vectors("test_index", query_vector, top_k=2)
        
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["score"] == 0.95
        assert results[0]["metadata"] == {"text": "doc1"}
        
        mock_pinecone_index.query.assert_called_once()
        call_args = mock_pinecone_index.query.call_args[1]
        assert call_args["vector"] == query_vector
        assert call_args["top_k"] == 2
        assert call_args["namespace"] == "test_namespace"
    
    async def test_search_vectors_with_filter(self, provider, mock_pinecone_index):
        """Test vector search with filter conditions."""
        provider._initialized = True
        
        mock_response = {"matches": []}
        mock_pinecone_index.query.return_value = mock_response
        
        query_vector = [0.1, 0.2]
        filter_conditions = {"category": {"$eq": "A"}}
        
        with patch.object(PineconeProvider, '_get_index', return_value=mock_pinecone_index):
            results = await provider.search_vectors("test_index", query_vector, top_k=5, filter_conditions=filter_conditions)
        
        call_args = mock_pinecone_index.query.call_args[1]
        assert call_args["filter"] == filter_conditions
    
    async def test_search_vectors_error(self, provider, mock_pinecone_index):
        """Test vector search with error."""
        provider._initialized = True
        mock_pinecone_index.query.side_effect = Exception("Query failed")
        
        with patch.object(PineconeProvider, '_get_index', return_value=mock_pinecone_index):
            with pytest.raises(ProviderError) as exc_info:
                await provider.search_vectors("test_index", [0.1, 0.2])
        
        assert "Failed to search vectors" in str(exc_info.value)
    
    async def test_delete_vectors_success(self, provider, mock_pinecone_index):
        """Test successful vector deletion."""
        provider._initialized = True
        
        ids = ["id1", "id2"]
        
        with patch.object(PineconeProvider, '_get_index', return_value=mock_pinecone_index):
            result = await provider.delete_vectors("test_index", ids)
        
        assert result is True
        mock_pinecone_index.delete.assert_called_once()
        call_args = mock_pinecone_index.delete.call_args[1]
        assert call_args["ids"] == ids
        assert call_args["namespace"] == "test_namespace"
    
    async def test_delete_vectors_error(self, provider, mock_pinecone_index):
        """Test vector deletion with error."""
        provider._initialized = True
        mock_pinecone_index.delete.side_effect = Exception("Delete failed")
        
        with patch.object(PineconeProvider, '_get_index', return_value=mock_pinecone_index):
            with pytest.raises(ProviderError) as exc_info:
                await provider.delete_vectors("test_index", ["id1"])
        
        assert "Failed to delete vectors" in str(exc_info.value)
    
    async def test_get_index_stats_success(self, provider, mock_pinecone_index):
        """Test successful index stats retrieval."""
        provider._initialized = True
        
        mock_stats = {
            "dimension": 384,
            "index_fullness": 0.1,
            "namespaces": {
                "test_namespace": {"vector_count": 1000}
            },
            "total_vector_count": 1000
        }
        mock_pinecone_index.describe_index_stats.return_value = mock_stats
        
        with patch.object(PineconeProvider, '_get_index', return_value=mock_pinecone_index):
            stats = await provider.get_index_stats("test_index")
        
        assert stats["total_vectors"] == 1000
        assert stats["dimension"] == 384
        assert stats["index_fullness"] == 0.1
        assert stats["namespace_stats"]["test_namespace"]["vector_count"] == 1000
    
    async def test_get_index_stats_error(self, provider, mock_pinecone_index):
        """Test index stats retrieval with error."""
        provider._initialized = True
        mock_pinecone_index.describe_index_stats.side_effect = Exception("Stats failed")
        
        with patch.object(PineconeProvider, '_get_index', return_value=mock_pinecone_index):
            with pytest.raises(ProviderError) as exc_info:
                await provider.get_index_stats("test_index")
        
        assert "Failed to get index stats" in str(exc_info.value)
    
    def test_provider_registration(self):
        """Test that provider is properly registered."""
        # This would be tested in integration tests
        # Here we just verify the decorator is applied
        assert hasattr(PineconeProvider, '__provider_type__')
        assert hasattr(PineconeProvider, '__provider_name__')
    
    def test_convert_filter_conditions(self, provider):
        """Test filter condition conversion to Pinecone format."""
        # Simple equality filter
        filter_input = {"category": "A"}
        expected_output = {"category": {"$eq": "A"}}
        result = provider._convert_filter_conditions(filter_input)
        assert result == expected_output
        
        # Already in Pinecone format
        pinecone_filter = {"category": {"$eq": "A"}, "score": {"$gte": 0.8}}
        result = provider._convert_filter_conditions(pinecone_filter)
        assert result == pinecone_filter
        
        # Complex filter
        complex_filter = {"category": "A", "tags": ["important", "urgent"]}
        result = provider._convert_filter_conditions(complex_filter)
        assert result["category"]["$eq"] == "A"
        assert result["tags"]["$in"] == ["important", "urgent"]


@pytest.mark.skipif(not PINECONE_AVAILABLE, reason="pinecone-client package not available")
@pytest.mark.integration
class TestPineconeProviderIntegration:
    """Integration tests for Pinecone provider.
    
    These tests require a valid Pinecone API key and environment.
    """
    
    @pytest.fixture
    def settings(self, pinecone_settings):
        """Create integration test settings from global config."""
        return pinecone_settings
    
    @pytest.fixture
    async def provider(self, settings):
        """Create and initialize provider for integration tests."""
        provider = PineconeProvider(name="integration_pinecone", provider_type="vector", settings=settings)
        
        try:
            # Note: This would fail without a real API key
            # await provider.initialize()
            yield provider
        finally:
            # Cleanup
            try:
                await provider.shutdown()
            except:
                pass
    
    def test_settings_integration(self, settings):
        """Test settings integration with environment-based values."""
        # Accept both cloud and local API keys
        assert settings.api_key in ["test_api_key", "pclocal"]  # Both cloud and local
        # Accept both cloud and local environments
        assert settings.environment in ["us-west1-gcp", "local"]  # Both cloud and local
        assert settings.index_name == "integration_test"  # Updated to match fixture
        # Namespace can be None, empty string, or "test" for the default fixture configuration
        assert settings.namespace in ["test", None, ""]
        # Dimension defaults to 1536 if not specified in the fixture
        assert settings.dimension in [128, 1536]
        assert settings.metric == "cosine"
    
    def test_provider_creation_integration(self, settings):
        """Test provider creation with real settings."""
        provider = PineconeProvider(name="integration_test", provider_type="vector", settings=settings)
        
        assert provider.name == "integration_test"
        # Accept both cloud and local API keys
        assert provider._settings.api_key in ["test_api_key", "pclocal"]
        assert not provider.initialized
    
    def test_vector_format_integration(self, settings):
        """Test vector format conversion for Pinecone."""
        provider = PineconeProvider(name="test", provider_type="vector", settings=settings)
        
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        metadata = [{"text": "doc1"}, {"text": "doc2"}]
        ids = ["id1", "id2"]
        
        formatted = provider._format_vectors_for_upsert(vectors, metadata, ids)
        
        assert len(formatted) == 2
        assert formatted[0]["id"] == "id1"
        assert formatted[0]["values"] == [0.1, 0.2]
        assert formatted[0]["metadata"] == {"text": "doc1"}
    
    def test_batch_size_handling_integration(self, settings):
        """Test batch size handling for large uploads."""
        provider = PineconeProvider(name="test", provider_type="vector", settings=settings)
        
        # Test batch splitting
        large_batch_size = 250
        vectors = [[0.1] * 128] * large_batch_size
        
        batches = provider._split_into_batches(vectors, batch_size=100)
        
        assert len(batches) == 3  # 100, 100, 50
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50