"""Tests for ChromaDB vector database provider."""
import pytest
import os
import tempfile
import shutil
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import uuid4

# Test both with and without chromadb installed
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    # Mock chromadb for testing
    chromadb = None

from flowlib.providers.vector.chroma.provider import (
    ChromaDBProvider,
    ChromaDBProviderSettings,
)
from flowlib.core.errors.errors import ProviderError


class TestChromaDBProviderSettings:
    """Test ChromaDB provider settings."""
    
    def test_default_settings(self):
        """Test default ChromaDB provider settings."""
        settings = ChromaDBProviderSettings()
        
        # Test ChromaDB-specific defaults
        assert settings.persist_directory is None
        assert settings.collection_name == "default"
        assert settings.client_type == "persistent"
        assert settings.http_host == "localhost"
        assert settings.http_port == 8000
        assert settings.http_headers is None
        assert settings.distance_function == "cosine"
        assert settings.anonymized_telemetry is True
        
        # Test inherited provider settings
        assert settings.timeout_seconds == 60.0
        assert settings.max_retries == 3
        assert settings.api_key is None
    
    def test_custom_settings(self):
        """Test custom ChromaDB provider settings."""
        settings = ChromaDBProviderSettings(
            persist_directory="/custom/chroma/data",
            collection_name="custom_collection",
            client_type="http",
            http_host="chroma.example.com",
            http_port=8080,
            http_headers={"Authorization": "Bearer token"},
            distance_function="l2",
            anonymized_telemetry=False,
            # Use valid base provider settings instead of invalid fields
            timeout_seconds=120.0,
            max_retries=5
        )
        
        assert settings.persist_directory == "/custom/chroma/data"
        assert settings.collection_name == "custom_collection"
        assert settings.client_type == "http"
        assert settings.http_host == "chroma.example.com"
        assert settings.http_port == 8080
        assert settings.http_headers == {"Authorization": "Bearer token"}
        assert settings.distance_function == "l2"
        assert settings.anonymized_telemetry is False
        # Test valid base provider settings
        assert settings.timeout_seconds == 120.0
        assert settings.max_retries == 5
    
    def test_settings_inheritance(self):
        """Test that ChromaDBProviderSettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        
        settings = ChromaDBProviderSettings()
        assert isinstance(settings, ProviderSettings)
    
    def test_client_type_validation(self):
        """Test client type options."""
        client_types = ["persistent", "http"]
        
        for client_type in client_types:
            settings = ChromaDBProviderSettings(client_type=client_type)
            assert settings.client_type == client_type
    
    def test_distance_function_validation(self):
        """Test distance function options."""
        distance_functions = ["cosine", "l2", "ip"]
        
        for distance_func in distance_functions:
            settings = ChromaDBProviderSettings(distance_function=distance_func)
            assert settings.distance_function == distance_func


class TestChromaDBProvider:
    """Test ChromaDB provider."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def settings(self, temp_dir):
        """Create test settings."""
        return ChromaDBProviderSettings(
            persist_directory=temp_dir,
            collection_name="test_collection",
            client_type="persistent",
            distance_function="cosine"
        )
    
    @pytest.fixture
    def provider(self, settings):
        """Create test provider."""
        return ChromaDBProvider(name="test_chroma", provider_type="vector", settings=settings)
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Create mock ChromaDB client."""
        mock = Mock()
        mock.get_or_create_collection = Mock()
        mock.delete_collection = Mock()
        mock.list_collections = Mock()
        mock.reset = Mock()
        return mock
    
    @pytest.fixture
    def mock_chroma_collection(self):
        """Create mock ChromaDB collection."""
        mock = Mock()
        mock.add = Mock()
        mock.query = Mock()
        mock.delete = Mock()
        mock.count = Mock()
        mock.peek = Mock()
        mock.get = Mock()
        mock.update = Mock()
        mock.upsert = Mock()
        return mock
    
    def test_provider_initialization(self, settings):
        """Test provider initialization."""
        provider = ChromaDBProvider(name="test_provider", provider_type="vector", settings=settings)
        
        assert provider.name == "test_provider"
        assert provider.provider_type == "vector"
        assert provider._settings == settings
        assert provider._client is None
        assert provider._collections == {}
    
    def test_provider_inheritance(self, provider):
        """Test that ChromaDBProvider inherits from VectorDBProvider."""
        from flowlib.providers.vector.base import VectorDBProvider
        assert isinstance(provider, VectorDBProvider)
    
    def test_invalid_settings_type(self):
        """Test provider with invalid settings type."""
        from flowlib.providers.vector.base import VectorDBProviderSettings
        
        invalid_settings = VectorDBProviderSettings()
        
        with pytest.raises(TypeError) as exc_info:
            ChromaDBProvider(name="test", provider_type="vector", settings=invalid_settings)
        
        assert "must be a ChromaDBProviderSettings instance" in str(exc_info.value)
    
    @patch('flowlib.providers.vector.chroma.provider.chromadb', None)
    async def test_initialize_without_chromadb(self, provider):
        """Test initialization without chromadb package."""
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "ChromaDB package not installed" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_chroma"
    
    @patch('flowlib.providers.vector.chroma.provider.chromadb')
    async def test_initialize_persistent_client_success(self, mock_chromadb, provider, mock_chroma_client):
        """Test successful initialization with persistent client."""
        mock_chromadb.PersistentClient.return_value = mock_chroma_client
        
        await provider.initialize()
        
        mock_chromadb.PersistentClient.assert_called_once()
        call_kwargs = mock_chromadb.PersistentClient.call_args[1]
        assert call_kwargs["path"] == provider._settings.persist_directory
        
        assert provider._client == mock_chroma_client
        assert provider.initialized is True
    
    @patch('flowlib.providers.vector.chroma.provider.chromadb')
    async def test_initialize_http_client_success(self, mock_chromadb, settings, mock_chroma_client):
        """Test successful initialization with HTTP client."""
        # Create new settings with http client type
        http_settings = ChromaDBProviderSettings(
            persist_directory=settings.persist_directory,
            collection_name=settings.collection_name,
            client_type="http",
            http_host="localhost",
            http_port=8000
        )
        provider = ChromaDBProvider(name="test_chroma", provider_type="vector", settings=http_settings)
        mock_chromadb.HttpClient.return_value = mock_chroma_client
        
        await provider.initialize()
        
        mock_chromadb.HttpClient.assert_called_once()
        call_kwargs = mock_chromadb.HttpClient.call_args[1]
        assert call_kwargs["host"] == provider._settings.http_host
        assert call_kwargs["port"] == provider._settings.http_port
        
        assert provider._client == mock_chroma_client
        assert provider.initialized is True
    
    @patch('flowlib.providers.vector.chroma.provider.chromadb')
    async def test_initialize_chromadb_error(self, mock_chromadb, provider):
        """Test initialization with ChromaDB error."""
        mock_chromadb.PersistentClient.side_effect = Exception("ChromaDB initialization failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "Failed to initialize ChromaDB provider" in str(exc_info.value)
    
    async def test_shutdown(self, provider, mock_chroma_client):
        """Test provider shutdown."""
        provider._client = mock_chroma_client
        provider._collections = {"test": Mock()}
        provider._initialized = True
        
        await provider.shutdown()
        
        assert provider._client is None
        assert provider._collections == {}
        assert provider.initialized is False
    
    async def test_get_or_create_collection_new(self, provider, mock_chroma_client, mock_chroma_collection):
        """Test creating a new collection."""
        provider._client = mock_chroma_client
        provider._initialized = True
        mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
        
        collection = await provider._get_or_create_collection("new_collection")
        
        assert collection == mock_chroma_collection
        assert "new_collection" in provider._collections
        mock_chroma_client.get_or_create_collection.assert_called_once_with(name="new_collection")
    
    async def test_get_or_create_collection_existing(self, provider, mock_chroma_collection):
        """Test retrieving existing collection."""
        provider._initialized = True
        provider._collections["existing_collection"] = mock_chroma_collection
        
        collection = await provider._get_or_create_collection("existing_collection")
        
        assert collection == mock_chroma_collection
    
    async def test_get_or_create_collection_not_initialized(self, provider):
        """Test collection access when not initialized."""
        provider._initialized = False
        
        with pytest.raises(ProviderError) as exc_info:
            await provider._get_or_create_collection("test_collection")
        
        assert "Provider not initialized" in str(exc_info.value)
    
    async def test_create_index_success(self, provider, mock_chroma_client, mock_chroma_collection):
        """Test successful index creation."""
        provider._client = mock_chroma_client
        provider._initialized = True
        mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
        
        result = await provider.create_index("test_index", 384, "cosine")
        
        assert result is True
        mock_chroma_client.get_or_create_collection.assert_called_once_with(name="test_index")
    
    async def test_create_index_error(self, provider, mock_chroma_client):
        """Test index creation with error."""
        provider._client = mock_chroma_client
        provider._initialized = True
        mock_chroma_client.get_or_create_collection.side_effect = Exception("Collection creation failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.create_index("test_index", 384)
        
        assert "Failed to create index" in str(exc_info.value)
    
    async def test_delete_index_success(self, provider, mock_chroma_client):
        """Test successful index deletion."""
        provider._client = mock_chroma_client
        provider._initialized = True
        provider._collections["test_index"] = Mock()
        
        result = await provider.delete_index("test_index")
        
        assert result is True
        mock_chroma_client.delete_collection.assert_called_once_with(name="test_index")
        assert "test_index" not in provider._collections
    
    async def test_delete_index_not_found(self, provider, mock_chroma_client):
        """Test deleting non-existent index."""
        provider._client = mock_chroma_client
        provider._initialized = True
        mock_chroma_client.delete_collection.side_effect = Exception("Collection not found")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.delete_index("nonexistent_index")
        
        assert "Failed to delete index" in str(exc_info.value)
    
    async def test_index_exists_true(self, provider, mock_chroma_client):
        """Test index exists check for existing index."""
        provider._client = mock_chroma_client
        provider._initialized = True
        
        mock_collection = Mock()
        mock_collection.name = "existing_index"
        mock_chroma_client.list_collections.return_value = [mock_collection]
        
        result = await provider.index_exists("existing_index")
        
        assert result is True
    
    async def test_index_exists_false(self, provider, mock_chroma_client):
        """Test index exists check for non-existent index."""
        provider._client = mock_chroma_client
        provider._initialized = True
        mock_chroma_client.list_collections.return_value = []
        
        result = await provider.index_exists("nonexistent_index")
        
        assert result is False
    
    async def test_insert_vectors_success(self, provider, mock_chroma_collection):
        """Test successful vector insertion."""
        provider._initialized = True
        
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        metadata = [{"text": "doc1"}, {"text": "doc2"}]
        ids = ["id1", "id2"]
        
        with patch.object(ChromaDBProvider, '_get_or_create_collection', return_value=mock_chroma_collection):
            result = await provider.insert_vectors("test_index", vectors, metadata, ids)
        
        assert result is True
        mock_chroma_collection.add.assert_called_once_with(
            embeddings=vectors,
            metadatas=metadata,
            ids=ids
        )
    
    async def test_insert_vectors_auto_ids(self, provider, mock_chroma_collection):
        """Test vector insertion with auto-generated IDs."""
        provider._initialized = True
        
        vectors = [[0.1, 0.2, 0.3]]
        metadata = [{"text": "doc1"}]
        
        with patch.object(ChromaDBProvider, '_get_or_create_collection', return_value=mock_chroma_collection):
            with patch('uuid.uuid4', return_value=Mock(hex="auto_id")):
                result = await provider.insert_vectors("test_index", vectors, metadata)
        
        assert result is True
        call_args = mock_chroma_collection.add.call_args[1]
        assert call_args["ids"] == ["auto_id"]
    
    async def test_insert_vectors_error(self, provider, mock_chroma_collection):
        """Test vector insertion with error."""
        provider._initialized = True
        mock_chroma_collection.add.side_effect = Exception("Insert failed")
        
        with patch.object(ChromaDBProvider, '_get_or_create_collection', return_value=mock_chroma_collection):
            with pytest.raises(ProviderError) as exc_info:
                await provider.insert_vectors("test_index", [[0.1, 0.2]], [{"text": "doc"}])
        
        assert "Failed to insert vectors" in str(exc_info.value)
    
    async def test_search_vectors_success(self, provider, mock_chroma_collection):
        """Test successful vector search."""
        provider._initialized = True
        
        # Mock ChromaDB query response
        mock_response = {
            "ids": [["id1", "id2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"text": "doc1"}, {"text": "doc2"}]],
            "embeddings": [[[0.1, 0.2], [0.4, 0.5]]]
        }
        mock_chroma_collection.query.return_value = mock_response
        
        query_vector = [0.1, 0.2]
        
        with patch.object(ChromaDBProvider, '_get_or_create_collection', return_value=mock_chroma_collection):
            results = await provider.search_vectors("test_index", query_vector, top_k=2)
        
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert results[0]["score"] == 0.9  # 1 - 0.1 distance
        assert results[0]["metadata"] == {"text": "doc1"}
        
        mock_chroma_collection.query.assert_called_once_with(
            query_embeddings=[query_vector],
            n_results=2,
            where=None
        )
    
    async def test_search_vectors_with_filter(self, provider, mock_chroma_collection):
        """Test vector search with filter conditions."""
        provider._initialized = True
        
        mock_response = {
            "ids": [["id1"]],
            "distances": [[0.1]],
            "metadatas": [[{"text": "doc1", "category": "A"}]],
            "embeddings": [[[0.1, 0.2]]]
        }
        mock_chroma_collection.query.return_value = mock_response
        
        query_vector = [0.1, 0.2]
        filter_conditions = {"category": "A"}
        
        with patch.object(ChromaDBProvider, '_get_or_create_collection', return_value=mock_chroma_collection):
            results = await provider.search_vectors("test_index", query_vector, top_k=5, filter_conditions=filter_conditions)
        
        assert len(results) == 1
        mock_chroma_collection.query.assert_called_once_with(
            query_embeddings=[query_vector],
            n_results=5,
            where=filter_conditions
        )
    
    async def test_search_vectors_error(self, provider, mock_chroma_collection):
        """Test vector search with error."""
        provider._initialized = True
        mock_chroma_collection.query.side_effect = Exception("Search failed")
        
        with patch.object(ChromaDBProvider, '_get_or_create_collection', return_value=mock_chroma_collection):
            with pytest.raises(ProviderError) as exc_info:
                await provider.search_vectors("test_index", [0.1, 0.2])
        
        assert "Failed to search vectors" in str(exc_info.value)
    
    async def test_delete_vectors_success(self, provider, mock_chroma_collection):
        """Test successful vector deletion."""
        provider._initialized = True
        
        ids = ["id1", "id2"]
        
        with patch.object(ChromaDBProvider, '_get_or_create_collection', return_value=mock_chroma_collection):
            result = await provider.delete_vectors("test_index", ids)
        
        assert result is True
        mock_chroma_collection.delete.assert_called_once_with(ids=ids)
    
    async def test_delete_vectors_error(self, provider, mock_chroma_collection):
        """Test vector deletion with error."""
        provider._initialized = True
        mock_chroma_collection.delete.side_effect = Exception("Delete failed")
        
        with patch.object(ChromaDBProvider, '_get_or_create_collection', return_value=mock_chroma_collection):
            with pytest.raises(ProviderError) as exc_info:
                await provider.delete_vectors("test_index", ["id1"])
        
        assert "Failed to delete vectors" in str(exc_info.value)
    
    async def test_get_index_stats_success(self, provider, mock_chroma_collection):
        """Test successful index stats retrieval."""
        provider._initialized = True
        
        mock_chroma_collection.count.return_value = 1000
        mock_chroma_collection.peek.return_value = {
            "embeddings": [[[0.1] * 384]]
        }
        
        with patch.object(ChromaDBProvider, '_get_or_create_collection', return_value=mock_chroma_collection):
            stats = await provider.get_index_stats("test_index")
        
        assert stats["total_vectors"] == 1000
        assert stats["dimension"] == 384
        assert stats["distance_function"] == "cosine"
    
    async def test_get_index_stats_error(self, provider, mock_chroma_collection):
        """Test index stats retrieval with error."""
        provider._initialized = True
        mock_chroma_collection.count.side_effect = Exception("Stats failed")
        
        with patch.object(ChromaDBProvider, '_get_or_create_collection', return_value=mock_chroma_collection):
            with pytest.raises(ProviderError) as exc_info:
                await provider.get_index_stats("test_index")
        
        assert "Failed to get index stats" in str(exc_info.value)
    
    def test_provider_registration(self):
        """Test that provider is properly registered."""
        # This would be tested in integration tests
        # Here we just verify the decorator is applied
        assert hasattr(ChromaDBProvider, '__provider_type__')
        assert hasattr(ChromaDBProvider, '__provider_name__')
    
    def test_distance_function_mapping(self, provider):
        """Test distance function mapping to ChromaDB space."""
        mappings = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip",
            "euclidean": "l2"  # Should map to l2
        }
        
        for input_metric, expected_space in mappings.items():
            result = provider._map_distance_function(input_metric)
            assert result == expected_space


@pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="chromadb package not available")
@pytest.mark.integration
class TestChromaDBProviderIntegration:
    """Integration tests for ChromaDB provider.
    
    These tests require a running ChromaDB instance or use in-memory mode.
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def settings(self, chroma_settings):
        """Create integration test settings from global config."""
        # Override collection name for integration tests
        settings_dict = chroma_settings.model_dump()
        settings_dict.update({
            "collection_name": "integration_test",
            # Remove invalid field: vector_dimension
            "distance_function": "cosine"
        })
        return ChromaDBProviderSettings(**settings_dict)
    
    @pytest.fixture
    async def provider(self, settings):
        """Create and initialize provider for integration tests."""
        provider = ChromaDBProvider(name="integration_chroma", provider_type="vector", settings=settings)
        
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
        """Test complete vector operations lifecycle with real ChromaDB."""
        # Create index
        await provider.create_index("integration_test", 128, "cosine")
        
        # Check index exists
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
        
        # Delete index
        await provider.delete_index("integration_test")
        
        # Verify index deleted
        exists_after_delete = await provider.index_exists("integration_test")
        assert exists_after_delete is False
    
    def test_settings_integration(self, settings):
        """Test settings integration with real values."""
        assert settings.client_type in ["persistent", "http"]
        # Removed invalid field assertion: vector_dimension 
        assert settings.distance_function == "cosine"
        assert os.path.exists(settings.persist_directory)
    
    def test_provider_creation_integration(self, settings):
        """Test provider creation with real settings."""
        provider = ChromaDBProvider(name="integration_test", provider_type="vector", settings=settings)
        
        assert provider.name == "integration_test"
        assert provider._settings.persist_directory == settings.persist_directory
        assert not provider.initialized