"""Simple working tests for Provider Factory."""

import pytest
from flowlib.providers.core.factory import PROVIDER_IMPLEMENTATIONS


class TestProviderImplementations:
    """Test PROVIDER_IMPLEMENTATIONS registry structure."""
    
    def test_provider_implementations_structure(self):
        """Test that PROVIDER_IMPLEMENTATIONS has expected structure."""
        # Check that main provider types exist
        expected_types = [
            "llm", "database", "message_queue", "cache", "vector_db",
            "storage", "embedding", "graph_db", "state_persister"
        ]
        
        for provider_type in expected_types:
            assert provider_type in PROVIDER_IMPLEMENTATIONS
            assert isinstance(PROVIDER_IMPLEMENTATIONS[provider_type], dict)
    
    def test_llm_implementations(self):
        """Test LLM provider implementations."""
        llm_impls = PROVIDER_IMPLEMENTATIONS["llm"]
        
        assert "llamacpp" in llm_impls
        assert "llama" in llm_impls
    
    def test_database_implementations(self):
        """Test database provider implementations."""
        db_impls = PROVIDER_IMPLEMENTATIONS["database"]
        
        assert "postgres" in db_impls
        assert "postgresql" in db_impls
        assert "mongodb" in db_impls
        assert "mongo" in db_impls
        assert "sqlite" in db_impls
        assert "sqlite3" in db_impls
    
    def test_vector_db_implementations(self):
        """Test vector database provider implementations."""
        vector_impls = PROVIDER_IMPLEMENTATIONS["vector_db"]
        
        assert "chroma" in vector_impls
        assert "chromadb" in vector_impls
        assert "pinecone" in vector_impls
        assert "qdrant" in vector_impls
    
    def test_cache_implementations(self):
        """Test cache provider implementations."""
        cache_impls = PROVIDER_IMPLEMENTATIONS["cache"]
        
        assert "redis" in cache_impls
        assert "memory" in cache_impls
        assert "inmemory" in cache_impls
    
    def test_storage_implementations(self):
        """Test storage provider implementations."""
        storage_impls = PROVIDER_IMPLEMENTATIONS["storage"]
        
        assert "s3" in storage_impls
        assert "aws" in storage_impls
        assert "local" in storage_impls
        assert "localfile" in storage_impls
        assert "file" in storage_impls
    
    def test_embedding_implementations(self):
        """Test embedding provider implementations."""
        embedding_impls = PROVIDER_IMPLEMENTATIONS["embedding"]
        
        assert "llamacpp" in embedding_impls
        assert "llamacpp_embedding" in embedding_impls
    
    def test_graph_db_implementations(self):
        """Test graph database provider implementations."""
        graph_impls = PROVIDER_IMPLEMENTATIONS["graph_db"]
        
        assert "neo4j" in graph_impls
    
    def test_message_queue_implementations(self):
        """Test message queue provider implementations."""
        mq_impls = PROVIDER_IMPLEMENTATIONS["message_queue"]
        
        assert "rabbitmq" in mq_impls
        assert "rabbit" in mq_impls
        assert "kafka" in mq_impls
    
    def test_state_persister_implementations(self):
        """Test state persister provider implementations."""
        sp_impls = PROVIDER_IMPLEMENTATIONS["state_persister"]
        
        assert "redis" in sp_impls
        assert "mongodb" in sp_impls
        assert "postgres" in sp_impls
        assert "file" in sp_impls


if __name__ == "__main__":
    pytest.main([__file__, "-v"])