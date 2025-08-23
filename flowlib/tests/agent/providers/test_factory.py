"""Tests for agent provider factory."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from flowlib.agent.providers.factory import AgentProviderFactory
from flowlib.agent.core.errors import ProviderError, ConfigurationError


class TestAgentProviderFactory:
    """Test AgentProviderFactory class."""
    
    @pytest.mark.asyncio
    async def test_create_provider_success(self):
        """Test successful provider creation."""
        mock_provider = Mock()
        
        with patch('flowlib.agent.providers.factory.base_create_provider', 
                  AsyncMock(return_value=mock_provider)) as mock_create:
            with patch('flowlib.agent.providers.factory.provider_registry.get', 
                      AsyncMock(side_effect=Exception("Not found"))):
                
                result = await AgentProviderFactory.create_provider(
                    provider_type="llm",
                    provider_name="llamacpp",
                    settings={"model_path": "/path/to/model.gguf"}
                )
                
                assert result == mock_provider
                mock_create.assert_called_once_with(
                    provider_type="llm",
                    name="llamacpp",
                    implementation="llamacpp",
                    register=True,
                    model_path="/path/to/model.gguf"
                )
    
    @pytest.mark.asyncio
    async def test_create_provider_reuse_existing(self):
        """Test reusing existing provider."""
        existing_provider = Mock()
        
        with patch('flowlib.agent.providers.factory.provider_registry.contains', return_value=True), \
             patch('flowlib.agent.providers.factory.provider_registry.get_by_config',
                  AsyncMock(return_value=existing_provider)):
            
            result = await AgentProviderFactory.create_provider(
                provider_type="llm",
                provider_name="llamacpp"
            )
            
            assert result == existing_provider
    
    @pytest.mark.asyncio
    async def test_create_provider_unknown_implementation(self):
        """Test creating provider with unknown implementation."""
        mock_provider = Mock()
        
        with patch('flowlib.agent.providers.factory.base_create_provider',
                  AsyncMock(return_value=mock_provider)):
            with patch('flowlib.agent.providers.factory.provider_registry.get',
                      AsyncMock(side_effect=Exception("Not found"))):
                
                # Unknown provider name should use the name as implementation
                result = await AgentProviderFactory.create_provider(
                    provider_type="llm",
                    provider_name="custom_llm"
                )
                
                assert result == mock_provider
    
    @pytest.mark.asyncio
    async def test_create_provider_error(self):
        """Test provider creation error handling."""
        with patch('flowlib.agent.providers.factory.base_create_provider',
                  AsyncMock(side_effect=Exception("Creation failed"))):
            with patch('flowlib.agent.providers.factory.provider_registry.get',
                      AsyncMock(side_effect=Exception("Not found"))):
                
                with pytest.raises(ProviderError) as exc_info:
                    await AgentProviderFactory.create_provider(
                        provider_type="llm",
                        provider_name="llamacpp"
                    )
                
                assert "Failed to create llm provider 'llamacpp'" in str(exc_info.value)
                assert exc_info.value.context["provider_name"] == "llamacpp"
                assert exc_info.value.context["operation"] == "create"
    
    @pytest.mark.asyncio
    async def test_create_memory_providers_full(self):
        """Test creating all memory providers."""
        mock_embedding = Mock()
        mock_vector = Mock()
        mock_graph = Mock()
        mock_llm = Mock()
        
        memory_config = {
            "vector_memory": {
                "embedding_provider_name": "llamacpp",
                "embedding_settings": {"model_path": "/embedding.gguf"},
                "vector_provider_name": "chroma",
                "vector_settings": {"persist_dir": "./chroma"}
            },
            "knowledge_memory": {
                "graph_provider_name": "neo4j",
                "provider_settings": {"uri": "bolt://localhost"}
            },
            "fusion_provider_name": "llamacpp",
            "fusion_settings": {"model_path": "/llm.gguf"}
        }
        
        # Mock the create_provider method to return different providers
        with patch.object(AgentProviderFactory, 'create_provider',
                         AsyncMock(side_effect=[
                             mock_embedding, mock_vector, mock_graph, mock_llm
                         ])) as mock_create:
            
            providers = await AgentProviderFactory.create_memory_providers(memory_config)
            
            assert providers["embedding"] == mock_embedding
            assert providers["vector"] == mock_vector
            assert providers["graph"] == mock_graph
            assert providers["fusion_llm"] == mock_llm
            
            # Verify calls
            assert mock_create.call_count == 4
            
            # Check embedding provider call
            mock_create.assert_any_call(
                "embedding", "llamacpp", 
                {"model_path": "/embedding.gguf"}
            )
            
            # Check vector provider call
            mock_create.assert_any_call(
                "vector_db", "chroma",
                {"persist_dir": "./chroma"}
            )
    
    @pytest.mark.asyncio
    async def test_create_memory_providers_partial(self):
        """Test creating partial memory providers."""
        mock_embedding = Mock()
        mock_vector = Mock()
        
        memory_config = {
            "vector_memory": {
                "embedding_provider_name": "openai",
                "vector_provider_name": "pinecone"
            }
        }
        
        with patch.object(AgentProviderFactory, 'create_provider',
                         AsyncMock(side_effect=[mock_embedding, mock_vector])):
            
            providers = await AgentProviderFactory.create_memory_providers(memory_config)
            
            assert len(providers) == 2
            assert "embedding" in providers
            assert "vector" in providers
            assert "graph" not in providers
            assert "fusion_llm" not in providers
    
    @pytest.mark.asyncio
    async def test_create_memory_providers_cleanup_on_failure(self):
        """Test cleanup of providers on failure."""
        mock_embedding = Mock()
        mock_embedding.shutdown = AsyncMock()
        
        memory_config = {
            "vector_memory": {"embedding_provider_name": "llamacpp"},
            "knowledge_memory": {"graph_provider_name": "neo4j"}
        }
        
        # First provider succeeds, second fails
        with patch.object(AgentProviderFactory, 'create_provider',
                         AsyncMock(side_effect=[
                             mock_embedding,
                             Exception("Graph creation failed")
                         ])):
            
            with pytest.raises(ProviderError) as exc_info:
                await AgentProviderFactory.create_memory_providers(memory_config)
            
            # Verify cleanup was called
            mock_embedding.shutdown.assert_called_once()
            assert "Failed to create memory providers" in str(exc_info.value)
    
    def test_get_default_settings_known_provider(self):
        """Test getting default settings for known provider."""
        # Test LLM defaults
        llm_defaults = AgentProviderFactory.get_default_settings("llm", "llamacpp")
        assert "model_path" in llm_defaults
        assert "n_ctx" in llm_defaults
        assert llm_defaults["n_ctx"] == 4096
        
        # Test vector DB defaults
        vector_defaults = AgentProviderFactory.get_default_settings("vector_db", "chroma")
        assert "persist_directory" in vector_defaults
        assert "collection_name" in vector_defaults
        
        # Test graph DB defaults
        graph_defaults = AgentProviderFactory.get_default_settings("graph_db", "neo4j")
        assert "uri" in graph_defaults
        assert "username" in graph_defaults
    
    def test_get_default_settings_unknown_provider(self):
        """Test getting default settings for unknown provider."""
        # Unknown provider type
        defaults = AgentProviderFactory.get_default_settings("unknown_type", "anything")
        assert defaults == {}
        
        # Unknown provider name
        defaults = AgentProviderFactory.get_default_settings("llm", "unknown_llm")
        assert defaults == {}
    
    def test_provider_implementations_mapping(self):
        """Test provider implementations mapping structure."""
        mappings = AgentProviderFactory.PROVIDER_IMPLEMENTATIONS
        
        # Verify expected provider types
        assert "llm" in mappings
        assert "embedding" in mappings
        assert "vector_db" in mappings
        assert "graph_db" in mappings
        assert "cache" in mappings
        assert "database" in mappings
        
        # Verify each has default
        for provider_type, implementations in mappings.items():
            assert "default" in implementations
            
        # Verify specific implementations
        assert mappings["llm"]["openai"] == "openai"
        assert mappings["llm"]["default"] == "llamacpp"
        assert mappings["vector_db"]["default"] == "chroma"
        assert mappings["graph_db"]["default"] == "memory_graph"