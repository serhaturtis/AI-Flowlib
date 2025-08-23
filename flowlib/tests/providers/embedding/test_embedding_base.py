"""Tests for embedding provider base classes."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Union

from flowlib.providers.embedding.base import EmbeddingProvider, EmbeddingProviderSettings
from flowlib.core.errors.errors import ProviderError, ErrorContext, ConfigurationError
from flowlib.core.errors.models import ProviderErrorContext


class ConcreteEmbeddingProvider(EmbeddingProvider[EmbeddingProviderSettings]):
    """Concrete implementation for testing."""
    
    def __init__(self, name: str = "test_embedding", settings=None):
        if settings is None:
            settings = EmbeddingProviderSettings()
        super().__init__(name=name, settings=settings, provider_type="embedding")
        self._initialized = False
        self._embedding_dimension = 384  # Common dimension size
    
    async def initialize(self):
        """Initialize the embedding provider."""
        self._initialized = True
    
    async def shutdown(self):
        """Shutdown the embedding provider."""
        self._initialized = False
    
    async def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """Generate mock embeddings."""
        if not self._initialized:
            raise ProviderError(
                message="Provider not initialized",
                context=ErrorContext.create(
                    flow_name="embedding_provider_base",
                    error_type="StateError",
                    error_location="embed",
                    component=self.name,
                    operation="embedding_generation"
                ),
                provider_context=ProviderErrorContext(
                    provider_name=self.name,
                    provider_type="embedding",
                    operation="embedding_generation",
                    retry_count=0
                )
            )
        
        # Handle single string input
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Generate mock embeddings (normalized random values)
        embeddings = []
        for i, txt in enumerate(texts):
            # Create a simple mock embedding based on text content and index
            # This is deterministic for testing purposes and guarantees uniqueness
            # Use full hash value to ensure uniqueness across different texts
            text_hash = abs(hash(txt))
            text_seed = i * 1000 + (text_hash % 1000)  # Larger range to avoid collisions
            embedding = [float(text_seed + j * 0.001) for j in range(self._embedding_dimension)]
            embeddings.append(embedding)
        
        return embeddings


class TestEmbeddingProvider:
    """Test EmbeddingProvider base class."""
    
    def test_initialization_default(self):
        """Test provider initialization with defaults."""
        provider = ConcreteEmbeddingProvider()
        
        assert provider.name == "test_embedding"
        assert provider.provider_type == "embedding"
        assert provider._initialized is False
    
    def test_initialization_custom(self):
        """Test provider initialization with custom name."""
        provider = ConcreteEmbeddingProvider("custom_embedding")
        
        assert provider.name == "custom_embedding"
        assert provider.provider_type == "embedding"
    
    @pytest.mark.asyncio
    async def test_initialization_lifecycle(self):
        """Test provider initialization lifecycle."""
        provider = ConcreteEmbeddingProvider()
        
        # Initially not initialized
        assert provider._initialized is False
        
        # Initialize
        await provider.initialize()
        assert provider._initialized is True
        
        # Shutdown
        await provider.shutdown()
        assert provider._initialized is False
    
    @pytest.mark.asyncio
    async def test_embed_single_string(self):
        """Test embedding generation for single string."""
        provider = ConcreteEmbeddingProvider()
        await provider.initialize()
        
        text = "Hello world"
        embeddings = await provider.embed(text)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) == 384  # Expected dimension
        assert all(isinstance(val, float) for val in embeddings[0])
    
    @pytest.mark.asyncio
    async def test_embed_multiple_strings(self):
        """Test embedding generation for multiple strings."""
        provider = ConcreteEmbeddingProvider()
        await provider.initialize()
        
        texts = ["Hello world", "How are you?", "AI embeddings"]
        embeddings = await provider.embed(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(val, float) for val in embedding)
    
    @pytest.mark.asyncio
    async def test_embed_empty_string(self):
        """Test embedding generation for empty string."""
        provider = ConcreteEmbeddingProvider()
        await provider.initialize()
        
        text = ""
        embeddings = await provider.embed(text)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) == 384
    
    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        """Test embedding generation for empty list."""
        provider = ConcreteEmbeddingProvider()
        await provider.initialize()
        
        texts = []
        embeddings = await provider.embed(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 0
    
    @pytest.mark.asyncio
    async def test_embed_not_initialized(self):
        """Test embedding generation when provider not initialized."""
        provider = ConcreteEmbeddingProvider()
        
        with pytest.raises(ProviderError, match="Provider not initialized"):
            await provider.embed("test text")
    
    @pytest.mark.asyncio
    async def test_embed_deterministic_output(self):
        """Test that embedding generation is deterministic for same input."""
        provider = ConcreteEmbeddingProvider()
        await provider.initialize()
        
        text = "test text"
        
        # Generate embeddings twice
        embeddings1 = await provider.embed(text)
        embeddings2 = await provider.embed(text)
        
        # Should be identical
        assert embeddings1 == embeddings2
    
    @pytest.mark.asyncio
    async def test_embed_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        provider = ConcreteEmbeddingProvider()
        await provider.initialize()
        
        text1 = "Hello world"
        text2 = "Goodbye world"
        
        embeddings1 = await provider.embed(text1)
        embeddings2 = await provider.embed(text2)
        
        # Should be different
        assert embeddings1 != embeddings2
        assert embeddings1[0] != embeddings2[0]
    
    @pytest.mark.asyncio
    async def test_embed_single_vs_batch_consistency(self):
        """Test that single embedding matches batch embedding."""
        provider = ConcreteEmbeddingProvider()
        await provider.initialize()
        
        text = "consistency test"
        
        # Single embedding
        single_embedding = await provider.embed(text)
        
        # Batch embedding with one item
        batch_embedding = await provider.embed([text])
        
        # Should be identical
        assert single_embedding == batch_embedding
    
    @pytest.mark.asyncio
    async def test_embed_large_batch(self):
        """Test embedding generation for large batch."""
        provider = ConcreteEmbeddingProvider()
        await provider.initialize()
        
        # Generate a large batch of texts
        texts = [f"text number {i}" for i in range(100)]
        embeddings = await provider.embed(texts)
        
        assert len(embeddings) == 100
        
        # Verify each embedding
        for i, embedding in enumerate(embeddings):
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(val, float) for val in embedding)
        
        # Verify embeddings are different
        unique_embeddings = set(tuple(emb) for emb in embeddings)
        assert len(unique_embeddings) == 100  # All should be unique
    
    @pytest.mark.asyncio
    async def test_embed_special_characters(self):
        """Test embedding generation for text with special characters."""
        provider = ConcreteEmbeddingProvider()
        await provider.initialize()
        
        texts = [
            "Hello! @#$%^&*()",
            "Unicode: café, naïve, résumé",
            "Numbers: 12345",
            "Mixed: abc123!@#"
        ]
        
        embeddings = await provider.embed(texts)
        
        assert len(embeddings) == 4
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(val, float) for val in embedding)


class AbstractEmbeddingProvider(EmbeddingProvider):
    """Abstract provider implementation to test NotImplementedError."""
    
    def __init__(self, name: str = "abstract_embedding", settings=None):
        if settings is None:
            settings = EmbeddingProviderSettings()
        super().__init__(name=name, settings=settings, provider_type="embedding")


class TestAbstractMethods:
    """Test that abstract methods raise NotImplementedError."""
    
    @pytest.mark.asyncio
    async def test_abstract_embed(self):
        """Test that embed method raises NotImplementedError."""
        provider = AbstractEmbeddingProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement 'embed'"):
            await provider.embed("test text")
    
    @pytest.mark.asyncio
    async def test_abstract_embed_with_list(self):
        """Test that embed method raises NotImplementedError with list input."""
        provider = AbstractEmbeddingProvider()
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement 'embed'"):
            await provider.embed(["test", "text"])


class TestEmbeddingProviderIntegration:
    """Integration tests for embedding provider functionality."""
    
    @pytest.mark.asyncio
    async def test_provider_context_manager_like_usage(self):
        """Test provider usage in context-manager-like pattern."""
        provider = ConcreteEmbeddingProvider()
        
        try:
            await provider.initialize()
            
            # Use the provider
            text = "context manager test"
            embeddings = await provider.embed(text)
            
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 384
            
        finally:
            await provider.shutdown()
            
        # Verify shutdown
        assert provider._initialized is False
    
    @pytest.mark.asyncio
    async def test_multiple_embed_calls(self):
        """Test multiple consecutive embed calls."""
        provider = ConcreteEmbeddingProvider()
        await provider.initialize()
        
        try:
            # Multiple calls should all work
            results = []
            for i in range(5):
                text = f"call number {i}"
                embedding = await provider.embed(text)
                results.append(embedding)
            
            # Verify all calls succeeded
            assert len(results) == 5
            for result in results:
                assert len(result) == 1  # Single text input
                assert len(result[0]) == 384  # Expected dimension
            
            # Verify results are different
            unique_results = set(tuple(tuple(emb) for emb in result) for result in results)
            assert len(unique_results) == 5
            
        finally:
            await provider.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling_during_embedding(self):
        """Test error handling when embedding fails."""
        class FailingEmbeddingProvider(EmbeddingProvider):
            def __init__(self):
                settings = EmbeddingProviderSettings()
                super().__init__(name="failing", settings=settings, provider_type="embedding")
            
            async def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
                raise Exception("Embedding failed")
        
        provider = FailingEmbeddingProvider()
        
        with pytest.raises(Exception, match="Embedding failed"):
            await provider.embed("test text")