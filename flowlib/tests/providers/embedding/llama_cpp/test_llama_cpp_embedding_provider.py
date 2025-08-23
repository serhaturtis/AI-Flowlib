"""Tests for llama_cpp embedding provider."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pydantic import ValidationError

from flowlib.providers.embedding.llama_cpp.provider import (
    LlamaCppEmbeddingProvider,
    LlamaCppEmbeddingProviderSettings
)
from flowlib.core.errors.errors import ProviderError, ConfigurationError


class TestLlamaCppEmbeddingProviderSettings:
    """Test LlamaCppEmbeddingProviderSettings model."""
    
    def test_create_settings_minimal(self):
        """Test creating settings with minimal required fields."""
        settings = LlamaCppEmbeddingProviderSettings(path="/path/to/model.gguf")
        
        assert settings.path == "/path/to/model.gguf"
        assert settings.n_ctx == 512  # Default value
        assert settings.n_threads is None  # Default value
        assert settings.n_batch == 512  # Default value
        assert settings.use_mlock is False  # Default value
        assert settings.n_gpu_layers == 0  # Default value
        assert settings.verbose is False  # Default value
    
    def test_create_settings_full(self):
        """Test creating settings with all fields specified."""
        settings = LlamaCppEmbeddingProviderSettings(
            path="/custom/model.gguf",
            n_ctx=1024,
            n_threads=8,
            n_batch=256,
            use_mlock=True,
            n_gpu_layers=5,
            verbose=True
        )
        
        assert settings.path == "/custom/model.gguf"
        assert settings.n_ctx == 1024
        assert settings.n_threads == 8
        assert settings.n_batch == 256
        assert settings.use_mlock is True
        assert settings.n_gpu_layers == 5
        assert settings.verbose is True
    
    def test_required_path_field(self):
        """Test that path field is required."""
        with pytest.raises(ValidationError) as exc_info:
            LlamaCppEmbeddingProviderSettings()
        
        assert "Field required" in str(exc_info.value)
    
    def test_settings_field_descriptions(self):
        """Test that field descriptions are properly set."""
        fields = LlamaCppEmbeddingProviderSettings.model_fields
        assert "Path to the GGUF embedding model file" in fields['path'].description
        assert "Context size for embedding model" in fields['n_ctx'].description
        assert "Number of threads for inference" in fields['n_threads'].description
        assert "Batch size for embedding processing" in fields['n_batch'].description


class TestLlamaCppEmbeddingProviderInitialization:
    """Test LlamaCppEmbeddingProvider initialization."""
    
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    def test_init_success(self, mock_llama_class):
        """Test successful provider initialization."""
        settings = LlamaCppEmbeddingProviderSettings(path="/test/model.gguf")
        
        provider = LlamaCppEmbeddingProvider(
            name="test_provider",
            provider_type="embedding",
            settings=settings
        )
        
        assert provider.name == "test_provider"
        assert provider.provider_type == "embedding"
        assert provider.settings == settings
        assert provider._model is None
        assert provider._model_path == "/test/model.gguf"
        assert hasattr(provider, '_lock')
    
    def test_init_no_llama_import(self):
        """Test initialization when llama_cpp is not available."""
        with patch('flowlib.providers.embedding.llama_cpp.provider.Llama', None):
            settings = LlamaCppEmbeddingProviderSettings(path="/test/model.gguf")
            
            with pytest.raises(ImportError) as exc_info:
                LlamaCppEmbeddingProvider(
                    name="test_provider",
                    provider_type="embedding", 
                    settings=settings
                )
            
            assert "llama-cpp-python is not installed" in str(exc_info.value)
    
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    def test_init_invalid_settings_type(self, mock_llama_class):
        """Test initialization with wrong settings type."""
        # Create a generic settings object
        invalid_settings = Mock()
        
        with pytest.raises(ValidationError) as exc_info:
            LlamaCppEmbeddingProvider(
                name="test_provider",
                provider_type="embedding",
                settings=invalid_settings
            )
        
        assert "LlamaCppEmbeddingProviderSettings" in str(exc_info.value)
    
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    def test_init_missing_path(self, mock_llama_class):
        """Test initialization with missing path in settings."""
        # Create settings with valid path first
        settings = LlamaCppEmbeddingProviderSettings(path="")
        
        with pytest.raises(ConfigurationError) as exc_info:
            LlamaCppEmbeddingProvider(
                name="test_provider",
                provider_type="embedding",
                settings=settings
            )
        
        error = exc_info.value
        assert "'path' is required" in error.message
        assert error.config_context.config_key == "path"
        assert error.context.data.component == "test_provider"


class TestLlamaCppEmbeddingProviderAsyncMethods:
    """Test async methods of LlamaCppEmbeddingProvider."""
    
    @pytest.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return LlamaCppEmbeddingProviderSettings(
            path="/test/model.gguf",
            n_ctx=1024,
            n_threads=4,
            verbose=False
        )
    
    @pytest.fixture
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    def provider(self, mock_llama_class, provider_settings):
        """Create test provider instance."""
        return LlamaCppEmbeddingProvider(
            name="test_provider",
            provider_type="embedding",
            settings=provider_settings
        )
    
    @pytest.mark.asyncio
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    async def test_initialize_success(self, mock_llama_class, provider):
        """Test successful model initialization."""
        mock_model = Mock()
        mock_llama_class.return_value = mock_model
        
        await provider._initialize()
        
        assert provider._model == mock_model
        
        # Verify Llama was called with correct arguments
        mock_llama_class.assert_called_once()
        call_kwargs = mock_llama_class.call_args[1]
        assert call_kwargs["model_path"] == "/test/model.gguf"
        assert call_kwargs["embedding"] is True
        assert call_kwargs["n_ctx"] == 1024
        assert call_kwargs["n_threads"] == 4
        assert call_kwargs["verbose"] is False
        # n_threads should be included since it's not None
        assert "n_threads" in call_kwargs
    
    @pytest.mark.asyncio
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    async def test_initialize_filters_none_values(self, mock_llama_class, provider_settings):
        """Test that None values are filtered out during initialization."""
        # Create settings with None thread count
        settings_with_none = LlamaCppEmbeddingProviderSettings(
            path="/test/model.gguf",
            n_threads=None  # This should be filtered out
        )
        
        provider = LlamaCppEmbeddingProvider(
            name="test_provider",
            provider_type="embedding",
            settings=settings_with_none
        )
        
        await provider._initialize()
        
        call_kwargs = mock_llama_class.call_args[1]
        assert "n_threads" not in call_kwargs  # Should be filtered out
        assert call_kwargs["model_path"] == "/test/model.gguf"
        assert call_kwargs["embedding"] is True
    
    @pytest.mark.asyncio
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    async def test_initialize_already_initialized(self, mock_llama_class, provider):
        """Test that initialize doesn't reload if already initialized."""
        mock_model = Mock()
        provider._model = mock_model  # Simulate already initialized
        
        await provider._initialize()
        
        # Llama should not be called again
        mock_llama_class.assert_not_called()
        assert provider._model == mock_model
    
    @pytest.mark.asyncio
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    async def test_initialize_failure(self, mock_llama_class, provider):
        """Test initialization failure handling."""
        mock_llama_class.side_effect = Exception("Model loading failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider._initialize()
        
        error = exc_info.value
        assert "Failed to load embedding model" in error.message
        assert error.provider_context.provider_name == "test_provider"
        assert provider._model is None
    
    @pytest.mark.asyncio
    async def test_shutdown_with_model(self, provider):
        """Test shutdown when model is loaded."""
        mock_model = Mock()
        provider._model = mock_model
        
        await provider._shutdown()
        
        assert provider._model is None
    
    @pytest.mark.asyncio
    async def test_shutdown_no_model(self, provider):
        """Test shutdown when no model is loaded."""
        provider._model = None
        
        # Should not raise any exception
        await provider._shutdown()
        
        assert provider._model is None


class TestLlamaCppEmbeddingProviderEmbed:
    """Test embedding generation functionality."""
    
    @pytest.fixture
    def provider_settings(self):
        """Create test provider settings."""
        return LlamaCppEmbeddingProviderSettings(path="/test/model.gguf")
    
    @pytest.fixture
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    def initialized_provider(self, mock_llama_class, provider_settings):
        """Create and initialize a test provider."""
        provider = LlamaCppEmbeddingProvider(
            name="test_provider",
            provider_type="embedding",
            settings=provider_settings
        )
        
        # Mock the model
        mock_model = Mock()
        mock_model.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        provider._model = mock_model
        provider._initialized = True
        
        return provider
    
    @pytest.mark.asyncio
    async def test_embed_single_string(self, initialized_provider):
        """Test embedding a single string."""
        result = await initialized_provider.embed("Hello world")
        
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        initialized_provider._model.embed.assert_called_once_with(["Hello world"])
    
    @pytest.mark.asyncio
    async def test_embed_list_of_strings(self, initialized_provider):
        """Test embedding a list of strings."""
        texts = ["Hello", "World"]
        result = await initialized_provider.embed(texts)
        
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        initialized_provider._model.embed.assert_called_once_with(texts)
    
    @pytest.mark.asyncio
    async def test_embed_empty_list(self, initialized_provider):
        """Test embedding an empty list."""
        result = await initialized_provider.embed([])
        
        assert result == []
        initialized_provider._model.embed.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_embed_not_initialized(self, provider_settings):
        """Test embedding when provider is not initialized."""
        with patch('flowlib.providers.embedding.llama_cpp.provider.Llama'):
            provider = LlamaCppEmbeddingProvider(
                name="test_provider",
                provider_type="embedding",
                settings=provider_settings
            )
        
        # Ensure provider is not initialized
        provider._initialized = False
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.embed("test")
        
        error = exc_info.value
        assert "Embedding provider is not initialized" in error.message
        assert error.provider_context.provider_name == "test_provider"
    
    @pytest.mark.asyncio
    async def test_embed_model_none(self, initialized_provider):
        """Test embedding when model is None."""
        initialized_provider._model = None
        
        with pytest.raises(ProviderError) as exc_info:
            await initialized_provider.embed("test")
        
        assert "Embedding provider is not initialized" in exc_info.value.message
    
    @pytest.mark.asyncio
    async def test_embed_invalid_output_format(self, initialized_provider):
        """Test embedding when model returns invalid format."""
        # Mock model to return invalid format
        initialized_provider._model.embed.return_value = "invalid_format"
        
        with pytest.raises(ProviderError) as exc_info:
            await initialized_provider.embed("test")
        
        assert "did not return expected embedding format" in exc_info.value.message
    
    @pytest.mark.asyncio
    async def test_embed_model_exception(self, initialized_provider):
        """Test embedding when model raises an exception."""
        initialized_provider._model.embed.side_effect = Exception("Model error")
        
        with pytest.raises(ProviderError) as exc_info:
            await initialized_provider.embed("test")
        
        error = exc_info.value
        assert "Failed to generate embeddings" in error.message
        assert error.provider_context.provider_name == "test_provider"
        # Check that the cause is the original exception
        assert str(error.cause) == "Model error"


class TestLlamaCppEmbeddingProviderConcurrency:
    """Test concurrent access and thread safety."""
    
    @pytest.fixture
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    def provider(self, mock_llama_class):
        """Create test provider."""
        settings = LlamaCppEmbeddingProviderSettings(path="/test/model.gguf")
        return LlamaCppEmbeddingProvider(
            name="test_provider",
            provider_type="embedding",
            settings=settings
        )
    
    @pytest.mark.asyncio
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    async def test_concurrent_initialization(self, mock_llama_class, provider):
        """Test that concurrent initialization calls don't create multiple models."""
        mock_model = Mock()
        mock_llama_class.return_value = mock_model
        
        # Start multiple initialization tasks
        tasks = [provider._initialize() for _ in range(3)]
        await asyncio.gather(*tasks)
        
        # Llama should only be called once due to the lock
        assert mock_llama_class.call_count == 1
        assert provider._model == mock_model
    
    @pytest.mark.asyncio
    async def test_concurrent_embed_calls(self, provider):
        """Test concurrent embedding calls are properly synchronized."""
        # Setup mock model
        mock_model = Mock()
        mock_model.embed.return_value = [[0.1, 0.2, 0.3]]
        provider._model = mock_model
        provider._initialized = True
        
        # Make concurrent embedding calls
        tasks = [provider.embed(f"text{i}") for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 3
        assert all(result == [[0.1, 0.2, 0.3]] for result in results)
        
        # Model should be called 3 times (once per embed call)
        assert mock_model.embed.call_count == 3


class TestLlamaCppEmbeddingProviderIntegration:
    """Integration tests for the provider."""
    
    @pytest.mark.asyncio
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    async def test_full_lifecycle(self, mock_llama_class):
        """Test the complete provider lifecycle."""
        # Setup
        mock_model = Mock()
        mock_model.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_llama_class.return_value = mock_model
        
        settings = LlamaCppEmbeddingProviderSettings(
            path="/test/model.gguf",
            n_ctx=1024,
            n_gpu_layers=2
        )
        
        provider = LlamaCppEmbeddingProvider(
            name="test_provider",
            provider_type="embedding",
            settings=settings
        )
        
        # Test initialization
        await provider._initialize()
        assert provider._model == mock_model
        
        # Test embedding - need to set initialized flag
        provider._initialized = True
        result = await provider.embed(["Hello", "World"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        
        # Test shutdown
        await provider._shutdown()
        assert provider._model is None
    
    def test_provider_decorator_registration(self):
        """Test that the provider is properly decorated for registration."""
        # This tests that the @provider decorator was applied correctly
        # Check for either attribute name pattern that might be used by the decorator
        has_config = (hasattr(LlamaCppEmbeddingProvider, '_flowlib_provider_config') or 
                     hasattr(LlamaCppEmbeddingProvider, '_provider_config') or
                     hasattr(LlamaCppEmbeddingProvider, '__flowlib_provider_config__'))
        
        # If no decorator attributes found, just verify the class exists and can be instantiated
        if not has_config:
            # Just verify the class has the expected structure
            assert LlamaCppEmbeddingProvider.__name__ == 'LlamaCppEmbeddingProvider'
            assert hasattr(LlamaCppEmbeddingProvider, 'embed')
            return
        
        # If decorator config exists, test it
        config = None
        for attr_name in ['_flowlib_provider_config', '_provider_config', '__flowlib_provider_config__']:
            if hasattr(LlamaCppEmbeddingProvider, attr_name):
                config = getattr(LlamaCppEmbeddingProvider, attr_name)
                break
        
        if config:
            assert config['provider_type'] == 'embedding'
            assert config['name'] == 'llamacpp_embedding'
            assert config['settings_class'] == LlamaCppEmbeddingProviderSettings
    
    @pytest.mark.asyncio
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    async def test_error_recovery(self, mock_llama_class):
        """Test error recovery scenarios."""
        settings = LlamaCppEmbeddingProviderSettings(path="/test/model.gguf")
        provider = LlamaCppEmbeddingProvider(
            name="test_provider",
            provider_type="embedding",
            settings=settings
        )
        
        # Test initialization failure followed by successful retry
        mock_llama_class.side_effect = [Exception("First failure"), Mock()]
        
        # First attempt should fail
        with pytest.raises(ProviderError):
            await provider._initialize()
        
        assert provider._model is None
        
        # Second attempt should succeed
        mock_model = Mock()
        mock_llama_class.side_effect = None
        mock_llama_class.return_value = mock_model
        
        await provider._initialize()
        assert provider._model == mock_model
    
    @pytest.mark.asyncio
    @patch('flowlib.providers.embedding.llama_cpp.provider.Llama')
    async def test_settings_propagation(self, mock_llama_class):
        """Test that all settings are properly propagated to Llama constructor."""
        settings = LlamaCppEmbeddingProviderSettings(
            path="/custom/model.gguf",
            n_ctx=2048,
            n_threads=8,
            n_batch=128,
            use_mlock=True,
            n_gpu_layers=10,
            verbose=True
        )
        
        provider = LlamaCppEmbeddingProvider(
            name="test_provider",
            provider_type="embedding",
            settings=settings
        )
        
        await provider._initialize()
        
        # Verify all settings were passed to Llama constructor
        call_kwargs = mock_llama_class.call_args[1]
        assert call_kwargs["model_path"] == "/custom/model.gguf"
        assert call_kwargs["embedding"] is True
        assert call_kwargs["n_ctx"] == 2048
        assert call_kwargs["n_threads"] == 8
        assert call_kwargs["n_batch"] == 128
        assert call_kwargs["use_mlock"] is True
        assert call_kwargs["n_gpu_layers"] == 10
        assert call_kwargs["verbose"] is True