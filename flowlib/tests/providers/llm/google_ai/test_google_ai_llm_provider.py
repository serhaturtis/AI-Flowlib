"""Tests for Google AI provider."""
import pytest
import json
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pydantic import BaseModel

# Test both with and without google-genai installed
try:
    from google import genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

from flowlib.providers.llm.google_ai.provider import (
    GoogleAIProvider,
    GoogleAISettings,
)
from flowlib.core.errors.errors import ProviderError
from flowlib.resources.decorators.decorators import PromptTemplate
from flowlib.providers.llm.base import PromptConfigOverride


class MockOutputModel(BaseModel):
    """Mock model for structured output."""
    name: str
    age: int
    description: str


class MockPrompt:
    """Mock prompt template."""
    template: str = "Generate a person named {{name}} who is {{age}} years old."
    config: PromptConfigOverride = PromptConfigOverride(temperature=0.7)


class TestGoogleAISettings:
    """Test Google AI settings."""
    
    def test_default_settings(self):
        """Test default Google AI settings."""
        settings = GoogleAISettings(api_key="test_key")
        
        # Test Google AI specific fields only (after architecture change to direct inheritance)
        assert settings.api_key == "test_key"
        assert settings.safety_settings is None
        assert settings.timeout == 300
        assert settings.max_concurrent_requests == 10
        
        # Test inherited provider settings
        assert settings.timeout_seconds == 60.0
        assert settings.max_retries == 3
    
    def test_custom_settings(self):
        """Test custom Google AI settings."""
        safety_settings = {"DANGEROUS_CONTENT": "BLOCK_NONE"}
        
        settings = GoogleAISettings(
            api_key="custom_key",
            safety_settings=safety_settings,
            timeout=600,
            max_concurrent_requests=20,
            timeout_seconds=120.0,
            max_retries=5
        )
        
        # Test Google AI specific fields
        assert settings.api_key == "custom_key"
        assert settings.safety_settings == safety_settings
        assert settings.timeout == 600
        assert settings.max_concurrent_requests == 20
        
        # Test inherited provider settings  
        assert settings.timeout_seconds == 120.0
        assert settings.max_retries == 5
    
    def test_settings_inheritance(self):
        """Test that GoogleAISettings inherits from ProviderSettings."""
        from flowlib.providers.core.base import ProviderSettings
        
        settings = GoogleAISettings(api_key="test_key")
        assert isinstance(settings, ProviderSettings)


class TestGoogleAIProvider:
    """Test Google AI provider."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return GoogleAISettings(
            api_key="test_api_key",
            safety_settings=None,
            timeout=300,
            max_concurrent_requests=10
        )
    
    @pytest.fixture
    def provider(self, settings):
        """Create test provider."""
        return GoogleAIProvider(name="test_google_ai", provider_type="llm", settings=settings)
    
    @pytest.fixture
    def mock_genai_model(self):
        """Create mock Google AI model."""
        mock = Mock()
        mock.generate_content = AsyncMock()
        mock.generate_content_async = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_genai_response(self):
        """Create mock Google AI response."""
        mock = Mock()
        mock.text = "Generated response text"
        mock.candidates = [Mock()]
        mock.candidates[0].content = Mock()
        mock.candidates[0].content.parts = [Mock()]
        mock.candidates[0].content.parts[0].text = "Generated response text"
        # Mock prompt_feedback to indicate no blocking
        mock.prompt_feedback = None  # No feedback means no blocking
        return mock
    
    def test_provider_initialization(self, settings):
        """Test provider initialization."""
        provider = GoogleAIProvider(name="test_provider", provider_type="llm", settings=settings)
        
        assert provider.name == "test_provider"
        assert provider.provider_type == "llm"
        assert provider._settings == settings
        assert provider._models == {}
    
    def test_provider_inheritance(self, provider):
        """Test that GoogleAIProvider inherits from LLMProvider."""
        from flowlib.providers.llm.base import LLMProvider
        assert isinstance(provider, LLMProvider)
    
    def test_invalid_settings_type(self):
        """Test provider with invalid settings type."""
        from flowlib.providers.core.base import ProviderSettings
        
        invalid_settings = ProviderSettings(timeout_seconds=60.0, max_retries=3)
        
        with pytest.raises(TypeError) as exc_info:
            GoogleAIProvider(name="test", provider_type="llm", settings=invalid_settings)
        
        assert "must be a GoogleAISettings instance" in str(exc_info.value)
    
    @patch('flowlib.providers.llm.google_ai.provider.GOOGLE_AI_AVAILABLE', False)
    async def test_initialize_without_google_ai(self, provider):
        """Test initialization without google-genai package."""
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "google-genai package not installed" in str(exc_info.value)
        assert exc_info.value.provider_context.provider_name == "test_google_ai"
    
    @patch('flowlib.providers.llm.google_ai.provider.GOOGLE_AI_AVAILABLE', True)
    @patch('flowlib.providers.llm.google_ai.provider.genai')
    async def test_initialize_success(self, mock_genai, provider):
        """Test successful initialization."""
        await provider.initialize()
        
        mock_genai.Client.assert_called_once_with(api_key="test_api_key")
        assert provider.initialized is True
    
    @patch('flowlib.providers.llm.google_ai.provider.GOOGLE_AI_AVAILABLE', True)
    async def test_initialize_no_api_key(self):
        """Test initialization without API key."""
        settings = GoogleAISettings(api_key="")
        provider = GoogleAIProvider(name="test_google_ai", provider_type="llm", settings=settings)
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "Google AI API key not configured" in str(exc_info.value)
    
    @patch('flowlib.providers.llm.google_ai.provider.GOOGLE_AI_AVAILABLE', True)
    @patch('flowlib.providers.llm.google_ai.provider.genai')
    async def test_initialize_genai_error(self, mock_genai, provider):
        """Test initialization with genai configuration error."""
        mock_genai.Client.side_effect = Exception("API configuration failed")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.initialize()
        
        assert "Failed to initialize Google AI provider" in str(exc_info.value)
    
    async def test_shutdown(self, provider):
        """Test provider shutdown."""
        object.__setattr__(provider, "_initialized", True)
        object.__setattr__(provider, "_genai_module", Mock())
        object.__setattr__(provider, "_models", {"test_model": {"model_instance": Mock()}})
        
        await provider.shutdown()
        
        assert provider.initialized is False
        assert provider._models == {}
    
    def test_get_or_cache_model_config_new(self, provider):
        """Test caching a new model configuration."""
        model_config = {
            "model_id": "gemini-1.5-pro",
            "temperature": 0.8,
            "max_tokens": 1000
        }
        
        result = provider._get_or_cache_model_config("test_model", model_config)
        
        assert result == "gemini-1.5-pro"
        assert "test_model" in provider._models
        assert provider._models["test_model"]["model_id"] == "gemini-1.5-pro"
        assert provider._models["test_model"]["config"] == model_config
    
    def test_get_or_cache_model_config_existing(self, provider):
        """Test retrieving existing model configuration."""
        provider._models["existing_model"] = {
            "model_id": "gemini-1.5-flash",
            "config": {"temperature": 0.5}
        }
        
        result = provider._get_or_cache_model_config("existing_model", {})
        
        assert result == "gemini-1.5-flash"
    
    def test_get_or_cache_model_config_error(self, provider):
        """Test model configuration error when model_id is missing."""
        model_config = {
            "temperature": 0.8,
            "max_tokens": 1000
            # Missing model_id
        }
        
        with pytest.raises(ValueError) as exc_info:
            provider._get_or_cache_model_config("test_model", model_config)
        
        assert "must specify 'model_id'" in str(exc_info.value)
    
    @patch('flowlib.providers.llm.google_ai.provider.types')
    def test_create_generation_config(self, mock_types, provider):
        """Test generation config creation."""
        model_config = {
            "temperature": 0.8,
            "max_tokens": 1000,
            "top_p": 0.9,
            "top_k": 40
        }
        
        provider._create_generation_config(model_config)
        
        mock_types.GenerateContentConfig.assert_called_once_with(
            temperature=0.8,
            max_output_tokens=1000,
            top_p=0.9,
            top_k=40
        )
    
    @patch('flowlib.providers.llm.google_ai.provider.types')
    def test_create_generation_config_defaults(self, mock_types, provider):
        """Test generation config with default values."""
        model_config = {}
        
        provider._create_generation_config(model_config)
        
        mock_types.GenerateContentConfig.assert_called_once_with(
            temperature=0.7,  # default value
            max_output_tokens=2048  # default value
            # top_p and top_k are None by default and filtered out
        )
    
    @patch('flowlib.providers.llm.base.resource_registry')
    @patch('flowlib.providers.llm.google_ai.provider.GOOGLE_AI_AVAILABLE', True)
    async def test_generate_success(self, mock_registry, provider, mock_genai_response):
        """Test successful text generation."""
        # Setup mocks - use object.__setattr__ for frozen models
        object.__setattr__(provider, '_initialized', True)
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_genai_response
        object.__setattr__(provider, '_client', mock_client)
        
        mock_registry.get.return_value = Mock()
        mock_registry.get.return_value.model_dump.return_value = {"model_id": "gemini-1.5-pro"}
        
        prompt = MockPrompt()
        result = await provider.generate(prompt, "test_model", {"name": "Alice", "age": "25"})
        
        assert result == "Generated response text"
        mock_client.models.generate_content.assert_called_once()
    
    @patch('flowlib.providers.llm.base.resource_registry')
    @patch('flowlib.providers.llm.google_ai.provider.GOOGLE_AI_AVAILABLE', True)
    async def test_generate_not_initialized(self, mock_registry, provider):
        """Test generation when not initialized."""
        provider._initialized = False
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.generate(MockPrompt(), "test_model")
        
        assert "Provider not initialized" in str(exc_info.value)
    
    @patch('flowlib.providers.llm.base.resource_registry')
    @patch('flowlib.providers.llm.google_ai.provider.GOOGLE_AI_AVAILABLE', True)
    async def test_generate_blocked_content(self, mock_registry, provider):
        """Test generation with blocked content via prompt_feedback."""
        object.__setattr__(provider, "_initialized", True)
        mock_client = Mock()
        object.__setattr__(provider, "_client", mock_client)
        
        mock_registry.get.return_value = Mock()
        mock_registry.get.return_value.model_dump.return_value = {"model_id": "gemini-1.5-pro"}
        
        # Mock response with blocked content
        mock_response = Mock()
        mock_response.prompt_feedback = Mock()
        mock_response.prompt_feedback.block_reason = "SAFETY"
        mock_response.prompt_feedback.block_reason_message = "Content blocked by safety filters"
        mock_client.models.generate_content.return_value = mock_response
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.generate(MockPrompt(), "test_model")
        
        assert "Prompt blocked by Google AI safety filters" in str(exc_info.value)
    
    @patch('flowlib.providers.llm.base.resource_registry')
    @patch('flowlib.providers.llm.google_ai.provider.GOOGLE_AI_AVAILABLE', True)
    async def test_generate_structured_success(self, mock_registry, provider):
        """Test successful structured generation with JSON response."""
        object.__setattr__(provider, "_initialized", True)
        mock_client = Mock()
        object.__setattr__(provider, "_client", mock_client)
        
        mock_registry.get.return_value = Mock()
        mock_registry.get.return_value.model_dump.return_value = {"model_id": "gemini-1.5-pro"}
        
        # Mock response with JSON text
        mock_response = Mock()
        mock_response.text = '{"name": "Alice", "age": 25, "description": "Test person"}'
        mock_response.prompt_feedback = None  # No blocking
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content.parts = [Mock()]  # Non-empty
        
        mock_client.models.generate_content.return_value = mock_response
        
        prompt = MockPrompt()
        result = await provider.generate_structured(prompt, MockOutputModel, "test_model")
        
        assert isinstance(result, MockOutputModel)
        assert result.name == "Alice"
        assert result.age == 25
    
    @patch('flowlib.providers.llm.base.resource_registry')
    @patch('flowlib.providers.llm.google_ai.provider.GOOGLE_AI_AVAILABLE', True)
    async def test_generate_structured_with_markdown_cleanup(self, mock_registry, provider):
        """Test structured generation with markdown JSON cleanup."""
        object.__setattr__(provider, "_initialized", True)
        mock_client = Mock()
        object.__setattr__(provider, "_client", mock_client)
        
        mock_registry.get.return_value = Mock()
        mock_registry.get.return_value.model_dump.return_value = {"model_id": "gemini-1.5-pro"}
        
        # Mock response with markdown-wrapped JSON
        mock_response = Mock()
        mock_response.text = '```json\n{"name": "Bob", "age": 30, "description": "Markdown test"}\n```'
        mock_response.prompt_feedback = None  # No blocking
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content.parts = [Mock()]  # Non-empty
        
        mock_client.models.generate_content.return_value = mock_response
        
        prompt = MockPrompt()
        result = await provider.generate_structured(prompt, MockOutputModel, "test_model")
        
        assert isinstance(result, MockOutputModel)
        assert result.name == "Bob"
        assert result.age == 30
    
    @patch('flowlib.providers.llm.base.resource_registry')
    @patch('flowlib.providers.llm.google_ai.provider.GOOGLE_AI_AVAILABLE', True)
    async def test_generate_structured_parsing_error(self, mock_registry, provider):
        """Test structured generation with JSON parsing error."""
        object.__setattr__(provider, "_initialized", True)
        mock_client = Mock()
        object.__setattr__(provider, "_client", mock_client)
        
        mock_registry.get.return_value = Mock()
        mock_registry.get.return_value.model_dump.return_value = {"model_id": "gemini-1.5-pro"}
        
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.text = "Invalid JSON response { this is not valid json"
        mock_response.prompt_feedback = None  # No blocking
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].content.parts = [Mock()]  # Non-empty
        
        mock_client.models.generate_content.return_value = mock_response
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.generate_structured(MockPrompt(), MockOutputModel, "test_model")
        
        assert "Failed to parse structured response" in str(exc_info.value)
    
    def test_provider_registration(self):
        """Test that provider is properly registered."""
        # This would be tested in integration tests
        # Here we just verify the decorator is applied
        assert hasattr(GoogleAIProvider, '__provider_type__')
        assert hasattr(GoogleAIProvider, '__provider_name__')


@pytest.mark.skipif(not GOOGLE_AI_AVAILABLE, reason="google-genai package not available")
@pytest.mark.integration
class TestGoogleAIProviderIntegration:
    """Integration tests for Google AI provider.
    
    These tests require a valid Google AI API key.
    """
    
    @pytest.fixture
    def settings(self):
        """Create integration test settings."""
        return GoogleAISettings(
            api_key="test_api_key",  # Would use real key in actual integration tests
            safety_settings=None,
            timeout=300,
            max_concurrent_requests=10
        )
    
    @pytest.fixture
    async def provider(self, settings):
        """Create and initialize provider for integration tests."""
        provider = GoogleAIProvider(name="integration_google_ai", provider_type="llm", settings=settings)
        
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
        """Test settings integration with real values."""
        assert settings.api_key == "test_api_key"
        assert settings.safety_settings is None
        assert settings.timeout == 300
        assert settings.max_concurrent_requests == 10
    
    def test_provider_creation_integration(self, settings):
        """Test provider creation with real settings."""
        provider = GoogleAIProvider(name="integration_test", provider_type="llm", settings=settings)
        
        assert provider.name == "integration_test"
        assert provider._settings.api_key == "test_api_key"
        assert not provider.initialized
    
    def test_generation_config_integration(self, settings):
        """Test generation config creation with real parameters."""
        provider = GoogleAIProvider(name="test", provider_type="llm", settings=settings)
        
        # Test with model-specific config (not provider settings)
        model_config = {
            "temperature": 0.8,
            "max_tokens": 1000,
            "top_p": 0.9,
            "top_k": 40
        }
        
        # This tests the config creation logic without requiring API calls
        with patch('flowlib.providers.llm.google_ai.provider.types') as mock_types:
            provider._create_generation_config(model_config)
            
            mock_types.GenerateContentConfig.assert_called_once()
            call_kwargs = mock_types.GenerateContentConfig.call_args[1]
            assert call_kwargs["temperature"] == 0.8
            assert call_kwargs["max_output_tokens"] == 1000
            assert call_kwargs["top_p"] == 0.9
            assert call_kwargs["top_k"] == 40