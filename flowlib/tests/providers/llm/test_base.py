"""Tests for LLM provider base class."""
import pytest
import asyncio
from typing import Any, Dict, Optional, Type
from unittest.mock import Mock, patch, AsyncMock
from pydantic import BaseModel, ValidationError

from flowlib.providers.llm.base import (
    LLMProvider,
    LLMProviderSettings,
)
from flowlib.providers.core.base import ProviderSettings
from flowlib.core.errors.errors import ProviderError
from flowlib.core.interfaces import PromptTemplate


class MockModel(BaseModel):
    """Mock Pydantic model for structured generation."""
    name: str
    age: int
    description: Optional[str] = None


class MockPrompt:
    """Mock prompt template."""
    template: str = "Hello {{name}}, you are {{age}} years old."
    config: Dict[str, Any] = {"temperature": 0.7}


class ConcreteLLMProvider(LLMProvider):
    """Concrete implementation for testing."""
    
    def __init__(self, name: str = "test_llm", settings: Optional[LLMProviderSettings] = None):
        super().__init__(name=name, provider_type="llm", settings=settings)
        # Use __dict__ to avoid Pydantic field validation
        self.__dict__["generate_calls"] = []
        self.__dict__["structured_calls"] = []
    
    async def generate(self, prompt: PromptTemplate, model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> str:
        """Mock generate implementation."""
        self.generate_calls.append({
            "prompt": prompt,
            "model_name": model_name,
            "variables": prompt_variables
        })
        return "Generated response"
    
    async def generate_structured(self, prompt: PromptTemplate, output_type: Type[BaseModel], model_name: str, prompt_variables: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Mock structured generation implementation."""
        self.structured_calls.append({
            "prompt": prompt,
            "output_type": output_type,
            "model_name": model_name,
            "variables": prompt_variables
        })
        # Return a valid instance of the output type
        if output_type == MockModel:
            return MockModel(name="Test", age=25, description="Generated")
        return output_type()


class TestLLMProviderSettings:
    """Test LLM provider settings."""
    
    def test_default_settings(self):
        """Test default LLM provider settings."""
        settings = LLMProviderSettings()
        
        # Test generation parameters
        assert settings.temperature == 0.7
        assert settings.max_tokens is None
        assert settings.top_p == 1.0
        assert settings.frequency_penalty == 0.0
        assert settings.presence_penalty == 0.0
        
        # Test token management
        assert settings.max_input_tokens is None
        assert settings.max_output_tokens is None
        
        # Test streaming settings
        assert settings.stream is False
        
        # Test advanced settings
        assert settings.stop_sequences == []
    
    def test_custom_settings(self):
        """Test custom LLM provider settings."""
        settings = LLMProviderSettings(
            temperature=0.8,
            max_tokens=2000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            max_input_tokens=4000,
            max_output_tokens=1000,
            stream=True,
            stop_sequences=["Human:", "AI:"]
        )
        
        assert settings.temperature == 0.8
        assert settings.max_tokens == 2000
        assert settings.top_p == 0.9
        assert settings.frequency_penalty == 0.1
        assert settings.presence_penalty == 0.1
        assert settings.max_input_tokens == 4000
        assert settings.max_output_tokens == 1000
        assert settings.stream is True
        assert settings.stop_sequences == ["Human:", "AI:"]
    
    def test_temperature_validation(self):
        """Test temperature validation."""
        # Valid temperatures
        LLMProviderSettings(temperature=0.0)
        LLMProviderSettings(temperature=1.0)
        LLMProviderSettings(temperature=2.0)
        
        # Invalid temperatures
        with pytest.raises(ValidationError):
            LLMProviderSettings(temperature=-0.1)
        
        with pytest.raises(ValidationError):
            LLMProviderSettings(temperature=2.1)
    
    def test_top_p_validation(self):
        """Test top_p validation."""
        # Valid top_p values
        LLMProviderSettings(top_p=0.0)
        LLMProviderSettings(top_p=0.5)
        LLMProviderSettings(top_p=1.0)
        
        # Invalid top_p values
        with pytest.raises(ValidationError):
            LLMProviderSettings(top_p=-0.1)
        
        with pytest.raises(ValidationError):
            LLMProviderSettings(top_p=1.1)
    
    def test_settings_inheritance(self):
        """Test that LLMProviderSettings inherits from ProviderSettings."""
        settings = LLMProviderSettings()
        assert isinstance(settings, ProviderSettings)
        
        # Should have base provider settings
        assert hasattr(settings, 'api_key')
        assert hasattr(settings, 'timeout_seconds')
        assert hasattr(settings, 'max_retries')
        assert hasattr(settings, 'retry_delay_seconds')


class TestLLMProvider:
    """Test LLM provider base class."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return LLMProviderSettings(
            temperature=0.8,
            max_tokens=1000,
            top_p=0.9
        )
    
    @pytest.fixture
    def provider(self, settings):
        """Create test provider."""
        return ConcreteLLMProvider(name="test_provider", settings=settings)
    
    @pytest.fixture
    def prompt_template(self):
        """Create test prompt template."""
        return MockPrompt()
    
    def test_provider_initialization(self, settings):
        """Test provider initialization."""
        provider = ConcreteLLMProvider(name="test_llm", settings=settings)
        
        assert provider.name == "test_llm"
        assert provider.provider_type == "llm"
        assert provider.settings == settings
        assert provider._initialized is False
        assert provider._models == {}
    
    def test_provider_inheritance(self, provider):
        """Test that LLMProvider inherits from Provider."""
        from flowlib.providers.core.base import Provider
        assert isinstance(provider, Provider)
    
    async def test_initialize(self, provider):
        """Test provider initialization."""
        assert provider.initialized is False
        
        await provider.initialize()
        
        assert provider.initialized is True
        assert provider._initialized is True
    
    async def test_shutdown(self, provider):
        """Test provider shutdown."""
        await provider.initialize()
        assert provider.initialized is True
        
        await provider.shutdown()
        
        assert provider.initialized is False
        assert provider._initialized is False
    
    @patch('flowlib.providers.llm.base.resource_registry')
    async def test_get_model_config_success(self, mock_registry, provider):
        """Test successful model config retrieval."""
        mock_config = {"temperature": 0.7, "max_tokens": 1000}
        mock_registry.get.return_value = mock_config
        
        config = await provider.get_model_config("test_model")
        
        assert config == mock_config
        mock_registry.get.assert_called_once_with("test_model")
    
    @patch('flowlib.providers.llm.base.resource_registry')
    async def test_get_model_config_class_instance(self, mock_registry, provider):
        """Test model config with class that needs instantiation."""
        class MockModelConfig:
            def __init__(self):
                self.temperature = 0.7
        
        mock_registry.get.return_value = MockModelConfig
        
        config = await provider.get_model_config("test_model")
        
        assert isinstance(config, MockModelConfig)
        assert config.temperature == 0.7
    
    @patch('flowlib.providers.llm.base.resource_registry')
    async def test_get_model_config_error(self, mock_registry, provider):
        """Test model config retrieval error."""
        mock_registry.get.side_effect = KeyError("Model not found")
        
        with pytest.raises(ProviderError) as exc_info:
            await provider.get_model_config("nonexistent_model")
        
        assert "Error retrieving model configuration" in str(exc_info.value)
        assert "nonexistent_model" in str(exc_info.value)
    
    async def test_generate_abstract_method(self):
        """Test that generate is abstract in base class."""
        # Create a minimal concrete provider that only implements abstract methods with NotImplementedError
        class AbstractTestProvider(LLMProvider):
            pass
        
        # This should fail during instantiation due to missing settings type
        with pytest.raises(TypeError, match="must specify settings type"):
            AbstractTestProvider(name="abstract", provider_type="llm")
    
    async def test_generate_structured_abstract_method(self):
        """Test that generate_structured is abstract in concrete provider."""
        # Use the concrete provider but test that methods raise NotImplementedError if not overridden
        class IncompleteProvider(LLMProvider):
            def __init__(self, name: str = "incomplete"):
                # Provide settings directly to avoid generic type discovery
                settings = LLMProviderSettings()
                super().__init__(name=name, provider_type="llm", settings=settings)
        
        provider = IncompleteProvider()
        prompt = MockPrompt()
        
        with pytest.raises(NotImplementedError):
            await provider.generate_structured(prompt, MockModel, "test_model")
    
    async def test_concrete_generate(self, provider, prompt_template):
        """Test concrete implementation of generate."""
        result = await provider.generate(prompt_template, "test_model", {"name": "Alice", "age": "30"})
        
        assert result == "Generated response"
        assert len(provider.generate_calls) == 1
        
        call = provider.generate_calls[0]
        assert call["prompt"] == prompt_template
        assert call["model_name"] == "test_model"
        assert call["variables"] == {"name": "Alice", "age": "30"}
    
    async def test_concrete_generate_structured(self, provider, prompt_template):
        """Test concrete implementation of generate_structured."""
        result = await provider.generate_structured(prompt_template, MockModel, "test_model", {"name": "Bob"})
        
        assert isinstance(result, MockModel)
        assert result.name == "Test"
        assert result.age == 25
        assert len(provider.structured_calls) == 1
        
        call = provider.structured_calls[0]
        assert call["output_type"] == MockModel
    
    def test_format_template_basic(self, provider):
        """Test basic template formatting."""
        template = "Hello {{name}}, you are {{age}} years old."
        variables = {"name": "Alice", "age": "30"}
        kwargs = {"variables": variables}
        
        result = provider.format_template(template, kwargs)
        
        assert result == "Hello Alice, you are 30 years old."
    
    def test_format_template_missing_variables(self, provider):
        """Test template formatting with missing variables."""
        template = "Hello {{name}}, you are {{age}} years old. City: {{city}}"
        variables = {"name": "Alice", "age": "30"}
        kwargs = {"variables": variables}
        
        result = provider.format_template(template, kwargs)
        
        # Should replace available variables, leave missing ones
        assert result == "Hello Alice, you are 30 years old. City: {{city}}"
    
    def test_format_template_no_variables(self, provider):
        """Test template formatting without variables."""
        template = "Hello world!"
        kwargs = {"variables": {}}
        
        result = provider.format_template(template, kwargs)
        
        assert result == "Hello world!"
    
    def test_format_template_invalid_variables(self, provider):
        """Test template formatting with invalid variables format."""
        template = "Hello {{name}}"
        kwargs = {"variables": "not_a_dict"}
        
        # Should raise TypeError with strict validation
        with pytest.raises(TypeError, match="Variables must be a dict"):
            provider.format_template(template, kwargs)
    
    def test_format_prompt_basic(self, provider):
        """Test basic prompt formatting."""
        prompt = "What is the weather like?"
        
        result = provider._format_prompt(prompt)
        
        assert result == prompt
    
    def test_format_prompt_with_output_type(self, provider):
        """Test prompt formatting with output type."""
        prompt = "Generate a person profile."
        
        result = provider._format_prompt(prompt, output_type=MockModel)
        
        assert "Generate a person profile." in result
        assert "JSON object" in result
        assert "structure" in result
    
    def test_format_prompt_with_model_type(self, provider):
        """Test prompt formatting with model type."""
        prompt = "Test prompt"
        
        result = provider._format_prompt(prompt, model_type="custom_model")
        
        # Base implementation should return unchanged
        assert result == prompt
    
    @patch('flowlib.providers.llm.base.model_to_simple_json_schema')
    def test_format_prompt_schema_error(self, mock_schema, provider):
        """Test prompt formatting when schema generation fails."""
        mock_schema.side_effect = Exception("Schema error")
        prompt = "Generate data"
        
        # Should raise exception with strict contract enforcement
        with pytest.raises(Exception, match="Schema error"):
            provider._format_prompt(prompt, output_type=MockModel)
    
    def test_get_model_templates_default(self, provider):
        """Test default model templates."""
        templates = provider._get_model_templates()
        
        assert "default" in templates
        assert templates["default"]["pre_prompt"] == ""
        assert templates["default"]["post_prompt"] == ""
    
    def test_provider_properties(self, provider):
        """Test provider properties."""
        # Test initialized property
        assert provider.initialized is False
        
        provider._initialized = True
        assert provider.initialized is True
    
    async def test_provider_lifecycle(self, provider):
        """Test complete provider lifecycle."""
        # Initial state
        assert not provider.initialized
        
        # Initialize
        await provider.initialize()
        assert provider.initialized
        
        # Use provider
        prompt = MockPrompt()
        result = await provider.generate(prompt, "test_model")
        assert result == "Generated response"
        
        # Shutdown
        await provider.shutdown()
        assert not provider.initialized


class TestLLMProviderIntegration:
    """Integration tests for LLM provider."""
    
    @pytest.fixture
    def provider(self):
        """Create provider for integration tests."""
        settings = LLMProviderSettings(
            temperature=0.7,
            max_tokens=1000,
            stream=False
        )
        return ConcreteLLMProvider(name="integration_provider", settings=settings)
    
    async def test_end_to_end_generation(self, provider):
        """Test end-to-end generation flow."""
        await provider.initialize()
        
        # Create prompt template
        prompt = MockPrompt()
        
        # Test generation
        result = await provider.generate(
            prompt, 
            "test_model", 
            {"name": "Integration", "age": "25"}
        )
        
        assert result == "Generated response"
        assert len(provider.generate_calls) == 1
        
        await provider.shutdown()
    
    async def test_end_to_end_structured_generation(self, provider):
        """Test end-to-end structured generation flow."""
        await provider.initialize()
        
        # Create prompt template
        prompt = MockPrompt()
        
        # Test structured generation
        result = await provider.generate_structured(
            prompt,
            MockModel,
            "test_model",
            {"name": "Structured", "age": "30"}
        )
        
        assert isinstance(result, MockModel)
        assert result.name == "Test"
        assert len(provider.structured_calls) == 1
        
        await provider.shutdown()
    
    def test_template_formatting_integration(self, provider):
        """Test template formatting with real scenarios."""
        scenarios = [
            {
                "template": "Write a story about {{character}} who is {{age}} years old.",
                "variables": {"character": "Alice", "age": "25"},
                "expected": "Write a story about Alice who is 25 years old."
            },
            {
                "template": "Generate {{count}} {{item_type}} for {{purpose}}.",
                "variables": {"count": "5", "item_type": "ideas", "purpose": "brainstorming"},
                "expected": "Generate 5 ideas for brainstorming."
            },
            {
                "template": "No variables here!",
                "variables": {},
                "expected": "No variables here!"
            }
        ]
        
        for scenario in scenarios:
            result = provider.format_template(
                scenario["template"],
                {"variables": scenario["variables"]}
            )
            assert result == scenario["expected"]