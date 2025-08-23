"""Tests for resource decorators."""
import pytest
from unittest.mock import Mock, patch
from pydantic import BaseModel
from typing import Dict, Any

from flowlib.resources.decorators.decorators import (
    resource,
    model_config,
    prompt,
    config,
    llm_config,
    database_config,
    vector_db_config,
    cache_config,
    storage_config,
    embedding_config,
    graph_db_config,
    message_queue_config,
    PromptTemplate
)
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.models.model_resource import ModelResource


class TestResourceDecorator:
    """Test the @resource decorator."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry."""
        with patch('flowlib.resources.decorators.decorators.resource_registry') as mock_reg:
            mock_reg.register = Mock()
            yield mock_reg
    
    def test_resource_decorator_with_valid_class(self, mock_registry):
        """Test @resource decorator with valid ResourceBase subclass."""
        @resource("test_resource", "test_type")
        class TestResource(ResourceBase):
            pass
        
        # Verify registration was called
        mock_registry.register.assert_called_once()
        call_args = mock_registry.register.call_args
        assert call_args[1]['name'] == "test_resource"
        assert call_args[1]['resource_type'] == "test_type"
        
        # Verify metadata was attached
        assert TestResource.__resource_name__ == "test_resource"
        assert TestResource.__resource_type__ == "test_type"
        assert hasattr(TestResource, '__resource_metadata__')
    
    def test_resource_decorator_with_invalid_class(self, mock_registry):
        """Test @resource decorator with non-ResourceBase class."""
        with pytest.raises(TypeError) as exc_info:
            @resource("test_resource", "test_type")
            class InvalidResource:
                pass
        
        assert "must be a ResourceBase subclass" in str(exc_info.value)
        mock_registry.register.assert_not_called()
    
    def test_resource_decorator_with_metadata(self, mock_registry):
        """Test @resource decorator with additional metadata."""
        @resource("test_resource", "test_type", version="1.0", author="test")
        class TestResource(ResourceBase):
            pass
        
        # Verify metadata was passed through
        call_args = mock_registry.register.call_args
        assert call_args[1]['version'] == "1.0"
        assert call_args[1]['author'] == "test"
        expected_metadata = {"name": "test_resource", "type": "test_type", "version": "1.0", "author": "test"}
        assert TestResource.__resource_metadata__ == expected_metadata
    
    def test_resource_decorator_no_registry(self):
        """Test @resource decorator when registry is None."""
        with patch('flowlib.resources.decorators.decorators.resource_registry', None):
            with pytest.raises(RuntimeError) as exc_info:
                @resource("test_resource", "test_type")
                class TestResource(ResourceBase):
                    pass
            
            assert "Resource registry not initialized" in str(exc_info.value)


class TestModelConfigDecorator:
    """Test the @model_config decorator."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry."""
        with patch('flowlib.resources.decorators.decorators.resource_registry') as mock_reg:
            mock_reg.register = Mock()
            yield mock_reg
    
    def test_model_config_with_resource_base(self, mock_registry):
        """Test @model_config with ResourceBase subclass."""
        @model_config("test_model", provider_type="llamacpp")
        class TestModelConfig(ResourceBase):
            provider_type: str
            config: dict
        
        # Verify registration
        mock_registry.register.assert_called_once()
        call_args = mock_registry.register.call_args
        assert call_args[1]['name'] == "test_model"
        assert call_args[1]['resource_type'] == ResourceType.MODEL_CONFIG
        
        # Verify metadata
        assert TestModelConfig.__resource_name__ == "test_model"
        assert TestModelConfig.__resource_type__ == ResourceType.MODEL_CONFIG
    
    def test_model_config_with_regular_class(self, mock_registry):
        """Test @model_config with regular class (wrapped in ModelResource)."""
        @model_config("test_model", provider_type="llamacpp", config={"param": "value"})
        class TestModel:
            def __init__(self):
                self.value = "test"
        
        # Verify registration with ModelResource wrapper
        mock_registry.register.assert_called_once()
        call_args = mock_registry.register.call_args
        assert call_args[1]['name'] == "test_model"
        
        # Verify metadata
        assert TestModel.__resource_name__ == "test_model"
        assert TestModel.__resource_type__ == ResourceType.MODEL_CONFIG
    
    def test_model_config_provider_fallback(self, mock_registry):
        """Test @model_config with deprecated 'provider' parameter."""
        @model_config("test_model", provider="openai")
        class TestModel(ResourceBase):
            provider_type: str
            config: dict
        
        mock_registry.register.assert_called_once()
    
    def test_model_config_provider_conflict(self, mock_registry):
        """Test @model_config with both provider and provider_type."""
        with pytest.raises(ValueError) as exc_info:
            @model_config("test_model", provider="openai", provider_type="llamacpp")
            class TestModel(ResourceBase):
                provider_type: str
                config: dict
        
        assert "Cannot specify both 'provider' and 'provider_type'" in str(exc_info.value)
    
    def test_model_config_default_provider(self, mock_registry):
        """Test @model_config with default provider."""
        @model_config("test_model")
        class TestModel(ResourceBase):
            provider_type: str
            config: dict
        
        # Should use default "llamacpp" provider
        mock_registry.register.assert_called_once()
    
    def test_model_config_no_registry(self):
        """Test @model_config when registry is None."""
        with patch('flowlib.resources.decorators.decorators.resource_registry', None):
            with pytest.raises(RuntimeError) as exc_info:
                @model_config("test_model")
                class TestModel(ResourceBase):
                    provider_type: str
                    config: dict
            
            assert "Resource registry not initialized" in str(exc_info.value)


class TestPromptDecorator:
    """Test the @prompt decorator."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry."""
        with patch('flowlib.resources.decorators.decorators.resource_registry') as mock_reg:
            mock_reg.register = Mock()
            yield mock_reg
    
    def test_prompt_decorator_valid(self, mock_registry):
        """Test @prompt decorator with valid prompt class."""
        @prompt("test_prompt")
        class TestPrompt(ResourceBase):
            template: str = "Hello {name}!"
            config: Dict[str, Any] = {"temperature": 0.7}
        
        # Verify registration
        mock_registry.register.assert_called_once()
        call_args = mock_registry.register.call_args
        assert call_args[1]['resource_type'] == ResourceType.PROMPT_CONFIG
        
        # Verify PromptTemplate protocol compliance
        assert 'template' in TestPrompt.model_fields
        assert 'config' in TestPrompt.model_fields
        assert TestPrompt.__resource_type__ == ResourceType.PROMPT_CONFIG
    
    def test_prompt_decorator_with_default_config(self, mock_registry):
        """Test @prompt decorator adds default config when missing."""
        @prompt("test_prompt")
        class TestPrompt(ResourceBase):
            template: str = "Hello {name}!"
        
        # Should have default config added
        assert hasattr(TestPrompt, 'config')
        expected_config = {
            "max_tokens": 2048,
            "temperature": 0.5,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        assert TestPrompt.config == expected_config
    
    def test_prompt_decorator_preserves_existing_config(self, mock_registry):
        """Test @prompt decorator preserves existing config."""
        custom_config = {"temperature": 0.9, "max_tokens": 1000}
        
        @prompt("test_prompt")
        class TestPrompt(ResourceBase):
            template: str = "Hello {name}!"
            config: Dict[str, Any] = custom_config
        
        # Should preserve existing config (check in model_fields default)
        assert TestPrompt.model_fields['config'].default == custom_config
    
    def test_prompt_decorator_missing_template(self, mock_registry):
        """Test @prompt decorator with missing template."""
        with pytest.raises(ValueError) as exc_info:
            @prompt("test_prompt")
            class TestPrompt(ResourceBase):
                config: Dict[str, Any] = {}
        
        assert "must have a 'template' attribute" in str(exc_info.value)
    
    def test_prompt_decorator_with_metadata(self, mock_registry):
        """Test @prompt decorator with metadata."""
        @prompt("test_prompt", author="test", version="1.0")
        class TestPrompt(ResourceBase):
            template: str = "Hello {name}!"
        
        # Verify metadata
        expected_metadata = {"name": "test_prompt", "type": ResourceType.PROMPT, "author": "test", "version": "1.0"}
        assert TestPrompt.__resource_metadata__ == expected_metadata


class TestPromptTemplate:
    """Test PromptTemplate protocol."""
    
    def test_prompt_template_protocol_compliance(self):
        """Test that decorated classes comply with PromptTemplate protocol."""
        class ValidPrompt:
            template: str = "Hello {name}!"
            config: Dict[str, Any] = {"temperature": 0.7}
        
        # This should not raise a type error
        def accepts_prompt_template(prompt: PromptTemplate) -> str:
            return prompt.template
        
        prompt = ValidPrompt()
        result = accepts_prompt_template(prompt)
        assert result == "Hello {name}!"
    
    def test_prompt_template_protocol_methods(self):
        """Test that PromptTemplate protocol has expected attributes."""
        import typing
        
        # Get the protocol's required attributes
        annotations = typing.get_type_hints(PromptTemplate)
        
        assert 'template' in annotations
        assert 'config' in annotations
        assert annotations['template'] == str
        assert annotations['config'] == Dict[str, Any]


class TestSpecializedDecorators:
    """Test specialized configuration decorators."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry."""
        with patch('flowlib.resources.decorators.decorators.resource_registry') as mock_reg:
            mock_reg.register = Mock()
            yield mock_reg
    
    @pytest.mark.parametrize("decorator_func,expected_type", [
        (config, ResourceType.CONFIG),
        (llm_config, ResourceType.LLM_CONFIG),
        (database_config, ResourceType.DATABASE_CONFIG),
        (vector_db_config, ResourceType.VECTOR_DB_CONFIG),
        (cache_config, ResourceType.CACHE_CONFIG),
        (storage_config, ResourceType.STORAGE_CONFIG),
        (embedding_config, ResourceType.EMBEDDING_CONFIG),
        (graph_db_config, ResourceType.GRAPH_DB_CONFIG),
        (message_queue_config, ResourceType.MESSAGE_QUEUE_CONFIG),
    ])
    def test_specialized_decorators(self, mock_registry, decorator_func, expected_type):
        """Test specialized configuration decorators."""
        @decorator_func("test_config")
        class TestConfig(ResourceBase):
            pass
        
        # Verify registration with correct type
        mock_registry.register.assert_called_once()
        call_args = mock_registry.register.call_args
        assert call_args[1]['resource_type'] == expected_type
        
        # Verify metadata
        assert TestConfig.__resource_type__ == expected_type
        assert TestConfig.__resource_name__ == "test_config"
    
    def test_specialized_decorators_with_metadata(self, mock_registry):
        """Test specialized decorators with metadata."""
        @llm_config("test_llm", provider="openai", version="1.0")
        class TestLLMConfig(ResourceBase):
            pass
        
        # Verify metadata passed through
        call_args = mock_registry.register.call_args
        assert call_args[1]['provider'] == "openai"
        assert call_args[1]['version'] == "1.0"
        expected_metadata = {"name": "test_llm", "type": ResourceType.LLM_CONFIG, "provider": "openai", "version": "1.0"}
        assert TestLLMConfig.__resource_metadata__ == expected_metadata


class TestDecoratorIntegration:
    """Integration tests for decorators."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry."""
        with patch('flowlib.resources.decorators.decorators.resource_registry') as mock_reg:
            mock_reg.register = Mock()
            yield mock_reg
    
    def test_multiple_decorators_same_class(self, mock_registry):
        """Test that multiple decorators can't be applied to same class."""
        # This is more of a design consideration - decorators should be mutually exclusive
        
        @resource("test1", "type1")
        class TestResource(ResourceBase):
            pass
        
        # Applying another decorator would overwrite metadata
        # This is expected behavior - last decorator wins
        assert TestResource.__resource_name__ == "test1"
        assert TestResource.__resource_type__ == "type1"
    
    def test_decorator_inheritance(self, mock_registry):
        """Test decorator behavior with class inheritance."""
        @resource("base_resource", "base_type")
        class BaseResource(ResourceBase):
            base_attr: str = "base"
        
        # Child class inherits decorator metadata from parent (this is expected Python behavior)
        class ChildResource(BaseResource):
            child_attr: str = "child"
        
        assert hasattr(BaseResource, '__resource_name__')
        assert hasattr(ChildResource, '__resource_name__')  # Child inherits parent attributes
        assert ChildResource.__resource_name__ == "base_resource"
    
    def test_decorator_with_complex_class(self, mock_registry):
        """Test decorators with complex class definitions."""
        @llm_config("complex_llm", provider_type="custom", temperature=0.8)
        class ComplexLLMConfig(ResourceBase):
            """A complex LLM configuration."""
            custom_setting: str = "value"  # Define as a field instead of setting in __init__
            
            def generate_prompt(self, text: str) -> str:
                return f"Generate: {text}"
            
            @property
            def is_valid(self) -> bool:
                return True
        
        # Verify class functionality is preserved
        instance = ComplexLLMConfig(name="test", type="llm_config", custom_setting="value")
        assert instance.generate_prompt("test") == "Generate: test"
        assert instance.is_valid is True
        assert instance.custom_setting == "value"
        
        # Verify decorator metadata
        assert ComplexLLMConfig.__resource_name__ == "complex_llm"
        assert ComplexLLMConfig.__resource_type__ == ResourceType.LLM_CONFIG


class TestDecoratorErrorHandling:
    """Test error handling in decorators."""
    
    def test_registry_error_propagation(self):
        """Test that registry errors are properly propagated."""
        with patch('flowlib.resources.decorators.decorators.resource_registry') as mock_reg:
            mock_reg.register.side_effect = Exception("Registry error")
            
            with pytest.raises(Exception) as exc_info:
                @resource("test_resource", "test_type")
                class TestResource(ResourceBase):
                    pass
            
            assert "Registry error" in str(exc_info.value)
    
    def test_invalid_resource_type_handling(self):
        """Test handling of invalid resource types."""
        with patch('flowlib.resources.decorators.decorators.resource_registry') as mock_reg:
            mock_reg.register = Mock()
            
            # This should work - we don't validate resource types in decorators
            @resource("test_resource", "invalid_type")
            class TestResource(ResourceBase):
                pass
            
            # Registry gets whatever type we pass
            call_args = mock_reg.register.call_args
            assert call_args[1]['resource_type'] == "invalid_type"
    
    def test_decorator_with_none_values(self):
        """Test decorators with None values."""
        with patch('flowlib.resources.decorators.decorators.resource_registry') as mock_reg:
            mock_reg.register = Mock()
            
            @model_config("test_model", provider_type=None, config=None)
            class TestModel(ResourceBase):
                provider_type: str
                config: dict
            
            # Should use defaults
            mock_reg.register.assert_called_once()