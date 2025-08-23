"""Tests for model resource."""
import pytest
from pydantic import ValidationError
from typing import Dict, Any

from flowlib.resources.models.model_resource import ModelResource
from flowlib.resources.models.base import ResourceBase


class TestModelResource:
    """Test ModelResource class."""
    
    def test_valid_model_resource(self):
        """Test valid model resource creation."""
        config = {"temperature": 0.7, "max_tokens": 1000}
        
        model = ModelResource(
            name="test_model",
            type="model",
            provider_type="llamacpp",
            config=config
        )
        
        assert model.name == "test_model"
        assert model.type == "model"
        assert model.provider_type == "llamacpp"
        assert model.config == config
    
    def test_model_resource_inheritance(self):
        """Test that ModelResource inherits from ResourceBase."""
        model = ModelResource(
            name="test_model",
            type="model",
            provider_type="openai",
            config={}
        )
        
        assert isinstance(model, ResourceBase)
        assert hasattr(model, 'name')
        assert hasattr(model, 'type')
    
    def test_model_resource_with_complex_config(self):
        """Test model resource with complex configuration."""
        complex_config = {
            "temperature": 0.8,
            "max_tokens": 2048,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stop": ["\\n", "Human:", "AI:"],
            "model_params": {
                "n_ctx": 4096,
                "n_batch": 512,
                "n_threads": 8
            },
            "generation_settings": {
                "use_mmap": True,
                "use_mlock": False,
                "low_vram": False
            }
        }
        
        model = ModelResource(
            name="complex_model",
            type="model",
            provider_type="llamacpp",
            config=complex_config
        )
        
        assert model.config == complex_config
        assert model.config["model_params"]["n_ctx"] == 4096
        assert model.config["generation_settings"]["use_mmap"] is True
    
    def test_model_resource_default_type(self):
        """Test model resource with default type."""
        model = ModelResource(
            name="default_type_model",
            provider_type="google_ai",
            config={}
        )
        
        assert model.type == "model"
    
    def test_model_resource_empty_config(self):
        """Test model resource with empty configuration."""
        model = ModelResource(
            name="empty_config_model",
            type="model",
            provider_type="openai",
            config={}
        )
        
        assert model.config == {}
        assert isinstance(model.config, dict)
    
    def test_model_resource_required_fields(self):
        """Test that required fields are validated."""
        # Missing name
        with pytest.raises(ValidationError) as exc_info:
            ModelResource(
                type="model",
                provider_type="llamacpp",
                config={}
            )
        assert "name" in str(exc_info.value)
        
        # Missing provider_type
        with pytest.raises(ValidationError) as exc_info:
            ModelResource(
                name="test_model",
                type="model",
                config={}
            )
        assert "provider_type" in str(exc_info.value)
        
        # Test that config defaults to empty dict when not provided
        model = ModelResource(
            name="test_model",
            type="model",
            provider_type="llamacpp"
        )
        assert model.config == {}
    
    def test_model_resource_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ModelResource(
                name="test_model",
                type="model",
                provider_type="llamacpp",
                config={},
                extra_field="not_allowed"
            )
        
        assert "extra_field" in str(exc_info.value)
    
    def test_model_resource_type_validation(self):
        """Test type validation for model resource fields."""
        # Invalid name type
        with pytest.raises(ValidationError):
            ModelResource(
                name=123,  # Should be string
                type="model",
                provider_type="llamacpp",
                config={}
            )
        
        # Invalid provider_type type
        with pytest.raises(ValidationError):
            ModelResource(
                name="test_model",
                type="model",
                provider_type=None,  # Should be string
                config={}
            )
        
        # Invalid config type
        with pytest.raises(ValidationError):
            ModelResource(
                name="test_model",
                type="model",
                provider_type="llamacpp",
                config="not_a_dict"  # Should be dict
            )
    
    def test_model_resource_serialization(self):
        """Test model resource serialization."""
        config = {"temperature": 0.7, "max_tokens": 1000}
        
        model = ModelResource(
            name="serializable_model",
            type="model",
            provider_type="openai",
            config=config
        )
        
        # Test dict conversion
        model_dict = model.model_dump()
        
        assert model_dict["name"] == "serializable_model"
        assert model_dict["type"] == "model"
        assert model_dict["provider_type"] == "openai"
        assert model_dict["config"] == config
        
        # Test JSON serialization
        import json
        json_str = model.model_dump_json()
        parsed = json.loads(json_str)
        
        assert parsed["name"] == "serializable_model"
        assert parsed["config"]["temperature"] == 0.7
    
    def test_model_resource_deserialization(self):
        """Test model resource deserialization."""
        model_data = {
            "name": "deserialized_model",
            "type": "model",
            "provider_type": "llamacpp",
            "config": {
                "temperature": 0.8,
                "max_tokens": 1500,
                "model_path": "/path/to/model.gguf"
            }
        }
        
        # Test from dict
        model = ModelResource(**model_data)
        
        assert model.name == "deserialized_model"
        assert model.provider_type == "llamacpp"
        assert model.config["model_path"] == "/path/to/model.gguf"
        
        # Test from JSON
        import json
        json_str = json.dumps(model_data)
        model_from_json = ModelResource.model_validate_json(json_str)
        
        assert model_from_json.name == "deserialized_model"
        assert model_from_json.config == model_data["config"]
    
    def test_model_resource_equality(self):
        """Test model resource equality comparison."""
        config = {"temperature": 0.7}
        
        model1 = ModelResource(
            name="test_model",
            type="model",
            provider_type="openai",
            config=config
        )
        
        model2 = ModelResource(
            name="test_model",
            type="model",
            provider_type="openai",
            config=config
        )
        
        model3 = ModelResource(
            name="different_model",
            type="model",
            provider_type="openai",
            config=config
        )
        
        assert model1 == model2
        assert model1 != model3
    
    def test_model_resource_copy(self):
        """Test model resource copying."""
        original_config = {"temperature": 0.7, "max_tokens": 1000}
        
        original = ModelResource(
            name="original_model",
            type="model",
            provider_type="llamacpp",
            config=original_config
        )
        
        # Test shallow copy
        copied = original.model_copy()
        
        assert copied == original
        assert copied is not original
        assert copied.config is original.config  # Shallow copy
        
        # Test deep copy
        deep_copied = original.model_copy(deep=True)
        
        assert deep_copied == original
        assert deep_copied is not original
        assert deep_copied.config is not original.config  # Deep copy
        
        # Modify deep copy config
        deep_copied.config["temperature"] = 0.9
        assert original.config["temperature"] == 0.7  # Original unchanged
    
    def test_model_resource_update(self):
        """Test model resource field updates."""
        model = ModelResource(
            name="updatable_model",
            type="model",
            provider_type="openai",
            config={"temperature": 0.7}
        )
        
        # Test field update
        updated = model.model_copy(update={"provider_type": "llamacpp"})
        
        assert updated.name == "updatable_model"  # Unchanged
        assert updated.provider_type == "llamacpp"  # Updated
        assert updated.config == {"temperature": 0.7}  # Unchanged
        
        # Test config update
        new_config = {"temperature": 0.9, "max_tokens": 2000}
        config_updated = model.model_copy(update={"config": new_config})
        
        assert config_updated.config == new_config
        assert model.config == {"temperature": 0.7}  # Original unchanged
    
    def test_model_resource_provider_types(self):
        """Test model resource with different provider types."""
        provider_types = ["llamacpp", "openai", "google_ai", "anthropic", "huggingface"]
        
        for provider_type in provider_types:
            model = ModelResource(
                name=f"{provider_type}_model",
                type="model",
                provider_type=provider_type,
                config={"provider_specific": f"{provider_type}_config"}
            )
            
            assert model.provider_type == provider_type
            assert model.config["provider_specific"] == f"{provider_type}_config"
    
    def test_model_resource_config_types(self):
        """Test model resource with different config value types."""
        complex_config = {
            "string_param": "text_value",
            "int_param": 42,
            "float_param": 3.14,
            "bool_param": True,
            "list_param": [1, 2, 3, "four"],
            "dict_param": {"nested": "value"},
            "none_param": None
        }
        
        model = ModelResource(
            name="complex_config_model",
            type="model",
            provider_type="custom",
            config=complex_config
        )
        
        assert model.config["string_param"] == "text_value"
        assert model.config["int_param"] == 42
        assert model.config["float_param"] == 3.14
        assert model.config["bool_param"] is True
        assert model.config["list_param"] == [1, 2, 3, "four"]
        assert model.config["dict_param"]["nested"] == "value"
        assert model.config["none_param"] is None


class TestModelResourceIntegration:
    """Integration tests for ModelResource."""
    
    def test_model_resource_as_base_resource(self):
        """Test using ModelResource as ResourceBase."""
        model = ModelResource(
            name="integration_model",
            type="model",
            provider_type="integration_provider",
            config={"test": "value"}
        )
        
        # Should work with functions expecting ResourceBase
        def process_resource(resource: ResourceBase) -> str:
            return f"Processing {resource.name} of type {resource.type}"
        
        result = process_resource(model)
        assert result == "Processing integration_model of type model"
    
    def test_model_resource_polymorphism(self):
        """Test polymorphic usage of ModelResource."""
        resources = [
            ModelResource(
                name="model1",
                type="model",
                provider_type="openai",
                config={"api_key": "key1"}
            ),
            ModelResource(
                name="model2", 
                type="model",
                provider_type="llamacpp",
                config={"model_path": "/path/to/model"}
            )
        ]
        
        # Process as ResourceBase instances
        for resource in resources:
            assert isinstance(resource, ResourceBase)
            assert hasattr(resource, 'name')
            assert hasattr(resource, 'type')
            
            # Also as ModelResource instances
            assert isinstance(resource, ModelResource)
            assert hasattr(resource, 'provider_type')
            assert hasattr(resource, 'config')
    
    def test_model_resource_validation_integration(self):
        """Test model resource validation with real-world scenarios."""
        # LLaMA.cpp model configuration
        llamacpp_model = ModelResource(
            name="llama2_7b",
            type="model",
            provider_type="llamacpp",
            config={
                "model_path": "/models/llama-2-7b-chat.gguf",
                "n_ctx": 4096,
                "n_batch": 512,
                "n_threads": 8,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "stop": ["Human:", "Assistant:"]
            }
        )
        
        assert llamacpp_model.provider_type == "llamacpp"
        assert llamacpp_model.config["n_ctx"] == 4096
        
        # OpenAI model configuration
        openai_model = ModelResource(
            name="gpt4_turbo",
            type="model",
            provider_type="openai",
            config={
                "model": "gpt-4-turbo-preview",
                "api_key": "sk-...",
                "organization": "org-...",
                "temperature": 0.3,
                "max_tokens": 4000,
                "top_p": 1.0,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
        )
        
        assert openai_model.provider_type == "openai"
        assert openai_model.config["model"] == "gpt-4-turbo-preview"
        
        # Both should be valid ModelResource instances
        models = [llamacpp_model, openai_model]
        for model in models:
            assert isinstance(model, ModelResource)
            assert model.type == "model"
            assert isinstance(model.config, dict)