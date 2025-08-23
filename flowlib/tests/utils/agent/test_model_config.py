"""Tests for model configuration utilities."""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from flowlib.utils.agent.model_config import (
    ensure_standard_models_registered,
    get_model_config,
    MODEL_USAGE_GUIDE
)
from flowlib.resources.models.constants import ResourceType


class TestEnsureStandardModelsRegistered:
    """Test ensure_standard_models_registered function."""
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_both_models_already_registered(self, mock_registry, caplog):
        """Test when both models are already registered."""
        mock_registry.contains.side_effect = [True, True]  # has_small, has_large
        
        with caplog.at_level(logging.DEBUG):
            ensure_standard_models_registered()
        
        # Verify only contains was called, no get or register
        assert mock_registry.contains.call_count == 2
        mock_registry.get.assert_not_called()
        mock_registry.register.assert_not_called()
        assert "Both agent-model-small and agent-model-large are already registered" in caplog.text
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_neither_model_registered(self, mock_registry, caplog):
        """Test when neither model is registered."""
        mock_registry.contains.side_effect = [False, False]  # has_small, has_large
        
        with caplog.at_level(logging.WARNING):
            ensure_standard_models_registered()
        
        # Verify warning was logged
        assert mock_registry.contains.call_count == 2
        mock_registry.get.assert_not_called()
        mock_registry.register.assert_not_called()
        assert "Neither agent-model-small nor agent-model-large are registered" in caplog.text
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_only_small_model_registered(self, mock_registry, caplog):
        """Test when only small model is registered."""
        mock_registry.contains.side_effect = [True, False]  # has_small, has_large
        
        # Create mock small model with proper class
        class MockModelConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        mock_small_model = Mock()
        mock_small_model.model_dump.return_value = {
            'name': 'agent-model-small',
            'provider': 'openai',
            'model': 'gpt-3.5-turbo'
        }
        mock_small_model.__class__ = MockModelConfig
        
        mock_registry.get.return_value = mock_small_model
        
        with caplog.at_level(logging.INFO):
            ensure_standard_models_registered()
        
        # Verify operations
        assert mock_registry.contains.call_count == 2
        mock_registry.get.assert_called_once_with("agent-model-small")
        mock_small_model.model_dump.assert_called_once()
        
        # Verify registration of large model
        mock_registry.register.assert_called_once()
        register_call = mock_registry.register.call_args
        assert register_call[1]['name'] == "agent-model-large"
        assert register_call[1]['resource_type'] == ResourceType.MODEL_CONFIG
        assert "Creating agent-model-large from agent-model-small configuration" in caplog.text
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_only_large_model_registered(self, mock_registry, caplog):
        """Test when only large model is registered."""
        mock_registry.contains.side_effect = [False, True]  # has_small, has_large
        
        # Create mock large model with proper class
        class MockModelConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        mock_large_model = Mock()
        mock_large_model.model_dump.return_value = {
            'name': 'agent-model-large',
            'provider': 'openai',
            'model': 'gpt-4'
        }
        mock_large_model.__class__ = MockModelConfig
        
        mock_registry.get.return_value = mock_large_model
        
        with caplog.at_level(logging.INFO):
            ensure_standard_models_registered()
        
        # Verify operations
        assert mock_registry.contains.call_count == 2
        mock_registry.get.assert_called_once_with("agent-model-large")
        mock_large_model.model_dump.assert_called_once()
        
        # Verify registration of small model
        mock_registry.register.assert_called_once()
        register_call = mock_registry.register.call_args
        assert register_call[1]['name'] == "agent-model-small"
        assert register_call[1]['resource_type'] == ResourceType.MODEL_CONFIG
        assert "Creating agent-model-small from agent-model-large configuration" in caplog.text
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_model_copy_preserves_data(self, mock_registry):
        """Test that model copy preserves all data except name."""
        mock_registry.contains.side_effect = [True, False]  # has_small, has_large
        
        # Create mock model with more complex data
        model_data = {
            'name': 'agent-model-small',
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 2000,
            'custom_field': 'custom_value'
        }
        
        # Create a proper mock class
        class MockModelConfig:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        mock_small_model = Mock()
        mock_small_model.model_dump.return_value = model_data.copy()
        mock_small_model.__class__ = MockModelConfig
        
        mock_registry.get.return_value = mock_small_model
        
        ensure_standard_models_registered()
        
        # Verify the registered object has correct data
        register_call = mock_registry.register.call_args
        registered_obj = register_call[1]['obj']
        
        assert registered_obj.name == "agent-model-large"
        assert registered_obj.provider == model_data['provider']
        assert registered_obj.model == model_data['model']
        assert registered_obj.temperature == model_data['temperature']
        assert registered_obj.max_tokens == model_data['max_tokens']
        assert registered_obj.custom_field == model_data['custom_field']
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_error_handling(self, mock_registry, caplog):
        """Test error handling in ensure_standard_models_registered."""
        mock_registry.contains.side_effect = Exception("Registry error")
        
        with caplog.at_level(logging.ERROR):
            ensure_standard_models_registered()
        
        assert "Error ensuring standard models are registered" in caplog.text
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_get_returns_none(self, mock_registry, caplog):
        """Test when get returns None."""
        mock_registry.contains.side_effect = [True, False]  # has_small, has_large
        mock_registry.get.return_value = None
        
        ensure_standard_models_registered()
        
        # Should not attempt to register when get returns None
        mock_registry.register.assert_not_called()


class TestGetModelConfig:
    """Test get_model_config function."""
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_get_model_config_success(self, mock_registry):
        """Test successfully getting model configuration."""
        # Create a real object with attributes instead of Mock
        class MockModel:
            def __init__(self):
                self.name = "agent-model-small"
                self.type = ResourceType.MODEL_CONFIG
                self.provider = "openai"
                self.model = "gpt-3.5-turbo"
                self.temperature = 0.7
                self.max_tokens = 2000
                self._private_attr = "should not be included"
            
            def some_method(self):
                return "should not be included"
        
        mock_model = MockModel()
        mock_registry.get.return_value = mock_model
        
        config = get_model_config("agent-model-small")
        
        assert config == {
            'provider': 'openai',
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 2000
        }
        
        mock_registry.get.assert_called_once_with("agent-model-small")
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_get_model_config_not_found(self, mock_registry):
        """Test when model is not found."""
        mock_registry.get.return_value = None
        
        config = get_model_config("agent-model-small")
        
        assert config is None
        mock_registry.get.assert_called_once_with("agent-model-small")
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_get_model_config_error(self, mock_registry, caplog):
        """Test error handling in get_model_config."""
        mock_registry.get.side_effect = Exception("Get error")
        
        with caplog.at_level(logging.ERROR):
            config = get_model_config("agent-model-small")
        
        assert config is None
        assert "Error getting model config for agent-model-small" in caplog.text
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_get_model_config_filters_attributes(self, mock_registry):
        """Test that get_model_config properly filters attributes."""
        # Create mock with various attribute types
        mock_model = Mock()
        mock_model.name = "agent-model-large"
        mock_model.type = ResourceType.MODEL_CONFIG
        mock_model.valid_attr = "included"
        mock_model.none_attr = None  # Should be excluded
        mock_model._private = "excluded"
        mock_model.callable_attr = Mock()  # Should be excluded
        
        # Mock dir() to return specific attributes
        with patch('builtins.dir', return_value=[
            'name', 'type', 'valid_attr', 'none_attr', '_private', 'callable_attr'
        ]):
            mock_registry.get.return_value = mock_model
            
            config = get_model_config("agent-model-large")
        
        assert config == {'valid_attr': 'included'}
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_get_model_config_with_complex_values(self, mock_registry):
        """Test getting config with complex attribute values."""
        mock_model = Mock()
        mock_model.name = "agent-model-small"
        mock_model.type = ResourceType.MODEL_CONFIG
        mock_model.provider = "openai"
        mock_model.model = "gpt-4"
        mock_model.params = {"temperature": 0.5, "top_p": 0.9}
        mock_model.tags = ["fast", "efficient"]
        
        mock_registry.get.return_value = mock_model
        
        config = get_model_config("agent-model-small")
        
        assert config['provider'] == "openai"
        assert config['model'] == "gpt-4"
        assert config['params'] == {"temperature": 0.5, "top_p": 0.9}
        assert config['tags'] == ["fast", "efficient"]


class TestModelUsageGuide:
    """Test MODEL_USAGE_GUIDE constant."""
    
    def test_model_usage_guide_content(self):
        """Test that MODEL_USAGE_GUIDE contains expected content."""
        assert "agent-model-small" in MODEL_USAGE_GUIDE
        assert "agent-model-large" in MODEL_USAGE_GUIDE
        assert "quick, simple tasks" in MODEL_USAGE_GUIDE
        assert "complex, thoughtful tasks" in MODEL_USAGE_GUIDE
        assert "automatically ensures both models are available" in MODEL_USAGE_GUIDE


class TestIntegration:
    """Integration tests for model config utilities."""
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_ensure_and_get_workflow(self, mock_registry):
        """Test the workflow of ensuring models then getting config."""
        # Setup: only small model registered
        mock_registry.contains.side_effect = [True, False]  # First call
        
        # Create mock model
        model_data = {
            'name': 'agent-model-small',
            'provider': 'anthropic',
            'model': 'claude-2'
        }
        
        mock_model = Mock()
        mock_model.model_dump.return_value = model_data.copy()
        mock_model.provider = model_data['provider']
        mock_model.model = model_data['model']
        
        MockModelClass = type('MockModelConfig', (), {
            '__init__': lambda self, **kwargs: [setattr(self, k, v) for k, v in kwargs.items()]
        })
        mock_model.__class__ = MockModelClass
        
        mock_registry.get.return_value = mock_model
        
        # Execute workflow
        ensure_standard_models_registered()
        
        # Reset mock for next operation
        mock_registry.get.reset_mock()
        mock_registry.get.return_value = mock_model
        
        config = get_model_config("agent-model-small")
        
        assert config['provider'] == 'anthropic'
        assert config['model'] == 'claude-2'
    
    @patch('flowlib.agent.utils.model_config.resource_registry')
    def test_model_usage_patterns(self, mock_registry):
        """Test different model usage patterns."""
        # Create different configs for small and large
        small_model = Mock()
        small_model.provider = "openai"
        small_model.model = "gpt-3.5-turbo"
        small_model.temperature = 0.7
        
        large_model = Mock()
        large_model.provider = "openai"
        large_model.model = "gpt-4"
        large_model.temperature = 0.3
        large_model.reasoning_depth = "deep"
        
        mock_registry.get.side_effect = [small_model, large_model]
        
        # Get configs for both
        small_config = get_model_config("agent-model-small")
        large_config = get_model_config("agent-model-large")
        
        # Verify they have different characteristics
        assert small_config['model'] == "gpt-3.5-turbo"
        assert large_config['model'] == "gpt-4"
        assert small_config['temperature'] == 0.7  # Higher for creativity
        assert large_config['temperature'] == 0.3  # Lower for accuracy
        assert 'reasoning_depth' not in small_config
        assert large_config.get('reasoning_depth') == "deep"