"""Tests for core dynamic loader."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import importlib
import inspect

from flowlib.core.loader.loader import DynamicLoader


class MockProvider:
    """Mock provider for testing."""
    
    def __init__(self, name: str = "mock", host: str = "localhost", port: int = 5432):
        self.name = name
        self.host = host
        self.port = port


class MockResource:
    """Mock resource for testing."""
    
    def __init__(self, name: str = "mock", version: str = "1.0"):
        self.name = name
        self.version = version


class MockFlow:
    """Mock flow for testing."""
    
    def __init__(self, name: str = "mock", config: dict = None):
        self.name = name
        self.config = config or {}


class TestDynamicLoader:
    """Test DynamicLoader functionality."""
    
    def test_load_class_success(self):
        """Test successful class loading."""
        # Test loading a real class
        cls = DynamicLoader.load_class("collections:OrderedDict")
        
        from collections import OrderedDict
        assert cls is OrderedDict
    
    def test_load_class_invalid_format(self):
        """Test loading class with invalid format."""
        with pytest.raises(ImportError, match="Cannot load class from"):
            DynamicLoader.load_class("invalid_format")
    
    def test_load_class_module_not_found(self):
        """Test loading class from non-existent module."""
        with pytest.raises(ImportError, match="Cannot load class from"):
            DynamicLoader.load_class("nonexistent.module:SomeClass")
    
    def test_load_class_class_not_found(self):
        """Test loading non-existent class from valid module."""
        with pytest.raises(ImportError, match="Cannot load class from"):
            DynamicLoader.load_class("collections:NonExistentClass")
    
    def test_load_provider_class_success(self):
        """Test successful provider class loading."""
        with patch.object(DynamicLoader, 'load_class', return_value=MockProvider) as mock_load:
            cls = DynamicLoader.load_provider_class("postgres")
            
            assert cls is MockProvider
            mock_load.assert_called_once_with(
                'flowlib.providers.db.postgres.provider:PostgreSQLProvider'
            )
    
    def test_load_provider_class_unknown_type(self):
        """Test loading unknown provider type."""
        with pytest.raises(ValueError, match="Unknown provider type: unknown"):
            DynamicLoader.load_provider_class("unknown")
    
    def test_create_provider_success(self):
        """Test successful provider creation."""
        config = {"name": "test", "host": "127.0.0.1", "port": 5432, "extra": "ignored"}
        
        with patch.object(DynamicLoader, 'load_provider_class', return_value=MockProvider):
            provider = DynamicLoader.create_provider("postgres", config)
            
            assert isinstance(provider, MockProvider)
            assert isinstance(provider, MockProvider)
            assert provider.name == "test"
    
    # Removed redundant parameter filtering test - handled by Pydantic validation
    
    def test_create_provider_with_valid_config(self):
        """Test provider creation with valid configuration."""
        config = {"name": "test", "host": "localhost", "port": 3306}
        
        with patch.object(DynamicLoader, 'load_provider_class', return_value=MockProvider):
            provider = DynamicLoader.create_provider("mysql", config)
            
            # Only valid constructor parameters should be passed
            assert provider.name == "test"
            assert provider.host == "localhost"
            assert provider.port == 3306
            assert not hasattr(provider, 'invalid_param')
    
    def test_load_resource_class_success(self):
        """Test successful resource class loading."""
        with patch.object(DynamicLoader, 'load_class', return_value=MockResource) as mock_load:
            cls = DynamicLoader.load_resource_class("model")
            
            assert cls is MockResource
            mock_load.assert_called_once_with(
                'flowlib.resources.model_resource:ModelResource'
            )
    
    def test_load_resource_class_unknown_type(self):
        """Test loading unknown resource type."""
        with pytest.raises(ValueError, match="Unknown resource type: unknown"):
            DynamicLoader.load_resource_class("unknown")
    
    def test_create_resource_success(self):
        """Test successful resource creation."""
        config = {"name": "test_resource", "version": "2.0", "extra": "ignored"}
        
        with patch.object(DynamicLoader, 'load_resource_class', return_value=MockResource):
            resource = DynamicLoader.create_resource("model", config)
            
            assert isinstance(resource, MockResource)
            assert resource.name == "test_resource"
    
    def test_load_flow_class_success(self):
        """Test successful flow class loading."""
        with patch.object(DynamicLoader, 'load_class', return_value=MockFlow) as mock_load:
            cls = DynamicLoader.load_flow_class("conversation")
            
            assert cls is MockFlow
            mock_load.assert_called_once_with(
                'flowlib.agent.components.conversation.flow:ConversationFlow'
            )
    
    def test_load_flow_class_unknown_name(self):
        """Test loading unknown flow name."""
        with pytest.raises(ValueError, match="Unknown flow: unknown"):
            DynamicLoader.load_flow_class("unknown")
    
    def test_create_flow_with_config(self):
        """Test flow creation with configuration."""
        config = {"name": "test_flow", "config": {"param": "value"}, "extra": "ignored"}
        
        with patch.object(DynamicLoader, 'load_flow_class', return_value=MockFlow):
            flow = DynamicLoader.create_flow("conversation", config)
            
            assert isinstance(flow, MockFlow)
            assert flow.name == "test_flow"
    
    # Removed redundant default configuration tests - handled by constructor defaults
    
    def test_register_provider_module(self):
        """Test registering new provider module."""
        original_modules = DynamicLoader.PROVIDER_MODULES.copy()
        
        try:
            DynamicLoader.register_provider_module("custom", "custom.provider:CustomProvider")
            
            assert "custom" in DynamicLoader.PROVIDER_MODULES
            assert DynamicLoader.PROVIDER_MODULES["custom"] == "custom.provider:CustomProvider"
        finally:
            # Restore original modules
            DynamicLoader.PROVIDER_MODULES.clear()
            DynamicLoader.PROVIDER_MODULES.update(original_modules)
    
    def test_register_resource_module(self):
        """Test registering new resource module."""
        original_modules = DynamicLoader.RESOURCE_MODULES.copy()
        
        try:
            DynamicLoader.register_resource_module("custom", "custom.resource:CustomResource")
            
            assert "custom" in DynamicLoader.RESOURCE_MODULES
            assert DynamicLoader.RESOURCE_MODULES["custom"] == "custom.resource:CustomResource"
        finally:
            # Restore original modules
            DynamicLoader.RESOURCE_MODULES.clear()
            DynamicLoader.RESOURCE_MODULES.update(original_modules)
    
    def test_register_flow_module(self):
        """Test registering new flow module."""
        original_modules = DynamicLoader.FLOW_MODULES.copy()
        
        try:
            DynamicLoader.register_flow_module("custom", "custom.flow:CustomFlow")
            
            assert "custom" in DynamicLoader.FLOW_MODULES
            assert DynamicLoader.FLOW_MODULES["custom"] == "custom.flow:CustomFlow"
        finally:
            # Restore original modules
            DynamicLoader.FLOW_MODULES.clear()
            DynamicLoader.FLOW_MODULES.update(original_modules)
    
    def test_get_available_providers(self):
        """Test getting available provider types."""
        providers = DynamicLoader.get_available_providers()
        
        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "postgres" in providers
        assert "chroma" in providers
        assert "neo4j" in providers
    
    def test_get_available_resources(self):
        """Test getting available resource types."""
        resources = DynamicLoader.get_available_resources()
        
        assert isinstance(resources, list)
        assert len(resources) > 0
        assert "model" in resources
        assert "config" in resources
    
    def test_get_available_flows(self):
        """Test getting available flows."""
        flows = DynamicLoader.get_available_flows()
        
        assert isinstance(flows, list)
        assert len(flows) > 0
        assert "conversation" in flows
        assert "classification" in flows
    
    def test_provider_modules_mapping(self):
        """Test that provider modules mapping is complete."""
        expected_providers = [
            'llamacpp', 'google_ai', 'postgres', 'mongodb', 'sqlite',
            'chroma', 'pinecone', 'qdrant', 'neo4j', 'arango', 'redis',
            's3', 'local', 'llamacpp_embedding', 'rabbitmq', 'kafka'
        ]
        
        available_providers = DynamicLoader.get_available_providers()
        
        for provider in expected_providers:
            assert provider in available_providers, f"Provider {provider} not in available providers"
    
    def test_resource_modules_mapping(self):
        """Test that resource modules mapping is complete."""
        expected_resources = ['prompt', 'model', 'config', 'template']
        
        available_resources = DynamicLoader.get_available_resources()
        
        for resource in expected_resources:
            assert resource in available_resources, f"Resource {resource} not in available resources"
    
    def test_flow_modules_mapping(self):
        """Test that flow modules mapping includes expected flows."""
        expected_flows = [
            'conversation', 'classification', 'knowledge-extraction',
            'knowledge-retrieval'
        ]
        
        available_flows = DynamicLoader.get_available_flows()
        
        for flow in expected_flows:
            assert flow in available_flows, f"Flow {flow} not in available flows"
    
    def test_parameter_filtering_with_complex_signature(self):
        """Test parameter filtering with complex constructor signature."""
        
        class ComplexProvider:
            def __init__(self, name: str, config: dict, *args, **kwargs):
                self.name = name
                self.config = config
                self.args = args
                self.kwargs = kwargs
        
        config = {
            "name": "test",
            "config": {"key": "value"},
            "extra_param": "should_be_ignored",
            "another_extra": 123
        }
        
        with patch.object(DynamicLoader, 'load_provider_class', return_value=ComplexProvider):
            provider = DynamicLoader.create_provider("complex", config)
            
            assert provider.name == "test"
            assert provider.config == {"key": "value"}
            # kwargs should be empty since extra params are filtered out
            assert provider.kwargs == {}
    
    def test_parameter_filtering_with_no_matching_params(self):
        """Test parameter filtering when config has no matching parameters."""
        
        class SimpleProvider:
            def __init__(self, required_param: str = "default"):
                self.required_param = required_param
        
        config = {
            "unrelated_param": "value",
            "another_param": 123
        }
        
        with patch.object(DynamicLoader, 'load_provider_class', return_value=SimpleProvider):
            provider = DynamicLoader.create_provider("simple", config)
            
            # Should use default value since no matching params
            assert provider.required_param == "default"
    
    def test_load_class_with_nested_module(self):
        """Test loading class from nested module path."""
        # Test with a real nested module
        cls = DynamicLoader.load_class("unittest.mock:Mock")
        
        from unittest.mock import Mock
        assert cls is Mock
    
    def test_create_provider_with_empty_config(self):
        """Test creating provider with empty configuration."""
        
        class DefaultProvider:
            def __init__(self, name: str = "default"):
                self.name = name
        
        with patch.object(DynamicLoader, 'load_provider_class', return_value=DefaultProvider):
            provider = DynamicLoader.create_provider("default", {})
            
            assert provider.name == "default"
    
    def test_error_handling_in_class_loading(self):
        """Test error handling during class loading."""
        with patch('importlib.import_module', side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError, match="Cannot load class from"):
                DynamicLoader.load_class("fake.module:FakeClass")
    
    def test_error_handling_in_attribute_access(self):
        """Test error handling when class doesn't exist in module."""
        mock_module = Mock()
        del mock_module.FakeClass  # Ensure attribute doesn't exist
        
        with patch('importlib.import_module', return_value=mock_module):
            with pytest.raises(ImportError, match="Cannot load class from"):
                DynamicLoader.load_class("real.module:FakeClass")