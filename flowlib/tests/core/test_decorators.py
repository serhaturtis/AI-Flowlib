"""Tests for core decorators."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from functools import wraps

from flowlib.core.decorators.decorators import (
    provider,
    resource,
    flow,
    config,
    llm_config,
    database_config,
    vector_db_config,
    graph_db_config,
    cache_config,
    embedding_config,
    model,
    prompt,
    template,
    singleton,
    lazy_init,
    inject
)


class TestProviderDecorator:
    """Test provider decorator functionality."""
    
    @patch('flowlib.core.decorators.decorators.emit_registration')
    def test_provider_decorator_basic(self, mock_emit):
        """Test basic provider decorator usage."""
        
        @provider('llm', 'test_provider')
        class TestProvider:
            def __init__(self):
                self.name = "test"
        
        # Check that emit_registration was called
        mock_emit.assert_called_once()
        event = mock_emit.call_args[0][0]
        
        assert event.item_type == 'provider'
        assert event.name == 'test_provider'
        assert event.metadata['provider_type'] == 'llm'
        assert event.metadata['class_name'] == 'TestProvider'
        assert 'module' in event.metadata
        
        # Test that factory works
        instance = event.factory()
        assert isinstance(instance, TestProvider)
        assert instance.name == "test"
    
    @patch('flowlib.core.decorators.decorators.emit_registration')
    def test_provider_decorator_with_metadata(self, mock_emit):
        """Test provider decorator with additional metadata."""
        
        @provider('database', 'test_db', version='1.0', author='test')
        class TestDatabase:
            pass
        
        event = mock_emit.call_args[0][0]
        
        assert event.metadata['provider_type'] == 'database'
        assert event.metadata['version'] == '1.0'
        assert event.metadata['author'] == 'test'
    
    def test_provider_decorator_returns_class(self):
        """Test that provider decorator returns the original class."""
        
        @provider('llm', 'test')
        class TestProvider:
            def method(self):
                return "test"
        
        # Class should be unchanged
        instance = TestProvider()
        assert instance.method() == "test"


class TestResourceDecorator:
    """Test resource decorator functionality."""
    
    @patch('flowlib.core.decorators.decorators.emit_registration')
    def test_resource_decorator_basic(self, mock_emit):
        """Test basic resource decorator usage."""
        
        @resource('test_resource', 'prompt')
        class TestResource:
            def __init__(self):
                self.content = "test content"
        
        event = mock_emit.call_args[0][0]
        
        assert event.item_type == 'resource'
        assert event.name == 'test_resource'
        assert event.metadata['resource_type'] == 'prompt'
        assert event.metadata['class_name'] == 'TestResource'
        
        # Test factory
        instance = event.factory()
        assert isinstance(instance, TestResource)
        assert instance.content == "test content"
    
    @patch('flowlib.core.decorators.decorators.emit_registration')
    def test_resource_decorator_with_metadata(self, mock_emit):
        """Test resource decorator with metadata."""
        
        @resource('config_resource', 'config', version='2.0')
        class ConfigResource:
            pass
        
        event = mock_emit.call_args[0][0]
        
        assert event.metadata['resource_type'] == 'config'
        assert event.metadata['version'] == '2.0'


class TestFlowDecorator:
    """Test flow decorator functionality."""
    
    @patch('flowlib.core.decorators.decorators.emit_registration')
    def test_flow_decorator_basic(self, mock_emit):
        """Test basic flow decorator usage."""
        
        @flow('test_flow')
        class TestFlow:
            def run(self):
                return "running"
        
        event = mock_emit.call_args[0][0]
        
        assert event.item_type == 'flow'
        assert event.name == 'test_flow'
        assert event.metadata['description'] is None
        assert event.metadata['class_name'] == 'TestFlow'
        
        # Test factory
        instance = event.factory()
        assert isinstance(instance, TestFlow)
        assert instance.run() == "running"
    
    @patch('flowlib.core.decorators.decorators.emit_registration')
    def test_flow_decorator_with_description(self, mock_emit):
        """Test flow decorator with description."""
        
        @flow('analysis_flow', description='Analyzes data', category='analysis')
        class AnalysisFlow:
            pass
        
        event = mock_emit.call_args[0][0]
        
        assert event.metadata['description'] == 'Analyzes data'
        assert event.metadata['category'] == 'analysis'


class TestConfigDecorator:
    """Test config decorator functionality."""
    
    @patch('flowlib.core.decorators.decorators.emit_configuration')
    def test_config_decorator_basic(self, mock_emit):
        """Test basic config decorator usage."""
        
        @config('test_config', 'llm', model_path='/path/to/model')
        class TestConfig:
            pass
        
        event = mock_emit.call_args[0][0]
        
        assert event.config_name == 'test_config'
        assert event.config_class == TestConfig
        assert event.provider_type == 'llm'
        assert event.settings['model_path'] == '/path/to/model'
    
    @patch('flowlib.core.decorators.decorators.emit_configuration')
    def test_config_decorator_multiple_settings(self, mock_emit):
        """Test config decorator with multiple settings."""
        
        @config('db_config', 'postgres', host='localhost', port=5432, database='test')
        class DatabaseConfig:
            pass
        
        event = mock_emit.call_args[0][0]
        
        assert event.provider_type == 'postgres'
        assert event.settings['host'] == 'localhost'
        assert event.settings['port'] == 5432
        assert event.settings['database'] == 'test'


class TestSpecificConfigDecorators:
    """Test specific configuration decorators."""
    
    @patch('flowlib.core.decorators.decorators.emit_configuration')
    def test_llm_config_default_provider(self, mock_emit):
        """Test LLM config with default provider type."""
        
        @llm_config('default_llm', model_path='/path/to/model')
        class DefaultLLM:
            pass
        
        event = mock_emit.call_args[0][0]
        
        assert event.provider_type == 'llamacpp'
        assert event.settings['model_path'] == '/path/to/model'
    
    @patch('flowlib.core.decorators.decorators.emit_configuration')
    def test_llm_config_custom_provider(self, mock_emit):
        """Test LLM config with custom provider type."""
        
        @llm_config('custom_llm', provider_type='openai', api_key='secret')
        class CustomLLM:
            pass
        
        event = mock_emit.call_args[0][0]
        
        assert event.provider_type == 'openai'
        assert event.settings['api_key'] == 'secret'
        assert 'provider_type' not in event.settings  # Should be removed from settings
    
    @patch('flowlib.core.decorators.decorators.emit_configuration')
    def test_database_config(self, mock_emit):
        """Test database config decorator."""
        
        @database_config('test_db', provider_type='postgres', host='localhost')
        class TestDB:
            pass
        
        event = mock_emit.call_args[0][0]
        assert event.provider_type == 'postgres'
        assert event.settings['host'] == 'localhost'
    
    @patch('flowlib.core.decorators.decorators.emit_configuration')
    def test_vector_db_config(self, mock_emit):
        """Test vector database config decorator."""
        
        @vector_db_config('test_vector', collection_name='test')
        class TestVector:
            pass
        
        event = mock_emit.call_args[0][0]
        assert event.provider_type == 'chroma'  # Default
        assert event.settings['collection_name'] == 'test'
    
    @patch('flowlib.core.decorators.decorators.emit_configuration')
    def test_graph_db_config(self, mock_emit):
        """Test graph database config decorator."""
        
        @graph_db_config('test_graph', uri='bolt://localhost:7687')
        class TestGraph:
            pass
        
        event = mock_emit.call_args[0][0]
        assert event.provider_type == 'neo4j'  # Default
        assert event.settings['uri'] == 'bolt://localhost:7687'
    
    @patch('flowlib.core.decorators.decorators.emit_configuration')
    def test_cache_config(self, mock_emit):
        """Test cache config decorator."""
        
        @cache_config('test_cache', host='localhost', port=6379)
        class TestCache:
            pass
        
        event = mock_emit.call_args[0][0]
        assert event.provider_type == 'redis'  # Default
        assert event.settings['host'] == 'localhost'
        assert event.settings['port'] == 6379
    
    @patch('flowlib.core.decorators.decorators.emit_configuration')
    def test_embedding_config(self, mock_emit):
        """Test embedding config decorator."""
        
        @embedding_config('test_embedding', model_path='/path/to/embedding')
        class TestEmbedding:
            pass
        
        event = mock_emit.call_args[0][0]
        assert event.provider_type == 'llamacpp_embedding'  # Default
        assert event.settings['model_path'] == '/path/to/embedding'


class TestLegacyDecorators:
    """Test legacy compatibility decorators."""
    
    @patch('flowlib.core.decorators.decorators.emit_registration')
    def test_model_decorator(self, mock_emit):
        """Test model decorator (legacy compatibility)."""
        
        @model('test_model', version='1.0')
        class TestModel:
            pass
        
        event = mock_emit.call_args[0][0]
        
        assert event.item_type == 'resource'
        assert event.name == 'test_model'
        assert event.metadata['resource_type'] == 'model'
        assert event.metadata['version'] == '1.0'
    
    @patch('flowlib.core.decorators.decorators.emit_registration')
    def test_prompt_decorator(self, mock_emit):
        """Test prompt decorator."""
        
        @prompt('test_prompt', category='system')
        class TestPrompt:
            pass
        
        event = mock_emit.call_args[0][0]
        
        assert event.item_type == 'resource'
        assert event.name == 'test_prompt'
        assert event.metadata['resource_type'] == 'prompt'
        assert event.metadata['category'] == 'system'
    
    @patch('flowlib.core.decorators.decorators.emit_registration')
    def test_template_decorator(self, mock_emit):
        """Test template decorator."""
        
        @template('test_template', format='jinja2')
        class TestTemplate:
            pass
        
        event = mock_emit.call_args[0][0]
        
        assert event.item_type == 'resource'
        assert event.name == 'test_template'
        assert event.metadata['resource_type'] == 'template'
        assert event.metadata['format'] == 'jinja2'


class TestUtilityDecorators:
    """Test utility decorators."""
    
    def test_singleton_decorator(self):
        """Test singleton decorator."""
        
        @singleton
        class SingletonClass:
            def __init__(self, value=None):
                self.value = value
        
        # Multiple calls should return same instance
        instance1 = SingletonClass("test")
        instance2 = SingletonClass("different")
        
        assert instance1 is instance2
        assert instance1.value == "test"  # First call wins
        assert instance2.value == "test"
    
    def test_lazy_init_decorator(self):
        """Test lazy initialization decorator."""
        
        class ExpensiveClass:
            def __init__(self, value):
                self.value = value
                self.initialized = True
        
        @lazy_init
        class LazyExpensive(ExpensiveClass):
            pass
        
        # Creating wrapper should not initialize the expensive class
        wrapper = LazyExpensive("test")
        assert wrapper._instance is None
        
        # Accessing attribute should trigger initialization
        value = wrapper.value
        assert value == "test"
        assert wrapper.initialized is True
    
    def test_lazy_init_method_access(self):
        """Test lazy initialization with method access."""
        
        @lazy_init
        class LazyClass:
            def __init__(self, value):
                self.value = value
            
            def get_value(self):
                return self.value
        
        wrapper = LazyClass("test")
        
        # Method should work after lazy initialization
        result = wrapper.get_value()
        assert result == "test"


class TestInjectDecorator:
    """Test dependency injection decorator."""
    
    @pytest.mark.asyncio
    async def test_inject_provider(self):
        """Test injecting provider dependency."""
        mock_provider = Mock()
        mock_container = Mock()
        mock_container.get_provider = AsyncMock(return_value=mock_provider)
        
        @inject(llm='default-llm')
        async def test_function(data, llm):
            return f"data: {data}, llm: {llm}"
        
        with patch('flowlib.core.container.container.get_container', return_value=mock_container):
            result = await test_function("test_data")
        
        assert result == f"data: test_data, llm: {mock_provider}"
        mock_container.get_provider.assert_called_once_with('default-llm')
    
    @pytest.mark.asyncio
    async def test_inject_resource(self):
        """Test injecting resource dependency."""
        mock_resource = Mock()
        mock_container = Mock()
        mock_container.get_resource = Mock(return_value=mock_resource)
        
        @inject(config='resource:app-config')
        async def test_function(data, config):
            return f"data: {data}, config: {config}"
        
        with patch('flowlib.core.container.container.get_container', return_value=mock_container):
            result = await test_function("test_data")
        
        assert result == f"data: test_data, config: {mock_resource}"
        mock_container.get_resource.assert_called_once_with('app-config')
    
    @pytest.mark.asyncio
    async def test_inject_flow(self):
        """Test injecting flow dependency."""
        mock_flow = Mock()
        mock_container = Mock()
        mock_container.get_flow = Mock(return_value=mock_flow)
        
        @inject(processor='flow:data-processor')
        async def test_function(data, processor):
            return f"data: {data}, processor: {processor}"
        
        with patch('flowlib.core.container.container.get_container', return_value=mock_container):
            result = await test_function("test_data")
        
        assert result == f"data: test_data, processor: {mock_flow}"
        mock_container.get_flow.assert_called_once_with('data-processor')
    
    @pytest.mark.asyncio
    async def test_inject_multiple_dependencies(self):
        """Test injecting multiple dependencies."""
        mock_provider = Mock()
        mock_resource = Mock()
        mock_container = Mock()
        mock_container.get_provider = AsyncMock(return_value=mock_provider)
        mock_container.get_resource = Mock(return_value=mock_resource)
        
        @inject(llm='default-llm', config='resource:app-config')
        async def test_function(data, llm, config):
            return f"data: {data}, llm: {llm}, config: {config}"
        
        with patch('flowlib.core.container.container.get_container', return_value=mock_container):
            result = await test_function("test_data")
        
        expected = f"data: test_data, llm: {mock_provider}, config: {mock_resource}"
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_inject_with_explicit_kwargs(self):
        """Test injection when dependency is explicitly provided."""
        mock_provider = Mock()
        mock_container = Mock()
        mock_container.get_provider = AsyncMock(return_value=mock_provider)
        
        @inject(llm='default-llm')
        async def test_function(data, llm):
            return f"data: {data}, llm: {llm}"
        
        explicit_llm = Mock()
        
        with patch('flowlib.core.container.container.get_container', return_value=mock_container):
            result = await test_function("test_data", llm=explicit_llm)
        
        # Should use explicit dependency, not inject
        assert result == f"data: test_data, llm: {explicit_llm}"
        mock_container.get_provider.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_inject_default_provider_type(self):
        """Test injection with default provider type."""
        mock_provider = Mock()
        mock_container = Mock()
        mock_container.get_provider = AsyncMock(return_value=mock_provider)
        
        @inject(llm='default-llm')  # No prefix = provider
        async def test_function(data, llm):
            return f"data: {data}, llm: {llm}"
        
        with patch('flowlib.core.container.container.get_container', return_value=mock_container):
            result = await test_function("test_data")
        
        mock_container.get_provider.assert_called_once_with('default-llm')
    
    def test_inject_preserves_function_metadata(self):
        """Test that inject decorator preserves function metadata."""
        
        @inject(llm='default-llm')
        async def test_function(data, llm):
            """Test function docstring."""
            return data
        
        assert test_function.__name__ == 'test_function'
        assert test_function.__doc__ == 'Test function docstring.'