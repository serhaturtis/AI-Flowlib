"""Tests for agent decorator base functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional, Union

from flowlib.agent.components.decorators.base import agent, agent_flow, dual_path_agent
from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.models.config import AgentConfig
from flowlib.flows.base.base import Flow
from flowlib.core.context.context import Context


# Mock classes for testing
class MockFlow(Flow):
    """Mock flow for testing."""
    
    def __init__(self, name="mock_flow"):
        super().__init__(name)
    
    def get_description(self):
        return "Mock flow for testing"


class MockImplementation:
    """Mock implementation class for agent testing."""
    
    def __init__(self, value="test"):
        self.value = value
        self.agent = None
    
    def set_agent(self, agent):
        """Allow agent to be set."""
        self.agent = agent
    
    def test_method(self):
        """Test method to verify delegation."""
        return f"test_method called with value: {self.value}"
    
    def get_agent_name(self):
        """Method that uses agent reference."""
        if self.agent:
            return self.agent.config.name
        return "no_agent"


class MockImplementationNoSetAgent:
    """Mock implementation without set_agent method."""
    
    def __init__(self):
        self.agent = None
    
    def simple_method(self):
        return "simple"


class MockDualPathAgent:
    """Mock DualPathAgent for testing."""
    
    def __init__(self, config=None):
        self.config = config
        self.name = "mock_dual_path"
    
    def execute(self):
        return "dual_path_executed"


class TestAgentDecorator:
    """Test @agent decorator functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Clear any registrations to avoid conflicts
        try:
            from flowlib.agent.registry import agent_registry
            agent_registry._agents.clear()
        except (ImportError, AttributeError):
            pass
    
    def test_agent_decorator_basic(self):
        """Test basic @agent decorator usage."""
        @agent(name="test_agent", provider_name="llamacpp")
        class TestAgent:
            def __init__(self):
                self.initialized = True
        
        # Check that decorator added attributes
        assert hasattr(TestAgent, 'create')
        assert hasattr(TestAgent, '__agent_class__')
        
        # Test creating an instance
        agent_instance = TestAgent.create()
        
        # Verify it's an BaseAgent instance
        assert isinstance(agent_instance, BaseAgent)
        assert agent_instance.config.name == "test_agent"
        assert agent_instance.config.provider_name == "llamacpp"
    
    def test_agent_decorator_with_model_name(self):
        """Test @agent decorator with model name."""
        @agent(name="model_agent", provider_name="llamacpp", model_name="test_model")
        class ModelAgent:
            pass
        
        agent_instance = ModelAgent.create()
        assert agent_instance.config.model_name == "test_model"
    
    def test_agent_decorator_default_name(self):
        """Test @agent decorator with default name from class."""
        @agent(provider_name="llamacpp")
        class DefaultNameAgent:
            pass
        
        agent_instance = DefaultNameAgent.create()
        assert agent_instance.config.name == "DefaultNameAgent"
    
    def test_agent_decorator_with_additional_kwargs(self):
        """Test @agent decorator with additional configuration."""
        @agent(
            name="kwargs_agent",
            provider_name="llamacpp",
            custom_param="custom_value",
            temperature=0.7
        )
        class KwargsAgent:
            pass
        
        agent_instance = KwargsAgent.create()
        # Additional kwargs should be available in config
        config_dict = agent_instance.config.model_dump()
        # Known fields go directly to config
        assert config_dict.get("temperature") == 0.7
        # Unknown fields go to additional_settings
        additional_settings = config_dict.get("additional_settings", {})
        assert additional_settings.get("custom_param") == "custom_value"
    
    def test_agent_decorator_with_flows(self):
        """Test @agent decorator with flows."""
        @agent(name="flow_agent", provider_name="llamacpp")
        class FlowAgent:
            pass
        
        mock_flow = MockFlow("test_flow")
        agent_instance = FlowAgent.create(flows=[mock_flow])
        
        # Check that flow was registered
        flows = agent_instance.get_flows()
        assert "test_flow" in flows
    
    def test_agent_decorator_with_dict_config(self):
        """Test @agent decorator with dict config override."""
        @agent(name="config_agent", provider_name="llamacpp")
        class ConfigAgent:
            pass
        
        config_override = {
            "provider_name": "different_provider",  # Should not override
            "new_param": "new_value"  # Should be added
        }
        
        agent_instance = ConfigAgent.create(config=config_override)
        
        # Original config should take precedence
        assert agent_instance.config.provider_name == "llamacpp"
        # New params should be added to additional_settings
        config_dict = agent_instance.config.model_dump()
        additional_settings = config_dict.get("additional_settings", {})
        assert additional_settings.get("new_param") == "new_value"
    
    def test_agent_decorator_with_agent_config(self):
        """Test @agent decorator with AgentConfig object."""
        @agent(name="obj_config_agent", provider_name="llamacpp")
        class ObjConfigAgent:
            pass
        
        config_obj = AgentConfig(
            name="override_name",  # Should not override
            provider_name="override_provider",  # Should not override
            persona="Test agent persona",
            description="test description"  # Should be added
        )
        
        agent_instance = ObjConfigAgent.create(config=config_obj)
        
        # Original config should take precedence
        assert agent_instance.config.name == "obj_config_agent"
        assert agent_instance.config.provider_name == "llamacpp"
        # New fields should be added
        assert agent_instance.config.description == "test description"
    
    def test_agent_decorator_method_delegation_with_set_agent(self):
        """Test method delegation to implementation with set_agent."""
        @agent(name="delegate_agent", provider_name="llamacpp")
        class DelegateAgent(MockImplementation):
            pass
        
        agent_instance = DelegateAgent.create()
        
        # Test that implementation methods are accessible
        result = agent_instance.test_method()
        assert "test_method called" in result
        
        # Test that agent was set on implementation
        assert agent_instance._impl.agent == agent_instance
        
        # Test method that uses agent reference
        agent_name = agent_instance.get_agent_name()
        assert agent_name == "delegate_agent"
    
    def test_agent_decorator_method_delegation_without_set_agent(self):
        """Test method delegation to implementation without set_agent."""
        @agent(name="simple_delegate_agent", provider_name="llamacpp")
        class SimpleDelegateAgent(MockImplementationNoSetAgent):
            pass
        
        agent_instance = SimpleDelegateAgent.create()
        
        # Test that implementation methods are accessible
        result = agent_instance.simple_method()
        assert result == "simple"
        
        # Test that agent was set as attribute
        assert agent_instance._impl.agent == agent_instance
    
    def test_agent_decorator_attribute_error(self):
        """Test AttributeError when accessing non-existent attribute."""
        @agent(name="error_agent", provider_name="llamacpp")
        class ErrorAgent:
            pass
        
        agent_instance = ErrorAgent.create()
        
        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            _ = agent_instance.nonexistent
    
    def test_agent_decorator_task_description(self):
        """Test agent creation with task description."""
        @agent(name="task_agent", provider_name="llamacpp")
        class TaskAgent:
            pass
        
        agent_instance = TaskAgent.create(task_description="Test task description")
        # Task description should be passed to BaseAgent
        # This would be tested via the BaseAgent's handling of task_description
        assert isinstance(agent_instance, BaseAgent)


class TestAgentFlowDecorator:
    """Test @agent_flow decorator functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Clear any registrations
        try:
            from flowlib.flows.registry.registry import flow_registry
            if flow_registry and hasattr(flow_registry, '_flows'):
                flow_registry._flows.clear()
        except (ImportError, AttributeError):
            pass
    
    def test_agent_flow_decorator_basic(self):
        """Test basic @agent_flow decorator usage."""
        @agent_flow(name="test_flow", description="Test agent flow")
        class TestAgentFlow:
            pass
        
        # Check that flow decorator was applied
        assert hasattr(TestAgentFlow, '__flow_metadata__')
        assert TestAgentFlow.__flow_metadata__['name'] == "test_flow"
        assert TestAgentFlow.__flow_metadata__['agent_flow'] is True
        assert TestAgentFlow.__flow_metadata__['category'] == "agent"
    
    def test_agent_flow_decorator_with_category(self):
        """Test @agent_flow decorator with custom category."""
        @agent_flow(
            name="category_flow",
            description="Test flow with category",
            category="conversation"
        )
        class CategoryFlow:
            pass
        
        assert CategoryFlow.__flow_metadata__['category'] == "conversation"
    
    def test_agent_flow_decorator_infrastructure(self):
        """Test @agent_flow decorator with infrastructure flag."""
        @agent_flow(
            name="infra_flow",
            description="Infrastructure flow",
            is_infrastructure=True
        )
        class InfraFlow:
            pass
        
        assert hasattr(InfraFlow, '__is_infrastructure__')
        assert InfraFlow.__is_infrastructure__ is True
    
    def test_agent_flow_decorator_no_description_with_method(self):
        """Test @agent_flow decorator with existing get_description method."""
        @agent_flow(name="method_flow")
        class MethodFlow:
            def get_description(self):
                return "Original description"
        
        # Should preserve original get_description method
        flow_instance = MethodFlow()
        assert flow_instance.get_description() == "Original description"
    
    def test_agent_flow_decorator_no_description_with_docstring(self):
        """Test @agent_flow decorator using class docstring as description."""
        @agent_flow(name="docstring_flow")
        class DocstringFlow:
            """This is a test flow with docstring."""
            pass
        
        # Should use docstring as description
        flow_instance = DocstringFlow()
        assert "This is a test flow with docstring." in flow_instance.get_description()
    
    def test_agent_flow_decorator_no_description_no_docstring(self):
        """Test @agent_flow decorator with no description or docstring."""
        @agent_flow(name="empty_flow")
        class EmptyFlow:
            pass
        
        # Should use empty string as description
        flow_instance = EmptyFlow()
        assert flow_instance.get_description() == ""
    
    @patch('flowlib.agent.components.decorators.base.flow_registry')
    def test_agent_flow_decorator_registry_registration(self, mock_flow_registry):
        """Test that agent flow is registered with flow_registry."""
        mock_flow_registry.register_flow = Mock()
        
        @agent_flow(name="registry_flow", description="Registry test flow")
        class RegistryFlow:
            pass
        
        # Check that registration was attempted
        mock_flow_registry.register_flow.assert_called_once_with(
            "registry_flow", RegistryFlow
        )
    
    @patch('flowlib.agent.components.decorators.base.flow_registry')
    @patch('flowlib.agent.components.decorators.base.logger')
    def test_agent_flow_decorator_registry_failure(self, mock_logger, mock_flow_registry):
        """Test handling of registry registration failure."""
        mock_flow_registry.register_flow = Mock(side_effect=Exception("Registry error"))
        
        @agent_flow(name="failing_flow", description="Flow that fails registration")
        class FailingFlow:
            pass
        
        # Should log warning but not raise exception
        mock_logger.warning.assert_called_once()
        assert "Failed to register agent flow" in str(mock_logger.warning.call_args[0][0])


class TestDualPathAgentDecorator:
    """Test @dual_path_agent decorator functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Clear any registrations
        try:
            from flowlib.agent.registry import agent_registry
            agent_registry._agents.clear()
        except (ImportError, AttributeError):
            pass
    
    def test_dual_path_agent_decorator_basic(self):
        """Test basic @dual_path_agent decorator usage."""
        @dual_path_agent(name="test_dual", description="Test dual path agent")
        class TestDualAgent(MockDualPathAgent):
            pass
        
        # Check that metadata was added
        assert hasattr(TestDualAgent, '__agent_name__')
        assert hasattr(TestDualAgent, '__agent_description__')
        assert hasattr(TestDualAgent, '__agent_metadata__')
        
        assert TestDualAgent.__agent_name__ == "test_dual"
        assert TestDualAgent.__agent_description__ == "Test dual path agent"
        assert TestDualAgent.__agent_metadata__['agent_type'] == 'dual_path'
    
    def test_dual_path_agent_decorator_with_metadata(self):
        """Test @dual_path_agent decorator with additional metadata."""
        @dual_path_agent(
            name="meta_dual",
            description="Dual agent with metadata",
            version="1.0",
            capabilities=["planning", "execution"]
        )
        class MetaDualAgent(MockDualPathAgent):
            pass
        
        metadata = MetaDualAgent.__agent_metadata__
        assert metadata['agent_type'] == 'dual_path'
        assert metadata['version'] == "1.0"
        assert metadata['capabilities'] == ["planning", "execution"]
    
    @patch('flowlib.agent.components.decorators.base.agent_registry')
    def test_dual_path_agent_decorator_registration(self, mock_agent_registry):
        """Test that dual path agent is registered."""
        mock_agent_registry.register = Mock()
        
        @dual_path_agent(name="registered_dual", description="Registered dual agent")
        class RegisteredDualAgent(MockDualPathAgent):
            pass
        
        # Check that registration was called
        mock_agent_registry.register.assert_called_once_with(
            name="registered_dual",
            agent_class=RegisteredDualAgent,
            metadata={'agent_type': 'dual_path'}
        )
    
    def test_dual_path_agent_decorator_returns_class(self):
        """Test that decorator returns the original class."""
        original_class = type('OriginalClass', (MockDualPathAgent,), {})
        
        @dual_path_agent(name="return_test", description="Return test agent")
        class ReturnTestAgent(MockDualPathAgent):
            pass
        
        # Should be the same class with added attributes
        assert ReturnTestAgent.__name__ == "ReturnTestAgent"
        assert callable(ReturnTestAgent)


class TestDecoratorIntegration:
    """Test integration between different decorators."""
    
    def test_agent_and_agent_flow_integration(self):
        """Test using both @agent and @agent_flow decorators together."""
        @agent_flow(name="integrated_flow", description="Integrated test flow")
        class IntegratedFlow(Flow):
            def __init__(self, name="integrated_flow"):
                super().__init__(name)
            
            def get_description(self):
                return "Integrated flow"
        
        @agent(name="integrated_agent", provider_name="llamacpp")
        class IntegratedAgent:
            pass
        
        # Create agent with the flow
        flow_instance = IntegratedFlow()
        agent_instance = IntegratedAgent.create(flows=[flow_instance])
        
        # Verify integration
        flows = agent_instance.get_flows()
        assert "integrated_flow" in flows
        assert flows["integrated_flow"].__class__.__flow_metadata__['agent_flow'] is True
    
    def test_multiple_agent_decorators(self):
        """Test creating multiple agents with decorators."""
        @agent(name="agent_one", provider_name="llamacpp")
        class AgentOne:
            def method_one(self):
                return "one"
        
        @agent(name="agent_two", provider_name="openai")
        class AgentTwo:
            def method_two(self):
                return "two"
        
        agent1 = AgentOne.create()
        agent2 = AgentTwo.create()
        
        # Verify they're independent
        assert agent1.config.name == "agent_one"
        assert agent2.config.name == "agent_two"
        assert agent1.config.provider_name == "llamacpp"
        assert agent2.config.provider_name == "openai"
        
        # Verify method delegation
        assert agent1.method_one() == "one"
        assert agent2.method_two() == "two"


class TestDecoratorErrorHandling:
    """Test error handling in decorators."""
    
    def test_agent_decorator_invalid_config(self):
        """Test agent decorator with invalid config."""
        @agent(name="invalid_agent", provider_name="llamacpp")
        class InvalidAgent:
            pass
        
        # Test with invalid config type
        with pytest.raises((TypeError, ValueError)):
            InvalidAgent.create(config="invalid_config_type")
    
    def test_agent_flow_decorator_flow_registry_none(self):
        """Test agent_flow decorator when flow_registry is None."""
        with patch('flowlib.agent.components.decorators.base.flow_registry', None):
            @agent_flow(name="no_registry_flow", description="No registry flow")
            class NoRegistryFlow:
                pass
            
            # Should not raise exception
            assert hasattr(NoRegistryFlow, '__flow_metadata__')
    
    def test_dual_path_agent_decorator_missing_registry(self):
        """Test dual_path_agent decorator when registry import fails."""
        with patch('flowlib.agent.components.decorators.base.agent_registry') as mock_registry:
            mock_registry.register = Mock(side_effect=Exception("Registry error"))
            
            # Should not raise exception during decoration
            @dual_path_agent(name="error_dual", description="Error dual agent")
            class ErrorDualAgent(MockDualPathAgent):
                pass
            
            # Decorator should complete successfully
            assert hasattr(ErrorDualAgent, '__agent_name__')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])