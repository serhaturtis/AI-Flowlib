"""Comprehensive tests for agent registry module."""

import pytest
import logging
from typing import Dict, Type, Any, Optional
from unittest.mock import Mock, patch

from flowlib.agent.registry import AgentRegistry, agent_registry


# Test helper classes
class MockAgent:
    """Mock agent class for testing."""
    
    def __init__(self, name: str = "mock_agent"):
        self.name = name
    
    def run(self):
        return f"Running {self.name}"


class AnotherMockAgent:
    """Another mock agent class for testing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    async def async_run(self):
        return "Async running"


class ComplexMockAgent:
    """Complex mock agent with metadata."""
    
    version = "1.0.0"
    description = "A complex test agent"
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)


class TestAgentRegistry:
    """Test AgentRegistry class."""
    
    def test_agent_registry_creation(self):
        """Test creating agent registry instance."""
        registry = AgentRegistry()
        
        assert registry._agents == {}
        assert isinstance(registry._agents, dict)
    
    def test_register_agent_basic(self):
        """Test basic agent registration."""
        registry = AgentRegistry()
        
        registry.register("test_agent", MockAgent)
        
        assert "test_agent" in registry._agents
        agent_info = registry._agents["test_agent"]
        assert agent_info.agent_class == MockAgent
        assert agent_info.metadata == {}
    
    def test_register_agent_with_metadata(self):
        """Test agent registration with metadata."""
        registry = AgentRegistry()
        metadata = {
            "version": "1.0.0",
            "description": "Test agent",
            "author": "Test Author",
            "tags": ["test", "mock"]
        }
        
        registry.register("test_agent", MockAgent, metadata)
        
        agent_info = registry._agents["test_agent"]
        assert agent_info.agent_class == MockAgent
        assert agent_info.metadata == metadata
    
    def test_register_agent_duplicate_warning(self):
        """Test registering duplicate agent logs warning."""
        registry = AgentRegistry()
        
        # Register first time
        registry.register("duplicate_agent", MockAgent)
        
        # Register again - should log warning but allow overwrite
        with patch('flowlib.agent.registry.logger') as mock_logger:
            registry.register("duplicate_agent", AnotherMockAgent)
            
            mock_logger.warning.assert_called_once()
            assert "already registered" in mock_logger.warning.call_args[0][0]
            assert "duplicate_agent" in mock_logger.warning.call_args[0][0]
        
        # Verify the agent was overwritten
        assert registry._agents["duplicate_agent"].agent_class == AnotherMockAgent
    
    def test_register_agent_none_metadata(self):
        """Test registering agent with None metadata."""
        registry = AgentRegistry()
        
        registry.register("none_metadata_agent", MockAgent, None)
        
        assert registry._agents["none_metadata_agent"].metadata == {}
    
    def test_register_agent_empty_metadata(self):
        """Test registering agent with empty metadata."""
        registry = AgentRegistry()
        
        registry.register("empty_metadata_agent", MockAgent, {})
        
        assert registry._agents["empty_metadata_agent"].metadata == {}
    
    def test_register_agent_logs_debug(self):
        """Test that registration logs debug message."""
        registry = AgentRegistry()
        
        with patch('flowlib.agent.registry.logger') as mock_logger:
            registry.register("debug_agent", MockAgent)
            
            mock_logger.debug.assert_called_once()
            debug_message = mock_logger.debug.call_args[0][0]
            assert "Registered agent: debug_agent" in debug_message
            assert "MockAgent" in debug_message
    
    def test_get_agent_class_existing(self):
        """Test getting existing agent class."""
        registry = AgentRegistry()
        registry.register("existing_agent", MockAgent)
        
        agent_class = registry.get_agent_class("existing_agent")
        
        assert agent_class == MockAgent
    
    def test_get_agent_class_nonexistent(self):
        """Test getting nonexistent agent class."""
        registry = AgentRegistry()
        
        agent_class = registry.get_agent_class("nonexistent_agent")
        
        assert agent_class is None
    
    def test_get_agent_class_multiple_agents(self):
        """Test getting agent class with multiple registered agents."""
        registry = AgentRegistry()
        registry.register("agent1", MockAgent)
        registry.register("agent2", AnotherMockAgent)
        registry.register("agent3", ComplexMockAgent)
        
        assert registry.get_agent_class("agent1") == MockAgent
        assert registry.get_agent_class("agent2") == AnotherMockAgent
        assert registry.get_agent_class("agent3") == ComplexMockAgent
        assert registry.get_agent_class("nonexistent") is None
    
    def test_get_agent_metadata_existing(self):
        """Test getting metadata for existing agent."""
        registry = AgentRegistry()
        metadata = {"version": "2.0", "type": "test"}
        registry.register("metadata_agent", MockAgent, metadata)
        
        retrieved_metadata = registry.get_agent_metadata("metadata_agent")
        
        assert retrieved_metadata == metadata
    
    def test_get_agent_metadata_existing_empty(self):
        """Test getting empty metadata for existing agent."""
        registry = AgentRegistry()
        registry.register("empty_meta_agent", MockAgent)
        
        metadata = registry.get_agent_metadata("empty_meta_agent")
        
        assert metadata == {}
    
    def test_get_agent_metadata_nonexistent(self):
        """Test getting metadata for nonexistent agent."""
        registry = AgentRegistry()
        
        metadata = registry.get_agent_metadata("nonexistent_agent")
        
        assert metadata is None
    
    def test_get_agent_info_existing(self):
        """Test getting full agent info for existing agent."""
        registry = AgentRegistry()
        metadata = {"version": "1.0", "description": "Test"}
        registry.register("info_agent", MockAgent, metadata)
        
        info = registry.get_agent_info("info_agent")
        
        assert info is not None
        assert info.agent_class == MockAgent
        assert info.metadata == metadata
    
    def test_get_agent_info_nonexistent(self):
        """Test getting info for nonexistent agent."""
        registry = AgentRegistry()
        
        info = registry.get_agent_info("nonexistent_agent")
        
        assert info is None
    
    def test_list_agents_empty(self):
        """Test listing agents from empty registry."""
        registry = AgentRegistry()
        
        agents = registry.list_agents()
        
        assert agents == []
    
    def test_list_agents_single(self):
        """Test listing single agent."""
        registry = AgentRegistry()
        registry.register("single_agent", MockAgent)
        
        agents = registry.list_agents()
        
        assert agents == ["single_agent"]
    
    def test_list_agents_multiple_sorted(self):
        """Test listing multiple agents are sorted."""
        registry = AgentRegistry()
        registry.register("zebra_agent", MockAgent)
        registry.register("alpha_agent", AnotherMockAgent)
        registry.register("beta_agent", ComplexMockAgent)
        
        agents = registry.list_agents()
        
        assert agents == ["alpha_agent", "beta_agent", "zebra_agent"]
    
    def test_list_agents_with_special_characters(self):
        """Test listing agents with special characters in names."""
        registry = AgentRegistry()
        registry.register("agent_1", MockAgent)
        registry.register("agent-2", AnotherMockAgent)
        registry.register("agent.3", ComplexMockAgent)
        registry.register("agent@4", MockAgent)
        
        agents = registry.list_agents()
        
        # Should be sorted lexicographically
        expected = ["agent-2", "agent.3", "agent@4", "agent_1"]
        assert agents == expected
    
    def test_clear_registry_empty(self):
        """Test clearing empty registry."""
        registry = AgentRegistry()
        
        registry.clear()
        
        assert registry._agents == {}
    
    def test_clear_registry_with_agents(self):
        """Test clearing registry with agents."""
        registry = AgentRegistry()
        registry.register("agent1", MockAgent)
        registry.register("agent2", AnotherMockAgent, {"version": "1.0"})
        
        # Verify agents are there
        assert len(registry._agents) == 2
        
        registry.clear()
        
        assert registry._agents == {}
        assert registry.list_agents() == []
    
    def test_clear_registry_logs_debug(self):
        """Test that clearing registry logs debug message."""
        registry = AgentRegistry()
        registry.register("test_agent", MockAgent)
        
        with patch('flowlib.agent.registry.logger') as mock_logger:
            registry.clear()
            
            mock_logger.debug.assert_called_once_with("Agent registry cleared.")
    
    def test_registry_state_isolation(self):
        """Test that registry instances are isolated."""
        registry1 = AgentRegistry()
        registry2 = AgentRegistry()
        
        registry1.register("agent1", MockAgent)
        registry2.register("agent2", AnotherMockAgent)
        
        assert registry1.list_agents() == ["agent1"]
        assert registry2.list_agents() == ["agent2"]
        assert registry1.get_agent_class("agent2") is None
        assert registry2.get_agent_class("agent1") is None
    
    def test_registry_with_complex_classes(self):
        """Test registry with complex agent classes."""
        registry = AgentRegistry()
        
        # Register class with class attributes and methods
        metadata = {
            "version": ComplexMockAgent.version,
            "description": ComplexMockAgent.description,
            "has_create_method": hasattr(ComplexMockAgent, 'create')
        }
        
        registry.register("complex_agent", ComplexMockAgent, metadata)
        
        # Verify registration
        agent_class = registry.get_agent_class("complex_agent")
        assert agent_class == ComplexMockAgent
        assert hasattr(agent_class, 'create')
        assert agent_class.version == "1.0.0"
        
        # Verify metadata
        retrieved_metadata = registry.get_agent_metadata("complex_agent")
        assert retrieved_metadata["version"] == "1.0.0"
        assert retrieved_metadata["has_create_method"] is True
    
    def test_registry_name_case_sensitivity(self):
        """Test that agent names are case sensitive."""
        registry = AgentRegistry()
        
        registry.register("TestAgent", MockAgent)
        registry.register("testagent", AnotherMockAgent)
        registry.register("TESTAGENT", ComplexMockAgent)
        
        assert registry.get_agent_class("TestAgent") == MockAgent
        assert registry.get_agent_class("testagent") == AnotherMockAgent
        assert registry.get_agent_class("TESTAGENT") == ComplexMockAgent
        assert registry.get_agent_class("TestAGENT") is None
        
        agents = registry.list_agents()
        assert "TestAgent" in agents
        assert "testagent" in agents
        assert "TESTAGENT" in agents
        assert len(agents) == 3
    
    def test_registry_edge_case_names(self):
        """Test registry with edge case agent names."""
        registry = AgentRegistry()
        
        edge_case_names = [
            "",  # Empty string
            " ",  # Whitespace
            "agent with spaces",
            "agent\twith\ttabs",
            "agent\nwith\nnewlines",
            "agent-with-dashes",
            "agent_with_underscores",
            "agent.with.dots",
            "agent123",
            "123agent",
            "αβγ",  # Unicode
            "🤖agent",  # Emoji
        ]
        
        for name in edge_case_names:
            registry.register(name, MockAgent)
            assert registry.get_agent_class(name) == MockAgent
        
        # All should be listed
        agents = registry.list_agents()
        assert len(agents) == len(edge_case_names)
        for name in edge_case_names:
            assert name in agents


class TestGlobalAgentRegistry:
    """Test the global agent registry instance."""
    
    def setup_method(self):
        """Clear the global registry before each test."""
        agent_registry.clear()
    
    def teardown_method(self):
        """Clear the global registry after each test."""
        agent_registry.clear()
    
    def test_global_registry_exists(self):
        """Test that global agent registry exists."""
        assert agent_registry is not None
        assert isinstance(agent_registry, AgentRegistry)
    
    def test_global_registry_singleton_behavior(self):
        """Test that global registry behaves like singleton."""
        from flowlib.agent.registry import agent_registry as imported_registry
        
        # Should be the same instance
        assert agent_registry is imported_registry
        
        # Registering in one should affect the other
        agent_registry.register("global_test", MockAgent)
        assert imported_registry.get_agent_class("global_test") == MockAgent
    
    def test_global_registry_basic_operations(self):
        """Test basic operations on global registry."""
        # Should start empty (due to setup_method)
        assert agent_registry.list_agents() == []
        
        # Register agent
        agent_registry.register("global_agent", MockAgent, {"global": True})
        
        # Verify registration
        assert agent_registry.get_agent_class("global_agent") == MockAgent
        assert agent_registry.get_agent_metadata("global_agent")["global"] is True
        assert "global_agent" in agent_registry.list_agents()
    
    def test_global_registry_persistence_across_imports(self):
        """Test that global registry persists across imports."""
        # Register in global registry
        agent_registry.register("persistent_agent", MockAgent)
        
        # Import in a different way and verify it's there
        from flowlib.agent.registry import agent_registry as fresh_import
        assert fresh_import.get_agent_class("persistent_agent") == MockAgent
    
    def test_global_registry_multiple_test_isolation(self):
        """Test that tests are properly isolated."""
        # This test depends on setup_method/teardown_method working
        assert agent_registry.list_agents() == []
        
        agent_registry.register("isolation_test", MockAgent)
        assert len(agent_registry.list_agents()) == 1


class TestAgentRegistryIntegration:
    """Test integration aspects of the agent registry."""
    
    def test_registry_with_actual_agent_pattern(self):
        """Test registry with realistic agent usage pattern."""
        registry = AgentRegistry()
        
        # Simulate typical agent registration pattern
        class ChatAgent:
            def __init__(self, model: str = "gpt-3.5-turbo"):
                self.model = model
            
            async def chat(self, message: str) -> str:
                return f"Response to: {message}"
        
        class CodeAgent:
            def __init__(self, language: str = "python"):
                self.language = language
            
            def generate_code(self, prompt: str) -> str:
                return f"# Generated {self.language} code for: {prompt}"
        
        # Register with realistic metadata
        chat_metadata = {
            "type": "conversational",
            "capabilities": ["chat", "qa"],
            "model_support": ["gpt-3.5-turbo", "gpt-4"],
            "version": "1.0.0"
        }
        
        code_metadata = {
            "type": "code_generation", 
            "capabilities": ["code_gen", "refactoring"],
            "languages": ["python", "javascript", "rust"],
            "version": "0.9.0"
        }
        
        registry.register("chat_agent", ChatAgent, chat_metadata)
        registry.register("code_agent", CodeAgent, code_metadata)
        
        # Test retrieval and usage
        chat_class = registry.get_agent_class("chat_agent")
        code_class = registry.get_agent_class("code_agent")
        
        assert chat_class == ChatAgent
        assert code_class == CodeAgent
        
        # Test instantiation
        chat_agent = chat_class(model="gpt-4")
        code_agent = code_class(language="rust")
        
        assert chat_agent.model == "gpt-4"
        assert code_agent.language == "rust"
        
        # Test metadata retrieval
        chat_meta = registry.get_agent_metadata("chat_agent")
        assert "gpt-4" in chat_meta["model_support"]
        assert chat_meta["type"] == "conversational"
        
        code_meta = registry.get_agent_metadata("code_agent")
        assert "rust" in code_meta["languages"]
        assert code_meta["version"] == "0.9.0"
    
    def test_registry_error_handling_robustness(self):
        """Test registry validation with strict contracts."""
        import pytest
        from pydantic_core import ValidationError
        
        registry = AgentRegistry()
        
        # Test with None class (should be rejected by strict validation)
        with pytest.raises(ValidationError, match="Input should be a type"):
            registry.register("none_class", None)
        
        # Test with non-class objects (should be rejected by strict validation)
        with pytest.raises(ValidationError, match="Input should be a type"):
            registry.register("string_object", "not_a_class")
        
        # Test with functions (should be rejected by strict validation)
        def mock_function():
            return "I'm a function"
        
        with pytest.raises(ValidationError, match="Input should be a type"):
            registry.register("function_agent", mock_function)
        
        # Test with lambda (should be rejected by strict validation)
        lambda_agent = lambda x: f"Lambda result: {x}"
        with pytest.raises(ValidationError, match="Input should be a type"):
            registry.register("lambda_agent", lambda_agent)
        
        # Registry should be empty since all registrations were rejected
        agents = registry.list_agents()
        assert agents == []
    
    def test_registry_large_scale_operations(self):
        """Test registry performance with many agents."""
        registry = AgentRegistry()
        
        # Register many agents
        num_agents = 100
        for i in range(num_agents):
            agent_name = f"agent_{i:03d}"
            metadata = {
                "id": i,
                "category": "batch" if i % 2 == 0 else "interactive",
                "priority": i % 5
            }
            registry.register(agent_name, MockAgent, metadata)
        
        # Test listing (should be sorted)
        agents = registry.list_agents()
        assert len(agents) == num_agents
        assert agents == sorted(agents)
        
        # Test retrieval of random agents
        test_indices = [0, 25, 50, 75, 99]
        for i in test_indices:
            agent_name = f"agent_{i:03d}"
            agent_class = registry.get_agent_class(agent_name)
            metadata = registry.get_agent_metadata(agent_name)
            
            assert agent_class == MockAgent
            assert metadata["id"] == i
            assert metadata["category"] in ["batch", "interactive"]
        
        # Test clearing large registry
        registry.clear()
        assert len(registry.list_agents()) == 0
    
    def test_registry_thread_safety_simulation(self):
        """Test registry operations that might occur concurrently."""
        registry = AgentRegistry()
        
        # Simulate operations that might happen in different threads
        operations = [
            ("register", "agent_a", MockAgent, {"thread": "1"}),
            ("register", "agent_b", AnotherMockAgent, {"thread": "2"}),
            ("get_class", "agent_a"),
            ("register", "agent_c", ComplexMockAgent, {"thread": "3"}),
            ("list", None),
            ("get_metadata", "agent_b"),
            ("register", "agent_a", ComplexMockAgent, {"thread": "1_update"}),  # Overwrite
            ("get_info", "agent_c"),
            ("clear", None),
            ("list", None)
        ]
        
        results = []
        for op in operations:
            if op[0] == "register":
                registry.register(op[1], op[2], op[3])
                results.append(f"registered_{op[1]}")
            elif op[0] == "get_class":
                result = registry.get_agent_class(op[1])
                results.append(result.__name__ if result else None)
            elif op[0] == "list":
                result = registry.list_agents()
                results.append(len(result))
            elif op[0] == "get_metadata":
                result = registry.get_agent_metadata(op[1])
                results.append(result.get("thread") if result else None)
            elif op[0] == "get_info":
                result = registry.get_agent_info(op[1])
                results.append(result is not None)
            elif op[0] == "clear":
                registry.clear()
                results.append("cleared")
        
        # Verify expected sequence results
        expected_results = [
            "registered_agent_a",     # register agent_a
            "registered_agent_b",     # register agent_b
            "MockAgent",              # get_class agent_a
            "registered_agent_c",     # register agent_c
            3,                        # list (3 agents)
            "2",                      # get_metadata agent_b (thread "2")
            "registered_agent_a",     # register agent_a (overwrite)
            True,                     # get_info agent_c (exists)
            "cleared",                # clear
            0                         # list (0 agents after clear)
        ]
        
        assert results == expected_results