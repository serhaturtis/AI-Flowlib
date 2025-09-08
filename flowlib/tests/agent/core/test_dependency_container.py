"""Tests for agent dependency container."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from flowlib.agent.core.dependency_container import (
    ComponentContainer,
    ComponentBuilder,
    AgentComponentRegistry
)
from flowlib.agent.core.errors import ComponentError


class MockComponent:
    """Mock component for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.shutdown_called = False
    
    async def initialize(self):
        """Mock initialize method."""
        self.initialized = True
    
    async def shutdown(self):
        """Mock shutdown method."""
        self.shutdown_called = True


class MockProvider:
    """Mock provider for testing."""
    
    def __init__(self, name: str):
        self.name = name


class TestComponentContainer:
    """Test ComponentContainer functionality."""
    
    def test_container_initialization(self):
        """Test container initialization."""
        container = ComponentContainer()
        
        assert container._components == {}
        assert container._providers == {}
        assert container._dependencies == {}
        assert container._initialized == set()
        assert container._initializing == set()
    
    @pytest.mark.asyncio
    async def test_register_component_success(self):
        """Test successful component registration."""
        container = ComponentContainer()
        component = MockComponent("test_component")
        
        await container.register_component("test", component)
        
        assert "test" in container._components
        assert container._components["test"] == component
    
    @pytest.mark.asyncio
    async def test_register_component_with_dependencies(self):
        """Test component registration with dependencies."""
        container = ComponentContainer()
        component = MockComponent("test_component")
        dependencies = ["dep1", "dep2"]
        
        await container.register_component("test", component, dependencies)
        
        assert "test" in container._components
        assert container._dependencies["test"] == dependencies
    
    @pytest.mark.asyncio
    async def test_register_component_duplicate_name(self):
        """Test registering component with duplicate name."""
        container = ComponentContainer()
        component1 = MockComponent("component1")
        component2 = MockComponent("component2")
        
        await container.register_component("test", component1)
        
        with pytest.raises(ComponentError, match="Component 'test' already registered"):
            await container.register_component("test", component2)
    
    @pytest.mark.asyncio
    async def test_register_provider_success(self):
        """Test successful provider registration."""
        container = ComponentContainer()
        provider = MockProvider("test_provider")
        
        await container.register_provider("test_provider", provider)
        
        assert "test_provider" in container._providers
        assert container._providers["test_provider"] == provider
    
    def test_get_component_by_type_success(self):
        """Test successful component retrieval by type."""
        container = ComponentContainer()
        component = MockComponent("test")
        container._components["test"] = component
        
        retrieved = container.get_component(MockComponent)
        assert retrieved == component
    
    def test_get_component_by_type_not_found(self):
        """Test component retrieval by type when not found."""
        container = ComponentContainer()
        
        with pytest.raises(ComponentError, match="No component of type MockComponent found"):
            container.get_component(MockComponent)
    
    def test_get_component_by_name_success(self):
        """Test successful component retrieval by name."""
        container = ComponentContainer()
        component = MockComponent("test")
        container._components["test"] = component
        
        retrieved = container.get_component_by_name("test")
        assert retrieved == component
    
    def test_get_component_by_name_not_found(self):
        """Test component retrieval by name when not found."""
        container = ComponentContainer()
        
        with pytest.raises(ComponentError, match="Component 'nonexistent' not found"):
            container.get_component_by_name("nonexistent")
    
    def test_get_provider_success(self):
        """Test successful provider retrieval."""
        container = ComponentContainer()
        provider = MockProvider("test")
        container._providers["test"] = provider
        
        retrieved = container.get_provider("test")
        assert retrieved == provider
    
    def test_get_provider_not_found(self):
        """Test provider retrieval when not found."""
        container = ComponentContainer()
        
        with pytest.raises(ComponentError, match="Provider 'nonexistent' not found"):
            container.get_provider("nonexistent")
    
    @pytest.mark.asyncio
    async def test_initialize_all_no_dependencies(self):
        """Test initializing components without dependencies."""
        container = ComponentContainer()
        
        comp1 = MockComponent("comp1")
        comp2 = MockComponent("comp2")
        
        await container.register_component("comp1", comp1)
        await container.register_component("comp2", comp2)
        
        await container.initialize_all()
        
        assert comp1.initialized is True
        assert comp2.initialized is True
        assert "comp1" in container._initialized
        assert "comp2" in container._initialized
    
    @pytest.mark.asyncio
    async def test_initialize_all_with_dependencies(self):
        """Test initializing components with dependencies."""
        container = ComponentContainer()
        
        base_comp = MockComponent("base")
        dependent_comp = MockComponent("dependent")
        
        await container.register_component("base", base_comp)
        await container.register_component("dependent", dependent_comp, ["base"])
        
        await container.initialize_all()
        
        assert base_comp.initialized is True
        assert dependent_comp.initialized is True
        assert "base" in container._initialized
        assert "dependent" in container._initialized
    
    @pytest.mark.asyncio
    async def test_initialize_all_circular_dependency(self):
        """Test initialization with circular dependencies."""
        container = ComponentContainer()
        
        comp1 = MockComponent("comp1")
        comp2 = MockComponent("comp2")
        
        await container.register_component("comp1", comp1, ["comp2"])
        await container.register_component("comp2", comp2, ["comp1"])
        
        with pytest.raises(ComponentError, match="Circular dependencies detected"):
            await container.initialize_all()
    
    @pytest.mark.asyncio
    async def test_initialize_component_already_initialized(self):
        """Test initializing already initialized component."""
        container = ComponentContainer()
        component = MockComponent("test")
        
        await container.register_component("test", component)
        container._initialized.add("test")
        
        # Should not re-initialize
        await container._initialize_component("test")
        assert component.initialized is False  # Should not be called again
    
    @pytest.mark.asyncio
    async def test_initialize_component_circular_detection(self):
        """Test circular dependency detection during initialization."""
        container = ComponentContainer()
        
        comp1 = MockComponent("comp1")
        comp2 = MockComponent("comp2")
        
        await container.register_component("comp1", comp1, ["comp2"])
        await container.register_component("comp2", comp2, ["comp1"])
        
        with pytest.raises(ComponentError, match="Component initialization failed: comp1"):
            await container._initialize_component("comp1")
    
    @pytest.mark.asyncio
    async def test_initialize_component_failure(self):
        """Test component initialization failure."""
        container = ComponentContainer()
        
        class FailingComponent:
            async def initialize(self):
                raise RuntimeError("Initialization failed")
        
        failing_comp = FailingComponent()
        await container.register_component("failing", failing_comp)
        
        with pytest.raises(ComponentError, match="Component initialization failed: failing"):
            await container._initialize_component("failing")
    
    @pytest.mark.asyncio
    async def test_shutdown_all_success(self):
        """Test successful shutdown of all components."""
        container = ComponentContainer()
        
        comp1 = MockComponent("comp1")
        comp2 = MockComponent("comp2")
        
        await container.register_component("comp1", comp1)
        await container.register_component("comp2", comp2, ["comp1"])
        
        await container.initialize_all()
        await container.shutdown_all()
        
        assert comp1.shutdown_called is True
        assert comp2.shutdown_called is True
        assert len(container._initialized) == 0
    
    @pytest.mark.asyncio
    async def test_shutdown_all_with_errors(self):
        """Test shutdown with component errors."""
        container = ComponentContainer()
        
        class FailingShutdownComponent:
            async def shutdown(self):
                raise RuntimeError("Shutdown failed")
        
        good_comp = MockComponent("good")
        failing_comp = FailingShutdownComponent()
        
        await container.register_component("good", good_comp)
        await container.register_component("failing", failing_comp)
        
        container._initialized.add("good")
        container._initialized.add("failing")
        
        # Should not raise exception, just log errors
        await container.shutdown_all()
        
        assert good_comp.shutdown_called is True
        assert len(container._initialized) == 0
    
    @pytest.mark.asyncio
    async def test_shutdown_component_without_shutdown_method(self):
        """Test shutdown of component without shutdown method."""
        container = ComponentContainer()
        
        class NoShutdownComponent:
            pass
        
        component = NoShutdownComponent()
        await container.register_component("no_shutdown", component)
        container._initialized.add("no_shutdown")
        
        # Should not raise exception
        await container.shutdown_all()
        assert len(container._initialized) == 0
    
    def test_get_initialization_order_simple(self):
        """Test getting initialization order for simple dependencies."""
        container = ComponentContainer()
        container._components = {"a": Mock(), "b": Mock(), "c": Mock()}
        container._dependencies = {"b": ["a"], "c": ["b"]}
        
        order = container._get_initialization_order()
        
        assert order == ["a", "b", "c"]
    
    def test_get_initialization_order_complex(self):
        """Test getting initialization order for complex dependencies."""
        container = ComponentContainer()
        container._components = {"a": Mock(), "b": Mock(), "c": Mock(), "d": Mock()}
        container._dependencies = {"b": ["a"], "c": ["a"], "d": ["b", "c"]}
        
        order = container._get_initialization_order()
        
        # 'a' must come first, 'd' must come last
        assert order[0] == "a"
        assert order[-1] == "d"
        # 'b' and 'c' can be in any order
        assert set(order[1:3]) == {"b", "c"}
    
    def test_get_initialization_order_circular_dependency(self):
        """Test circular dependency detection in initialization order."""
        container = ComponentContainer()
        container._components = {"a": Mock(), "b": Mock()}
        container._dependencies = {"a": ["b"], "b": ["a"]}
        
        with pytest.raises(ComponentError, match="Circular dependencies detected"):
            container._get_initialization_order()
    
    def test_get_dependency_graph(self):
        """Test getting dependency graph."""
        container = ComponentContainer()
        container._dependencies = {"a": ["b"], "c": ["a", "b"]}
        
        graph = container.get_dependency_graph()
        
        assert graph == {"a": ["b"], "c": ["a", "b"]}
        # Should return a copy, not the original
        assert graph is not container._dependencies
    
    def test_get_component_status(self):
        """Test getting component status."""
        container = ComponentContainer()
        container._components = {"a": Mock(), "b": Mock(), "c": Mock()}
        container._initialized.add("a")
        container._initializing.add("b")
        
        status = container.get_component_status()
        
        assert status == {
            "a": "initialized",
            "b": "initializing", 
            "c": "not_initialized"
        }


class TestComponentBuilder:
    """Test ComponentBuilder functionality."""
    
    def test_builder_initialization(self):
        """Test builder initialization."""
        container = ComponentContainer()
        builder = ComponentBuilder(container)
        
        assert builder.container == container
        assert builder._pending_registrations == []
    
    def test_add_component(self):
        """Test adding component to builder."""
        container = ComponentContainer()
        builder = ComponentBuilder(container)
        component = MockComponent("test")
        
        result = builder.add_component("test", component, ["dep1"])
        
        assert result == builder  # Should return self for chaining
        assert len(builder._pending_registrations) == 1
        assert builder._pending_registrations[0] == ("test", component, ["dep1"])
    
    def test_add_provider(self):
        """Test adding provider to builder."""
        container = ComponentContainer()
        builder = ComponentBuilder(container)
        provider = MockProvider("test")
        
        result = builder.add_provider("test_provider", provider)
        
        assert result == builder  # Should return self for chaining
        assert len(builder._pending_registrations) == 1
        assert builder._pending_registrations[0] == ("test_provider", provider, None, True)
    
    @pytest.mark.asyncio
    async def test_build_success(self):
        """Test successful build process."""
        container = ComponentContainer()
        builder = ComponentBuilder(container)
        
        component = MockComponent("test_comp")
        provider = MockProvider("test_prov")
        
        builder.add_component("comp", component)
        builder.add_provider("prov", provider)
        
        result = await builder.build()
        
        assert result == container
        assert "comp" in container._components
        assert "prov" in container._providers
        assert component.initialized is True
    
    @pytest.mark.asyncio
    async def test_build_with_dependencies(self):
        """Test build process with component dependencies."""
        container = ComponentContainer()
        builder = ComponentBuilder(container)
        
        base_comp = MockComponent("base")
        dependent_comp = MockComponent("dependent")
        
        builder.add_component("base", base_comp)
        builder.add_component("dependent", dependent_comp, ["base"])
        
        result = await builder.build()
        
        assert base_comp.initialized is True
        assert dependent_comp.initialized is True


class TestAgentComponentRegistry:
    """Test AgentComponentRegistry functionality."""
    
    def test_create_standard_container(self):
        """Test creating standard container."""
        container = AgentComponentRegistry.create_standard_container()
        
        assert isinstance(container, ComponentContainer)
        assert container._components == {}
        assert container._providers == {}
    
    @pytest.mark.asyncio
    async def test_setup_memory_components_basic(self):
        """Test setting up basic memory components."""
        container = ComponentContainer()
        config = {}
        
        with patch('flowlib.agent.components.memory.working.WorkingMemory') as MockWorking, \
             patch('flowlib.agent.components.memory.vector.VectorMemory') as MockVector, \
             patch('flowlib.agent.components.memory.knowledge.KnowledgeMemory') as MockKnowledge:
            
            working_instance = Mock()
            vector_instance = Mock()
            knowledge_instance = Mock()
            
            MockWorking.return_value = working_instance
            MockVector.return_value = vector_instance
            MockKnowledge.return_value = knowledge_instance
            
            await AgentComponentRegistry.setup_memory_components(container, config)
            
            # Verify components were registered
            assert "working_memory" in container._components
            assert "vector_memory" in container._components
            assert "knowledge_memory" in container._components
            
            # Verify dependencies
            assert container._dependencies["vector_memory"] == ["working_memory"]
            assert set(container._dependencies["knowledge_memory"]) == {"working_memory", "vector_memory"}
    
    @pytest.mark.asyncio
    async def test_setup_memory_components_disabled_vector(self):
        """Test setting up memory components with vector memory disabled."""
        container = ComponentContainer()
        config = {"enable_vector_memory": False}
        
        with patch('flowlib.agent.components.memory.working.WorkingMemory') as MockWorking, \
             patch('flowlib.agent.components.memory.knowledge.KnowledgeMemory') as MockKnowledge:
            
            working_instance = Mock()
            knowledge_instance = Mock()
            
            MockWorking.return_value = working_instance
            MockKnowledge.return_value = knowledge_instance
            
            await AgentComponentRegistry.setup_memory_components(container, config)
            
            # Verify only working and knowledge memory registered
            assert "working_memory" in container._components
            assert "vector_memory" not in container._components
            assert "knowledge_memory" in container._components
            
            # Verify knowledge memory only depends on working memory
            assert container._dependencies["knowledge_memory"] == ["working_memory"]
    
    @pytest.mark.asyncio
    async def test_setup_memory_components_disabled_knowledge(self):
        """Test setting up memory components with knowledge memory disabled."""
        container = ComponentContainer()
        config = {"enable_knowledge_memory": False}
        
        with patch('flowlib.agent.components.memory.working.WorkingMemory') as MockWorking, \
             patch('flowlib.agent.components.memory.vector.VectorMemory') as MockVector:
            
            working_instance = Mock()
            vector_instance = Mock()
            
            MockWorking.return_value = working_instance
            MockVector.return_value = vector_instance
            
            await AgentComponentRegistry.setup_memory_components(container, config)
            
            # Verify working and vector memory registered, but not knowledge
            assert "working_memory" in container._components
            assert "vector_memory" in container._components
            assert "knowledge_memory" not in container._components
    
    @pytest.mark.asyncio
    async def test_setup_planning_components(self):
        """Test setting up planning components."""
        container = ComponentContainer()
        config = {}
        
        # First register working memory as dependency
        working_memory = MockComponent("working")
        await container.register_component("working_memory", working_memory)
        
        with patch('flowlib.agent.components.task_decomposition.planner.AgentPlanner') as MockPlanning:
            planning_instance = Mock()
            MockPlanning.return_value = planning_instance
            
            await AgentComponentRegistry.setup_planning_components(container, config)
            
            # Verify planning component was registered
            assert "planning" in container._components
            assert container._dependencies["planning"] == ["working_memory"]
    


class TestComponentContainerIntegration:
    """Test component container integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_lifecycle_integration(self):
        """Test full component lifecycle integration."""
        container = ComponentContainer()
        
        # Create components with dependencies
        base_comp = MockComponent("base")
        middle_comp = MockComponent("middle")
        top_comp = MockComponent("top")
        
        # Register in reverse dependency order to test sorting
        await container.register_component("top", top_comp, ["middle"])
        await container.register_component("middle", middle_comp, ["base"])
        await container.register_component("base", base_comp)
        
        # Add a provider
        provider = MockProvider("test_provider")
        await container.register_provider("provider", provider)
        
        # Initialize all
        await container.initialize_all()
        
        # Verify initialization order was correct
        assert base_comp.initialized is True
        assert middle_comp.initialized is True
        assert top_comp.initialized is True
        
        # Verify we can retrieve components
        assert container.get_component(MockComponent) in [base_comp, middle_comp, top_comp]
        assert container.get_component_by_name("top") == top_comp
        assert container.get_provider("provider") == provider
        
        # Shutdown all
        await container.shutdown_all()
        
        assert base_comp.shutdown_called is True
        assert middle_comp.shutdown_called is True
        assert top_comp.shutdown_called is True
    
    @pytest.mark.asyncio
    async def test_builder_pattern_integration(self):
        """Test builder pattern integration."""
        container = ComponentContainer()
        
        base_comp = MockComponent("base")
        dependent_comp = MockComponent("dependent")
        provider = MockProvider("provider")
        
        # Use builder pattern
        result_container = await (ComponentBuilder(container)
                                  .add_component("base", base_comp)
                                  .add_component("dependent", dependent_comp, ["base"])
                                  .add_provider("provider", provider)
                                  .build())
        
        assert result_container == container
        assert base_comp.initialized is True
        assert dependent_comp.initialized is True
        assert "provider" in container._providers
    
    @pytest.mark.asyncio
    async def test_status_monitoring_during_lifecycle(self):
        """Test component status monitoring during lifecycle."""
        container = ComponentContainer()
        
        component = MockComponent("test")
        await container.register_component("test", component)
        
        # Check initial status
        status = container.get_component_status()
        assert status["test"] == "not_initialized"
        
        # Initialize
        await container.initialize_all()
        status = container.get_component_status()
        assert status["test"] == "initialized"
        
        # Shutdown
        await container.shutdown_all()
        # After shutdown, component is no longer in initialized set
        status = container.get_component_status()
        assert status["test"] == "not_initialized"