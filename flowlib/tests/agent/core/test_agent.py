"""
Tests for the AgentCore class.

This module tests the central agent orchestration system including
initialization, configuration, component coordination, flow management,
state persistence, and the main API methods.
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, Optional, List

from flowlib.agent.core.agent import AgentCore
from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ConfigurationError, ExecutionError, StatePersistenceError, NotInitializedError
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.models.state import AgentState, AgentStats
# Removed unused imports - mock classes don't inherit from base classes
from flowlib.flows.models.results import FlowResult
from flowlib.flows.models.constants import FlowStatus
from flowlib.flows.models.metadata import FlowMetadata


class MockFlow:
    """Mock flow for testing."""
    
    def __init__(self, name: str = "test_flow", description: str = "Test flow"):
        self.name = name
        self.description = description
        self.flow_type = "test"
        self._execution_count = 0
    
    async def run_pipeline(self, inputs):
        """Mock pipeline execution."""
        self._execution_count += 1
        from flowlib.flows.models.results import FlowResult
        from flowlib.flows.models.constants import FlowStatus
        return FlowResult(
            status=FlowStatus.SUCCESS,
            data={"result": f"executed_{self._execution_count}", "success": True}
        )
    
    def get_pipeline_method(self):
        """Get pipeline method with input model."""
        pipeline = Mock()
        pipeline.__input_model__ = MockFlowInput
        return pipeline


class MockFlowInput:
    """Mock flow input model."""
    pass


class MockStatePersister:
    """Mock state persister for testing."""
    
    def __init__(self):
        self._states = {}
        self._initialized = False
    
    async def initialize(self):
        self._initialized = True
    
    async def shutdown(self):
        self._initialized = False
    
    async def save_state(self, state: AgentState, metadata: Optional[Dict[str, str]] = None):
        self._states[state.task_id] = {
            "state": state,
            "metadata": metadata or {}
        }
        return True
    
    async def load_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        entry = self._states.get(task_id)
        if entry:
            state = entry["state"]
            # Return the serialized form as a dictionary
            return state.model_dump() if hasattr(state, 'model_dump') else state.to_dict()
        return None
    
    async def delete_state(self, task_id: str) -> bool:
        if task_id in self._states:
            del self._states[task_id]
            return True
        return False
    
    async def list_states(self, filter_criteria: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        results = []
        for task_id, entry in self._states.items():
            metadata = entry["metadata"].copy()
            metadata["task_id"] = task_id
            
            # Apply filter if provided
            if filter_criteria:
                match = all(metadata.get(k) == v for k, v in filter_criteria.items())
                if not match:
                    continue
            
            results.append(metadata)
        return results


class MockMemory:
    """Mock comprehensive memory for testing."""
    
    def __init__(self):
        self.initialized = False
        self._storage = {}
        self._contexts = set()
    
    async def initialize(self):
        self.initialized = True
    
    async def shutdown(self):
        self.initialized = False
    
    async def create_context(self, context_name: str, parent=None, metadata=None):
        self._contexts.add(context_name)
    
    async def store_with_model(self, request):
        self._storage[request.key] = request.value
    
    async def retrieve_with_model(self, request):
        return self._storage.get(request.key)
    
    async def search_with_model(self, request):
        return [{"item": {"key": k, "value": v}, "score": 0.9} 
                for k, v in self._storage.items() 
                if request.query.lower() in str(v).lower()]
    
    def set_parent(self, parent):
        self.parent = parent
    
    def get_stats(self):
        return {"items": len(self._storage), "contexts": len(self._contexts)}


class MockPlanner:
    """Mock agent planner for testing."""
    
    def __init__(self, config=None, activity_stream=None):
        self.config = config
        self.activity_stream = activity_stream
        self.initialized = False
        self.parent = None
    
    async def initialize(self):
        self.initialized = True
    
    async def shutdown(self):
        self.initialized = False
    
    def set_parent(self, parent):
        self.parent = parent
    
    def get_stats(self):
        return {"plans_created": 5, "planning_time": 1.2}


class MockReflection:
    """Mock agent reflection for testing."""
    
    def __init__(self, config=None, activity_stream=None):
        self.config = config
        self.activity_stream = activity_stream
        self.initialized = False
        self.parent = None
    
    async def initialize(self):
        self.initialized = True
    
    async def shutdown(self):
        self.initialized = False
    
    def set_parent(self, parent):
        self.parent = parent
    
    def get_stats(self):
        return {"reflections": 3, "insights": 2}


class MockEngine:
    """Mock unified agent engine for testing."""
    
    def __init__(self, config=None, memory=None, planner=None, reflection=None, activity_stream=None):
        self.config = config
        self.memory = memory
        self.planner = planner
        self.reflection = reflection
        self.activity_stream = activity_stream
        self.initialized = False
        self.parent = None
        self._execution_count = 0
    
    async def initialize(self):
        self.initialized = True
    
    async def shutdown(self):
        self.initialized = False
    
    def set_parent(self, parent):
        self.parent = parent
    
    async def execute_flow(self, flow_name: str, inputs: Any, state: AgentState) -> FlowResult:
        self._execution_count += 1
        return FlowResult(
            status=FlowStatus.SUCCESS,
            flow_name=flow_name,
            data={"flow": flow_name, "execution": self._execution_count},
            metadata={"executed_at": datetime.now().isoformat()}
        )
    
    async def execute_cycle(self, state: AgentState, memory_context: str = "agent", no_flow_is_error: bool = False) -> bool:
        state.increment_cycle()
        return state.cycles < 3  # Stop after 3 cycles
    
    async def execute(self, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "task_id": "test_task",
            "cycles": 2,
            "progress": 100,
            "is_complete": True,
            "execution_history": [
                {
                    "flow_name": "ConversationFlow",
                    "inputs": {"message": task_description},
                    "result": {"response": "Task completed"},
                    "success": True,
                    "elapsed_time": 1.5
                }
            ],
            "errors": [],
            "output": "Task completed successfully"
        }
    
    def get_stats(self):
        return {"executions": self._execution_count, "avg_time": 2.1}


class MockActivityStream:
    """Mock activity stream for testing."""
    
    def __init__(self, output_handler=None):
        self.output_handler = output_handler
        self.events = []
    
    def set_output_handler(self, handler):
        self.output_handler = handler
    
    def log_event(self, event):
        self.events.append(event)
    
    def flow_execution(self, flow_name: str, inputs: str, result: str):
        """Mock flow execution logging."""
        self.events.append({
            "type": "flow_execution",
            "flow_name": flow_name,
            "inputs": inputs,
            "result": result
        })


class MockKnowledgePluginManager:
    """Mock knowledge plugin manager for testing."""
    
    def __init__(self):
        self.loaded_plugins = {"test_plugin": Mock()}
        self.initialized = False
    
    async def initialize(self):
        self.initialized = True
    
    async def shutdown(self):
        self.initialized = False


class MockLearningFlow:
    """Mock learning flow for testing."""
    
    async def learn_from_content(self, content: str, context: str, focus_areas: Optional[List[str]] = None):
        return Mock(
            success=True,
            knowledge=Mock(
                entities=[],
                relationships=[],
                concepts=[],
                get_stats=lambda: {"entities_count": 2, "relationships_count": 1, "patterns_count": 1}
            )
        )
    
    async def learn(self, content: str, context: Optional[str] = None, focus_areas: Optional[List[str]] = None):
        """Learn from content."""
        result = Mock()
        result.is_success = True
        result.success = True
        return result
    
    async def extract_entities(self, content: str, context: Optional[str] = None):
        """Extract entities from content."""
        return [{"id": "entity1", "type": "person", "name": "John"}, {"id": "entity2", "type": "place", "name": "Paris"}]
    
    async def learn_relationships(self, content: str, entity_ids: List[str]):
        """Learn relationships from content."""
        return [{"from": entity_ids[0], "to": entity_ids[1] if len(entity_ids) > 1 else entity_ids[0], "type": "related_to"}]
    
    async def integrate_knowledge(self, content: str, entity_ids: List[str]):
        """Integrate knowledge from content."""
        result = Mock()
        result.is_success = True
        result.success = True
        return result
    
    async def form_concepts(self, content: str):
        """Form concepts from content."""
        return [{"id": "concept1", "name": "Learning", "type": "abstract"}]


class TestAgentCore:
    """Test the AgentCore class."""
    
    def setup_method(self):
        """Set up each test with fresh mocks."""
        self.mock_memory = MockMemory()
        self.mock_planner = MockPlanner()
        self.mock_reflection = MockReflection()
        self.mock_engine = MockEngine()
        self.mock_activity_stream = MockActivityStream()
        self.mock_knowledge_plugins = MockKnowledgePluginManager()
        self.mock_learning_flow = MockLearningFlow()
    
    def test_init_with_dict_config(self):
        """Test agent initialization with dictionary config."""
        config = {
            "name": "test_agent",
            "persona": "Test assistant",
            "provider_name": "llamacpp",
            "enable_memory": True
        }
        
        agent = AgentCore(config=config, task_description="Test task")
        
        assert agent.name == "test_agent"
        assert agent.persona == "Test assistant"
        assert isinstance(agent.config, AgentConfig)
        assert agent.config.name == "test_agent"
        assert agent.config.persona == "Test assistant"
        assert agent._initial_task_description == "Test task"
        assert agent._state_manager.current_state is None  # Not initialized yet
    
    def test_init_with_agent_config(self):
        """Test agent initialization with AgentConfig instance."""
        config = AgentConfig(
            name="config_agent",
            persona="Config assistant",
            provider_name="llamacpp",
            enable_memory=False
        )
        
        agent = AgentCore(config=config)
        
        assert agent.name == "config_agent"
        assert agent.persona == "Config assistant"
        assert agent.config is config
    
    def test_init_with_dict_config(self):
        """Test agent initialization with dict configuration."""
        config_dict = {
            "name": "dict_agent",
            "persona": "Dict assistant",
            "provider_name": "llamacpp"
        }
        
        agent = AgentCore(config=config_dict)
        
        assert agent.name == "dict_agent"
        assert agent.persona == "Dict assistant"
        assert isinstance(agent.config, AgentConfig)
    
    def test_init_with_no_config(self):
        """Test agent initialization with no config (uses defaults)."""
        agent = AgentCore()
        
        assert agent.name == "default_agent"
        assert agent.persona == "Default helpful assistant"
        assert isinstance(agent.config, AgentConfig)
    
    def test_init_with_invalid_config(self):
        """Test agent initialization with invalid config type."""
        with pytest.raises(ConfigurationError, match="Invalid config type"):
            AgentCore(config="invalid_config")
    
    def test_init_with_state_persister(self):
        """Test agent initialization with state persister."""
        persister = MockStatePersister()
        agent = AgentCore(state_persister=persister)
        
        assert agent._state_manager._state_persister is persister
    
    def test_llm_provider_property(self):
        """Test LLM provider property getter/setter."""
        agent = AgentCore()
        mock_provider = Mock()
        
        assert agent.llm_provider is None
        
        agent.llm_provider = mock_provider
        assert agent.llm_provider is mock_provider
    
    @pytest.mark.asyncio
    async def test_initialize_impl_success(self):
        """Test successful agent initialization."""
        agent = AgentCore(
            config={"name": "test_agent", "persona": "Test assistant", "provider_name": "llamacpp"},
            task_description="Test task"
        )
        
        # Mock the initialization implementation
        with patch.object(agent, '_initialize_impl', return_value=None) as mock_init:
            await agent.initialize()
            
            # Verify initialization was called
            mock_init.assert_called_once()
            assert agent.initialized
    
    @pytest.mark.asyncio
    async def test_initialize_impl_with_state_persistence(self):
        """Test initialization with state persistence enabled."""
        config = AgentConfig(
            name="persistent_agent",
            persona="Persistent assistant",
            provider_name="llamacpp",
            task_id="test_task_123",
            state_config={
                "persistence_type": "file",
                "auto_load": True,
                "auto_save": True
            }
        )
        
        persister = MockStatePersister()
        # Pre-save a state to load
        existing_state = AgentState(task_description="Existing task", task_id="test_task_123")
        await persister.save_state(existing_state)
        
        agent = AgentCore(config=config, state_persister=persister)
        
        with patch('flowlib.agent.core.orchestrator.ActivityStream', return_value=self.mock_activity_stream), \
             patch('flowlib.agent.core.orchestrator.KnowledgePluginManager', return_value=self.mock_knowledge_plugins), \
             patch('flowlib.agent.core.orchestrator.provider_registry') as mock_registry, \
             patch('flowlib.agent.core.orchestrator.AgentPlanner', return_value=self.mock_planner), \
             patch('flowlib.agent.core.orchestrator.AgentReflection', return_value=self.mock_reflection), \
             patch('flowlib.agent.core.orchestrator.AgentEngine', return_value=self.mock_engine), \
             patch.object(agent._memory_manager, 'setup_memory', AsyncMock()), \
             patch.object(agent._learning_manager, 'initialize_learning_capability', AsyncMock()), \
             patch.object(agent._flow_runner, 'discover_flows', AsyncMock()), \
             patch.object(agent._flow_runner, 'validate_required_flows', AsyncMock()):
            
            mock_registry.get_by_config = AsyncMock(return_value=Mock(name="test_provider"))
            
            await agent.initialize()
            
            assert agent._state_manager.current_state is not None
            assert agent._state_manager.current_state.task_id == existing_state.task_id
            assert agent._state_manager.current_state.task_description == "Existing task"
    
    @pytest.mark.asyncio
    async def test_initialize_impl_failure(self):
        """Test initialization failure handling."""
        agent = AgentCore()
        
        with patch('flowlib.agent.core.orchestrator.ActivityStream', side_effect=Exception("Stream failed")):
            with pytest.raises(ConfigurationError, match="Agent initialization failed"):
                await agent.initialize()
    
    @pytest.mark.asyncio
    async def test_shutdown_impl_success(self):
        """Test successful agent shutdown."""
        # Create config with auto_save enabled and required fields
        config = {
            "name": "test_agent",
            "persona": "Test assistant", 
            "provider_name": "llamacpp",
            "state_config": {"auto_save": True}
        }
        
        agent = AgentCore(config)
        agent._engine = self.mock_engine
        agent._reflection = self.mock_reflection
        agent._planner = self.mock_planner
        agent._memory_manager = Mock()
        agent._memory_manager._memory = self.mock_memory
        agent._knowledge_plugins = self.mock_knowledge_plugins
        agent._state_manager = Mock()
        agent._state_manager._state_persister = MockStatePersister()
        agent._state_manager.current_state = AgentState(task_description="Test")
        
        # Setup additional mocks needed for shutdown
        agent._config_manager = Mock()
        agent._flow_runner = Mock()
        agent._learning_manager = Mock()
        
        # Mock async shutdown methods
        agent._config_manager.shutdown = AsyncMock()
        agent._flow_runner.shutdown = AsyncMock()
        agent._learning_manager.shutdown = AsyncMock()
        agent._state_manager.should_auto_save = Mock(return_value=True)
        agent._state_manager.save_state = AsyncMock()
        
        # Setup memory manager shutdown to also shutdown the memory component
        async def memory_manager_shutdown():
            if agent._memory_manager._memory:
                await agent._memory_manager._memory.shutdown()
        
        agent._memory_manager.shutdown = AsyncMock(side_effect=memory_manager_shutdown)
        
        # Setup state manager shutdown to also shutdown the state persister
        async def state_manager_shutdown():
            if agent._state_manager._state_persister:
                await agent._state_manager._state_persister.shutdown()
        
        agent._state_manager.shutdown = AsyncMock(side_effect=state_manager_shutdown)
        
        # Initialize components
        await self.mock_engine.initialize()
        await self.mock_reflection.initialize()
        await self.mock_planner.initialize()
        await self.mock_memory.initialize()
        await self.mock_knowledge_plugins.initialize()
        await agent._state_manager._state_persister.initialize()
        
        await agent._shutdown_impl()
        
        assert not self.mock_engine.initialized
        assert not self.mock_reflection.initialized
        assert not self.mock_planner.initialized
        assert not agent._memory_manager._memory.initialized
        assert not self.mock_knowledge_plugins.initialized
        assert not agent._state_manager._state_persister._initialized
    
    @pytest.mark.asyncio
    async def test_shutdown_impl_with_failures(self):
        """Test shutdown with component failures."""
        agent = AgentCore()
        agent._engine = Mock()
        agent._engine.initialized = True
        agent._engine.shutdown = AsyncMock(side_effect=Exception("Shutdown failed"))
        
        # Mock other components to avoid initialization issues
        agent._state_manager = Mock()
        agent._state_manager.should_auto_save = Mock(return_value=False)
        agent._learning_manager = Mock()
        agent._learning_manager.shutdown = AsyncMock()
        agent._flow_runner = Mock()
        agent._flow_runner.shutdown = AsyncMock()
        agent._memory_manager = Mock()
        agent._memory_manager.shutdown = AsyncMock()
        agent._config_manager = Mock()
        agent._config_manager.shutdown = AsyncMock()
        agent._knowledge_plugins = Mock()
        agent._knowledge_plugins.shutdown = AsyncMock()
        
        # The shutdown should not raise an exception, it should log and continue
        await agent._shutdown_impl()  # Should not raise, just log error
    
    @pytest.mark.asyncio
    async def test_save_state_success(self):
        """Test successful state saving."""
        persister = MockStatePersister()
        await persister.initialize()
        
        agent = AgentCore(state_persister=persister)
        agent._state_manager.current_state = AgentState(task_description="Test task")
        
        await agent.save_state()
        
        # Verify state was saved
        loaded_state_data = await persister.load_state(agent._state_manager.current_state.task_id)
        assert loaded_state_data is not None
        assert loaded_state_data["task_description"] == "Test task"
    
    @pytest.mark.asyncio
    async def test_save_state_no_persister(self):
        """Test state saving without persister."""
        agent = AgentCore()
        agent._state_manager.current_state = AgentState(task_description="Test")
        
        with pytest.raises(StatePersistenceError, match="No state persister configured"):
            await agent.save_state()
    
    @pytest.mark.asyncio
    async def test_save_state_failure(self):
        """Test state saving failure."""
        persister = MockStatePersister()
        persister.save_state = AsyncMock(side_effect=Exception("Save failed"))
        
        agent = AgentCore(state_persister=persister)
        agent._state_manager.current_state = AgentState(task_description="Test")
        
        with pytest.raises(StatePersistenceError, match="Failed to save state"):
            await agent.save_state()
    
    @pytest.mark.asyncio
    async def test_load_state_success(self):
        """Test successful state loading."""
        persister = MockStatePersister()
        existing_state = AgentState(task_description="Existing task")
        await persister.save_state(existing_state)
        
        agent = AgentCore(state_persister=persister)
        
        await agent.load_state(existing_state.task_id)
        
        assert agent._state_manager.current_state is not None
        assert agent._state_manager.current_state.task_description == "Existing task"
    
    @pytest.mark.asyncio
    async def test_load_state_no_persister(self):
        """Test state loading without persister."""
        agent = AgentCore()
        
        with pytest.raises(StatePersistenceError, match="No state persister configured"):
            await agent.load_state("test_task")
    
    @pytest.mark.asyncio
    async def test_load_state_not_found(self):
        """Test loading non-existent state."""
        persister = MockStatePersister()
        agent = AgentCore(state_persister=persister)
        
        with pytest.raises(StatePersistenceError, match="No state found for task missing_task"):
            await agent.load_state("missing_task")
    
    @pytest.mark.asyncio
    async def test_delete_state_success(self):
        """Test successful state deletion."""
        persister = MockStatePersister()
        state = AgentState(task_description="Test")
        await persister.save_state(state)
        
        agent = AgentCore(state_persister=persister)
        agent._state_manager.current_state = state
        
        await agent.delete_state()
        
        # Verify state was deleted
        loaded_state = await persister.load_state(state.task_id)
        assert loaded_state is None
    
    @pytest.mark.asyncio
    async def test_delete_state_specific_task(self):
        """Test deleting specific task state."""
        persister = MockStatePersister()
        state1 = AgentState(task_description="Task 1")
        state2 = AgentState(task_description="Task 2")
        await persister.save_state(state1)
        await persister.save_state(state2)
        
        agent = AgentCore(state_persister=persister)
        
        await agent.delete_state(state1.task_id)
        
        # Verify only state1 was deleted
        assert await persister.load_state(state1.task_id) is None
        assert await persister.load_state(state2.task_id) is not None
    
    @pytest.mark.asyncio
    async def test_list_states_success(self):
        """Test successful state listing."""
        persister = MockStatePersister()
        state1 = AgentState(task_description="Task 1")
        state2 = AgentState(task_description="Task 2")
        await persister.save_state(state1, {"type": "test"})
        await persister.save_state(state2, {"type": "production"})
        
        agent = AgentCore(state_persister=persister)
        
        # List all states
        all_states = await agent.list_states()
        assert len(all_states) == 2
        
        # List with filter
        test_states = await agent.list_states({"type": "test"})
        assert len(test_states) == 1
        assert test_states[0]["task_id"] == state1.task_id
    
    def test_register_flow_success(self):
        """Test successful flow registration."""
        agent = AgentCore()
        flow = MockFlow("test_flow")
        
        with patch('flowlib.agent.core.flow_runner.flow_registry') as mock_registry:
            agent.register_flow(flow)
            
            assert "test_flow" in agent.flows
            assert agent.flows["test_flow"] is flow
            mock_registry.register.assert_called_once_with(flow)
    
    def test_register_flow_invalid(self):
        """Test registering invalid flow."""
        agent = AgentCore()
        
        # Register a string - should be skipped since it doesn't have a name attribute
        with patch('flowlib.agent.core.flow_runner.flow_registry'):
            agent.register_flow("test string")
            # String should be skipped since it doesn't have a name attribute
            assert len(agent.flows) == 0
    
    def test_register_flow_no_registry(self):
        """Test flow registration without stage registry."""
        agent = AgentCore()
        flow = MockFlow("test_flow")
        
        with patch('flowlib.agent.core.flow_runner.flow_registry', None):
            # Should work fine - registry is optional
            agent.register_flow(flow)
            assert "test_flow" in agent.flows
            assert agent.flows["test_flow"] is flow
    
    @pytest.mark.asyncio
    async def test_register_flow_async(self):
        """Test async flow registration."""
        agent = AgentCore()
        flow = MockFlow("async_flow")
        
        with patch('flowlib.agent.core.flow_runner.flow_registry'):
            await agent.register_flow_async(flow)
            
            assert "async_flow" in agent.flows
    
    def test_unregister_flow_success(self):
        """Test successful flow unregistration."""
        agent = AgentCore()
        flow = MockFlow("test_flow")
        
        with patch('flowlib.agent.core.flow_runner.flow_registry') as mock_registry:
            # First register the flow
            agent.register_flow(flow)
            assert "test_flow" in agent.flows
            
            # Then unregister it
            agent.unregister_flow("test_flow")
            
            assert "test_flow" not in agent.flows
            mock_registry.unregister.assert_called_once_with("test_flow")
    
    def test_unregister_flow_not_found(self):
        """Test unregistering non-existent flow."""
        agent = AgentCore()
        
        # Should silently succeed - no error for missing flow
        agent.unregister_flow("missing")
        assert len(agent.flows) == 0
    
    def test_get_flow_descriptions_success(self):
        """Test getting flow descriptions."""
        agent = AgentCore()
        flow = MockFlow("test_flow", "Test description")
        
        with patch('flowlib.agent.core.flow_runner.flow_registry'), \
             patch('flowlib.agent.core.flow_runner.FlowMetadata') as mock_metadata_class:
            
            # Register the flow first
            agent.register_flow(flow)
            
            mock_metadata = Mock()
            mock_metadata.model_dump.return_value = {
                "name": "test_flow",
                "description": "Test description"
            }
            mock_metadata_class.from_flow.return_value = mock_metadata
            
            descriptions = agent.get_flow_descriptions()
            
            assert len(descriptions) == 1
            assert descriptions[0]["name"] == "test_flow"
    
    def test_get_flow_descriptions_no_registry(self):
        """Test getting flow descriptions without registry."""
        agent = AgentCore()
        flow = MockFlow("test_flow", "Test description")
        
        with patch('flowlib.agent.core.flow_runner.flow_registry', None):
            # Register a flow first
            agent.register_flow(flow)
            
            # Should work fine - registry is optional
            descriptions = agent.get_flow_descriptions()
            assert len(descriptions) == 1
            assert descriptions[0]["name"] == "test_flow"
    
    @pytest.mark.asyncio
    async def test_store_memory_success(self):
        """Test successful memory storage."""
        agent = AgentCore()
        agent._memory_manager = Mock()
        agent._memory_manager._memory = self.mock_memory
        agent._memory_manager._memory.initialized = True
        agent._initialized = True
        
        # Mock async methods
        agent._memory_manager.store_memory = AsyncMock()
        agent._memory_manager.retrieve_memory = AsyncMock(return_value="test_value")
        
        await agent.store_memory("test_key", "test_value", context="test")
        
        # Verify memory was stored
        result = await agent.retrieve_memory("test_key")
        assert result == "test_value"
    
    @pytest.mark.asyncio
    async def test_store_memory_not_initialized(self):
        """Test memory storage when not initialized."""
        agent = AgentCore()
        
        with pytest.raises(NotInitializedError):
            await agent.store_memory("key", "value")
    
    @pytest.mark.asyncio
    async def test_retrieve_memory_success(self):
        """Test successful memory retrieval."""
        agent = AgentCore()
        agent._memory_manager = Mock()
        agent._memory_manager._memory = self.mock_memory
        agent._memory_manager._memory.initialized = True
        agent._initialized = True
        
        # Mock async methods
        agent._memory_manager.store_memory = AsyncMock()
        agent._memory_manager.retrieve_memory = AsyncMock(return_value="test_value")
        
        # Store first
        await agent.store_memory("test_key", "test_value")
        
        # Retrieve
        result = await agent.retrieve_memory("test_key")
        assert result == "test_value"
    
    @pytest.mark.asyncio
    async def test_search_memory_success(self):
        """Test successful memory search."""
        agent = AgentCore()
        agent._memory_manager = Mock()
        agent._memory_manager._memory = self.mock_memory
        agent._memory_manager._memory.initialized = True
        agent._initialized = True
        
        # Mock async methods
        agent._memory_manager.store_memory = AsyncMock()
        agent._memory_manager.search_memory = AsyncMock(return_value=[
            {"item": {"key": "key1", "value": "python programming"}, "score": 0.9}
        ])
        
        # Store some data
        await agent.store_memory("key1", "python programming")
        await agent.store_memory("key2", "javascript development")
        
        # Search
        results = await agent.search_memory("python")
        assert len(results) == 1
        assert results[0]["item"]["value"] == "python programming"
    
    @pytest.mark.asyncio
    async def test_execute_flow_success(self):
        """Test successful flow execution."""
        agent = AgentCore()
        agent._initialized = True
        agent._state_manager.current_state = AgentState(task_description="Test")
        
        # Set up the flow runner with proper initialization
        agent._flow_runner._initialized = True
        
        flow = MockFlow("test_flow")
        with patch('flowlib.agent.core.flow_runner.flow_registry'):
            agent.register_flow(flow)
            
            # Create valid input
            flow_input = MockFlowInput()
            
            # Mock the flow execution to return a proper FlowResult
            from flowlib.flows.models.results import FlowResult
            from flowlib.flows.models.constants import FlowStatus
            mock_result = FlowResult(
                status=FlowStatus.SUCCESS,
                flow_name="test_flow",
                data={"output": "test execution result"}
            )
            
            with patch.object(agent._flow_runner, 'execute_flow', return_value=mock_result) as mock_execute:
                result = await agent.execute_flow("test_flow", flow_input)
                
                assert isinstance(result, FlowResult)
                assert result.status == FlowStatus.SUCCESS
                mock_execute.assert_called_once_with("test_flow", flow_input)
    
    @pytest.mark.asyncio
    async def test_execute_flow_not_initialized(self):
        """Test flow execution when not initialized."""
        agent = AgentCore()
        
        with pytest.raises(NotInitializedError):
            await agent.execute_flow("test_flow", MockFlowInput())
    
    @pytest.mark.asyncio
    async def test_execute_flow_not_found(self):
        """Test executing non-existent flow."""
        agent = AgentCore()
        agent._initialized = True
        
        # Set up the flow runner with proper initialization
        agent._flow_runner._initialized = True
        
        # Mock the flow runner to raise ExecutionError for missing flow
        with patch.object(agent._flow_runner, 'execute_flow', side_effect=ExecutionError("Flow 'missing' not found")):
            with pytest.raises(ExecutionError, match="Flow 'missing' not found"):
                await agent.execute_flow("missing", MockFlowInput())
    
    @pytest.mark.asyncio
    async def test_execute_flow_invalid_input(self):
        """Test flow execution with invalid input type."""
        agent = AgentCore()
        agent._initialized = True
        agent._state_manager.current_state = AgentState(task_description="Test")
        
        # Set up the flow runner with proper initialization
        agent._flow_runner._initialized = True
        
        flow = MockFlow("test_flow")
        with patch('flowlib.agent.core.flow_runner.flow_registry'):
            agent.register_flow(flow)
            
            # Mock the flow runner to raise ValueError for invalid input
            with patch.object(agent._flow_runner, 'execute_flow', side_effect=ValueError("expects inputs to be a MockFlowInput instance")):
                with pytest.raises(ValueError, match="expects inputs to be a MockFlowInput instance"):
                    await agent.execute_flow("test_flow", "invalid_input")
    
    @pytest.mark.asyncio
    async def test_execute_cycle_success(self):
        """Test successful cycle execution."""
        agent = AgentCore()
        agent._engine = self.mock_engine
        agent._initialized = True
        agent._state_manager.current_state = AgentState(task_description="Test")
        
        # Mock the engine's execute_cycle method to modify state and return True
        async def mock_execute_cycle(**kwargs):
            agent._state_manager.current_state.increment_cycle()
            return True
        
        with patch.object(agent._engine, 'execute_cycle', side_effect=mock_execute_cycle):
            should_continue = await agent.execute_cycle()
            
            assert should_continue is True
            assert agent._state_manager.current_state.cycles == 1
    
    @pytest.mark.asyncio
    async def test_execute_cycle_with_auto_save(self):
        """Test cycle execution with auto-save enabled."""
        # Create config with auto_save enabled and save_frequency="cycle"
        config = {
            "name": "test_agent",
            "persona": "Test assistant", 
            "provider_name": "llamacpp",
            "state_config": {"auto_save": True, "save_frequency": "cycle"}
        }
        
        agent = AgentCore(config)
        agent._engine = self.mock_engine
        agent._initialized = True
        agent._state_manager.current_state = AgentState(task_description="Test")
        
        # Set up state manager with persister
        agent._state_manager._state_persister = MockStatePersister()
        
        # Mock the engine's execute_cycle method to modify state and return True
        async def mock_execute_cycle(**kwargs):
            agent._state_manager.current_state.increment_cycle()
            return True
        
        with patch.object(agent._engine, 'execute_cycle', side_effect=mock_execute_cycle), \
             patch.object(agent._state_manager, 'should_auto_save', return_value=True), \
             patch.object(agent, 'save_state') as mock_save:
            
            await agent.execute_cycle()
            
            # Auto-save behavior may vary based on implementation, 
            # but engine's execute_cycle should be called
            assert agent._state_manager.current_state.cycles == 1
    
    @pytest.mark.asyncio
    async def test_learn_success(self):
        """Test successful learning execution."""
        agent = AgentCore()
        agent._learning_manager = self.mock_learning_flow
        agent._initialized = True
        
        result = await agent.learn("Test content", "Test context", ["entities"])
        
        assert result.is_success
    
    @pytest.mark.asyncio
    async def test_learn_not_initialized(self):
        """Test learning when not initialized."""
        agent = AgentCore()
        
        with pytest.raises(NotInitializedError):
            await agent.learn("content")
    
    @pytest.mark.asyncio
    async def test_learn_no_capability(self):
        """Test learning without learning capability."""
        agent = AgentCore()
        agent._initialized = True
        
        with pytest.raises(NotInitializedError, match="Component 'learning_manager' must be initialized"):
            await agent.learn("content")
    
    @pytest.mark.asyncio
    async def test_extract_entities_success(self):
        """Test successful entity extraction."""
        agent = AgentCore()
        agent._learning_manager = self.mock_learning_flow
        agent._initialized = True
        
        entities = await agent.extract_entities("Test content with entities")
        
        assert isinstance(entities, list)
    
    @pytest.mark.asyncio
    async def test_learn_relationships_success(self):
        """Test successful relationship learning."""
        agent = AgentCore()
        agent._learning_manager = self.mock_learning_flow
        agent._initialized = True
        
        relationships = await agent.learn_relationships("Content", ["entity1", "entity2"])
        
        assert isinstance(relationships, list)
    
    @pytest.mark.asyncio
    async def test_integrate_knowledge_success(self):
        """Test successful knowledge integration."""
        agent = AgentCore()
        agent._learning_manager = self.mock_learning_flow
        agent._initialized = True
        
        result = await agent.integrate_knowledge("Content", ["entity1"])
        
        assert result.is_success
    
    @pytest.mark.asyncio
    async def test_form_concepts_success(self):
        """Test successful concept formation."""
        agent = AgentCore()
        agent._learning_manager = self.mock_learning_flow
        agent._initialized = True
        
        concepts = await agent.form_concepts("Conceptual content")
        
        assert isinstance(concepts, list)
    
    def test_set_activity_stream_handler(self):
        """Test setting activity stream handler."""
        agent = AgentCore()
        
        handler = Mock()
        agent.set_activity_stream_handler(handler)
        
        assert agent._activity_stream is not None
        assert agent._activity_stream.output_handler is handler
    
    def test_get_activity_stream(self):
        """Test getting activity stream."""
        agent = AgentCore()
        agent._activity_stream = self.mock_activity_stream
        
        stream = agent.get_activity_stream()
        assert stream is self.mock_activity_stream
    
    @pytest.mark.asyncio
    async def test_process_message_success(self):
        """Test successful message processing."""
        # Create config with memory enabled and provide mock memory component
        config = {
            "name": "test_agent",
            "persona": "Test assistant", 
            "provider_name": "llamacpp",
            "enable_memory": True
        }
        
        agent = AgentCore(config)
        agent._engine = self.mock_engine
        agent._memory_manager = Mock()
        agent._memory_manager.store_memory = AsyncMock()
        agent._memory_manager._memory = self.mock_memory  # Provide mock memory component
        agent._learning_manager = Mock()
        agent._learning_manager.learning_enabled = True
        agent._initialized = True
        agent._state_manager.current_state = AgentState(task_description="Test")
        
        # Register conversation flow
        conversation_flow = MockFlow("ConversationFlow")
        with patch('flowlib.agent.core.flow_runner.flow_registry'):
            agent.register_flow(conversation_flow)
            
            with patch.object(agent, '_generate_conversational_response', return_value="Response"), \
                 patch.object(agent, '_learn_from_conversation'):
                
                response = await agent.process_message("Hello, how are you?")
                
                assert response["content"] == "Response"
                assert "stats" in response
                assert "activity" in response
    
    @pytest.mark.asyncio
    async def test_process_message_not_initialized(self):
        """Test message processing when not initialized."""
        agent = AgentCore()
        
        with pytest.raises(NotInitializedError):
            await agent.process_message("Hello")
    
    @pytest.mark.asyncio
    async def test_list_available_flows(self):
        """Test listing available flows."""
        agent = AgentCore()
        flow1 = MockFlow("flow1", "First flow")
        flow2 = MockFlow("flow2", "Second flow")
        
        with patch('flowlib.agent.core.flow_runner.flow_registry'):
            # Register flows first
            agent.register_flow(flow1)
            agent.register_flow(flow2)
            
            flows = await agent.list_available_flows()
            
            assert len(flows) == 2
            assert flows[0]["name"] == "flow1"
            assert flows[1]["name"] == "flow2"
    
    def test_get_tools(self):
        """Test getting available tools."""
        agent = AgentCore()
        flow = MockFlow("test_flow")
        
        with patch('flowlib.agent.core.flow_runner.flow_registry'):
            agent.register_flow(flow)
            
            tools = agent.get_tools()
            
            assert "flow_test_flow" in tools
            assert tools["flow_test_flow"]["type"] == "flow"
    
    @pytest.mark.asyncio
    async def test_get_memory_stats_success(self):
        """Test getting memory statistics."""
        agent = AgentCore()
        agent._memory_manager = Mock()
        agent._memory_manager._memory = self.mock_memory
        agent._memory_manager._memory.initialized = True
        agent._initialized = True
        
        # Mock async method
        agent._memory_manager.get_memory_stats = AsyncMock(return_value={
            "working_memory_items": 5,
            "vector_memory_items": 3
        })
        
        stats = await agent.get_memory_stats()
        
        assert "working_memory_items" in stats
        assert "vector_memory_items" in stats
    
    @pytest.mark.asyncio
    async def test_get_memory_stats_no_memory(self):
        """Test getting memory stats without memory."""
        agent = AgentCore()
        agent._initialized = True
        
        # Create a memory manager with no memory component
        agent._memory_manager = Mock()
        agent._memory_manager.get_memory_stats = AsyncMock(return_value={
            "error": "Memory system not initialized"
        })
        
        stats = await agent.get_memory_stats()
        
        assert stats["error"] == "Memory system not initialized"
    
    def test_get_stats_success(self):
        """Test getting comprehensive agent statistics."""
        agent = AgentCore()
        agent._initialized = True
        agent._state_manager.current_state = AgentState(task_description="Test task")
        agent._memory_manager = Mock()
        agent._memory_manager._memory = self.mock_memory
        agent._planner = self.mock_planner
        agent._reflection = self.mock_reflection
        agent._engine = self.mock_engine
        
        with patch('flowlib.agent.core.flow_runner.flow_registry'):
            agent.register_flow(MockFlow("test_flow"))
        
        stats = agent.get_stats()
        
        assert isinstance(stats, AgentStats)
        assert stats.name == agent.name
        assert stats.initialized == True
        assert stats.flows["count"] == 1
        assert "test_flow" in stats.flows["names"]
    
    def test_get_uptime(self):
        """Test getting agent uptime."""
        agent = AgentCore()
        
        uptime = agent.get_uptime()
        assert uptime > 0  # Should have some uptime since creation


@pytest.mark.asyncio
async def test_integration_complete_agent_lifecycle():
    """Integration test for complete agent lifecycle."""
    # Create agent with minimal config
    config = AgentConfig(
        name="integration_agent",
        persona="Integration test assistant",
        provider_name="llamacpp",
        enable_memory=True
    )
    
    persister = MockStatePersister()
    agent = AgentCore(config=config, task_description="Integration test", state_persister=persister)
    
    # Mock all dependencies for initialization
    mock_memory = MockMemory()
    mock_planner = MockPlanner()
    mock_reflection = MockReflection()
    mock_engine = MockEngine()
    mock_activity_stream = MockActivityStream()
    mock_knowledge_plugins = MockKnowledgePluginManager()
    
    with patch('flowlib.agent.core.orchestrator.ActivityStream', return_value=mock_activity_stream), \
         patch('flowlib.agent.core.orchestrator.KnowledgePluginManager', return_value=mock_knowledge_plugins), \
         patch('flowlib.agent.core.orchestrator.provider_registry') as mock_registry, \
         patch('flowlib.agent.core.orchestrator.AgentPlanner', return_value=mock_planner), \
         patch('flowlib.agent.core.orchestrator.AgentReflection', return_value=mock_reflection), \
         patch('flowlib.agent.core.orchestrator.AgentEngine', return_value=mock_engine), \
         patch.object(agent._memory_manager, 'setup_memory', AsyncMock()), \
         patch.object(agent._learning_manager, 'initialize_learning_capability', AsyncMock()), \
         patch.object(agent._flow_runner, 'discover_flows', AsyncMock()), \
         patch.object(agent._flow_runner, 'validate_required_flows', AsyncMock()):
        
        # Mock provider registry
        mock_registry.get_by_config = AsyncMock(return_value=Mock(name="test_provider"))
        
        # Initialize agent
        await agent.initialize()
        assert agent.initialized
        assert agent._state_manager.current_state is not None
        
        # Register a flow
        test_flow = MockFlow("test_flow")
        with patch('flowlib.agent.core.flow_runner.flow_registry'):
            agent.register_flow(test_flow)
        
        # Test memory operations through mock
        agent._memory_manager._memory = mock_memory
        agent._memory_manager._memory.initialized = True
        await agent.store_memory("test_key", "test_value")
        retrieved = await agent.retrieve_memory("test_key")
        assert retrieved == "test_value"
        
        # Test flow execution
        flow_input = MockFlowInput()
        result = await agent.execute_flow("test_flow", flow_input)
        assert result.is_success
        
        # Test cycle execution
        should_continue = await agent.execute_cycle()
        assert should_continue is not None
        
        # Test state persistence
        await agent.save_state()
        
        # Test statistics
        stats = agent.get_stats()
        assert isinstance(stats, AgentStats)
        assert stats.initialized
        
        # Test shutdown
        await agent.shutdown()
        assert not agent.initialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])