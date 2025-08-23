"""
Agent orchestrator component.

This module provides the main coordination component that replaces the monolithic
AgentCore class, orchestrating focused manager components.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from flowlib.flows.base.base import Flow

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ConfigurationError, NotInitializedError
from flowlib.agent.core.config_manager import AgentConfigManager
from flowlib.agent.core.state_manager import AgentStateManager
from flowlib.agent.core.memory_manager import AgentMemoryManager
from flowlib.agent.core.flow_runner import AgentFlowRunner
from flowlib.agent.core.learning_manager import AgentLearningManager
from flowlib.agent.core.activity_stream import ActivityStream
from flowlib.agent.core.agent_activity_formatter import AgentActivityFormatter
from flowlib.agent.components.engine.engine import AgentEngine
from flowlib.agent.components.planning.planner import AgentPlanner
from flowlib.agent.components.reflection.base import AgentReflection
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.models.state import AgentState, AgentStats, ComponentStats
from flowlib.agent.components.persistence.base import BaseStatePersister
from flowlib.flows.models.results import FlowResult
from flowlib.providers.knowledge.plugin_manager import KnowledgePluginManager
from flowlib.providers.core.registry import provider_registry

logger = logging.getLogger(__name__)


class AgentOrchestrator(AgentComponent):
    """Orchestrates agent components and provides high-level API.
    
    This component coordinates the focused manager components and provides
    the main agent interface while delegating specific responsibilities
    to specialized managers.
    """
    
    def __init__(self,
                 config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
                 task_description: str = "",
                 name: str = None,
                 state_persister: Optional[BaseStatePersister] = None):
        """Initialize the agent orchestrator.
        
        Args:
            config: Agent configuration
            task_description: Task description for the agent
            name: Name for the agent
            state_persister: State persister for agent
        """
        # Initialize managers
        self._config_manager = AgentConfigManager()
        self._state_manager = AgentStateManager(state_persister)
        self._memory_manager = AgentMemoryManager()
        self._flow_runner = AgentFlowRunner()
        self._learning_manager = AgentLearningManager()
        
        # Prepare configuration
        agent_config = self._config_manager.prepare_config(config)
        
        # Initialize base component - fail-fast approach
        if name:
            super().__init__(name)
        elif agent_config.name:
            super().__init__(agent_config.name)
        else:
            raise ValueError("Either 'name' parameter or agent_config.name must be provided")
        
        # Store initial task description
        self._initial_task_description = task_description
        
        # Core components
        self._planner: Optional[AgentPlanner] = None
        self._reflection: Optional[AgentReflection] = None
        self._engine: Optional[AgentEngine] = None
        self._llm_provider: Optional[Any] = None
        
        # Activity stream and formatting
        self._activity_stream: Optional[ActivityStream] = None
        self._activity_formatter: Optional[AgentActivityFormatter] = None
        
        # Knowledge plugins
        self._knowledge_plugins: Optional[KnowledgePluginManager] = None
        
        # Track start time
        self._start_time = datetime.now()
        
        # Last result storage
        self.last_result = None
    
    async def _initialize_impl(self) -> None:
        """Initialize the agent orchestrator and all managers."""
        try:
            # Initialize activity stream
            if not self._activity_stream:
                self._activity_stream = ActivityStream()
            
            # Initialize knowledge plugin manager
            if not self._knowledge_plugins:
                self._knowledge_plugins = KnowledgePluginManager()
                await self._knowledge_plugins.initialize()
                logger.info(f"Initialized knowledge plugins: {list(self._knowledge_plugins.loaded_plugins.keys())}")
            
            # Set up cross-references between managers
            self._memory_manager._activity_stream = self._activity_stream
            self._memory_manager._knowledge_plugins = self._knowledge_plugins
            self._flow_runner._activity_stream = self._activity_stream
            self._learning_manager._activity_stream = self._activity_stream
            self._learning_manager._memory_manager = self._memory_manager
            
            # Initialize all managers
            await self._config_manager.initialize()
            await self._state_manager.initialize()
            await self._memory_manager.initialize()
            await self._flow_runner.initialize()
            await self._learning_manager.initialize()
            
            # Set parent relationships
            self._config_manager.set_parent(self)
            self._state_manager.set_parent(self)
            self._memory_manager.set_parent(self)
            self._flow_runner.set_parent(self)
            self._learning_manager.set_parent(self)
            
            # Setup state persistence from config
            self._state_manager.setup_persister(self._config_manager.config)
            
            # Handle state loading/creation
            await self._initialize_state()
            
            # Setup memory system
            await self._memory_manager.setup_memory(self._config_manager.config)
            
            # Initialize learning capability
            await self._learning_manager.initialize_learning_capability(self._config_manager.config)
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Discover and validate flows
            await self._flow_runner.discover_flows()
            await self._flow_runner.validate_required_flows()
            
            logger.info(f"Agent orchestrator '{self._name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent orchestrator: {e}")
            raise ConfigurationError(f"Agent initialization failed: {e}") from e
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the agent orchestrator and all managers."""
        try:
            # Shutdown core components
            if self._engine and self._engine.initialized:
                await self._engine.shutdown()
            
            if self._planner and self._planner.initialized:
                await self._planner.shutdown()
            
            if self._reflection and self._reflection.initialized:
                await self._reflection.shutdown()
            
            # Auto-save state if configured
            if (self._state_manager.should_auto_save(self._config_manager.config) and 
                self._state_manager.current_state):
                await self._state_manager.save_state()
            
            # Shutdown all managers
            await self._learning_manager.shutdown()
            await self._flow_runner.shutdown()
            await self._memory_manager.shutdown()
            await self._state_manager.shutdown()
            await self._config_manager.shutdown()
            
            # Shutdown knowledge plugins
            if self._knowledge_plugins:
                await self._knowledge_plugins.shutdown()
            
            logger.info(f"Agent orchestrator '{self._name}' shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during agent shutdown: {e}")
    
    async def _initialize_state(self) -> None:
        """Initialize agent state (load or create)."""
        config = self._config_manager.config
        
        # Check if we should auto-load existing state
        if self._state_manager.should_auto_load(config) and config.task_id:
            try:
                await self._state_manager.load_state(config.task_id)
                logger.info(f"Loaded existing state for task_id: {config.task_id}")
            except Exception as e:
                logger.warning(f"Failed to load state for task {config.task_id}, creating new state: {e}")
                await self._state_manager.create_state(self._initial_task_description)
        else:
            # Create new state
            await self._state_manager.create_state(self._initial_task_description)
    
    async def _initialize_core_components(self) -> None:
        """Initialize core agent components."""
        config = self._config_manager.config
        
        # Initialize planner
        self._planner = AgentPlanner(
            config=config.planner_config,
            activity_stream=self._activity_stream
        )
        self._planner.set_parent(self)
        
        # Initialize reflection
        self._reflection = AgentReflection(
            config=config.reflection_config,
            activity_stream=self._activity_stream
        )
        self._reflection.set_parent(self)
        
        # Initialize engine
        self._engine = AgentEngine(
            config=config.engine_config,
            memory=self._memory_manager.memory,
            planner=self._planner,
            reflection=self._reflection,
            activity_stream=self._activity_stream,
            agent_config=config
        )
        self._engine.set_parent(self)
        
        # Initialize all core components
        await self._planner.initialize()
        await self._reflection.initialize()
        await self._engine.initialize()
    
    # Delegate configuration operations to ConfigManager
    @property
    def config(self) -> AgentConfig:
        """Get agent configuration."""
        return self._config_manager.config
    
    @property
    def persona(self) -> str:
        """Get agent persona."""
        return self.config.persona
    
    # All state operations delegated to StateManager - access via agent._state_manager.current_state
    
    async def save_state(self) -> None:
        """Save agent state."""
        await self._state_manager.save_state()
    
    async def load_state(self, task_id: str) -> None:
        """Load agent state."""
        await self._state_manager.load_state(task_id)
    
    async def delete_state(self, task_id: Optional[str] = None) -> None:
        """Delete agent state."""
        await self._state_manager.delete_state(task_id)
    
    async def list_states(self, filter_criteria: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """List available states."""
        return await self._state_manager.list_states(filter_criteria)
    
    # Delegate memory operations to MemoryManager
    async def store_memory(self, key: str, value: Any, **kwargs) -> None:
        """Store a value in memory."""
        await self._memory_manager.store_memory(key, value, **kwargs)
    
    async def retrieve_memory(self, key: str, **kwargs) -> Any:
        """Retrieve a value from memory."""
        return await self._memory_manager.retrieve_memory(key, **kwargs)
    
    async def search_memory(self, query: str, **kwargs) -> List[Any]:
        """Search memory for relevant information."""
        return await self._memory_manager.search_memory(query, **kwargs)
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return await self._memory_manager.get_memory_stats()
    
    # Delegate flow operations to FlowRunner
    @property
    def flows(self) -> Dict[str, Flow]:
        """Get registered flows."""
        return self._flow_runner.flows
    
    def register_flow(self, flow: Any) -> None:
        """Register a flow with the agent."""
        self._flow_runner.register_flow(flow)
    
    async def register_flow_async(self, flow: Any) -> None:
        """Register a flow asynchronously."""
        await self._flow_runner.register_flow_async(flow)
    
    def unregister_flow(self, flow_name: str) -> None:
        """Unregister a flow from the agent."""
        self._flow_runner.unregister_flow(flow_name)
    
    def get_flow_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all registered flows."""
        return self._flow_runner.get_flow_descriptions()
    
    async def execute_flow(self, flow_name: str, inputs: Any, **kwargs) -> FlowResult:
        """Execute a flow with given inputs."""
        return await self._flow_runner.execute_flow(flow_name, inputs, **kwargs)
    
    async def list_available_flows(self) -> List[Dict[str, Any]]:
        """List all available flows."""
        return await self._flow_runner.list_available_flows()
    
    def get_flow_registry(self):
        """Get the flow registry."""
        return self._flow_runner.get_flow_registry()
    
    # Delegate learning operations to LearningManager
    async def learn(self, content: str, context: Optional[str] = None, focus_areas: Optional[List[str]] = None) -> Any:
        """Learn from content."""
        return await self._learning_manager.learn(content, context, focus_areas)
    
    async def extract_entities(self, content: str, context: Optional[str] = None) -> List[Any]:
        """Extract entities from content."""
        return await self._learning_manager.extract_entities(content, context)
    
    async def learn_relationships(self, content: str, entity_ids: List[str]) -> List[Any]:
        """Learn relationships from content."""
        return await self._learning_manager.learn_relationships(content, entity_ids)
    
    async def integrate_knowledge(self, content: str, entity_ids: List[str]) -> Any:
        """Integrate knowledge from content."""
        return await self._learning_manager.integrate_knowledge(content, entity_ids)
    
    async def form_concepts(self, content: str) -> List[Any]:
        """Form concepts from content."""
        return await self._learning_manager.form_concepts(content)
    
    # Engine delegation
    async def execute_cycle(self, **kwargs) -> bool:
        """Execute a single agent cycle."""
        if not self._engine:
            raise NotInitializedError(
                component_name=self._name,
                operation="execute_cycle"
            )
        
        # Pass current state to engine if not provided
        if 'state' not in kwargs:
            kwargs['state'] = self._state_manager.current_state
            
        return await self._engine.execute_cycle(**kwargs)
    
    # High-level operations
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message and return response."""
        self._check_initialized("process_message")
        
        # Store in memory if enabled
        if self.config.enable_memory:
            await self.store_memory(
                key=f"message_{datetime.now().isoformat()}",
                value={"message": message, "context": context}
            )
        
        # Generate conversational response  
        response = await self._generate_conversational_response(message)
        
        # Learn from conversation if learning is enabled
        if self._learning_manager.learning_enabled:
            await self._learn_from_conversation(message, response, context)
        
        return {
            "content": response,
            "task_id": self._state_manager.current_state.task_id if self._state_manager.current_state else None,
            "stats": self.get_stats().model_dump(),
            "activity": self.get_activity_stream().events if self.get_activity_stream() else []
        }
    
    # Statistics and monitoring
    def get_stats(self) -> AgentStats:
        """Get comprehensive agent statistics."""
        flows_info = {
            "count": len(self.flows),
            "names": list(self.flows.keys())
        }
        
        # Collect component stats
        memory_stats = None
        planner_stats = None
        reflection_stats = None
        engine_stats = None
        
        if self._initialized:
            if self._memory_manager and hasattr(self._memory_manager, 'memory') and self._memory_manager.memory:
                try:
                    mem_stats = self._memory_manager.memory.get_stats() if hasattr(self._memory_manager.memory, 'get_stats') else {}
                    # Ensure it's a dict, not a Mock
                    if hasattr(mem_stats, 'items'):  # Duck typing for dict-like object
                        mem_stats_dict = dict(mem_stats) if not isinstance(mem_stats, dict) else mem_stats
                    else:
                        mem_stats_dict = {}
                    memory_stats = ComponentStats(
                        initialized=True,
                        name="AgentMemory",
                        stats=mem_stats_dict
                    )
                except Exception:
                    memory_stats = ComponentStats(initialized=True, name="AgentMemory", stats={})
            
            if self._planner:
                try:
                    planner_stats_data = self._planner.get_stats() if hasattr(self._planner, 'get_stats') else {}
                    if hasattr(planner_stats_data, 'items'):
                        planner_stats_dict = dict(planner_stats_data) if not isinstance(planner_stats_data, dict) else planner_stats_data
                    else:
                        planner_stats_dict = {}
                    planner_stats = ComponentStats(
                        initialized=self._planner.initialized if hasattr(self._planner, 'initialized') else True,
                        name="AgentPlanner",
                        stats=planner_stats_dict
                    )
                except Exception:
                    planner_stats = ComponentStats(initialized=True, name="AgentPlanner", stats={})
            
            if self._reflection:
                try:
                    reflection_stats_data = self._reflection.get_stats() if hasattr(self._reflection, 'get_stats') else {}
                    if hasattr(reflection_stats_data, 'items'):
                        reflection_stats_dict = dict(reflection_stats_data) if not isinstance(reflection_stats_data, dict) else reflection_stats_data
                    else:
                        reflection_stats_dict = {}
                    reflection_stats = ComponentStats(
                        initialized=self._reflection.initialized if hasattr(self._reflection, 'initialized') else True,
                        name="AgentReflection",
                        stats=reflection_stats_dict
                    )
                except Exception:
                    reflection_stats = ComponentStats(initialized=True, name="AgentReflection", stats={})
            
            if self._engine:
                try:
                    engine_stats_data = self._engine.get_stats() if hasattr(self._engine, 'get_stats') else {}
                    if hasattr(engine_stats_data, 'items'):
                        engine_stats_dict = dict(engine_stats_data) if not isinstance(engine_stats_data, dict) else engine_stats_data
                    else:
                        engine_stats_dict = {}
                    engine_stats = ComponentStats(
                        initialized=self._engine.initialized if hasattr(self._engine, 'initialized') else True,
                        name="AgentEngine", 
                        stats=engine_stats_dict
                    )
                except Exception:
                    engine_stats = ComponentStats(initialized=True, name="AgentEngine", stats={})
        
        return AgentStats(
            name=self._name,
            initialized=self._initialized,
            uptime_seconds=self.get_uptime(),
            config=self.config.model_dump() if self.config else {},
            flows=flows_info,
            state={
                "task_id": self._state_manager.current_state.task_id if self._state_manager.current_state else None,
                "cycles": self._state_manager.current_state.cycles if self._state_manager.current_state else 0,
                "progress": self._state_manager.current_state.progress if self._state_manager.current_state else 0,
                "is_complete": self._state_manager.current_state.is_complete if self._state_manager.current_state else False
            },
            memory=memory_stats,
            planner=planner_stats,
            reflection=reflection_stats,
            engine=engine_stats,
            performance={}
        )
    
    def get_uptime(self) -> float:
        """Get agent uptime in seconds."""
        return (datetime.now() - self._start_time).total_seconds()
    
    # Activity stream management
    def set_activity_stream_handler(self, handler: Optional[Any] = None):
        """Set activity stream handler."""
        if not self._activity_stream:
            self._activity_stream = ActivityStream()
        
        if handler:
            self._activity_stream.set_output_handler(handler)
            self._activity_formatter = AgentActivityFormatter()
    
    def get_activity_stream(self) -> Optional[ActivityStream]:
        """Get activity stream."""
        return self._activity_stream
    
    # Provider access
    @property
    def llm_provider(self) -> Optional[Any]:
        """Get the LLM provider."""
        return self._llm_provider
    
    @llm_provider.setter
    def llm_provider(self, provider: Any):
        """Set the LLM provider."""
        self._llm_provider = provider
    
    def get_tools(self) -> Dict[str, Any]:
        """Get available tools/capabilities."""
        tools = {}
        
        # Add flow tools
        for flow_name in self.flows.keys():
            tools[f"flow_{flow_name}"] = {
                "type": "flow",
                "name": flow_name,
                "description": getattr(self.flows[flow_name], 'description', 'Flow execution tool')
            }
        
        return tools
    
    def _check_initialized(self, operation: str) -> None:
        """Check if the agent is initialized before performing operation."""
        if not self._initialized:
            raise NotInitializedError(
                component_name=self._name,
                operation=operation
            )
    
    async def _generate_conversational_response(self, message: str) -> str:
        """Generate a conversational response to a message."""
        # Simple fallback response generation
        return f"I understand your message: {message}"
    
    async def _learn_from_conversation(self, message: str, response: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Learn from a conversation exchange."""
        # Simple learning implementation
        if self._learning_manager and self._learning_manager.learning_enabled:
            content = f"User: {message}\nAssistant: {response}"
            await self._learning_manager.learn(content, "conversation", ["dialogue"])