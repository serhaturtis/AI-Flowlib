"""
Agent orchestrator component.

This module provides the main coordination component that replaces the monolithic
BaseAgent class, orchestrating focused manager components.
"""

import asyncio
import logging
import os
import threading
import time
import queue
import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from flowlib.flows.base.base import Flow

from flowlib.agent.core.component_registry import ComponentRegistry
from flowlib.agent.core.errors import ConfigurationError, NotInitializedError, ExecutionError
from flowlib.agent.core.config_manager import AgentConfigManager
from flowlib.agent.core.state_manager import AgentStateManager
from flowlib.agent.components.memory.manager import AgentMemoryManager
from flowlib.agent.core.flow_runner import AgentFlowRunner
from flowlib.agent.core.context.manager import AgentContextManager
from flowlib.agent.core.activity_stream import ActivityStream
from flowlib.agent.core.agent_activity_formatter import AgentActivityFormatter
from flowlib.agent.components.engine import EngineComponent
from flowlib.agent.components.task.decomposition import TaskDecompositionComponent
from flowlib.agent.components.task.execution import TaskExecutionComponent
from flowlib.agent.components.task.debriefing import TaskDebrieferComponent
from flowlib.agent.components.knowledge import KnowledgeComponent
from flowlib.agent.models.config import AgentConfig
from flowlib.agent.models.state import AgentStats, ComponentStats, ComponentStatistics, AgentConfiguration, FlowRegistryStats, PerformanceMetrics, AgentState, AgentExecutionState
from flowlib.agent.models.conversation import ConversationMessage, MessageRole
from flowlib.agent.components.persistence.base import BaseStatePersister
from flowlib.flows.models.results import FlowResult
from flowlib.flows.registry.registry import FlowRegistry
from flowlib.providers.knowledge.plugin_manager import KnowledgePluginManager
from flowlib.agent.core.models.messages import (
    AgentMessage, AgentResponse, AgentMessageType
)

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base agent implementation with full functionality.
    
    This is the core agent implementation that provides all fundamental
    agent capabilities including configuration, state, memory, flows,
    and learning through coordinated manager components.
    
    BaseAgent owns and coordinates components through a private registry,
    but is not itself a component.
    """
    
    def __init__(self,
                 config: AgentConfig,
                 task_description: str = "",
                 state_persister: Optional[BaseStatePersister] = None):
        """Initialize the base agent.
        
        Args:
            config: Agent configuration
            task_description: Task description for the agent
            state_persister: State persister for agent
        """
        # Initialize agent properties (not from AgentComponent)
        self._name = config.name
        
        self._initialized = False
        self._logger = logging.getLogger(f"{__name__}.{self._name}")
        
        # Create this agent's private component registry
        self._registry = ComponentRegistry(agent_name=self._name)
        
        # Create and register managers
        self._config_manager = AgentConfigManager()
        self._config_manager.set_registry(self._registry)
        self._registry.register("config_manager", self._config_manager, AgentConfigManager)
        
        # Set the config in config_manager
        self._config_manager.prepare_config(config)
        
        self._state_manager = AgentStateManager(state_persister)
        self._state_manager.set_registry(self._registry)
        self._registry.register("state_manager", self._state_manager, AgentStateManager)
        
        self._memory_manager = AgentMemoryManager()
        self._memory_manager.set_registry(self._registry)
        self._registry.register("memory_manager", self._memory_manager, AgentMemoryManager)
        
        self._context_manager = AgentContextManager(config.context)
        self._context_manager.set_registry(self._registry)
        self._registry.register("context_manager", self._context_manager, AgentContextManager)
        
        self._flow_runner = AgentFlowRunner()
        self._flow_runner.set_registry(self._registry)
        self._registry.register("flow_runner", self._flow_runner, AgentFlowRunner)
        
        # Learning is now handled by KnowledgeComponent
        
        # Store initial task description
        self._initial_task_description = task_description if task_description else "Interactive agent session"
        
        # Core components  
        self._task_decomposer: Optional[TaskDecompositionComponent] = None
        self._task_executor: Optional[TaskExecutionComponent] = None
        self._task_debriefer: Optional[TaskDebrieferComponent] = None
        self._engine: Optional[EngineComponent] = None
        
        # Activity stream and formatting
        self._activity_stream: Optional[ActivityStream] = None
        self._activity_formatter: Optional[AgentActivityFormatter] = None
        
        # Knowledge plugins
        self._knowledge_plugins: Optional[KnowledgePluginManager] = None
        
        # Track start time
        self._start_time = datetime.now()
        
        # Queue-based I/O using thread-safe queues for cross-event-loop communication
        self.input_queue: queue.Queue[AgentMessage] = queue.Queue()
        self.output_queue: queue.Queue[AgentResponse] = queue.Queue()
        
        # Thread management
        self._agent_thread: Optional[threading.Thread] = None
        self._agent_loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize the base agent and all managers."""
        if self._initialized:
            return
        try:
            
            # Initialize new tool system
            logger.info("Initialized new tool system architecture")
            
            # Initialize activity stream
            if not self._activity_stream:
                self._activity_stream = ActivityStream()
            
            # Initialize knowledge plugin manager
            if not self._knowledge_plugins:
                self._knowledge_plugins = KnowledgePluginManager()
                await self._knowledge_plugins.initialize()
                logger.info(f"Initialized knowledge plugins: {list(self._knowledge_plugins.loaded_plugins.keys())}")
            
            # Register utility components in registry for access by other components
            self._registry.register("activity_stream", self._activity_stream)
            self._registry.register("knowledge_plugins", self._knowledge_plugins)
            
            # Components now access each other through registry, not direct references
            
            # Initialize all managers
            await self._config_manager.initialize()
            await self._state_manager.initialize()
            await self._memory_manager.initialize()
            await self._context_manager.initialize()
            await self._flow_runner.initialize()
            # Learning now handled by knowledge component
            
            # No parent relationships needed - components use registry
            
            # Setup state persistence from config
            self._state_manager.setup_persister(self._config_manager.config)
            
            # Handle state loading/creation
            await self._initialize_state()
            
            # Setup memory system
            await self._memory_manager.setup_memory(self._config_manager.config.memory)
            
            # Initialize session context
            current_state = self._state_manager.current_state
            if current_state is None:
                raise RuntimeError("Agent state not initialized - cannot get task_id for session")

            await self._context_manager.initialize_session(
                session_id=current_state.task_id,
                agent_name=self._config_manager.config.name,
                agent_persona=self._config_manager.config.persona,
                working_directory=os.getcwd(),
                user_id=None
            )
            
            # Learning capability is now part of KnowledgeComponent
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Discover flows
            await self._flow_runner.discover_flows()
            
            self._initialized = True
            logger.info(f"Agent '{self._name}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise ConfigurationError(f"Agent initialization failed: {e}") from e
    
    async def shutdown(self) -> None:
        """Shutdown the agent and all managers."""
        if not self._initialized:
            return
        try:
            # Shutdown core components
            if self._engine and self._engine.initialized:
                await self._engine.shutdown()
            
            if self._task_generator and self._task_generator.initialized:
                await self._task_generator.shutdown()
            
            if self._task_decomposer and self._task_decomposer.initialized:
                await self._task_decomposer.shutdown()
            
            if self._task_executor and self._task_executor.initialized:
                await self._task_executor.shutdown()
                
            if self._task_debriefer and self._task_debriefer.initialized:
                await self._task_debriefer.shutdown()
            
            
            if self._knowledge and self._knowledge.initialized:
                await self._knowledge.shutdown()
            
            # Auto-save state if configured
            if (self._state_manager.should_auto_save(self._config_manager.config) and 
                self._state_manager.current_state):
                await self._state_manager.save_state()
            
            # Shutdown all managers
            await self._flow_runner.shutdown()
            await self._context_manager.shutdown()
            await self._memory_manager.shutdown()
            await self._state_manager.shutdown()
            await self._config_manager.shutdown()
            
            # Shutdown knowledge plugins
            if self._knowledge_plugins:
                await self._knowledge_plugins.shutdown()
            
            self._initialized = False
            logger.info(f"Agent '{self._name}' shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during agent shutdown: {e}")
    
    async def _initialize_state(self) -> None:
        """Initialize agent state (load or create)."""
        agent_config = self._config_manager.config
        
        # Check if we should auto-load existing state
        if self._state_manager.should_auto_load(agent_config) and agent_config.task_id:
            try:
                await self._state_manager.load_state(agent_config.task_id)
                logger.info(f"Loaded existing state for task_id: {agent_config.task_id}")
            except Exception as e:
                logger.warning(f"Failed to load state for task {agent_config.task_id}, creating new state: {e}")
                await self._state_manager.create_state(self._initial_task_description)
        else:
            # Create new state
            await self._state_manager.create_state(self._initial_task_description)
    
    async def _initialize_core_components(self) -> None:
        """Initialize core agent components."""
        config = self._config_manager.config
        
        # Initialize task generator
        from flowlib.agent.components.task.generation import TaskGeneratorComponent
        self._task_generator = TaskGeneratorComponent()
        self._registry.register("task_generator", self._task_generator, TaskGeneratorComponent)

        # Initialize task thinking component
        from flowlib.agent.components.task.thinking import TaskThinkingComponent
        self._task_thinking = TaskThinkingComponent()
        self._registry.register("task_thinking", self._task_thinking, TaskThinkingComponent)

        # Initialize task_decomposer
        self._task_decomposer = TaskDecompositionComponent(
            config=config,  # Use consolidated config directly
            activity_stream=self._activity_stream
        )
        self._task_decomposer.set_registry(self._registry)
        self._registry.register("task_decomposer", self._task_decomposer, TaskDecompositionComponent)
        
        # Initialize task executor
        self._task_executor = TaskExecutionComponent()
        self._registry.register("task_executor", self._task_executor, TaskExecutionComponent)
        
        # Initialize task debriefer
        self._task_debriefer = TaskDebrieferComponent(activity_stream=self._activity_stream)
        self._registry.register("task_debriefer", self._task_debriefer, TaskDebrieferComponent)
        
        
        # Initialize knowledge component if configured
        self._knowledge = None
        if config.knowledge:
            self._knowledge = KnowledgeComponent(
                config=config.knowledge,
                name="agent_knowledge"
            )
            self._registry.register("knowledge", self._knowledge, KnowledgeComponent)
            logger.info("Knowledge component initialized")
        
        # Initialize engine
        self._engine = EngineComponent(
            agent_name=config.name,
            agent_persona=config.persona,
            max_iterations=config.max_iterations,
            memory=self._memory_manager.memory,
            knowledge=self._knowledge,
            task_generator=self._task_generator,
            task_thinking=self._task_thinking,
            task_decomposer=self._task_decomposer,
            task_executor=self._task_executor,
            task_debriefer=self._task_debriefer,
            activity_stream=self._activity_stream,
            context_manager=self._context_manager
        )
        self._engine.set_registry(self._registry)
        self._registry.register("engine", self._engine, EngineComponent)
        
        # Initialize all core components
        await self._task_generator.initialize()
        await self._task_thinking.initialize()
        await self._task_decomposer.initialize()
        await self._task_executor.initialize()
        await self._task_debriefer.initialize()
        if self._knowledge:
            await self._knowledge.initialize()
        await self._engine.initialize()
    
    # Agent properties (no longer inherited from AgentComponent)
    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name
    
    @property
    def initialized(self) -> bool:
        """Check if agent is initialized."""
        return self._initialized
    
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
    async def store_memory(self, key: str, value: Any, **kwargs: Any) -> None:
        """Store a value in memory."""
        await self._memory_manager.store_memory(key, value, **kwargs)

    async def retrieve_memory(self, key: str, **kwargs: Any) -> Any:
        """Retrieve a value from memory."""
        return await self._memory_manager.retrieve_memory(key, **kwargs)

    async def search_memory(self, query: str, **kwargs: Any) -> Any:
        """Search memory for relevant information."""
        return await self._memory_manager.search_memory(query, **kwargs)
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return await self._memory_manager.get_memory_stats()
    
    def _generate_semantic_key(self, content: str, prefix: str) -> str:
        """Generate semantic key based on message content instead of timestamp.
        
        Args:
            content: Message content
            prefix: Key prefix (e.g., 'conversation', 'task', 'response')
            
        Returns:
            Semantic key in format: prefix_type_hash
        """
        # Clean and normalize content
        clean_content = re.sub(r'[^\w\s]', '', content.lower().strip())
        words = clean_content.split()[:5]  # First 5 words for semantic meaning
        
        # Determine content type - questions take priority over greetings
        if '?' in content:
            content_type = 'question'
        elif len(content) <= 20 and any(greeting in content.lower() for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            content_type = 'greeting'
        elif any(keyword in content.lower() for keyword in ['create', 'make', 'build', 'generate']):
            content_type = 'creation'
        elif any(keyword in content.lower() for keyword in ['list', 'show', 'display', 'find']):
            content_type = 'query'
        else:
            content_type = 'message'
        
        # Create semantic identifier from first few words
        if words:
            semantic_part = '_'.join(words[:3])  # First 3 words max
        else:
            semantic_part = 'empty'
            
        # Add content hash for uniqueness (short)
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return f"{prefix}_{content_type}_{semantic_part}_{content_hash}"
    
    # Delegate flow operations to FlowRunner
    @property
    def flows(self) -> Dict[str, "Flow"]:
        """Get registered flows."""
        return self._flow_runner.flows
    
    def register_flow(self, flow: Flow) -> None:
        """Register a flow with the agent."""
        self._flow_runner.register_flow(flow)
    
    async def register_flow_async(self, flow: Flow) -> None:
        """Register a flow asynchronously."""
        await self._flow_runner.register_flow_async(flow)
    
    def unregister_flow(self, flow_name: str) -> None:
        """Unregister a flow from the agent."""
        self._flow_runner.unregister_flow(flow_name)
    
    def get_flow_descriptions(self) -> List[Dict[str, Any]]:
        """Get descriptions of all registered flows."""
        return self._flow_runner.get_flow_descriptions()
    
    async def execute_flow(self, flow_name: str, inputs: Dict[str, str], **kwargs: str) -> FlowResult:
        """Execute a flow with given inputs."""
        return await self._flow_runner.execute_flow(flow_name, inputs, **kwargs)
    
    async def list_available_flows(self) -> List[Dict[str, Any]]:
        """List all available flows."""
        return await self._flow_runner.list_available_flows()
    
    def get_flow_registry(self) -> FlowRegistry:
        """Get the flow registry."""
        return self._flow_runner.get_flow_registry()
    
    # Delegate learning operations to LearningManager
    async def learn(self, content: str, context: Optional[str] = None, focus_areas: Optional[List[str]] = None) -> Any:
        """Learn from content."""
        # Learning operations now handled by knowledge component
        if self._knowledge:
            if not context:
                context = "general"
            result = await self._knowledge.learn_from_content(content, context, focus_areas)
            return result
        raise NotImplementedError("Learning requires knowledge component to be configured")
    
    
    # Engine delegation
    async def execute_cycle(self, state: Optional[AgentState] = None, conversation_history: Optional[str] = None, memory_context: str = "agent", no_flow_is_error: bool = False) -> bool:
        """Execute a single agent cycle."""
        if not self._engine:
            raise NotInitializedError(
                component_name=self._name,
                operation="execute_cycle"
            )
        
        # Use current state if not provided
        if state is None:
            state = self._state_manager.current_state
            if state is None:
                raise ExecutionError("No current state available for execution cycle")

        return await self._engine.execute_cycle(
            state=state,
            conversation_history=conversation_history,
            memory_context=memory_context,
            no_flow_is_error=no_flow_is_error
        )

    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process a single message and return the response.

        This is the public interface for message processing that wraps
        the internal message lifecycle handling.

        Args:
            message: The message to process
            context: Optional context for the message

        Returns:
            The agent's response as a string
        """
        result = await self._handle_message_lifecycle(message, context)
        return cast(str, result.get("content", ""))

    @property
    def agent_id(self) -> str:
        """Get the agent identifier (name)."""
        return self._name

    # High-level operations are now handled through queue-based messaging
    # The old process_message is replaced by _handle_message_lifecycle
    
    # Statistics and monitoring
    def get_stats(self) -> AgentStats:
        """Get comprehensive agent statistics."""
        flows_info = FlowRegistryStats(
            total_flows=len(self.flows),
            active_flows=len(self.flows),  # Assume all flows are active
            infrastructure_flows=0,  # TODO: categorize flows properly
            agent_flows=len(self.flows)
        )
        
        # Collect component stats
        memory_stats = None
        task_decomposer_stats = None
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
                        stats=ComponentStatistics(**mem_stats_dict) if mem_stats_dict else ComponentStatistics()
                    )
                except Exception:
                    memory_stats = ComponentStats(initialized=True, name="AgentMemory", stats=ComponentStatistics())
            
            if self._task_decomposer:
                try:
                    task_decomposer_stats_data = self._task_decomposer.get_stats() if hasattr(self._task_decomposer, 'get_stats') else {}
                    if hasattr(task_decomposer_stats_data, 'items'):
                        task_decomposer_stats_dict = dict(task_decomposer_stats_data) if not isinstance(task_decomposer_stats_data, dict) else task_decomposer_stats_data
                    else:
                        task_decomposer_stats_dict = {}
                    task_decomposer_stats = ComponentStats(
                        initialized=self._task_decomposer.initialized if hasattr(self._task_decomposer, 'initialized') else True,
                        name="TaskDecomposer",
                        stats=ComponentStatistics(**task_decomposer_stats_dict) if task_decomposer_stats_dict else ComponentStatistics()
                    )
                except Exception:
                    task_decomposer_stats = ComponentStats(initialized=True, name="TaskDecomposer", stats=ComponentStatistics())
            
            
            if self._engine:
                try:
                    engine_stats_data = self._engine.get_stats() if hasattr(self._engine, 'get_stats') else {}
                    if hasattr(engine_stats_data, 'items'):
                        engine_stats_dict = dict(engine_stats_data) if not isinstance(engine_stats_data, dict) else engine_stats_data
                    else:
                        engine_stats_dict = {}
                    engine_stats = ComponentStats(
                        initialized=self._engine.initialized if hasattr(self._engine, 'initialized') else True,
                        name="EngineComponent",
                        stats=ComponentStatistics(**engine_stats_dict) if engine_stats_dict else ComponentStatistics()
                    )
                except Exception:
                    engine_stats = ComponentStats(initialized=True, name="AgentEngine", stats=ComponentStatistics())
        
        # Create proper AgentConfiguration with only the required fields
        if self.config:
            config_data = AgentConfiguration(
                name=self.config.name,
                persona=self.config.persona,
                provider_name=self.config.provider_name,
                system_prompt=self.config.persona,  # Use persona as system prompt
                max_iterations=self.config.max_iterations,
                debug_mode=False  # Default to False
            )
        else:
            config_data = AgentConfiguration(
                name=self._name,
                persona="Default Agent",
                provider_name="default",
                system_prompt="You are a helpful AI assistant",
                max_iterations=10,
                debug_mode=False
            )

        # Create AgentExecutionState from current state manager data
        execution_state = None
        if self._state_manager.current_state:
            execution_state = AgentExecutionState(
                current_task=getattr(self._state_manager.current_state, 'task_description', ''),
                execution_phase="running" if self._initialized else "idle",
                last_action="",
                error_state=False,
                iteration_count=getattr(self._state_manager.current_state, 'cycles', 0)
            )

        return AgentStats(
            name=self._name,
            initialized=self._initialized,
            uptime_seconds=self.get_uptime(),
            config=config_data,
            state=execution_state,
            flows=flows_info,
            memory=memory_stats,
            planner=task_decomposer_stats,  # Changed from task_decomposer to planner
            engine=engine_stats,
            performance=PerformanceMetrics()
        )
    
    def get_uptime(self) -> float:
        """Get agent uptime in seconds."""
        return (datetime.now() - self._start_time).total_seconds()
    
    # Activity stream management
    def set_activity_stream_handler(self, handler: Optional[Any] = None) -> None:
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
    
    async def _get_recent_conversation_history(self, limit: int = 5) -> List[ConversationMessage]:
        """Get recent conversation history for task generation context.
        
        Args:
            limit: Maximum number of recent messages to retrieve
            
        Returns:
            List of conversation messages as Pydantic models
        """
        # Memory is always required - fail fast if not available
        if not self._memory_manager:
            raise ExecutionError("Memory manager is required but not initialized")
        if not self._memory_manager.memory:
            raise ExecutionError("Memory component is required but not initialized")
        
        try:
            # Search for recent conversation messages - use context-based search
            recent_messages = await self._memory_manager.search_memory(
                query="user messages and responses",
                limit=limit,
                context="conversation"  # Filter by conversation context
            )
            
            logger.info(f"Retrieved {len(recent_messages) if recent_messages else 0} messages from memory")
            
            # Convert memory items to conversation history format
            conversation_history = []
            if recent_messages:
                for item in recent_messages[-limit:]:  # Get most recent
                    try:
                        # Memory items must have value with message_type
                        message = ConversationMessage(
                            role=MessageRole.USER if item.value.message_type.value == "conversation" else MessageRole.SYSTEM,
                            content=item.value.content,
                            timestamp=item.value.created_at.isoformat()
                        )
                        conversation_history.append(message)
                    except Exception as e:
                        logger.warning(f"Failed to convert memory item to conversation message: {e}")
                        continue
            else:
                logger.info("No previous conversation messages found in memory")
            
            return conversation_history
            
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            raise ExecutionError(f"Failed to retrieve conversation history: {e}") from e
    
    async def _send_to_engine(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Send message to engine for task decomposition and execution."""
        self._check_initialized("message processing")
        
        if not self._engine:
            raise NotInitializedError("Engine not initialized for message processing")
        
        try:
            # Ensure we have agent state for task decomposition
            if not self._state_manager.current_state:
                await self._state_manager.create_state(message)
            
            # Get recent conversation history for task generation context
            conversation_history = await self._get_recent_conversation_history()
            
            # ALL messages go through engine (which uses existing task decomposition)
            result = await self._engine.execute(message, conversation_history)
            
            # Extract the actual response content from ExecutionResult
            # ExecutionResult must have an output field
            return result.output
                
        except Exception as e:
            self._logger.error(f"Error processing message through engine: {e}")
            raise ExecutionError(f"Failed to process message: {e}") from e
    
    
    async def _store_conversation_turn(self, message: str, response: str, tool_calls: list, tool_results: list) -> None:
        """Store conversation turn in state manager."""
        from flowlib.agent.models.state import ConversationTurn
        
        turn = ConversationTurn(
            user_message=message,
            agent_response=response
            # Note: tool_calls, tool_results, context_snapshot not supported by ConversationTurn
        )
        
        await self._state_manager.add_conversation_turn(turn)
    
    
    # Thread and Queue Management
    
    def start(self) -> None:
        """Start agent in dedicated thread."""
        if self._running:
            raise RuntimeError(f"Agent {self._name} already running")
        
        self._running = True
        self._agent_thread = threading.Thread(
            target=self._run_agent_thread,
            name=f"agent-{self._name}",
            daemon=False
        )
        self._agent_thread.start()
        logger.info(f"Started agent {self._name} in thread {self._agent_thread.name}")
    
    def stop(self) -> None:
        """Stop agent thread."""
        if not self._running:
            logger.warning(f"Agent {self._name} is not running")
            return
        
        self._running = False
        
        # Send shutdown message
        if self._agent_thread:
            shutdown_msg = AgentMessage(
                message_type=AgentMessageType.SHUTDOWN,
                content="",
                response_queue_id=""
            )
            self.input_queue.put(shutdown_msg)
        
        # Wait for thread to finish
        if self._agent_thread:
            self._agent_thread.join(timeout=10)
            if self._agent_thread.is_alive():
                logger.error(f"Agent {self._name} thread did not stop cleanly")
            else:
                logger.info(f"Agent {self._name} stopped")
    
    def is_running(self) -> bool:
        """Check if agent is running."""
        return bool(self._running and self._agent_thread and self._agent_thread.is_alive())
    
    def _run_agent_thread(self) -> None:
        """Main agent thread entry point."""
        # Create new event loop for this thread
        self._agent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._agent_loop)
        
        try:
            # Run the agent processing loop
            self._agent_loop.run_until_complete(self._agent_processing_loop())
        except Exception as e:
            logger.error(f"Agent {self._name} thread error: {e}")
        finally:
            # Clean up event loop
            self._agent_loop.close()
            self._agent_loop = None
    
    async def _agent_processing_loop(self) -> None:
        """Main agent processing loop - runs in agent thread."""
        logger.info(f"[Agent-{self._name}] Starting processing loop")
        try:
            # Initialize agent
            try:
                logger.debug(f"[Agent-{self._name}] Initializing agent...")
                await self.initialize()
                logger.info(f"[Agent-{self._name}] Initialized successfully and ready for messages")
            except Exception as init_error:
                logger.error(f"[Agent-{self._name}] Initialization failed: {init_error}")
                # Send error response for any pending messages
                while True:
                    try:
                        message = self.input_queue.get(timeout=0.1)
                        error_response = AgentResponse(
                            message_id=message.message_id,
                            success=False,
                            error=f"Agent initialization failed: {init_error}",
                            response_data={},
                            processing_time=0,
                            activity_stream=[]
                        )
                        logger.debug(f"[Agent-{self._name}] Sending error response for message {message.message_id}")
                        self.output_queue.put(error_response)
                    except queue.Empty:
                        break
                raise init_error
            
            logger.debug(f"[Agent-{self._name}] Entering main message processing loop")
            while self._running:
                try:
                    # Get message from input queue with timeout to allow shutdown checks
                    logger.debug(f"[Agent-{self._name}] Waiting for message from input queue...")
                    
                    # Use thread-safe queue with timeout
                    try:
                        message = self.input_queue.get(timeout=1.0)
                        logger.info(f"[Agent-{self._name}] Received message {message.message_id} of type {message.message_type}")
                    except queue.Empty:
                        continue
                    
                    # Check for shutdown message
                    if message.message_type == AgentMessageType.SHUTDOWN:
                        logger.info(f"[Agent-{self._name}] Received shutdown message, exiting loop")
                        break
                    
                    # Process message
                    logger.debug(f"[Agent-{self._name}] Processing message {message.message_id}...")
                    response = await self._handle_message_routing(message)
                    logger.debug(f"[Agent-{self._name}] Message processed, response success={response.success}")
                    
                    # Send response to output queue
                    logger.debug(f"[Agent-{self._name}] Putting response in output queue...")
                    self.output_queue.put(response)
                    logger.info(f"[Agent-{self._name}] Response for message {message.message_id} sent to output queue")
                except Exception as e:
                    logger.error(f"Error processing message in agent {self._name}: {e}")
                    # Send error response
                    if 'message' in locals():
                        error_response = AgentResponse(
                            message_id=message.message_id,
                            success=False,
                            error=str(e),
                            response_data={},
                            processing_time=0
                        )
                        self.output_queue.put(error_response)
        
        finally:
            # Shutdown agent
            await self.shutdown()
            logger.info(f"Agent {self._name} processing loop ended")
    
    async def _handle_message_routing(self, message: AgentMessage) -> AgentResponse:
        """Route message by type and create AgentResponse wrapper.
        
        Args:
            message: Message to route
            
        Returns:
            Agent response with proper wrapping
        """
        start_time = time.time()
        logger.debug(f"[Agent-{self._name}] Starting to process message {message.message_id}")
        
        # Clear activity stream buffer for new message
        if self._activity_stream:
            logger.debug(f"[Agent-{self._name}] Clearing activity stream buffer")
            self._activity_stream.activity_buffer.clear()
        
        try:
            # Route based on message type
            logger.debug(f"[Agent-{self._name}] Routing message type: {message.message_type}")
            if message.message_type == AgentMessageType.CONVERSATION:
                logger.debug(f"[Agent-{self._name}] Processing conversation message")
                result = await self._handle_message_lifecycle(
                    message.content,
                    message.context
                )
            elif message.message_type == AgentMessageType.TASK:
                logger.debug(f"[Agent-{self._name}] Processing task message")
                result = await self._execute_task_message(message)
            elif message.message_type == AgentMessageType.COMMAND:
                logger.debug(f"[Agent-{self._name}] Processing command message")
                result = await self._execute_command_message(message)
            elif message.message_type == AgentMessageType.SYSTEM:
                logger.debug(f"[Agent-{self._name}] Processing system message")
                result = await self._execute_system_message(message)
            else:
                raise ValueError(f"Unknown message type: {message.message_type}")
            
            logger.debug(f"[Agent-{self._name}] Message processing completed successfully")
            
            # Build successful response
            response = AgentResponse(
                message_id=message.message_id,
                success=True,
                response_data=result if isinstance(result, dict) else {"content": str(result)},
                processing_time=time.time() - start_time,
                activity_stream=[dict(entry) for entry in self._activity_stream.activity_buffer] if self._activity_stream else []
            )
            logger.debug(f"[Agent-{self._name}] Built response: success=True, has_content={bool(response.response_data)}")
            return response
            
        except Exception as e:
            logger.error(f"[Agent-{self._name}] Error processing message {message.message_id}: {e}")
            return AgentResponse(
                message_id=message.message_id,
                success=False,
                error=str(e),
                response_data={},
                processing_time=time.time() - start_time,
                activity_stream=[dict(entry) for entry in self._activity_stream.activity_buffer] if self._activity_stream else []
            )
    
    async def _handle_message_lifecycle(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle complete message lifecycle: memory → engine → learning → metadata wrapping.
        
        This orchestrates the full message handling pipeline.
        """
        logger.debug(f"[Agent-{self._name}] Processing internal message: {message[:50]}...")
        
        # Check initialized
        self._check_initialized("process_message")
        
        # Store in memory if enabled
        if self.config.enable_memory:
            logger.debug(f"[Agent-{self._name}] Storing message in memory")
            # Create proper AgentMessage model for storage
            agent_message = AgentMessage(
                message_type=AgentMessageType.CONVERSATION,
                content=message,
                context=context,
                response_queue_id="memory_storage"  # Required field
            )
            
            # Create semantic key based on message content
            semantic_key = self._generate_semantic_key(message, "conversation")
            
            await self.store_memory(
                key=semantic_key,
                value=agent_message,
                context=AgentMessageType.CONVERSATION.value
            )
        
        # Use flow-based message processing (sequential as before)
        logger.debug(f"[Agent-{self._name}] Processing message with flows")
        response = await self._send_to_engine(message, context)
        logger.debug(f"[Agent-{self._name}] Flow processing complete, response: {str(response)[:100]}...")
        
        # Build response with metadata
        response_content = response if isinstance(response, str) else str(response)
        
        response_data = {
            "content": response_content,
            "metadata": {
                "flows_executed": self._flow_runner._last_executed_flows if hasattr(self._flow_runner, '_last_executed_flows') else [],
                "memory_enabled": self.config.enable_memory,
                "learning_enabled": bool(self._knowledge)
            }
        }
        
        # Include activity if available
        if self._activity_stream:
            response_data["activity"] = [str(entry) for entry in self._activity_stream.activity_buffer]
        
        logger.debug(f"[Agent-{self._name}] Built response data with content length: {len(response_data.get('content', ''))}")
        return response_data
    
    async def _execute_task_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Execute a task message."""
        # Extract task details from message
        task_data = message.metadata.get("task", {})
        
        # Execute task through engine if available
        if self._engine:
            # Engine's execute method takes task_description and conversation_history
            task_description = task_data.get("description", message.content)
            result = await self._engine.execute(
                task_description=task_description,
                conversation_history=None  # Will be managed by state manager
            )
            return {"task_result": result}
        else:
            # Fallback to conversation processing
            return await self._handle_message_lifecycle(
                f"Execute task: {message.content}",
                message.context
            )
    
    async def _execute_command_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Execute a command message."""
        # Commands are processed through flows
        # Context is required for command messages
        command_context = message.context or {}
        command_context["command"] = True
        
        return await self._handle_message_lifecycle(
            message.content,
            command_context
        )
    
    async def _execute_system_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Execute a system message."""
        # System messages for agent control
        system_command = message.metadata.get("command", "")
        
        if system_command == "status":
            return self.get_stats().model_dump()
        elif system_command == "reset":
            # Reset functionality not implemented
            return {"status": "reset_not_available", "message": "Reset functionality not implemented"}
        elif system_command == "reload_config":
            # Reload configuration
            new_config = message.metadata.get("config")
            if new_config:
                self._config_manager.update_config(new_config)
            return {"status": "config_reloaded"}
        else:
            return {"error": f"Unknown system command: {system_command}"}