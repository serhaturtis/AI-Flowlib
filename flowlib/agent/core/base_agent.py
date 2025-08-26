"""
Agent orchestrator component.

This module provides the main coordination component that replaces the monolithic
BaseAgent class, orchestrating focused manager components.
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from flowlib.flows.base.base import Flow

from flowlib.agent.core.component_registry import ComponentRegistry
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
from flowlib.agent.models.state import AgentStats, ComponentStats, AgentConfiguration, FlowRegistryStats, PerformanceMetrics, AgentStatusSummary
from flowlib.agent.components.persistence.base import BaseStatePersister
from flowlib.flows.models.results import FlowResult
from flowlib.providers.knowledge.plugin_manager import KnowledgePluginManager
from flowlib.providers.core.registry import provider_registry
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
                 config: Optional[Union[Dict[str, Any], AgentConfig]] = None,
                 task_description: str = "",
                 name: str = None,
                 state_persister: Optional[BaseStatePersister] = None):
        """Initialize the base agent.
        
        Args:
            config: Agent configuration
            task_description: Task description for the agent
            name: Name for the agent
            state_persister: State persister for agent
        """
        # Initialize agent properties (not from AgentComponent)
        if name:
            self._name = name
        elif config and hasattr(config, 'name'):
            self._name = config.name
        else:
            self._name = "BaseAgent"
        
        self._initialized = False
        self._logger = logging.getLogger(f"{__name__}.{self._name}")
        
        # Create this agent's private component registry
        self._registry = ComponentRegistry(agent_name=self._name)
        
        # Create and register managers
        self._config_manager = AgentConfigManager()
        self._config_manager.set_registry(self._registry)
        self._registry.register("config_manager", self._config_manager, AgentConfigManager)
        
        self._state_manager = AgentStateManager(state_persister)
        self._state_manager.set_registry(self._registry)
        self._registry.register("state_manager", self._state_manager, AgentStateManager)
        
        self._memory_manager = AgentMemoryManager()
        self._memory_manager.set_registry(self._registry)
        self._registry.register("memory_manager", self._memory_manager, AgentMemoryManager)
        
        self._flow_runner = AgentFlowRunner()
        self._flow_runner.set_registry(self._registry)
        self._registry.register("flow_runner", self._flow_runner, AgentFlowRunner)
        
        self._learning_manager = AgentLearningManager()
        self._learning_manager.set_registry(self._registry)
        self._registry.register("learning_manager", self._learning_manager, AgentLearningManager)
        
        # Prepare configuration
        agent_config = self._config_manager.prepare_config(config)
        
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
        
        # Queue-based I/O
        self.input_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.output_queue: asyncio.Queue[AgentResponse] = asyncio.Queue()
        
        # Thread management
        self._agent_thread: Optional[threading.Thread] = None
        self._agent_loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize the base agent and all managers."""
        if self._initialized:
            return
        try:
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
            await self._flow_runner.initialize()
            await self._learning_manager.initialize()
            
            # No parent relationships needed - components use registry
            
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
            
            self._initialized = False
            logger.info(f"Agent '{self._name}' shutdown complete")
            
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
        self._planner.set_registry(self._registry)
        self._registry.register("planner", self._planner, AgentPlanner)
        
        # Initialize reflection
        self._reflection = AgentReflection(
            config=config.reflection_config,
            activity_stream=self._activity_stream
        )
        self._reflection.set_registry(self._registry)
        self._registry.register("reflection", self._reflection, AgentReflection)
        
        # Initialize engine
        self._engine = AgentEngine(
            config=config.engine_config,
            memory=self._memory_manager.memory,
            planner=self._planner,
            reflection=self._reflection,
            activity_stream=self._activity_stream,
            agent_config=config
        )
        self._engine.set_registry(self._registry)
        self._registry.register("engine", self._engine, AgentEngine)
        
        # Initialize all core components
        await self._planner.initialize()
        await self._reflection.initialize()
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
    
    # High-level operations are now handled through queue-based messaging
    # The old process_message is replaced by _process_message_internal
    
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
        
        # Create proper AgentConfiguration with only the required fields
        if self.config:
            config_data = AgentConfiguration(
                name=self.config.name,
                persona=self.config.persona,
                provider_name=self.config.provider_name,
                system_prompt=self.config.persona,  # Use persona as system prompt
                max_iterations=self.config.engine_config.max_iterations if self.config.engine_config else 10,
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

        # Create AgentStatusSummary from current state manager data
        status_summary = None
        if self._state_manager.current_state:
            status_summary = AgentStatusSummary(
                task_id=getattr(self._state_manager.current_state, 'task_id', 'unknown'),
                task_description=getattr(self._state_manager.current_state, 'task_description', 'Unknown task'),
                cycle=getattr(self._state_manager.current_state, 'cycles', 0),
                progress=getattr(self._state_manager.current_state, 'progress', 0),
                is_complete=getattr(self._state_manager.current_state, 'is_complete', False),
                execution_phase="running" if self._initialized else "idle",
                last_action=None,
                error_state=False
            )

        return AgentStats(
            name=self._name,
            initialized=self._initialized,
            uptime_seconds=self.get_uptime(),
            config=config_data,
            state=status_summary,
            flows=flows_info,
            memory=memory_stats,
            planner=planner_stats,
            reflection=reflection_stats,
            engine=engine_stats,
            performance=PerformanceMetrics()
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
    
    async def _process_message_with_flows(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process message using proper flow-based architecture."""
        from flowlib.agent.components.classification.models import MessageClassifierInput
        from flowlib.agent.components.conversation.models import ConversationInput
        from flowlib.agent.components.shell_command.models import ShellCommandIntentInput
        
        # Step 1: Classify the message using flow runner  
        try:
            # Get conversation history from context if available
            conversation_history = context.get("conversation_history", []) if context else []
            # Convert to ConversationMessage objects if they're dicts
            conversation_messages = []
            for msg in conversation_history:
                if isinstance(msg, dict):
                    from flowlib.agent.components.classification.models import ConversationMessage
                    conversation_messages.append(ConversationMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", "")
                    ))
                else:
                    conversation_messages.append(msg)
            
            classifier_input = MessageClassifierInput(
                message=message,
                conversation_history=conversation_messages
            )
            classification_result = await self._flow_runner.execute_flow(
                "message-classifier-flow", 
                classifier_input
            )
            
            # classification_result is a FlowResult containing MessageClassification in data
            if classification_result.is_error():
                logger.error(f"Classification flow failed: {classification_result.error}")
                raise RuntimeError(f"Classification failed: {classification_result.error}")
            
            message_classification = classification_result.data
            logger.info(f"Message classification: execute_task={message_classification.execute_task}, category={message_classification.category}")
            
        except Exception as e:
            logger.error(f"Message classification failed: {e}")
            raise
        
        # Step 2: Route to appropriate flow based on classification
        if message_classification.execute_task:
            # Handle task execution - check category for specific task types
            if message_classification.category in ["shell_command", "command", "technical"]:
                command_input = ShellCommandIntentInput(message=message)
                result = await self._flow_runner.execute_flow("shell-command", command_input)
                return result.data.response
            else:
                # Other task types - for now default to conversation
                logger.info(f"Task execution requested but category '{message_classification.category}' not handled, using conversation")
                # Get conversation history from context if available
                conversation_history = context.get("conversation_history", []) if context else []
                conv_input = ConversationInput(
                    message=message, 
                    conversation_history=conversation_history
                )
                result = await self._flow_runner.execute_flow("conversation", conv_input)
                return result.data.response
        else:
            # Handle conversational messages  
            # Get conversation history from context if available
            conversation_history = context.get("conversation_history", []) if context else []
            conv_input = ConversationInput(
                message=message,
                conversation_history=conversation_history  
            )
            result = await self._flow_runner.execute_flow("conversation", conv_input)
            return result.data.response
    
    async def _learn_from_conversation(self, message: str, response: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Learn from a conversation exchange.
        
        Returns:
            True if learning failed, False otherwise
        """
        if self._learning_manager and self._learning_manager.learning_enabled:
            try:
                content = f"User: {message}\nAssistant: {response}"
                # Delegate to learning manager which will handle worthiness evaluation
                await self._learning_manager.learn(content, "conversation", ["dialogue"])
                return False  # Success
            except Exception as e:
                logger.warning(f"Learning system error: {e}")
                return True  # Failed
        return False
    
    # Thread and Queue Management
    
    def start(self):
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
    
    def stop(self):
        """Stop agent thread."""
        if not self._running:
            logger.warning(f"Agent {self._name} is not running")
            return
        
        self._running = False
        
        # Send shutdown message
        if self._agent_loop:
            shutdown_msg = AgentMessage(
                message_type=AgentMessageType.SHUTDOWN,
                content="",
                response_queue_id=""
            )
            asyncio.run_coroutine_threadsafe(
                self.input_queue.put(shutdown_msg),
                self._agent_loop
            )
        
        # Wait for thread to finish
        if self._agent_thread:
            self._agent_thread.join(timeout=10)
            if self._agent_thread.is_alive():
                logger.error(f"Agent {self._name} thread did not stop cleanly")
            else:
                logger.info(f"Agent {self._name} stopped")
    
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._running and self._agent_thread and self._agent_thread.is_alive()
    
    def _run_agent_thread(self):
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
    
    async def _agent_processing_loop(self):
        """Main agent processing loop - runs in agent thread."""
        try:
            # Initialize agent
            await self.initialize()
            logger.info(f"Agent {self._name} initialized and ready for messages")
            
            while self._running:
                try:
                    # Get message from input queue with timeout to allow shutdown checks
                    message = await asyncio.wait_for(
                        self.input_queue.get(),
                        timeout=1.0
                    )
                    
                    # Check for shutdown message
                    if message.message_type == AgentMessageType.SHUTDOWN:
                        logger.info(f"Agent {self._name} received shutdown message")
                        break
                    
                    # Process message
                    response = await self._process_agent_message(message)
                    
                    # Send response to output queue
                    await self.output_queue.put(response)
                    
                except asyncio.TimeoutError:
                    # Timeout is normal, allows checking _running flag
                    continue
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
                        await self.output_queue.put(error_response)
        
        finally:
            # Shutdown agent
            await self.shutdown()
            logger.info(f"Agent {self._name} processing loop ended")
    
    async def _process_agent_message(self, message: AgentMessage) -> AgentResponse:
        """Process a single agent message.
        
        Args:
            message: Message to process
            
        Returns:
            Agent response
        """
        start_time = time.time()
        
        # Clear activity stream buffer for new message
        if self._activity_stream:
            self._activity_stream.activity_buffer.clear()
        
        try:
            # Route based on message type
            if message.message_type == AgentMessageType.CONVERSATION:
                result = await self._process_message_internal(
                    message.content,
                    message.context
                )
            elif message.message_type == AgentMessageType.TASK:
                result = await self._execute_task_message(message)
            elif message.message_type == AgentMessageType.COMMAND:
                result = await self._execute_command_message(message)
            elif message.message_type == AgentMessageType.SYSTEM:
                result = await self._execute_system_message(message)
            else:
                raise ValueError(f"Unknown message type: {message.message_type}")
            
            # Build successful response
            return AgentResponse(
                message_id=message.message_id,
                success=True,
                response_data=result if isinstance(result, dict) else {"content": str(result)},
                processing_time=time.time() - start_time,
                activity_stream=self._activity_stream.activity_buffer if self._activity_stream else []
            )
            
        except Exception as e:
            logger.error(f"Error processing message {message.message_id}: {e}")
            return AgentResponse(
                message_id=message.message_id,
                success=False,
                error=str(e),
                response_data={},
                processing_time=time.time() - start_time,
                activity_stream=self._activity_stream.activity_buffer if self._activity_stream else []
            )
    
    async def _process_message_internal(self, message: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Internal message processing - existing logic.
        
        This replaces the old process_message method with internal processing.
        """
        # Check initialized
        self._check_initialized("process_message")
        
        # Store in memory if enabled
        if self.config.enable_memory:
            await self.store_memory(
                key=f"message_{datetime.now().isoformat()}",
                value={"message": message, "context": context}
            )
        
        # Use flow-based message processing (sequential as before)
        response = await self._process_message_with_flows(message, context)
        
        # Learn from conversation if learning is enabled
        if self._learning_manager.learning_enabled:
            learning_error = await self._learn_from_conversation(message, response, context)
            if learning_error:
                response = f"Learning system error occurred while processing message: {message}"
        
        # Build response with metadata
        response_data = {
            "content": response if isinstance(response, str) else str(response),
            "metadata": {
                "flows_executed": self._flow_runner._last_executed_flows if hasattr(self._flow_runner, '_last_executed_flows') else [],
                "memory_enabled": self.config.enable_memory,
                "learning_enabled": self._learning_manager.learning_enabled
            }
        }
        
        # Include activity if available
        if self._activity_stream:
            response_data["activity"] = self._activity_stream.activity_buffer
        
        return response_data
    
    async def _execute_task_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Execute a task message."""
        # Extract task details from message
        task_data = message.metadata.get("task", {})
        
        # Execute task through engine if available
        if self._engine:
            result = await self._engine.execute_task(
                task_id=task_data.get("id", message.message_id),
                task_type=task_data.get("type", "general"),
                task_data=task_data
            )
            return {"task_result": result}
        else:
            # Fallback to conversation processing
            return await self._process_message_internal(
                f"Execute task: {message.content}",
                message.context
            )
    
    async def _execute_command_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Execute a command message."""
        # Commands are processed through flows
        command_context = message.context or {}
        command_context["command"] = True
        
        return await self._process_message_internal(
            message.content,
            command_context
        )
    
    async def _execute_system_message(self, message: AgentMessage) -> Dict[str, Any]:
        """Execute a system message."""
        # System messages for agent control
        system_command = message.metadata.get("command", "")
        
        if system_command == "status":
            return self.get_agent_status()
        elif system_command == "reset":
            await self.reset_state()
            return {"status": "reset_complete"}
        elif system_command == "reload_config":
            # Reload configuration
            new_config = message.metadata.get("config")
            if new_config:
                await self._config_manager.update_config(new_config)
            return {"status": "config_reloaded"}
        else:
            return {"error": f"Unknown system command: {system_command}"}