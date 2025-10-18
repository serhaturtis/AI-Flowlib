"""Unified agent launcher - replaces all run_* scripts and runners."""

import logging
from typing import Any, Dict, Optional, cast

from pydantic import Field

from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.execution.strategy import ExecutionMode, ExecutionStrategy
from flowlib.agent.models.config import AgentConfig, StatePersistenceConfig
from flowlib.core.models import StrictBaseModel
from flowlib.core.project import Project
from flowlib.resources.models.agent_config_resource import AgentConfigResource
from flowlib.resources.registry.registry import resource_registry

logger = logging.getLogger(__name__)


class LauncherConfig(StrictBaseModel):
    """Configuration for agent launcher."""

    project_path: Optional[str] = Field(
        default=None,
        description="Project path. None uses ~/.flowlib/"
    )
    agent_config_name: str = Field(
        ...,
        description="Agent configuration name from resource registry"
    )
    execution_mode: ExecutionMode = Field(
        ...,
        description="Execution mode (repl, daemon, autonomous, remote)"
    )
    execution_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mode-specific configuration"
    )


class AgentLauncher:
    """Unified agent launcher.

    Replaces all run_* scripts and runner wrappers with a single
    consistent interface for launching agents in any execution mode.

    Example:
        >>> launcher = AgentLauncher(project_path="./projects/my-agent")
        >>> await launcher.launch(
        ...     agent_config_name="my-agent",
        ...     mode=ExecutionMode.REPL
        ... )
    """

    def __init__(self, project_path: Optional[str] = None):
        """Initialize launcher with project.

        Args:
            project_path: Optional project path. None uses ~/.flowlib/
        """
        self.project = Project(project_path)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize project and load configurations."""
        if self._initialized:
            return

        logger.info(f"Initializing project at {self.project.flowlib_path}")
        self.project.initialize()
        self.project.load_configurations()
        self._initialized = True
        logger.info("Project initialization complete")

    async def launch(
        self,
        agent_config_name: str,
        mode: ExecutionMode,
        execution_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Launch agent in specified execution mode.

        Args:
            agent_config_name: Name of agent config from resource registry
            mode: Execution mode (REPL, DAEMON, AUTONOMOUS, REMOTE)
            execution_config: Mode-specific configuration dict

        Raises:
            ValueError: If agent config not found
            RuntimeError: If launcher not initialized
        """
        if not self._initialized:
            await self.initialize()

        logger.info(
            f"Launching agent '{agent_config_name}' in mode '{mode.value}'"
        )

        # Load agent configuration
        agent = await self._create_agent(agent_config_name)

        # Create execution strategy
        strategy = self._create_strategy(mode, execution_config or {})

        # Execute
        try:
            await strategy.execute(agent)
        finally:
            await strategy.cleanup()
            await agent.shutdown()

    async def _create_agent(self, agent_config_name: str) -> BaseAgent:
        """Create and initialize agent from config name.

        Args:
            agent_config_name: Agent config name in resource registry

        Returns:
            Initialized BaseAgent instance

        Raises:
            ValueError: If agent config not found
        """
        # Try to resolve through role assignment first
        from flowlib.config.role_manager import role_manager

        try:
            actual_config_name = role_manager.get_role_assignment(
                agent_config_name
            )
            if actual_config_name:
                agent_config_resource = resource_registry.get(actual_config_name)
                logger.info(
                    f"Loaded agent config '{agent_config_name}' -> "
                    f"'{actual_config_name}'"
                )
            else:
                agent_config_resource = resource_registry.get(agent_config_name)
                logger.info(f"Loaded agent config '{agent_config_name}' directly")
        except Exception as e:
            raise ValueError(
                f"Could not load agent configuration '{agent_config_name}': {e}"
            )

        # Cast and extract configuration
        config_resource = cast(AgentConfigResource, agent_config_resource)

        # Build memory config
        from flowlib.agent.components.memory.component import AgentMemoryConfig
        from flowlib.agent.components.memory.knowledge import (
            KnowledgeMemoryConfig,
        )
        from flowlib.agent.components.memory.vector import VectorMemoryConfig
        from flowlib.agent.components.memory.working import WorkingMemoryConfig

        # Memory is always required
        memory_config = AgentMemoryConfig(
            working_memory=WorkingMemoryConfig(default_ttl_seconds=3600),
            vector_memory=VectorMemoryConfig(
                vector_provider_config="default-vector-db",
                embedding_provider_config="default-embedding"
            ),
            knowledge_memory=KnowledgeMemoryConfig(
                graph_provider_config="default-graph-db"
            ),
            fusion_llm_config=config_resource.llm_name,
            store_execution_history=True
        )

        # Build AgentConfig
        config = AgentConfig(
            name=config_resource.name or agent_config_name,
            persona=config_resource.persona,
            profile_name=config_resource.profile_name,
            provider_name=config_resource.llm_name,
            model_name=config_resource.model_name,
            temperature=config_resource.temperature,
            max_iterations=config_resource.max_iterations,
            enable_learning=config_resource.enable_learning,
            memory=memory_config,
            state_config=StatePersistenceConfig(
                persistence_type="file",
                base_path="./agent_states",
                auto_save=True,
                auto_load=False
            )
        )

        # Create and initialize agent
        agent = BaseAgent(config)
        await agent.initialize()

        logger.info(f"Agent '{agent.name}' initialized successfully")
        return agent

    def _create_strategy(
        self,
        mode: ExecutionMode,
        execution_config: Dict[str, Any]
    ) -> ExecutionStrategy:
        """Create execution strategy for specified mode.

        Args:
            mode: Execution mode
            execution_config: Mode-specific configuration

        Returns:
            ExecutionStrategy instance

        Raises:
            ValueError: If mode is invalid or config is invalid
        """
        if mode == ExecutionMode.AUTONOMOUS:
            from flowlib.agent.execution.strategies import (
                AutonomousConfig,
                AutonomousStrategy,
            )

            config = AutonomousConfig(**execution_config)
            return AutonomousStrategy(config)

        elif mode == ExecutionMode.REPL:
            from flowlib.agent.execution.strategies import REPLConfig, REPLStrategy

            repl_config = REPLConfig(**execution_config)
            return REPLStrategy(repl_config)

        elif mode == ExecutionMode.DAEMON:
            from flowlib.agent.execution.strategies.daemon import (
                DaemonConfig,
                DaemonStrategy,
            )

            daemon_config = DaemonConfig(**execution_config)
            return DaemonStrategy(daemon_config)

        elif mode == ExecutionMode.REMOTE:
            from flowlib.agent.execution.strategies import (
                RemoteConfig,
                RemoteStrategy,
            )

            remote_config = RemoteConfig(**execution_config)
            return RemoteStrategy(remote_config)

        else:
            raise ValueError(f"Unsupported execution mode: {mode}")
