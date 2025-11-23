"""Unified agent launcher - replaces all run_* scripts and runners."""

import logging
from typing import Any, cast

from pydantic import Field

from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.execution.strategy import ExecutionMode, ExecutionStrategy
from flowlib.agent.models.config import AgentConfig, StatePersistenceConfig
from flowlib.core.models import StrictBaseModel
from flowlib.core.project import Project
from flowlib.resources.models.agent_config_resource import AgentConfigResource
from flowlib.resources.registry.registry import resource_registry
from flowlib.config.required_resources import RequiredAlias

logger = logging.getLogger(__name__)


def build_agent_config(project: Project, agent_config_name: str) -> AgentConfig:
    """Convert a registered agent config resource into a runtime AgentConfig."""

    from flowlib.config.alias_manager import alias_manager
    from flowlib.agent.components.memory.component import AgentMemoryConfig
    from flowlib.agent.components.memory.knowledge import KnowledgeMemoryConfig
    from flowlib.agent.components.memory.vector import VectorMemoryConfig
    from flowlib.agent.components.memory.working import WorkingMemoryConfig

    try:
        actual_config_name = alias_manager.get_alias_target(agent_config_name)
        if actual_config_name:
            agent_config_resource = resource_registry.get(actual_config_name)
            logger.info(f"Loaded agent config '{agent_config_name}' -> '{actual_config_name}'")
        else:
            agent_config_resource = resource_registry.get(agent_config_name)
            logger.info(f"Loaded agent config '{agent_config_name}' directly")
    except Exception as e:
        raise ValueError(f"Could not load agent configuration '{agent_config_name}': {e}") from e

    config_resource = cast(AgentConfigResource, agent_config_resource)

    memory_config = AgentMemoryConfig(
        working_memory=WorkingMemoryConfig(default_ttl_seconds=3600),
        vector_memory=VectorMemoryConfig(
            vector_provider_config=RequiredAlias.DEFAULT_VECTOR_DB.value,
            embedding_provider_config=RequiredAlias.DEFAULT_EMBEDDING.value,
        ),
        knowledge_memory=KnowledgeMemoryConfig(graph_provider_config=RequiredAlias.DEFAULT_GRAPH_DB.value),
        fusion_llm_config=config_resource.llm_name,
        store_execution_history=True,
    )

    return AgentConfig(
        name=config_resource.name or agent_config_name,
        persona=config_resource.persona,
        allowed_tool_categories=config_resource.allowed_tool_categories,
        working_directory=str(project.flowlib_path),
        provider_name=config_resource.llm_name,
        model_name=config_resource.model_name,
        temperature=config_resource.temperature,
        max_iterations=config_resource.max_iterations,
        enable_learning=config_resource.enable_learning,
        memory=memory_config,
        state_config=StatePersistenceConfig(
            persistence_type="file", base_path="./agent_states", auto_save=True, auto_load=False
        ),
    )


class LauncherConfig(StrictBaseModel):
    """Configuration for agent launcher."""

    project_path: str | None = Field(
        default=None, description="Project path. None uses ~/.flowlib/"
    )
    agent_config_name: str = Field(
        ..., description="Agent configuration name from resource registry"
    )
    execution_mode: ExecutionMode = Field(
        ..., description="Execution mode (repl, daemon, autonomous, remote)"
    )
    execution_config: dict[str, Any] = Field(
        default_factory=dict, description="Mode-specific configuration"
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

    def __init__(self, project_path: str | None = None):
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
        execution_config: dict[str, Any] | None = None,
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

        # Validate required aliases before launching
        self.project._validate_required_aliases()

        logger.info(f"Launching agent '{agent_config_name}' in mode '{mode.value}'")

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
        """Create and initialize agent from config name."""
        config = build_agent_config(self.project, agent_config_name)
        agent = BaseAgent(config)
        await agent.initialize()

        logger.info(f"Agent '{agent.name}' initialized successfully")
        return agent

    def _create_strategy(
        self, mode: ExecutionMode, execution_config: dict[str, Any]
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
