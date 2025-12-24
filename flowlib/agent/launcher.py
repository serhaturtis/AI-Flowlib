"""Unified agent launcher - replaces all run_* scripts and runners."""

import logging
from typing import Any, cast

from pydantic import Field

from flowlib.agent.components.memory.component import AgentMemoryConfig
from flowlib.agent.components.memory.knowledge import KnowledgeMemoryConfig
from flowlib.agent.components.memory.vector import VectorMemoryConfig
from flowlib.agent.components.memory.working import WorkingMemoryConfig
from flowlib.agent.core.base_agent import BaseAgent
from flowlib.agent.execution.strategy import ExecutionMode, ExecutionStrategy
from flowlib.agent.execution.strategies import (
    AutonomousConfig,
    AutonomousStrategy,
    REPLConfig,
    REPLStrategy,
    RemoteConfig,
    RemoteStrategy,
)
from flowlib.agent.execution.strategies.daemon import DaemonConfig, DaemonStrategy
from flowlib.agent.models.config import AgentConfig, StatePersistenceConfig
from flowlib.config.alias_manager import alias_manager
from flowlib.config.required_resources import RequiredAlias
from flowlib.core.message_source_config import MessageSourceConfig
from flowlib.core.models import StrictBaseModel
from flowlib.core.project import Project
from flowlib.resources.models.agent_config_resource import AgentConfigResource
from flowlib.resources.models.message_source_resource import MessageSourceResource
from flowlib.resources.registry.registry import resource_registry

logger = logging.getLogger(__name__)


def build_agent_config(project: Project, agent_config_name: str) -> AgentConfig:
    """Convert a registered agent config resource into a runtime AgentConfig."""
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
        activity_handler: Any | None = None,
    ) -> None:
        """Launch agent in specified execution mode.

        Args:
            agent_config_name: Name of agent config from resource registry
            mode: Execution mode (REPL, DAEMON, AUTONOMOUS, REMOTE)
            execution_config: Mode-specific configuration dict
            activity_handler: Optional callback for activity stream events.
                The handler receives formatted activity strings from the agent.
                Use this to stream agent activity to external systems.

        Raises:
            ValueError: If agent config not found
            RuntimeError: If launcher not initialized
        """
        if not self._initialized:
            await self.initialize()

        # Validate required aliases before launching
        self.project._validate_required_aliases()

        logger.info(f"Launching agent '{agent_config_name}' in mode '{mode.value}'")

        # Load agent config resource for strategy creation
        agent_config_resource = self._get_agent_config_resource(agent_config_name)

        # Load agent configuration
        agent = await self._create_agent(agent_config_name)

        # Set activity handler if provided
        if activity_handler is not None:
            agent.set_activity_stream_handler(activity_handler)
            logger.debug("Activity stream handler configured for agent '%s'", agent_config_name)

        # Create execution strategy (pass agent_config_resource for daemon mode)
        strategy = self._create_strategy(
            mode, execution_config or {}, agent_config_resource
        )

        # Execute
        try:
            await strategy.execute(agent)
        finally:
            await strategy.cleanup()
            await agent.shutdown()

    def _get_agent_config_resource(self, agent_config_name: str) -> AgentConfigResource:
        """Get agent config resource from registry.

        Args:
            agent_config_name: Name of agent config (may be alias)

        Returns:
            AgentConfigResource instance

        Raises:
            ValueError: If config not found or not an AgentConfigResource
        """
        try:
            actual_config_name = alias_manager.get_alias_target(agent_config_name)
            if actual_config_name:
                resource = resource_registry.get(actual_config_name)
            else:
                resource = resource_registry.get(agent_config_name)
        except Exception as e:
            raise ValueError(
                f"Could not load agent configuration '{agent_config_name}': {e}"
            ) from e

        if not isinstance(resource, AgentConfigResource):
            raise ValueError(
                f"Resource '{agent_config_name}' is not an AgentConfigResource. "
                f"Got type: {type(resource).__name__}"
            )

        return resource

    async def _create_agent(self, agent_config_name: str) -> BaseAgent:
        """Create and initialize agent from config name."""
        config = build_agent_config(self.project, agent_config_name)
        agent = BaseAgent(config)
        await agent.initialize()

        logger.info(f"Agent '{agent.name}' initialized successfully")
        return agent

    def _resolve_message_sources(
        self,
        agent_config_resource: AgentConfigResource,
        execution_config: dict[str, Any],
    ) -> list[MessageSourceConfig]:
        """Resolve message source names to runtime configs from registry.

        Priority: execution_config overrides agent_config_resource.
        If execution_config already contains 'message_sources' as list of
        MessageSourceConfig objects, use those directly. Otherwise, resolve
        names from agent_config_resource.message_sources.

        Args:
            agent_config_resource: Agent config resource with message_sources names
            execution_config: Mode-specific configuration that may override sources

        Returns:
            List of resolved MessageSourceConfig instances

        Raises:
            ValueError: If a source name is not found or not a MESSAGE_SOURCE resource
        """
        # Check if execution_config provides pre-resolved configs
        exec_sources = execution_config.get("message_sources", [])
        if exec_sources and isinstance(exec_sources[0], MessageSourceConfig):
            # Already resolved configs passed via execution_config
            return exec_sources

        # Get source names - execution_config overrides agent config
        if exec_sources:
            # Names provided in execution_config
            source_names = exec_sources
        else:
            # Fall back to agent config resource
            source_names = agent_config_resource.message_sources

        # Resolve each name from registry
        resolved_sources: list[MessageSourceConfig] = []
        for name in source_names:
            try:
                resource = resource_registry.get(name)
            except KeyError as e:
                raise ValueError(
                    f"Message source '{name}' not found in registry. "
                    f"Ensure it is defined with a @timer_source, @email_source, "
                    f"@webhook_source, or @queue_source decorator."
                ) from e

            if not isinstance(resource, MessageSourceResource):
                actual_type = getattr(resource, "type", type(resource).__name__)
                raise ValueError(
                    f"Resource '{name}' is not a MESSAGE_SOURCE resource. "
                    f"Got type: {actual_type}. "
                    f"Use @timer_source, @email_source, etc. decorators to define message sources."
                )

            # Convert resource to runtime config
            runtime_config = resource.to_runtime_config()
            resolved_sources.append(runtime_config)
            logger.debug(f"Resolved message source '{name}' -> {type(runtime_config).__name__}")

        return resolved_sources

    def _create_strategy(
        self,
        mode: ExecutionMode,
        execution_config: dict[str, Any],
        agent_config_resource: AgentConfigResource | None = None,
    ) -> ExecutionStrategy:
        """Create execution strategy for specified mode.

        Args:
            mode: Execution mode
            execution_config: Mode-specific configuration
            agent_config_resource: Agent config resource (required for DAEMON mode)

        Returns:
            ExecutionStrategy instance

        Raises:
            ValueError: If mode is invalid, config is invalid, or daemon has no sources
        """
        if mode == ExecutionMode.AUTONOMOUS:
            config = AutonomousConfig(**execution_config)
            return AutonomousStrategy(config)

        elif mode == ExecutionMode.REPL:
            repl_config = REPLConfig(**execution_config)
            return REPLStrategy(repl_config)

        elif mode == ExecutionMode.DAEMON:
            if agent_config_resource is None:
                raise ValueError(
                    "DAEMON mode requires agent_config_resource to resolve message sources"
                )

            # Resolve message sources from registry
            resolved_sources = self._resolve_message_sources(
                agent_config_resource, execution_config
            )

            # Validate: daemon mode must have at least one source
            if not resolved_sources:
                raise ValueError(
                    f"DAEMON mode requires at least one message source. "
                    f"Agent '{agent_config_resource.name}' has no message_sources configured. "
                    f"Either add message_sources to the agent config or pass them via execution_config."
                )

            # Build daemon config with resolved sources
            daemon_config_dict = {
                k: v for k, v in execution_config.items() if k != "message_sources"
            }
            daemon_config_dict["message_sources"] = resolved_sources
            daemon_config = DaemonConfig(**daemon_config_dict)

            logger.info(
                f"Daemon mode configured with {len(resolved_sources)} message sources: "
                f"{[s.name for s in resolved_sources]}"
            )

            return DaemonStrategy(daemon_config)

        elif mode == ExecutionMode.REMOTE:
            remote_config = RemoteConfig(**execution_config)
            return RemoteStrategy(remote_config)

        else:
            raise ValueError(f"Unsupported execution mode: {mode}")
