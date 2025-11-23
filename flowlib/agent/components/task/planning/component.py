"""Structured planning component for Plan-and-Execute architecture."""

import logging
import time

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.context.models import ExecutionContext
from flowlib.agent.core.errors import ExecutionError
from flowlib.flows.registry import flow_registry

from .models import PlanningInput, PlanningOutput

logger = logging.getLogger(__name__)


class StructuredPlannerComponent(AgentComponent):
    """Generates complete structured execution plans in a single LLM call.

    This component implements the Plan-and-Execute pattern optimized for local LLMs:
    - Replaces TaskGenerator + TaskThinker + TaskDecomposer
    - Single LLM call instead of 3 separate calls
    - Generates complete plan with multiple steps
    - Reduces bias cascade from multi-phase planning
    """

    def __init__(self, name: str = "structured_planner"):
        """Initialize structured planner component.

        Args:
            name: Component name
        """
        super().__init__(name)

    async def _initialize_impl(self) -> None:
        """Initialize the structured planner."""
        # Verify flows exist in registry
        execution_flow = flow_registry.get_flow("execution-planning")
        if not execution_flow:
            raise RuntimeError("ExecutionPlanningFlow not found in registry")

        clarification_flow = flow_registry.get_flow("clarification-planning")
        if not clarification_flow:
            raise RuntimeError("ClarificationPlanningFlow not found in registry")

        logger.info("StructuredPlanner initialized with execution and clarification flows")

    async def _shutdown_impl(self) -> None:
        """Shutdown the structured planner."""
        logger.info("StructuredPlanner shutdown")

    async def create_plan(
        self, context: ExecutionContext, validation_result=None
    ) -> PlanningOutput:
        """Create a structured execution plan from user message.

        Args:
            context: Unified execution context containing all necessary information
            validation_result: Optional context validation result (from ContextValidatorComponent)

        Returns:
            PlanningOutput with complete structured plan
        """
        self._check_initialized()

        start_time = time.time()

        try:
            # Extract information from unified context
            user_message = context.session.current_message
            conversation_history = context.session.conversation_history or []
            # Get available tools for agent's allowed categories
            allowed_categories = context.session.allowed_tool_categories or []
            available_tools = self._get_available_tools_for_categories(
                allowed_categories
            )
            working_directory = context.session.working_directory

            # FIX: Extract domain state from execution context
            # Get shared context (workspace, session state) and domain_state (tool-specific state)
            domain_state: dict = {}
            shared_variables: dict = {}

            # Discover workspace artifacts if component is available
            if self._registry:
                from flowlib.agent.components.workspace import WorkspaceDiscoveryComponent

                workspace_discovery = self._registry.get("workspace_discovery")

                if (
                    workspace_discovery
                    and isinstance(workspace_discovery, WorkspaceDiscoveryComponent)
                    and workspace_discovery.initialized
                    and working_directory
                ):
                    try:
                        # Scan workspace to get lightweight manifest of available artifacts
                        manifest = await workspace_discovery.scan_workspace(working_directory)

                        # Add manifest to domain state if any artifacts discovered
                        if manifest.domains:
                            domain_state["workspace_manifest"] = {
                                "working_directory": manifest.working_directory,
                                "discovered_artifacts": {
                                    domain: [
                                        {
                                            "name": artifact.name,
                                            "path": artifact.path,
                                            "type": artifact.artifact_type,
                                            "metadata": artifact.metadata,
                                        }
                                        for artifact in artifacts
                                    ]
                                    for domain, artifacts in manifest.domains.items()
                                },
                                "scan_timestamp": manifest.scan_timestamp,
                            }
                            logger.info(
                                f"Workspace scan found {len(manifest.domains)} domain(s) with "
                                f"{sum(len(arts) for arts in manifest.domains.values())} total artifact(s)"
                            )
                        else:
                            logger.debug("Workspace scan found no artifacts")
                    except Exception as e:
                        # Don't fail planning if workspace scan fails - just log warning
                        logger.warning(f"Workspace scan failed (continuing without manifest): {e}")

            # Session-level shared context
            if context.session.shared_context:
                domain_state.update(context.session.shared_context)

            # Task-level execution results
            if context.task.execution_results:
                domain_state["previous_execution_results"] = [
                    result.model_dump() if hasattr(result, "model_dump") else result
                    for result in context.task.execution_results
                ]

            # Use enriched task context if validation provided one (e.g., after clarification delegation)
            # This ensures planner sees the full context including delegation/clarification responses
            effective_user_message = user_message
            if validation_result and hasattr(validation_result, "enriched_task_context"):
                if validation_result.enriched_task_context:
                    effective_user_message = validation_result.enriched_task_context
                    logger.info(
                        f"Using enriched task context from validation: {effective_user_message[:100]}..."
                    )

            # Create input for planning flow
            planning_input = PlanningInput(
                user_message=effective_user_message,
                conversation_history=conversation_history,
                available_tools=available_tools,
                agent_persona=context.session.agent_persona,
                working_directory=working_directory,
                domain_state=domain_state,  # FIX: Pass domain state
                shared_variables=shared_variables,
                validation_result=validation_result,  # Pass validation result to planner
            )

            # Route to appropriate planning flow based on validation result
            if validation_result and validation_result.next_action == "clarify":
                # Use clarification planning - simple conversation-based plan
                logger.info("Using clarification planning (context insufficient)")
                planning_flow = flow_registry.get_flow("clarification-planning")
                if planning_flow is None:
                    raise RuntimeError("ClarificationPlanningFlow not found in registry")
            else:
                # Use execution planning - complex multi-step plan with parameter extraction
                logger.info("Using execution planning (context sufficient)")
                planning_flow = flow_registry.get_flow("execution-planning")
                if planning_flow is None:
                    raise RuntimeError("ExecutionPlanningFlow not found in registry")

            result = await planning_flow.run_pipeline(planning_input)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            logger.info(
                f"Generated {result.plan.message_type} plan with {len(result.plan.steps)} steps "
                f"in {processing_time:.2f}ms"
            )

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Structured planning failed: {e}")

            # Fail fast - no fallbacks allowed in flowlib
            raise ExecutionError(f"Structured planning failed: {str(e)}") from e

    def _get_available_tools_for_categories(
        self, allowed_categories: list[str]
    ) -> list[str]:
        """Get tools available for the agent based on allowed categories."""
        from flowlib.agent.components.task.execution.tool_access_manager import tool_access_manager

        return tool_access_manager.get_allowed_tools(allowed_categories)
