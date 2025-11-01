"""Classification-based planning component for Plan-and-Execute architecture.

This component implements the classification-based planning pattern:
1. Classify request type (conversation/single_tool/multi_step)
2. Route to specialized planner with type-specific schema
3. Prevents contradictions through mutually exclusive schemas
"""

import logging
import time

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.context.models import ExecutionContext
from flowlib.agent.core.errors import ExecutionError
from flowlib.flows.registry import flow_registry

from .models import PlanningInput, PlanningOutput

logger = logging.getLogger(__name__)


class ClassificationBasedPlannerComponent(AgentComponent):
    """Generates structured execution plans using classification-based routing.

    This component implements a two-stage planning approach:
    - Stage 1: Classify the request (conversation/single_tool/multi_step)
    - Stage 2: Route to specialized planner for that type

    Benefits:
    - Prevents contradictions (can't claim single_tool with multiple steps)
    - Specialized prompts and schemas for each type
    - Clearer separation of concerns
    - Better structured outputs
    """

    def __init__(self, name: str = "classification_based_planner"):
        """Initialize classification-based planner component.

        Args:
            name: Component name
        """
        super().__init__(name)

    async def _initialize_impl(self) -> None:
        """Initialize the classification-based planner."""
        # Verify flows exist in registry
        classification_flow = flow_registry.get_flow("classification-based-planning")
        if not classification_flow:
            raise RuntimeError("ClassificationBasedPlanningFlow not found in registry")

        clarification_flow = flow_registry.get_flow("clarification-planning")
        if not clarification_flow:
            raise RuntimeError("ClarificationPlanningFlow not found in registry")

        logger.info("ClassificationBasedPlanner initialized")

    async def _shutdown_impl(self) -> None:
        """Shutdown the classification-based planner."""
        logger.info("ClassificationBasedPlanner shutdown")

    async def create_plan(
        self, context: ExecutionContext, validation_result=None
    ) -> PlanningOutput:
        """Create a structured execution plan using classification-based approach.

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
            conversation_history = [
                {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
                for msg in context.session.conversation_history
            ]
            # Get available tools for agent role
            available_tools = self._get_available_tools_for_role(context.session.agent_role)
            agent_role = context.session.agent_role or "assistant"
            working_directory = context.session.working_directory

            # Extract domain state from execution context
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

            # Use enriched task context if validation provided one
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
                agent_role=agent_role,
                working_directory=working_directory,
                domain_state=domain_state,
                shared_variables=shared_variables,
                validation_result=validation_result,
            )

            # Route to appropriate planning flow based on validation result
            if validation_result and validation_result.next_action == "clarify":
                # Use clarification planning - simple conversation-based plan
                logger.info("Using clarification planning (context insufficient)")
                planning_flow = flow_registry.get_flow("clarification-planning")
                if planning_flow is None:
                    raise RuntimeError("ClarificationPlanningFlow not found in registry")
            else:
                # Use classification-based planning - classify then route to specialized planner
                logger.info("Using classification-based planning (context sufficient)")
                planning_flow = flow_registry.get_flow("classification-based-planning")
                if planning_flow is None:
                    raise RuntimeError("ClassificationBasedPlanningFlow not found in registry")

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
            logger.error(f"Classification-based planning failed: {e}")

            # Fail fast - no fallbacks allowed in flowlib
            raise ExecutionError(f"Classification-based planning failed: {str(e)}") from e

    def _get_available_tools_for_role(self, agent_role: str) -> list[str]:
        """Get tools available for the agent's role.

        Args:
            agent_role: Agent role string

        Returns:
            List of tool names available to this role
        """
        # Import here to avoid circular imports
        from flowlib.agent.components.task.execution.tool_role_manager import (
            tool_role_manager,
        )

        return tool_role_manager.get_allowed_tools(agent_role)
