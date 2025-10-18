"""Agent execution engine with Plan-Execute-Evaluate architecture."""

import logging
import time
from typing import Any, Optional

from flowlib.agent.components.knowledge.component import KnowledgeComponent
from flowlib.agent.components.memory.component import MemoryComponent
from flowlib.agent.components.task.evaluation.component import (
    CompletionEvaluatorComponent,
)
from flowlib.agent.components.task.execution.models import ToolStatus
from flowlib.agent.components.task.planning.component import StructuredPlannerComponent
from flowlib.agent.components.task.reflection.component import (
    ExecutionReflectorComponent,
)
from flowlib.agent.core.activity_stream import ActivityStream
from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.context.manager import AgentContextManager
from flowlib.agent.core.errors import ExecutionError
from flowlib.agent.models.state import AgentState, ExecutionResult

logger = logging.getLogger(__name__)


class EngineComponent(AgentComponent):
    """Engine that orchestrates Plan → Execute → Reflect → Evaluate loop.

    This implements the Plan-Execute-Reflect pattern optimized for local LLMs:
    - Single planning call generates complete structured plan
    - Execute all steps in the plan
    - Reflect on failures to decide if re-planning is needed
    - Evaluate completion for successful executions
    - Enables adaptive re-planning when execution fails
    """

    def __init__(self, agent_name: str, agent_persona: str, max_iterations: int = 10,
                 memory: Optional[MemoryComponent] = None,
                 knowledge: Optional[KnowledgeComponent] = None,
                 planner: Optional[StructuredPlannerComponent] = None,
                 evaluator: Optional[CompletionEvaluatorComponent] = None,
                 reflector: Optional[ExecutionReflectorComponent] = None,
                 activity_stream: Optional[ActivityStream] = None,
                 context_manager: Optional[AgentContextManager] = None,
                 name: str = "agent_engine"):
        """Initialize the agent engine with Plan-Execute-Reflect-Evaluate architecture.

        Args:
            agent_name: Name of the agent
            agent_persona: Agent's persona description
            max_iterations: Maximum plan-execute-evaluate cycles
            memory: Memory component
            knowledge: Knowledge component
            planner: Structured planner component
            evaluator: Completion evaluator component
            reflector: Execution reflector component
            activity_stream: Activity stream
            context_manager: Context manager (required)
            name: Component name
        """
        super().__init__(name)

        # Context manager is required - fail fast if missing
        if not context_manager:
            raise ExecutionError("EngineComponent requires AgentContextManager - fix configuration")

        self._agent_name = agent_name
        self._agent_persona = agent_persona
        self._max_iterations = max_iterations
        self._memory = memory
        self._knowledge = knowledge
        self._planner = planner
        self._evaluator = evaluator
        self._reflector = reflector
        self._activity_stream = activity_stream
        self._context_manager = context_manager
        self._iteration = 0
        self._consecutive_failures = 0  # Track consecutive failures for escalation
        self._max_retries_before_escalation = 3  # Escalate after 3 consecutive failures

        # Persistent shared data across all execution cycles
        # This ensures session state persists between Plan-Execute-Evaluate cycles
        self._session_shared_data: Optional[Any] = None

    async def _initialize_impl(self) -> None:
        """Initialize the engine components."""
        if not self._memory:
            raise ExecutionError("Memory component is required")
        if not self._planner:
            raise ExecutionError("Structured planner component is required")
        if not self._evaluator:
            raise ExecutionError("Completion evaluator component is required")
        if not self._reflector:
            raise ExecutionError("Execution reflector component is required")
        # Knowledge component is optional

        logger.info("Engine initialized with Plan-Execute-Reflect-Evaluate architecture")

    async def _shutdown_impl(self) -> None:
        """Shutdown the engine."""
        logger.info("Engine shutdown")

    async def execute(self, task_description: str, conversation_history: Optional[Any] = None) -> ExecutionResult:
        """Execute a task by orchestrating Plan-Execute-Evaluate cycles."""
        self._check_initialized()

        # Create agent state for this task
        state = AgentState(
            task_id=f"task_{int(time.time())}",
            task_description=task_description
        )

        try:
            # Execute cycles until completion or max iterations
            memory_context = "agent_execution"
            continue_execution = True

            while continue_execution and state.cycles < self._max_iterations:
                continue_execution = await self.execute_cycle(
                    state=state,
                    conversation_history=conversation_history,
                    memory_context=memory_context
                )

            # Check if max iterations reached without completion
            if not state.is_complete and state.cycles >= self._max_iterations:
                state.set_complete("Maximum iterations reached")

            # Extract output from executed flows
            output = state.completion_reason

            # Convert ExecutionHistoryEntry to ExecutionHistoryStep
            from flowlib.agent.models.state import ExecutionHistoryStep
            execution_steps = []
            for entry in state.execution_history:
                step = ExecutionHistoryStep(
                    step_id=f"step_{entry.cycle}_{entry.flow_name}",
                    step_type="flow_execution",
                    flow_name=entry.flow_name,
                    started_at=entry.timestamp,
                    completed_at=entry.timestamp,
                    success=entry.success,
                    error_message="" if entry.success else f"Flow {entry.flow_name} failed"
                )
                execution_steps.append(step)

            # Prepare result as Pydantic model
            result = ExecutionResult(
                output=output,
                task_id=state.task_id,
                cycles=state.cycles,
                progress=state.progress,
                is_complete=state.is_complete,
                errors=state.errors,
                execution_history=execution_steps,
                stats={
                    "cycles_executed": state.cycles,
                    "flows_executed": len(state.execution_history),
                    "errors_encountered": len(state.errors)
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error during task execution: {e}", exc_info=True)
            raise

    async def execute_cycle(self, state: AgentState, conversation_history: Optional[Any] = None, memory_context: str = "agent", no_flow_is_error: bool = False) -> bool:
        """Execute a single Plan-Execute-Reflect-Evaluate cycle.

        Implements the Plan-Execute-Reflect pattern:
        1. PLAN: Generate complete structured plan (1 LLM call)
        2. EXECUTE: Run all steps in the plan (no LLM calls)
        3. REFLECT: If failures occurred, analyze and decide next action (1 LLM call)
        4. EVALUATE: If no failures, check if task is complete (1 LLM call)

        Returns:
            True if more cycles needed, False if task complete
        """
        try:
            # Update state
            self._iteration += 1
            state.increment_cycle()
            await self._context_manager.increment_cycle()

            logger.info(f"Starting Plan-Execute-Evaluate cycle {state.cycles}")

            if self._activity_stream:
                self._activity_stream.execution(
                    f"Starting cycle {state.cycles} for task: {state.task_description[:50]}...",
                    cycle=state.cycles
                )

            # Start task in context manager
            await self._context_manager.start_task(state.task_description)

            # ===================================================================
            # STEP 1: PLAN - Generate structured execution plan (1 LLM call)
            # ===================================================================
            planning_context = await self._context_manager.create_execution_context(
                component_type="task_planning"
            )

            if not self._planner:
                raise ExecutionError("Structured planner is required but not initialized")

            planning_result = await self._planner.create_plan(context=planning_context)

            await self._context_manager.update_from_execution(
                component_type="task_planning",
                execution_result=planning_result,
                success=planning_result.success
            )

            if not planning_result.success:
                logger.error("Planning failed")
                state.set_complete("Planning failed")
                return False

            plan = planning_result.plan
            logger.info(
                f"Generated {plan.message_type} plan with {len(plan.steps)} steps: {plan.reasoning}"
            )

            # ===================================================================
            # STEP 2: EXECUTE - Run all steps in the plan (no re-planning)
            # ===================================================================
            executed_steps = []

            for step in plan.steps:
                # Check if this step depends on another step
                if step.depends_on_step is not None:
                    # Validate dependency index
                    if step.depends_on_step < 0 or step.depends_on_step >= len(executed_steps):
                        logger.warning(f"Step {step.step_id} has invalid dependency: {step.depends_on_step}")
                        step.result = f"Skipped: Invalid dependency index {step.depends_on_step}"
                        executed_steps.append({
                            "tool_name": step.tool_name,
                            "step_description": step.step_description,
                            "result": step.result,
                            "success": False
                        })
                        continue

                    # Check if dependency succeeded
                    dependency_step = executed_steps[step.depends_on_step]
                    if not dependency_step.get("success", False):
                        logger.info(f"Skipping step {step.step_id} because dependency step {step.depends_on_step} failed")
                        step.result = f"Skipped: Dependency step {step.depends_on_step} ({dependency_step['step_description']}) failed"
                        executed_steps.append({
                            "tool_name": step.tool_name,
                            "step_description": step.step_description,
                            "result": step.result,
                            "success": False
                        })
                        continue

                logger.info(f"Executing step: {step.step_description}")

                # Execute the tool for this step
                try:
                    # Create execution context for this step
                    step_context = await self._context_manager.create_execution_context(
                        component_type="task_execution"
                    )

                    # Convert plan step to TodoItem format expected by tools
                    from flowlib.agent.components.task.core.todo import (
                        TodoItem,
                        TodoPriority,
                    )
                    from flowlib.agent.components.task.execution.models import (
                        ToolExecutionContext,
                    )

                    # Create TodoItem from plan step
                    todo = TodoItem(
                        content=step.step_description,
                        assigned_tool=step.tool_name,
                        priority=TodoPriority.MEDIUM,
                        execution_context=step.parameters  # Store parameters in execution_context
                    )

                    # Get conversation history from context manager and convert to dicts
                    conversation_history = [
                        msg.model_dump() if hasattr(msg, 'model_dump') else dict(msg)
                        for msg in (step_context.session.conversation_history or [])
                    ]

                    # Create tool execution context
                    from flowlib.agent.components.task.execution.models import ToolExecutionSharedData

                    # Reuse existing session shared data or create new one
                    # This ensures state persists across all steps in all Plan-Execute cycles
                    if self._session_shared_data is None:
                        self._session_shared_data = ToolExecutionSharedData()
                        logger.debug(f"ENGINE: Created NEW session shared_data: {id(self._session_shared_data)}")

                    tool_context = ToolExecutionContext(
                        working_directory=step_context.session.working_directory,
                        agent_id=step_context.session.agent_name,
                        agent_persona=step_context.session.agent_persona,
                        agent_role=step_context.session.agent_role,
                        session_id=step_context.session.session_id,
                        execution_id=f"exec_{state.task_id}_{state.cycles}_{step.step_id}",
                        original_user_message=step_context.session.current_message,  # FIX: Populate from session context
                        conversation_history=conversation_history,
                        shared_data=self._session_shared_data  # Reuse persistent shared data
                    )

                    # Get tool factory from registry and create tool instance
                    from flowlib.agent.components.task.execution.registry import (
                        tool_registry,
                    )
                    tool_factory = tool_registry.get(step.tool_name)
                    if not tool_factory:
                        raise ExecutionError(f"Tool '{step.tool_name}' not found in registry")

                    # Create tool instance from factory
                    tool_instance = tool_factory.create_tool(tool_context)

                    # Execute the tool with proper signature
                    tool_result = await tool_instance.execute(todo, tool_context)

                    # Mark step as executed
                    step.executed = True
                    # Use protocol method - all ToolResult subclasses implement get_display_content()
                    step.result = tool_result.get_display_content()

                    # Check success status using enum comparison
                    is_success = tool_result.status == ToolStatus.SUCCESS if hasattr(tool_result, 'status') else True

                    executed_steps.append({
                        "tool_name": step.tool_name,
                        "step_description": step.step_description,
                        "result": step.result,
                        "success": is_success
                    })

                    logger.info(f"Step completed: {step.step_description}")

                except Exception as e:
                    logger.error(f"Step execution failed: {e}")
                    step.result = f"Error: {str(e)}"
                    executed_steps.append({
                        "tool_name": step.tool_name,
                        "step_description": step.step_description,
                        "result": f"Error: {str(e)}",
                        "success": False
                    })
                    # Continue executing other steps even if one fails

            # ===================================================================
            # SYNC SESSION STATE: Flow domain_state back to session shared_context
            # ===================================================================
            # After tool execution, sync tool-level domain_state back to session-level
            # shared_context so it's available for next planning cycle
            if self._session_shared_data and self._session_shared_data.domain_state:
                logger.debug(f"Syncing domain_state to session: {list(self._session_shared_data.domain_state.keys())}")
                # Access session context directly and update shared_context
                if self._context_manager._session_context:
                    self._context_manager._session_context.shared_context.update(
                        self._session_shared_data.domain_state
                    )
                    logger.debug("✅ Synced domain_state to session.shared_context")

            # Extract knowledge if component is available (only exists if enable_learning=True)
            if self._knowledge and executed_steps:
                try:
                    content = self._format_steps_for_learning(executed_steps)
                    if content:
                        learning_result = await self._knowledge.learn_from_content(
                            content=content,
                            context=f"Task execution cycle {state.cycles}: {state.task_description}",
                            domain_hint="task_execution"
                        )
                        if learning_result.success:
                            logger.debug(f"Extracted {learning_result.knowledge.total_items} knowledge items")
                except Exception as e:
                    logger.warning(f"Knowledge extraction failed, continuing: {e}")

            # ===================================================================
            # STEP 3: REFLECT - Analyze execution results and decide next action
            # ===================================================================
            # Check if any steps failed or if we have partial completion
            failed_steps = [s for s in executed_steps if not s.get("success", False)]
            has_failures = len(failed_steps) > 0
            partial_completion = len(executed_steps) > 0 and has_failures

            # Only reflect if there are failures or partial completion
            if has_failures:
                self._consecutive_failures += 1
                logger.info(f"Reflecting on execution: {len(failed_steps)} failed steps (consecutive failures: {self._consecutive_failures})")

                # Check if we should escalate to user after repeated failures
                if self._consecutive_failures >= self._max_retries_before_escalation:
                    escalation_message = await self._escalate_to_user(state, failed_steps, state.cycles)
                    await self._context_manager.add_assistant_response(escalation_message)
                    state.set_complete(escalation_message)
                    return False

                if not self._reflector:
                    raise ExecutionError("Execution reflector is required but not initialized")

                # Convert plan steps to dict format for reflection
                plan_steps_dict = [
                    {
                        "tool_name": step.tool_name,
                        "step_description": step.step_description
                    }
                    for step in plan.steps
                ]

                reflection_result = await self._reflector.reflect_on_execution(
                    original_goal=state.task_description,
                    plan_steps=plan_steps_dict,
                    execution_results=executed_steps,
                    partial_completion=partial_completion
                )

                if not reflection_result.success:
                    logger.error("Reflection failed")
                    state.set_complete("Reflection failed")
                    return False

                reflection = reflection_result.result
                logger.info(
                    f"Reflection result: {reflection.next_action} - {reflection.reasoning}"
                )

                # Act based on reflection decision
                if reflection.next_action == "replan":
                    # Re-planning needed - prepare guidance for next cycle
                    guidance = reflection.replanning_guidance or "Adjust approach based on failures"
                    logger.info(f"Re-planning: {guidance}")
                    # Update task description with replanning guidance
                    updated_description = f"{state.task_description}\n\nPrevious attempt failed. {guidance}"
                    state.set_task_description(updated_description)
                    return True  # Continue to next cycle with new plan

                elif reflection.next_action == "clarify":
                    # Need user input
                    clarification = reflection.clarification_question or "Need more information"
                    await self._context_manager.add_assistant_response(clarification)
                    state.set_complete(clarification)
                    logger.info(f"Awaiting user clarification: {clarification}")
                    return False

                # If reflection says "continue", proceed to evaluation
                logger.info("Reflection suggests continuing - proceeding to evaluation")
            else:
                # Reset consecutive failures on success
                self._consecutive_failures = 0

            # ===================================================================
            # STEP 4: EVALUATE - Check if task is complete (1 LLM call)
            # ===================================================================
            if not self._evaluator:
                raise ExecutionError("Completion evaluator is required but not initialized")

            evaluation_result = await self._evaluator.evaluate_completion(
                original_goal=state.task_description,
                plan_reasoning=plan.reasoning,
                executed_steps=executed_steps,
                expected_outcome=plan.expected_outcome
            )

            if not evaluation_result.success:
                logger.error("Evaluation failed")
                state.set_complete("Evaluation failed")
                return False

            evaluation = evaluation_result.result
            logger.info(
                f"Evaluation: {evaluation.next_action} "
                f"(confidence: {evaluation.completion_confidence:.2f}) - {evaluation.reasoning}"
            )

            # Act based on evaluation decision
            if evaluation.next_action == "done":
                # Task is complete - extract final response from executed steps
                logger.info(f"Extracting final response from {len(executed_steps)} steps")
                logger.debug(f"Executed steps structure: {executed_steps}")
                final_response = self._extract_final_response(executed_steps)
                logger.info(f"Final response extracted: '{final_response[:200]}'")
                await self._context_manager.add_assistant_response(final_response)

                state.set_progress(100)
                state.set_complete(final_response)
                logger.info(f"Task completed: {evaluation.reasoning}")
                return False

            elif evaluation.next_action == "clarify":
                # Need user input
                clarification = evaluation.clarification_question or "Need more information"
                await self._context_manager.add_assistant_response(clarification)

                state.set_complete(clarification)
                logger.info(f"Awaiting user clarification: {clarification}")
                return False

            else:  # "continue"
                # Need more cycles - keep task description the same
                state.set_progress(50)  # Partial progress
                logger.info(f"Continuing execution: {evaluation.reasoning}")
                return True

        except Exception as e:
            error_message = f"Error during execution cycle {state.cycles}: {str(e)}"
            state.add_error(error_message)
            state.set_complete(f"Task failed: {str(e)}")
            logger.error(error_message, exc_info=True)

            # Update context manager with cycle failure
            await self._context_manager.update_from_execution(
                component_type="execution_cycle",
                execution_result=str(e),
                success=False
            )

            return False

    def _extract_final_response(self, executed_steps: list[dict]) -> str:
        """Extract final user-facing response from executed steps."""
        if not executed_steps:
            return "Task completed with no results"

        # For conversation-type tasks, return the conversation response
        for step in executed_steps:
            if step.get("tool_name") == "conversation" and step.get("success"):
                result = step.get("result", "Task completed")
                logger.info(f"Extracted conversation response: {result[:100]}...")
                return result

        # For other tasks, summarize results
        successful_steps = [s for s in executed_steps if s.get("success")]
        if successful_steps:
            # Return the last successful result
            return successful_steps[-1].get("result", "Task completed successfully")

        return "Task completed"

    async def _escalate_to_user(
        self,
        state: AgentState,
        failed_steps: list[dict],
        cycle: int
    ) -> str:
        """Escalate to user after repeated failures.

        Args:
            state: Current agent state
            failed_steps: Steps that failed
            cycle: Current cycle number

        Returns:
            Escalation message for the user
        """
        # Format failure summary
        failure_summary = "\n".join([
            f"  • {step.get('tool_name')}: {step.get('result')}"
            for step in failed_steps
        ])

        escalation_message = f"""I've attempted this task {cycle} times but encountered persistent issues:

{failure_summary}

I need your help to proceed. Could you please:
1. Check if the required tools are working correctly
2. Provide any missing information or configuration
3. Clarify the task requirements if needed

What would you like me to do?"""

        logger.info(f"Escalated to user after {cycle} failed cycles")
        return escalation_message

    def _format_steps_for_learning(self, executed_steps: list[dict]) -> str:
        """Format executed steps for knowledge extraction."""
        if not executed_steps:
            return ""

        content_parts = []
        for idx, step in enumerate(executed_steps, 1):
            tool = step.get("tool_name", "unknown")
            description = step.get("step_description", "")
            result = step.get("result", "")
            success = step.get("success", False)
            status = "SUCCESS" if success else "FAILED"

            content_parts.append(
                f"Step {idx} ({tool}): {description}\n"
                f"Status: {status}\n"
                f"Result: {result}"
            )

        return "\n\n".join(content_parts)
