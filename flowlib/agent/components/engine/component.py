"""Agent execution engine with Plan-Execute-Evaluate architecture."""

import logging
import time
from typing import Any

from flowlib.agent.components.knowledge.component import KnowledgeComponent
from flowlib.agent.components.memory.component import MemoryComponent
from flowlib.agent.components.task.evaluation.component import (
    CompletionEvaluatorComponent,
)
from flowlib.agent.components.task.execution.models import ToolStatus
from flowlib.agent.components.task.planning.classification_component import (
    ClassificationBasedPlannerComponent,
)
from flowlib.agent.components.task.reflection.component import (
    ExecutionReflectorComponent,
)
from flowlib.agent.components.task.validation.component import (
    ContextValidatorComponent,
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

    def __init__(
        self,
        agent_name: str,
        agent_persona: str,
        max_iterations: int = 10,
        memory: MemoryComponent | None = None,
        knowledge: KnowledgeComponent | None = None,
        planner: ClassificationBasedPlannerComponent | None = None,
        evaluator: CompletionEvaluatorComponent | None = None,
        reflector: ExecutionReflectorComponent | None = None,
        validator: ContextValidatorComponent | None = None,
        activity_stream: ActivityStream | None = None,
        context_manager: AgentContextManager | None = None,
        name: str = "agent_engine",
    ):
        """Initialize the agent engine with Validate-Plan-Execute-Reflect-Evaluate architecture.

        Args:
            agent_name: Name of the agent
            agent_persona: Agent's persona description
            max_iterations: Maximum plan-execute-evaluate cycles
            memory: Memory component
            knowledge: Knowledge component
            planner: Classification-based planner component
            evaluator: Completion evaluator component
            reflector: Execution reflector component
            validator: Context validator component (required for proactive information gathering)
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
        self._validator = validator  # Context validator for proactive information gathering
        self._activity_stream = activity_stream
        self._context_manager = context_manager
        self._iteration = 0
        self._consecutive_failures = 0  # Track consecutive failures for escalation
        self._max_retries_before_escalation = 3  # Escalate after 3 consecutive failures

        # Persistent shared data across all execution cycles
        # This ensures session state persists between Plan-Execute-Evaluate cycles
        self._session_shared_data: Any | None = None

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
        if not self._validator:
            raise ExecutionError("Context validator component is required")
        # Knowledge component is optional

        logger.info("Engine initialized with Validate-Plan-Execute-Reflect-Evaluate architecture")

    async def _shutdown_impl(self) -> None:
        """Shutdown the engine."""
        logger.info("Engine shutdown")

    async def execute(
        self, task_description: str, conversation_history: Any | None = None
    ) -> ExecutionResult:
        """Execute a task by orchestrating Plan-Execute-Evaluate cycles."""
        self._check_initialized()

        # Create agent state for this task
        state = AgentState(task_id=f"task_{int(time.time())}", task_description=task_description)

        try:
            # Execute cycles until completion or max iterations
            memory_context = "agent_execution"
            continue_execution = True

            while continue_execution and state.cycles < self._max_iterations:
                continue_execution = await self.execute_cycle(
                    state=state,
                    conversation_history=conversation_history,
                    memory_context=memory_context,
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
                    error_message="" if entry.success else f"Flow {entry.flow_name} failed",
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
                    "errors_encountered": len(state.errors),
                },
            )

            return result

        except Exception as e:
            logger.error(f"Error during task execution: {e}", exc_info=True)
            raise

    async def execute_cycle(
        self,
        state: AgentState,
        conversation_history: Any | None = None,
        memory_context: str = "agent",
        no_flow_is_error: bool = False,
    ) -> bool:
        """Execute a single message handling cycle with branched control flow.

        Control flow branches based on validation and plan type:

        BRANCH A - Clarification (validation says "clarify"):
            1. VALIDATE → clarify
            2. PLAN → create clarification plan
            3. EXECUTE → deliver questions
            4. DONE (no evaluation - questions delivered = complete)

        BRANCH B - Simple execution (conversation/single_tool):
            1. VALIDATE → proceed
            2. PLAN → conversation or single_tool
            3. EXECUTE → deliver response/run tool
            4. DONE (no evaluation - execution success = complete)

        BRANCH C - Complex execution (multi_step):
            1. VALIDATE → proceed
            2. PLAN → multi-step plan
            3. EXECUTE → run all steps
            4. REFLECT (if failures)
            5. EVALUATE (check completion, may need more cycles)

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
                    cycle=state.cycles,
                )

            # Check if this is a replanning cycle (context already validated)
            is_replanning = "Previous attempt failed" in state.task_description

            # Start task in context manager (only for fresh requests, not replanning)
            if not is_replanning:
                await self._context_manager.start_task(state.task_description)

            # ===================================================================
            # STEP 0: VALIDATE CONTEXT - Check information sufficiency
            # ===================================================================
            # Skip validation on replanning cycles - context was already validated
            if is_replanning:
                logger.info(
                    "Replanning cycle detected - skipping validation (context already validated)"
                )
                # Create a mock validation result indicating proceed
                from flowlib.agent.components.task.validation.models import (
                    ValidationResult,
                )

                validation_result = ValidationResult(
                    has_sufficient_context=True,
                    confidence=1.0,
                    reasoning="Replanning cycle - using previously validated context",
                    next_action="proceed",
                    missing_information=[],
                    clarification_questions=[],
                    validation_time_ms=0.0,
                    llm_calls_made=0,
                )
            else:
                logger.info("Running context validation before planning")

                # Get conversation history from context manager
                session_context = self._context_manager._session_context
                conversation_history = (
                    session_context.conversation_history if session_context else []
                )

                # Get domain state from shared context
                domain_state = session_context.shared_context if session_context else {}

                # Run validation
                assert self._validator is not None  # Checked in _initialize_impl
                validation_output = await self._validator.validate_context(
                    user_message=state.task_description,
                    conversation_history=[msg.model_dump() for msg in conversation_history],
                    domain_state=domain_state,
                )

                validation_result = validation_output.result

                logger.info(
                    f"Validation complete: {validation_result.next_action} "
                    f"(confidence: {validation_result.confidence:.2f})"
                )

                # If validation provided enriched context, use it for planning ONLY
                # Do NOT call start_task again - that would pollute conversation history
                if validation_result.enriched_task_context:
                    logger.info("Using enriched task context from clarification response")
                    # Update task description with enriched context for planning
                    state.set_task_description(validation_result.enriched_task_context)

            # ===================================================================
            # CONTROL FLOW DECISION: Route based on validation result
            # ===================================================================
            # Track if this is a clarification task (affects completion logic)
            is_clarification_task = validation_result.next_action == "clarify"

            # ===================================================================
            # STEP 1: PLAN - Generate structured execution plan (1 LLM call)
            # ===================================================================
            planning_context = await self._context_manager.create_execution_context(
                component_type="task_planning"
            )

            if not self._planner:
                raise ExecutionError("Structured planner is required but not initialized")

            # Pass validation result to planner
            planning_result = await self._planner.create_plan(
                context=planning_context, validation_result=validation_result
            )

            await self._context_manager.update_from_execution(
                component_type="task_planning",
                execution_result=planning_result,
                success=planning_result.success,
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
                        logger.warning(
                            f"Step {step.step_id} has invalid dependency: {step.depends_on_step}"
                        )
                        step.result = f"Skipped: Invalid dependency index {step.depends_on_step}"
                        executed_steps.append(
                            {
                                "tool_name": step.tool_name,
                                "step_description": step.step_description,
                                "result": step.result,
                                "success": False,
                            }
                        )
                        continue

                    # Check if dependency succeeded
                    dependency_step = executed_steps[step.depends_on_step]
                    if not dependency_step.get("success", False):
                        logger.info(
                            f"Skipping step {step.step_id} because dependency step {step.depends_on_step} failed"
                        )
                        step.result = f"Skipped: Dependency step {step.depends_on_step} ({dependency_step['step_description']}) failed"
                        executed_steps.append(
                            {
                                "tool_name": step.tool_name,
                                "step_description": step.step_description,
                                "result": step.result,
                                "success": False,
                            }
                        )
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
                        execution_context=step.parameters,  # Store parameters in execution_context
                    )

                    # Get conversation history from context manager and convert to dicts
                    conversation_history = [
                        msg.model_dump() if hasattr(msg, "model_dump") else dict(msg)
                        for msg in (step_context.session.conversation_history or [])
                    ]

                    # Create tool execution context
                    from flowlib.agent.components.task.execution.models import (
                        ToolExecutionSharedData,
                    )

                    # Reuse existing session shared data or create new one
                    # This ensures state persists across all steps in all Plan-Execute cycles
                    if self._session_shared_data is None:
                        self._session_shared_data = ToolExecutionSharedData()
                        logger.debug(
                            f"ENGINE: Created NEW session shared_data: {id(self._session_shared_data)}"
                        )

                    tool_context = ToolExecutionContext(
                        working_directory=step_context.session.working_directory,
                        agent_id=step_context.session.agent_name,
                        agent_persona=step_context.session.agent_persona,
                        allowed_tool_categories=step_context.session.allowed_tool_categories,
                        session_id=step_context.session.session_id,
                        execution_id=f"exec_{state.task_id}_{state.cycles}_{step.step_id}",
                        original_user_message=step_context.session.current_message,  # FIX: Populate from session context
                        conversation_history=conversation_history,
                        shared_data=self._session_shared_data,  # Reuse persistent shared data
                    )

                    # Get tool factory and metadata from registry
                    from flowlib.agent.components.task.execution.registry import (
                        tool_registry,
                    )

                    tool_factory = tool_registry.get(step.tool_name)
                    if not tool_factory:
                        raise ExecutionError(f"Tool '{step.tool_name}' not found in registry")

                    tool_metadata = tool_registry.get_metadata(step.tool_name)
                    if not tool_metadata:
                        raise ExecutionError(f"Tool metadata for '{step.tool_name}' not found")

                    # Generate parameters BEFORE calling tool (following flow @pipeline pattern)
                    parameter_type = tool_metadata.parameter_type

                    # Option 1: Use structured parameters from planner (execution_context)
                    if todo.execution_context and isinstance(todo.execution_context, dict):
                        try:
                            parameters = parameter_type(**todo.execution_context)
                            logger.debug(
                                f"✅ Using planner-provided parameters for {step.tool_name}: {todo.execution_context}"
                            )
                        except Exception as e:
                            raise ExecutionError(
                                f"Invalid execution_context for {step.tool_name}: {e}. "
                                f"Expected {parameter_type.__name__} fields, got {todo.execution_context}"
) from e
                    # Option 2: Fall back to LLM parameter extraction from natural language
                    else:
                        logger.debug(
                            f"ℹ️  No execution_context for {step.tool_name}, using LLM extraction from: '{todo.content}'"
                        )
                        from flowlib.flows.registry.registry import flow_registry

                        param_generation_flow = flow_registry.get(
                            "writing-tool-parameter-generation"
                        )
                        if not param_generation_flow:
                            raise ExecutionError("writing-tool-parameter-generation flow not found")

                        # Get current book from tool context if available
                        current_book = None
                        if (
                            hasattr(tool_context.shared_data, "variables")
                            and "current_book" in tool_context.shared_data.variables
                        ):
                            current_book = tool_context.shared_data.variables["current_book"]

                        from tools.writing.parameter_generation import (  # type: ignore[import-not-found]
                            WritingToolParameterGenerationInput,
                        )

                        param_result = await param_generation_flow.run_pipeline(
                            WritingToolParameterGenerationInput(
                                task_content=todo.content,
                                working_directory=tool_context.working_directory,
                                tool_name=step.tool_name,
                                tool_description=tool_metadata.description,
                                parameter_type=parameter_type,
                                shared_variables=tool_context.shared_data.variables,
                                current_book=current_book,
                                conversation_history=tool_context.conversation_history,
                                original_user_message=tool_context.original_user_message,
                            )
                        )
                        parameters = param_result.parameters

                    # Create tool instance from factory
                    tool_instance = tool_factory.create_tool(tool_context)

                    # Execute the tool with TodoItem, validated parameters, and context
                    tool_result = await tool_instance.execute(todo, parameters, tool_context)

                    # Mark step as executed
                    step.executed = True
                    # Use protocol method - all ToolResult subclasses implement get_display_content()
                    step.result = tool_result.get_display_content()

                    # Check success status using enum comparison
                    is_success = (
                        tool_result.status == ToolStatus.SUCCESS
                        if hasattr(tool_result, "status")
                        else True
                    )

                    executed_steps.append(
                        {
                            "tool_name": step.tool_name,
                            "step_description": step.step_description,
                            "result": step.result,
                            "success": is_success,
                        }
                    )

                    logger.info(f"Step completed: {step.step_description}")

                except Exception as e:
                    logger.error(f"Step execution failed: {e}")
                    step.result = f"Error: {str(e)}"
                    executed_steps.append(
                        {
                            "tool_name": step.tool_name,
                            "step_description": step.step_description,
                            "result": f"Error: {str(e)}",
                            "success": False,
                        }
                    )
                    # Continue executing other steps even if one fails

            # ===================================================================
            # SYNC SESSION STATE: Flow domain_state back to session shared_context
            # ===================================================================
            # After tool execution, sync tool-level domain_state back to session-level
            # shared_context so it's available for next planning cycle
            if self._session_shared_data and self._session_shared_data.domain_state:
                logger.debug(
                    f"Syncing domain_state to session: {list(self._session_shared_data.domain_state.keys())}"
                )
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
                            domain_hint="task_execution",
                        )
                        if learning_result.success:
                            logger.debug(
                                f"Extracted {learning_result.knowledge.total_items} knowledge items"
                            )
                except Exception as e:
                    logger.warning(f"Knowledge extraction failed, continuing: {e}")

            # ===================================================================
            # CONTROL FLOW BRANCH: Skip evaluation for single-call tasks
            # ===================================================================
            # BRANCH A: Clarification tasks - complete immediately after delivery
            if is_clarification_task:
                logger.info("Clarification task: questions delivered, task complete")
                # Extract the delivered message
                final_response = self._extract_final_response(executed_steps)
                await self._context_manager.add_assistant_response(final_response)
                state.set_progress(100)
                state.set_complete(final_response)
                return False  # Task complete, no more cycles

            # BRANCH B: Single-call tasks (conversation/single_tool) - complete after execution
            # Note: single_tool can handle complex workflows (e.g., create_complete_book)
            # but they don't need evaluation because they're self-contained
            if plan.message_type in ["conversation", "single_tool"]:
                logger.info(f"{plan.message_type} task: execution complete")
                # Check if execution succeeded
                all_success = all(s.get("success", False) for s in executed_steps)
                if all_success:
                    final_response = self._extract_final_response(executed_steps)
                    await self._context_manager.add_assistant_response(final_response)
                    state.set_progress(100)
                    state.set_complete(final_response)
                    return False  # Task complete, no more cycles
                else:
                    # Single-call task failed - report to user
                    error_msg = f"Task failed: {executed_steps[0].get('result', 'Unknown error')}"
                    await self._context_manager.add_assistant_response(error_msg)
                    state.set_complete(error_msg)
                    return False  # Stop, don't retry single-call tasks

            # ===================================================================
            # BRANCH C: Multi-step tasks - need reflection and evaluation
            # ===================================================================
            logger.info("Complex multi-step task: proceeding to reflection and evaluation")

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
                logger.info(
                    f"Reflecting on execution: {len(failed_steps)} failed steps (consecutive failures: {self._consecutive_failures})"
                )

                # Check if we should escalate to user after repeated failures
                if self._consecutive_failures >= self._max_retries_before_escalation:
                    escalation_message = await self._escalate_to_user(
                        state, failed_steps, state.cycles
                    )
                    await self._context_manager.add_assistant_response(escalation_message)
                    state.set_complete(escalation_message)
                    return False

                if not self._reflector:
                    raise ExecutionError("Execution reflector is required but not initialized")

                # Convert plan steps to dict format for reflection
                plan_steps_dict = [
                    {"tool_name": step.tool_name, "step_description": step.step_description}
                    for step in plan.steps
                ]

                reflection_result = await self._reflector.reflect_on_execution(
                    original_goal=state.task_description,
                    plan_steps=plan_steps_dict,
                    execution_results=executed_steps,
                    partial_completion=partial_completion,
                )

                if not reflection_result.success:
                    logger.error("Reflection failed")
                    state.set_complete("Reflection failed")
                    return False

                reflection = reflection_result.result
                logger.info(f"Reflection result: {reflection.next_action} - {reflection.reasoning}")

                # Act based on reflection decision
                if reflection.next_action == "replan":
                    # Re-planning needed - prepare guidance for next cycle
                    guidance = reflection.replanning_guidance or "Adjust approach based on failures"
                    logger.info(f"Re-planning: {guidance}")
                    # Update task description with replanning guidance
                    updated_description = (
                        f"{state.task_description}\n\nPrevious attempt failed. {guidance}"
                    )
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
                expected_outcome=plan.expected_outcome,
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
                component_type="execution_cycle", execution_result=str(e), success=False
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
        self, state: AgentState, failed_steps: list[dict], cycle: int
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
        failure_summary = "\n".join(
            [f"  • {step.get('tool_name')}: {step.get('result')}" for step in failed_steps]
        )

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
                f"Step {idx} ({tool}): {description}\nStatus: {status}\nResult: {result}"
            )

        return "\n\n".join(content_parts)
