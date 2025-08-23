"""
Dual-Path Agent Implementation.

This module provides a clean implementation of an agent with separate paths
for conversation and tasks, eliminating the monolithic approach and reducing complexity.
"""

from typing import Dict, Any, Optional, List, Union

from flowlib.flows.base.base import Flow
import logging
import uuid
import asyncio
from datetime import datetime

from ..core.agent import AgentCore
from ..models.config import AgentConfig
from flowlib.agent.models.state import AgentState
from ..components.classification.flow import MessageClassifierFlow, MessageClassifierInput
from ..components.conversation.handler import DirectConversationHandler
from ..components.tasks.handler import TaskExecutionHandler
from ..components.conversation.flow import ConversationFlow
from ..components.shell_command.flow import ShellCommandFlow
from flowlib.flows.models.results import FlowResult
from flowlib.flows.models.constants import FlowStatus
from ...flows.base import Flow
from ..models.plan import PlanExecutionOutcome
from ..components.planning.models import Plan
from ..components.planning.models import PlanStep
from ...flows.registry import flow_registry
from flowlib.core.context.context import Context
from pydantic import BaseModel
from ..components.reflection.models import StepReflectionInput, StepReflectionResult, PlanReflectionContext


class EmptyInput(BaseModel):
    """Empty input for flows that don't require specific input."""
    pass
from ...utils.formatting.conversation import format_execution_history
# Import ShellCommandOutput for type checking
from ..components.shell_command.flow import ShellCommandOutput

logger = logging.getLogger(__name__)


class DualPathAgent:
    """Agent with separate paths for conversation and tasks using composition"""
    
    def __init__(
        self, 
        config: Optional[Union[Dict[str, Any], AgentConfig]] = None, 
        task_description: str = ""
    ):
        """Initialize the dual-path agent.
        
        Args:
            config: Agent configuration
            task_description: Task description for the agent
        """
        # Use composition instead of inheritance
        self._agent_core = AgentCore(config, task_description)
        
        # Components will be initialized during initialize()
        self._classifier = None
        self._conversation_handler = None
        self._task_handler = None
        
    async def initialize(self) -> None:
        """Initialize agent components."""
        # First initialize the core agent
        await self._agent_core.initialize()
        
        # Ensure memory is initialized (needed before use)
        if not self._memory or not self._memory.initialized:
            await self._memory.initialize()
        
        # Create classifier flow
        self._classifier = MessageClassifierFlow()
        self._agent_core.register_flow(self._classifier)
        
        # Create conversation handler with ConversationFlow
        conversation_flow = self._agent_core.flows.get("ConversationFlow")
        if not conversation_flow:
            raise ValueError("ConversationFlow must be registered with the agent before initializing DualPathAgent")
        self._conversation_handler = DirectConversationHandler(conversation_flow)
        
        # Create task handler
        self._task_handler = TaskExecutionHandler(
            planner=self._agent_core._planner,
            reflection=self._agent_core._reflection
        )
        
        logger.info("Dual-path agent initialized with conversation and task handlers")
    
    # Delegate common properties to agent core
    @property
    def initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._agent_core.initialized
    
    # All state access should go through agent._agent_core._state_manager.current_state
    
    @property
    def config(self) -> AgentConfig:
        """Get the agent configuration."""
        return self._agent_core.config
    
    @property
    def flows(self) -> Dict[str, Flow]:
        """Get the flows registry."""
        return self._agent_core.flows
    
    @property
    def _memory(self) -> Optional[Any]:
        """Get the memory component."""
        return self._agent_core._memory_manager._memory
    
    @property
    def state(self) -> Optional[AgentState]:
        """Get the current agent state."""
        return self._agent_core._state_manager.current_state
    
    # All component access should go through agent._agent_core._component_name
    
    async def save_state(self):
        """Delegate save_state to agent core."""
        return await self._agent_core.save_state()
    
    async def process_message(self, message: str) -> FlowResult:
        """Public entry point for processing a single message in an interactive session.

        This method primarily delegates to _handle_single_input.
        It ensures the agent is initialized and provides a consistent API.
        """
        if not self.initialized:
            # Initialization should happen before calling this
            # Consider raising NotInitializedError or logging a warning
            logger.error("Agent not initialized. Cannot process message.")
            # Return an error FlowResult
            return FlowResult(status="ERROR", error="Agent not initialized")

        # Delegate the core logic
        return await self._handle_single_input(message)

    async def _handle_single_input(self, message: str) -> FlowResult:
        """Core logic for processing a single input message (e.g., from a queue)."""
        # This logic is moved directly from the original process_message

        self._agent_core._state_manager.current_state.add_user_message(message)
        
        # Set default task description if needed (can be overridden by classifier)
        if not self._agent_core._state_manager.current_state.task_description:
            default_task = "Respond to user messages and complete tasks."
            self._agent_core._state_manager.current_state.task_description = self.config.task_description if self.config else default_task

        memory_context_summary = "No relevant memories found."
        try:
            # 1. Pre-Classification Memory Check
            if self._memory:
                # Ensure context exists before trying to access memory
                try:
                    # Create context for this session/task if it doesn't exist
                    # This is needed for memory operations to work correctly
                    if self._agent_core._state_manager.current_state.task_id:
                        # Create the context with task description as metadata
                        context_metadata = {"task_description": self._agent_core._state_manager.current_state.task_description}
                        await self._memory.create_context(
                            context_name=self._agent_core._state_manager.current_state.task_id,
                            metadata=context_metadata
                        )
                        logger.debug(f"Created memory context for task: {self._agent_core._state_manager.current_state.task_id}")
                except Exception as ctx_err:
                    # Log if there's an issue, but continue (might already exist)
                    logger.debug(f"Note on memory context creation: {str(ctx_err)}")
                
                # Simple query based on the new message for initial context
                relevant_memories = await self._memory.retrieve_relevant(
                    query=message, 
                    context=self._agent_core._state_manager.current_state.task_id, # Scope to current task/session
                    limit=3 
                )
                if relevant_memories:
                    summary_items = [f"- {mem[:100]}..." if len(mem) > 100 else f"- {mem}" for mem in relevant_memories]
                    memory_context_summary = "Relevant Memories Found:\n" + "\n".join(summary_items)
            else:
                 memory_context_summary = "Memory component not available."
        except Exception as e:
            logger.warning(f"Error retrieving relevant memories pre-classification: {str(e)}")
            memory_context_summary = "Error retrieving memories."
            
        final_result: FlowResult = None
        try:
            # 2. Classification (with Memory Context)
            history = self._get_formatted_history()
            classifier_input = MessageClassifierInput(
                message=message,
                conversation_history=history[-10:], # Limit history for classification prompt
                memory_context_summary=memory_context_summary
            )
            classification = await self._classifier.run_pipeline(classifier_input)
            logger.info(f"Message classified as {'TASK' if classification.execute_task else 'CONVERSATION'} "
                       f"(confidence: {classification.confidence:.2f}, category: {classification.category})")

            # 3. Path Selection & Execution
            if not classification.execute_task:
                # 4a. Conversation Path
                logger.info("Using direct conversation path")
                final_result = await self._conversation_handler.handle_conversation(
                    message=message,
                    state=self.state,
                    memory_context_summary=memory_context_summary,
                    task_result_summary=None # No task result here
                )
                
                # Store conversation result in execution history
                if final_result and final_result.status == "SUCCESS":
                    # Properly serialize the FlowResult with nested Pydantic model
                    result_dict = final_result.model_dump()
                    # If data is a Pydantic model, serialize it properly
                    if hasattr(final_result.data, 'model_dump'):
                        result_dict['data'] = final_result.data.model_dump()
                    
                    self._agent_core._state_manager.current_state.add_execution_result(
                        flow_name="conversation",
                        inputs={"message": message},  # Store the user's message as input
                        result=result_dict,  # Store properly serialized FlowResult
                        success=True
                    )
            else:
                # 4b. Task Path
                logger.info("Using task execution path")
                if classification.task_description:
                    original_task = self._agent_core._state_manager.current_state.task_description
                    self._agent_core._state_manager.current_state.task_description = classification.task_description
                    logger.info(f"Using task description from classification: {self._agent_core._state_manager.current_state.task_description}")
                else:
                    # Fallback if LLM didn't generate task description
                    # Fail-fast: task_description is required for task execution
                    current_description = self._agent_core._state_manager.current_state.task_description
                    if not current_description:
                        raise ValueError("Task execution requires a valid task_description but none was provided by classifier or exists in state")

                # Execute Task - Modified to use plan loop
                plan_outcome = await self._execute_plan_loop(self.state)
                task_flow_result = plan_outcome.result # Get the final result (or error result)

                # Prepare summary of task result for conversational feedback
                task_result_summary = "Task execution completed."
                if plan_outcome.status == "SUCCESS" or plan_outcome.status == "NO_ACTION_NEEDED":
                    # If the plan succeeded and produced a result, format it clearly
                    if task_flow_result and hasattr(task_flow_result.data, 'success') and task_flow_result.data.success is True:
                        stdout = getattr(task_flow_result.data, 'stdout', '').strip()
                        stderr = getattr(task_flow_result.data, 'stderr', '').strip()
                        command_executed = getattr(task_flow_result.data, 'command', 'Unknown command')
                        if stdout:
                            task_result_summary = f"Successfully executed '{command_executed}'. Output:\n```\n{stdout}\n```"
                        elif stderr:
                            task_result_summary = f"Executed '{command_executed}' with errors or warnings:\n```\n{stderr}\n```" # Report stderr if stdout is empty
                        else:
                            task_result_summary = f"Successfully executed '{command_executed}', but it produced no output."
                    elif plan_outcome.status == "NO_ACTION_NEEDED":
                        task_result_summary = "Task plan completed, no action taken or needed."
                    else:
                        # Fallback if successful but result format is unexpected
                        task_result_summary = "Task plan execution finished successfully, but the result format was unexpected."
                elif plan_outcome.status == "ERROR":
                    # Plan execution failed - must have error message
                    if not plan_outcome.error:
                        raise ValueError("Plan execution failed but no error message provided")
                    task_result_summary = f"Task execution failed during planning or execution. Error: {plan_outcome.error}"
                    # Include details from the step result if it exists
                    if task_flow_result and task_flow_result.error:
                        task_result_summary += f" (Details: {task_flow_result.error})"
                else: # Should not happen based on _execute_plan_loop logic
                    task_result_summary = f"Task execution finished with unknown status: {plan_outcome.status}"

                # 5. Task Result Feedback Loop (Call Conversation Handler)
                # --- Reflection Step --- 
                plan_reflect_context = None # Initialize context
                reflection_result = None
                if self._agent_core._reflection:
                    try:
                        logger.info("Reflecting on plan execution outcome...")
                        # Prepare the context for overall plan reflection
                        state_summary = f"Task: {self._agent_core._state_manager.current_state.task_description}\nProgress: {self._agent_core._state_manager.current_state.progress}%\nCycle: {self._agent_core._state_manager.current_state.cycles}\nComplete: {self._agent_core._state_manager.current_state.is_complete}"
                        # Get history from state (might need adjustment if AgentState stores it differently)
                        history_text = format_execution_history(self._agent_core._state_manager.current_state.execution_history)
                        
                        plan_reflect_context = PlanReflectionContext(
                            task_description=self._agent_core._state_manager.current_state.task_description,
                            plan_status=plan_outcome.status,
                            plan_error=plan_outcome.error,
                            step_reflections=plan_outcome.step_reflections, # Get from outcome
                            state_summary=state_summary,
                            execution_history_text=history_text,
                            current_progress=self._agent_core._state_manager.current_state.progress
                        )
                        
                        reflection_result = await self._agent_core._reflection.reflect(
                            plan_context=plan_reflect_context # Pass the new context object
                        )
                        logger.info(f"Reflection complete. Task complete: {reflection_result.is_complete}, Progress: {reflection_result.progress}%")
                        # Update state based on reflection
                        self._agent_core._state_manager.current_state.progress = reflection_result.progress
                        if reflection_result.is_complete:
                            completion_reason = reflection_result.completion_reason
                            if not completion_reason:
                                raise ValueError("Reflection indicated completion but provided no completion reason")
                            self._agent_core._state_manager.current_state.set_complete(completion_reason)
                    except Exception as reflect_err:
                        logger.error(f"Reflection failed: {reflect_err}", exc_info=True)
                        self._agent_core._state_manager.current_state.add_error(f"Reflection failed: {reflect_err}")
                else:
                    logger.warning("Reflection component not available, skipping reflection.")
                # ---------------------

                logger.info("Generating conversational response for task result.")
                final_result = await self._conversation_handler.handle_conversation(
                    message="[Internal Task Result Notification]", # Internal message
                    state=self.state, 
                    memory_context_summary=memory_context_summary, # Pass original memory summary
                    task_result_summary=task_result_summary # Pass the task result summary
                )

            # Add the final assistant response to state *after* all processing
            if final_result and final_result.status == "SUCCESS" and hasattr(final_result.data, 'response'):
                 self._agent_core._state_manager.current_state.add_system_message(final_result.data.response)
            elif final_result:
                 # Log if the final result wasn't successful or didn't have a response field
                 logger.warning(f"Final result status was {final_result.status}, no system message added.")

            # Save state after processing cycle
            await self._save_state()
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing message in DualPathAgent: {str(e)}", exc_info=True)
            # Error saving should happen in the runner or calling context
            # self._agent_core._state_manager.current_state.add_system_message(f"I encountered an error: {str(e)}") 
            # await self._save_state() 

            return FlowResult(
                flow_name="process_message_error_handler",
                status="ERROR",
                error=f"Error processing message: {str(e)}",
                data={"error": str(e)}
            )

    def _get_formatted_history(self) -> List[Dict[str, str]]:
        """Helper to get conversation history from state."""
        history = []
        # Ensure we pair messages correctly
        num_user = len(self._agent_core._state_manager.current_state.user_messages)
        num_system = len(self._agent_core._state_manager.current_state.system_messages)
        for i in range(num_user):
            history.append({"role": "user", "content": self._agent_core._state_manager.current_state.user_messages[i]})
            if i < num_system:
                history.append({"role": "assistant", "content": self._agent_core._state_manager.current_state.system_messages[i]})
        return history
            
    async def _save_state(self) -> None:
        """Save the agent state if a persister is available."""
        if self._agent_core._state_manager._state_persister and self._agent_core._state_manager.current_state.task_id:
            try:
                await self.save_state()
            except Exception as e:
                logger.warning(f"Failed to save state: {str(e)}")
                
    # Override execute_cycle to use dual-path approach
    async def execute_cycle(self, **kwargs) -> bool:
        """Execute a single agent cycle via process_message."""
        if not self.initialized:
             await self.initialize()
             
        if "message" in kwargs:
            await self.process_message(kwargs["message"])
            return not self._agent_core._state_manager.current_state.is_complete # Continue if task not marked complete
        else:
            logger.warning("execute_cycle called without 'message', cannot use dual-path logic.")
            # Maybe trigger a default action like asking for input? Or raise error?
            # For now, return False to stop processing if no message.
            return False 

    async def _execute_plan_loop(self, state: AgentState) -> PlanExecutionOutcome:
        """Executes the steps in the current plan iteratively."""
        outcome_status = "UNKNOWN"
        last_successful_result: Optional[FlowResult] = None # Store last good result
        outcome_error: Optional[str] = None
        step_reflections: List[StepReflectionResult] = [] # List to store step reflections
        while True: # Loop until plan completes, fails, or needs replanning
            # 1. Check for active plan, generate if needed
            current_plan: Optional[Plan] = state.current_plan
            if not current_plan:
                logger.info("No active plan. Generating new plan...")
                try:
                    # Planner now returns a Plan object
                    new_plan = await self._agent_core._planner.plan(state)
                    if not new_plan or not new_plan.steps:
                        logger.info("Planner returned no steps. Task requires no action or is complete.")
                        outcome_status = "NO_ACTION_NEEDED"
                        outcome_result = FlowResult(status=FlowStatus.SUCCESS, data={"message": "No action needed"}, flow_name="planner")
                        break # Exit loop, proceed to reflection/response
                    state.current_plan = new_plan
                    state.current_step_index = 0
                    current_plan = new_plan # Use the newly generated plan
                    logger.info(f"Generated plan ID '{current_plan.plan_id}' with {len(current_plan.steps)} steps.")
                except Exception as planning_err:
                    logger.error(f"Planning failed: {planning_err}", exc_info=True)
                    outcome_status = "ERROR"
                    outcome_error = f"Planning failed: {planning_err}"
                    outcome_result = FlowResult(status=FlowStatus.ERROR, error=f"Planning failed: {planning_err}", flow_name="planner")
                    state.add_error(f"Planning failed: {planning_err}")
                    state.current_plan = None # Ensure plan is cleared on error
                    state.current_step_index = 0
                    break # Exit loop
            # 2. Check if plan is complete
            if state.current_step_index >= len(current_plan.steps):
                logger.info(f"Plan ID '{current_plan.plan_id}' completed successfully.")
                outcome_status = "SUCCESS"
                # Use the result from the last successfully executed step
                outcome_result = last_successful_result
                state.current_plan = None # Clear completed plan
                state.current_step_index = 0
                break # Exit loop
            # 3. Get and execute the next step
            step: PlanStep = current_plan.steps[state.current_step_index]
            step_idx = state.current_step_index
            logger.info(f"Executing step {step_idx + 1}/{len(current_plan.steps)} (ID: {step.step_id}): Flow='{step.flow_name}'")
            step_result: Optional[FlowResult] = None
            # --- Generate Inputs Just-In-Time --- 
            parsed_input: Optional[BaseModel] = None # Holds the specific input model instance
            try:
                # We need the planner instance for generate_inputs
                if not self._agent_core._planner:
                     raise RuntimeError("Planner component is not available for input generation.")
                
                logger.debug(f"Generating inputs for step {step.step_id}, flow '{step.flow_name}' with intent: '{step.step_intent}'")
                # Adapt generate_inputs call (this method needs modification - Step 2.5)
                parsed_input = await self._agent_core._planner.generate_inputs(
                    state=state,
                    flow_name=step.flow_name,
                    step_intent=step.step_intent, # Pass intent
                    step_rationale=step.rationale, # Pass rationale
                    memory_context_id=state.task_id # Pass context id for memory lookup
                )
                logger.info(f"Generated inputs for flow '{step.flow_name}': {str(parsed_input.model_dump())[:100]}...")
                
            except Exception as input_gen_err:
                 # Handle input generation failure
                 error_message = f"Input generation failed for step {state.current_step_index + 1}: {input_gen_err}"
                 logger.error(error_message, exc_info=True)
                 outcome_status = "ERROR"
                 outcome_error = error_message
                 outcome_result = FlowResult(status=FlowStatus.ERROR, error=f"Input generation failed: {input_gen_err}", flow_name=step.flow_name)
                 state.add_error(error_message)
                 state.current_plan = None # Clear plan
                 state.current_step_index = 0
                 break # Exit loop
            # -------------------------------------
            try:
                # Create context for flow execution
                input_context = Context(data=parsed_input) 
               
                # --- Retrieve the Flow Instance ---
                flow_to_run = flow_registry.get_flow(step.flow_name)
                if not flow_to_run:
                    raise RuntimeError(f"Flow '{step.flow_name}' not found in registry.")
                # ----------------------------------
                
                # -------------------------------------------
                
                # --- Execute the Flow --- 
                step_result = await flow_to_run.execute(input_context)
                # ------------------------

                # Handle step success
                if step_result and step_result.status == "SUCCESS":
                    logger.info(f"Step {state.current_step_index + 1} succeeded.")
                    # Add execution result (might need adjustment based on run_flow return)
                    self._agent_core._state_manager.current_state.add_execution_result(
                        flow_name=step.flow_name,
                        inputs=parsed_input.model_dump(), # Log the generated inputs
                        result=step_result.model_dump(), # Store full FlowResult as dict
                        success=True
                    )
                    
                    # --- Perform Step Reflection --- 
                    if self._agent_core._reflection:
                        step_reflect_input = StepReflectionInput(
                            task_description=state.task_description,
                            step_id=step.step_id,
                            step_intent=step.step_intent,
                            step_rationale=step.rationale,
                            flow_name=step.flow_name,
                            flow_inputs=parsed_input, # Pass the generated input model
                            flow_result=step_result,
                            current_progress=state.progress
                        )
                        try:
                            step_reflection = await self._agent_core._reflection.step_reflect(step_reflect_input)
                            step_reflections.append(step_reflection)
                            logger.info(f"Step {step_idx + 1} reflection added.")
                        except Exception as reflect_err:
                            logger.error(f"Step reflection failed for step {step_idx + 1}: {reflect_err}", exc_info=True)
                            # Optionally add a placeholder reflection on error
                            step_reflections.append(StepReflectionResult(
                                step_id=step.step_id, 
                                reflection=f"Reflection error: {reflect_err}", 
                                step_success=True, # Step succeeded, reflection failed
                                key_observation="Reflection process failed."))
                    # ----------------------------- 
                    
                    last_successful_result = step_result # Store this result
                    state.current_step_index += 1
                    # Continue to the next iteration of the while loop for the next step
                    continue 
                else:
                    # Handle step failure (non-exception case)
                    error_message = f"Step {state.current_step_index + 1} failed. Status: {step_result.status if step_result else 'UNKNOWN'}. Error: {step_result.error if step_result and step_result.error else 'No error details'}."
                    logger.error(error_message)
                    outcome_status = "ERROR"
                    outcome_error = error_message
                    # Use the failed result if available, otherwise keep last good one? Or None?
                    # Let's store the *failed* result here for diagnosis
                    outcome_result = step_result
                    state.add_error(error_message)
                    state.current_plan = None # Clear plan on failure
                    state.current_step_index = 0
                    
                    # --- Perform Step Reflection (on step failure) --- 
                    if self._agent_core._reflection:
                        # Ensure step_result is a FlowResult, even if None
                        if not step_result:
                            step_result = FlowResult(status=FlowStatus.ERROR, error="Step failed before result object was created.", flow_name=step.flow_name)
                        step_reflect_input = StepReflectionInput(
                            task_description=state.task_description,
                            step_id=step.step_id,
                            step_intent=step.step_intent,
                            step_rationale=step.rationale,
                            flow_name=step.flow_name,
                            flow_inputs=parsed_input if parsed_input else EmptyInput(), # Pass dummy if needed
                            flow_result=step_result, # Pass the failed result
                            current_progress=state.progress
                        )
                        try:
                            step_reflection = await self._agent_core._reflection.step_reflect(step_reflect_input)
                            step_reflections.append(step_reflection)
                            logger.info(f"Step {step_idx + 1} (failed) reflection added.")
                        except Exception as reflect_err:
                            logger.error(f"Step reflection failed for step {step_idx + 1}: {reflect_err}", exc_info=True)
                            step_reflections.append(StepReflectionResult(
                                step_id=step.step_id, 
                                reflection=f"Reflection error: {reflect_err}", 
                                step_success=False, # Step failed
                                key_observation="Reflection process failed."))
                    # ------------------------------------------------
                    
                    break # Exit loop
            except Exception as step_err:
                # Handle step failure (exception case)
                error_message = f"Step {state.current_step_index + 1} raised an exception: {step_err}"
                logger.error(error_message, exc_info=True)
                outcome_status = "ERROR"
                outcome_error = error_message
                # Create a minimal FlowResult for the error, keep last_successful_result potentially
                outcome_result = FlowResult(status=FlowStatus.ERROR, error=str(step_err), flow_name=step.flow_name) 
                state.add_error(error_message)
                # Add failed execution attempt to history
                self._agent_core._state_manager.current_state.add_execution_result(
                    flow_name=step.flow_name,
                    inputs=parsed_input.model_dump() if parsed_input else {"error": "Input generation failed"}, # Log generated inputs or error
                    result=outcome_result.model_dump(),
                    success=False
                )
                state.current_plan = None # Clear plan on failure
                state.current_step_index = 0
                
                # --- Perform Step Reflection (on step exception) --- 
                if self._agent_core._reflection:
                    step_reflect_input = StepReflectionInput(
                        task_description=state.task_description,
                        step_id=step.step_id,
                        step_intent=step.step_intent,
                        step_rationale=step.rationale,
                        flow_name=step.flow_name,
                        flow_inputs=parsed_input if parsed_input else EmptyInput(), # Pass dummy if needed
                        flow_result=outcome_result, # Pass the error FlowResult
                        current_progress=state.progress
                    )
                    try:
                        step_reflection = await self._agent_core._reflection.step_reflect(step_reflect_input)
                        step_reflections.append(step_reflection)
                        logger.info(f"Step {step_idx + 1} (exception) reflection added.")
                    except Exception as reflect_err:
                        logger.error(f"Step reflection failed for step {step_idx + 1}: {reflect_err}", exc_info=True)
                        step_reflections.append(StepReflectionResult(
                            step_id=step.step_id, 
                            reflection=f"Reflection error: {reflect_err}", 
                            step_success=False, # Step failed
                            key_observation="Reflection process failed."))
                # ---------------------------------------------------

                break # Exit loop
        # Return the final outcome of the plan execution attempt
        return PlanExecutionOutcome(
            status=outcome_status,
            result=outcome_result, # This will be last successful or the error result
            step_reflections=step_reflections, # Include collected reflections
            error=outcome_error
        ) 