"""Agent execution engine with TODO-driven capabilities."""

import logging
import os
import time
from typing import Dict, Any, Optional

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ExecutionError
from flowlib.agent.models.state import AgentState, ExecutionResult
from flowlib.agent.components.task.decomposition import TodoManager
# Using existing AgentMessage from core models for conversation history

logger = logging.getLogger(__name__)


class EngineComponent(AgentComponent):
    """Engine that orchestrates TaskGeneration → TaskDecomposition → TaskExecution → TaskDebriefing."""
    
    def __init__(self, agent_name: str, agent_persona: str, max_iterations: int = 10,
                 memory=None, knowledge=None, task_generator=None,
                 task_decomposer=None, task_executor=None, task_debriefer=None,
                 activity_stream=None, context_manager=None,
                 name: str = "agent_engine"):
        """Initialize the agent engine.
        
        Args:
            agent_name: Name of the agent
            agent_persona: Agent's persona description
            max_iterations: Maximum execution iterations
            memory: Memory component
            knowledge: Knowledge component
            task_generator: Task generator component
            task_decomposer: Task decomposer component
            task_executor: Task executor component
            task_debriefer: Task debriefer component
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
        self._task_generator = task_generator
        self._task_decomposer = task_decomposer
        self._task_executor = task_executor
        self._task_debriefer = task_debriefer
        self._activity_stream = activity_stream
        self._context_manager = context_manager
        self._todo_manager = TodoManager("EngineToDoManager", activity_stream=activity_stream)
        self._iteration = 0
            
    async def _initialize_impl(self) -> None:
        """Initialize the engine components."""
        if not self._memory:
            raise ExecutionError("Memory component is required")
        if not self._task_generator:
            raise ExecutionError("Task generator component is required")
        if not self._task_decomposer:
            raise ExecutionError("Task decomposer component is required")
        if not self._task_executor:
            raise ExecutionError("Task executor component is required")
        if not self._task_debriefer:
            raise ExecutionError("Task debriefer component is required")
        # Knowledge component is optional
            
        logger.info("Engine initialized")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the engine."""
        logger.info("Engine shutdown")
    
    async def execute(self, task_description: str, conversation_history=None):
        """Execute a task by orchestrating components."""
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
            # completion_reason must be set when task completes
            output = state.completion_reason
            
            # Prepare result as Pydantic model
            result = ExecutionResult(
                output=output,
                task_id=state.task_id,
                cycles=state.cycles,
                progress=state.progress,
                is_complete=state.is_complete,
                errors=state.errors,
                execution_history=state.execution_history,
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
    
    async def execute_cycle(self, state, conversation_history=None, memory_context: str = "agent", no_flow_is_error: bool = False) -> bool:
        """Execute a single execution cycle with clean context management.
        
        Orchestrates: TaskGenerator → TaskDecomposer → TaskExecution using unified context.
        """
        try:
            # Update state
            self._iteration += 1
            state.increment_cycle()
            await self._context_manager.increment_cycle()
            
            logger.info(f"Starting execution cycle {state.cycles}")
            
            if self._activity_stream:
                self._activity_stream.execution(f"Starting cycle {state.cycles} for task: {state.task_description[:50]}...", cycle=state.cycles)
            
            # Start task in context manager
            await self._context_manager.start_task(state.task_description)
            
            # 1. Task Generation with managed context
            task_gen_context = await self._context_manager.create_execution_context(
                component_type="task_generation"
            )
            
            task_generation_result = await self._task_generator.convert_message_to_task(
                context=task_gen_context
            )
            
            await self._context_manager.update_from_execution(
                component_type="task_generation",
                execution_result=task_generation_result,
                success=task_generation_result.success
            )
            
            if not task_generation_result.success:
                logger.error(f"Task generation failed: {task_generation_result.error_message}")
                state.set_complete("Task generation failed")
                return False
            
            generated_task = task_generation_result.generated_task
            logger.info(f"Generated task: {generated_task.task_description}")
            
            # 2. Task Decomposition with managed context
            decomp_context = await self._context_manager.create_execution_context(
                component_type="task_decomposition",
                task_description=generated_task.task_description
            )
            
            todos = await self._task_decomposer.decompose_task(context=decomp_context)
            
            await self._context_manager.update_from_execution(
                component_type="task_decomposition", 
                execution_result=todos,
                success=len(todos) > 0
            )
            
            # 3. Check if we got TODOs
            if not todos:
                logger.info("No TODOs generated - task may be complete")
                state.set_complete("No actionable items generated")
                return False
            
            # 4. Task Execution with managed context  
            exec_context = await self._context_manager.create_execution_context(
                component_type="task_execution",
                todos=todos
            )
            
            # Execute the TODOs
            try:
                execution_result = await self._task_executor.execute_todos(context=exec_context)
                
                await self._context_manager.update_from_execution(
                    component_type="task_execution",
                    execution_result=execution_result,
                    success=getattr(execution_result, 'overall_success', True)
                )
                
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                await self._context_manager.update_from_execution(
                    component_type="task_execution",
                    execution_result=str(e),
                    success=False
                )
                # Create failed execution result
                from flowlib.agent.components.task.execution.models import TaskExecutionResult, ToolResult, ToolStatus
                execution_result = TaskExecutionResult(
                    task_description=generated_task.task_description,
                    todos_executed=todos,
                    tool_results=[
                        ToolResult(
                            status=ToolStatus.ERROR,
                            message=f"Execution failed: {str(e)}",
                            execution_time_ms=0
                        )
                    ],
                    final_response=f"Task execution failed: {str(e)}"
                )
            
            # 5. Extract knowledge if component is available
            if self._knowledge and execution_result:
                try:
                    # Combine execution results into content for learning
                    content = self._format_execution_for_learning(execution_result)
                    if content:
                        learning_result = await self._knowledge.learn_from_content(
                            content=content,
                            context=f"Task execution cycle {state.cycles}: {state.task_description}",
                            domain_hint="task_execution"
                        )
                        if learning_result.success:
                            logger.debug(f"Extracted {learning_result.knowledge.total_items} knowledge items from execution")
                except Exception as e:
                    logger.warning(f"Knowledge extraction failed, continuing: {e}")
            
            # 6. Use TaskDebriefer to analyze execution and decide next action
            from flowlib.agent.components.task.debriefing.models import DebriefingInput
            
            # Create debriefing input model
            debriefing_input = DebriefingInput(
                original_user_message=state.task_description,
                generated_task_description=generated_task.task_description,
                execution_results=execution_result.tool_results,
                todos_executed=todos,
                cycle_number=state.cycles,
                agent_persona=self._agent_persona,
                working_directory=os.getcwd(),
                max_cycles=self._max_iterations
            )
            
            debriefing_result = await self._task_debriefer.analyze_and_decide(debriefing_input)
            
            # Act based on debriefing decision
            if debriefing_result.decision.value == "present_success":
                # Add assistant response to conversation history
                await self._context_manager.add_assistant_response(debriefing_result.user_response)
                
                state.progress = 100
                state.set_complete(debriefing_result.user_response)
                logger.info(f"Task completed successfully: {debriefing_result.reasoning}")
                return False
            elif debriefing_result.decision.value == "retry_with_correction":
                # Add assistant response to conversation history
                response_content = self._format_execution_response(execution_result)
                await self._context_manager.add_assistant_response(response_content)
                
                # Update task description for next cycle
                state.task_description = debriefing_result.corrective_task
                state.progress = 30  # Some progress made
                logger.info(f"Retrying with correction: {debriefing_result.reasoning}")
                return True
            else:  # present_failure
                # Add assistant response to conversation history
                await self._context_manager.add_assistant_response(debriefing_result.user_response)
                
                state.set_complete(debriefing_result.user_response)
                logger.info(f"Task failed after analysis: {debriefing_result.reasoning}")
                return False
            
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
    
    def _format_execution_response(self, execution_result) -> str:
        """Format execution result for user response.
        
        Args:
            execution_result: Results from task execution
            
        Returns:
            Formatted response string for user
        """
        if not execution_result:
            return "Task execution completed with no results."
        
        response_parts = []
        
        # Format tool results if available
        if hasattr(execution_result, 'tool_results') and execution_result.tool_results:
            successful_tools = [tr for tr in execution_result.tool_results if hasattr(tr, 'status') and tr.status.value == 'SUCCESS']
            failed_tools = [tr for tr in execution_result.tool_results if hasattr(tr, 'status') and tr.status.value == 'FAILURE']
            
            if successful_tools:
                response_parts.append(f"Successfully executed {len(successful_tools)} operations.")
                
            if failed_tools:
                response_parts.append(f"Failed to execute {len(failed_tools)} operations.")
                for tool_result in failed_tools[:3]:  # Show first 3 failures
                    if hasattr(tool_result, 'message'):
                        response_parts.append(f"- {tool_result.message}")
        
        # Use final response if available
        if hasattr(execution_result, 'final_response') and execution_result.final_response:
            response_parts.append(execution_result.final_response)
        
        # Fallback to string representation
        if not response_parts:
            response_parts.append(str(execution_result))
            
        return " ".join(response_parts)
    
    def _format_execution_for_learning(self, execution_result) -> str:
        """Format execution results for knowledge extraction.
        
        Args:
            execution_result: Results from task execution
            
        Returns:
            Formatted content string for learning
        """
        if not execution_result:
            return ""
            
        content_parts = []
        
        # Extract information from execution result
        if hasattr(execution_result, 'outputs') and execution_result.outputs:
            content_parts.append(f"Execution outputs: {execution_result.outputs}")
        
        if hasattr(execution_result, 'errors') and execution_result.errors:
            content_parts.append(f"Errors encountered: {execution_result.errors}")
        
        if hasattr(execution_result, 'summary'):
            content_parts.append(f"Summary: {execution_result.summary}")
            
        # Convert to string representation if needed
        if not content_parts and hasattr(execution_result, '__dict__'):
            content_parts.append(str(execution_result.__dict__))
        elif not content_parts:
            content_parts.append(str(execution_result))
            
        return " ".join(content_parts)
    
    def _is_task_complete(self, execution_result) -> bool:
        """Check if the task is complete based on execution results.
        
        Args:
            execution_result: Results from task execution
            
        Returns:
            True if task appears complete, False otherwise
        """
        if not execution_result:
            return False
            
        # Check if execution result indicates completion
        if hasattr(execution_result, 'overall_success') and execution_result.overall_success:
            return True
            
        # Check if final response indicates completion
        if hasattr(execution_result, 'final_response') and execution_result.final_response:
            completion_indicators = [
                "completed successfully", "task complete", "finished", 
                "done", "success", "accomplished"
            ]
            response_lower = execution_result.final_response.lower()
            if any(indicator in response_lower for indicator in completion_indicators):
                return True
        
        # Check if all tools succeeded
        if hasattr(execution_result, 'tool_results') and execution_result.tool_results:
            successful_tools = [
                tr for tr in execution_result.tool_results 
                if hasattr(tr, 'status') and tr.status.value == 'SUCCESS'
            ]
            failed_tools = [
                tr for tr in execution_result.tool_results 
                if hasattr(tr, 'status') and tr.status.value == 'FAILURE'
            ]
            
            # If we have some successful tools and no failures, consider complete
            if successful_tools and not failed_tools:
                return True
        
        return False
    
    async def _create_cycle_context(self, memory_context: str, iteration: int) -> str:
        """Create context string for this execution cycle."""
        return f"{memory_context}_cycle_{iteration}"
    
    async def _create_initial_state(self, task_description: str, context: Dict[str, Any]) -> AgentState:
        """Create initial agent state for task execution."""
        state = AgentState(
            task_id=f"task_{int(time.time())}",
            task_description=task_description
        )
        # Add context to metadata if provided
        if context:
            state.metadata.update(context)
        return state
    
    def get_todo_manager(self) -> Optional[TodoManager]:
        """Get the TODO manager instance."""
        return self._todo_manager