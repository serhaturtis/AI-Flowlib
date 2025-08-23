"""Agent execution engine with TODO-driven capabilities."""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import Field
from flowlib.core.models import StrictBaseModel
from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ExecutionError
from flowlib.agent.components.engine.interfaces import EngineInterface
from flowlib.agent.models.state import AgentState
from flowlib.agent.models.config import AgentConfig, EngineConfig  
from flowlib.agent.components.planning.planner import AgentPlanner
from flowlib.agent.components.planning.todo import TodoManager, TodoItem, TodoStatus
from flowlib.agent.components.reflection.base import AgentReflection
from flowlib.flows.models.results import FlowResult, AgentResult
from flowlib.flows.models.constants import FlowStatus

logger = logging.getLogger(__name__)


class ExecutionResultData(StrictBaseModel):
    """Execution result data model."""
    
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data")
    

class CompletionSummaryData(StrictBaseModel):
    """Completion summary data model."""
    
    progress: float = Field(default=0.0, description="Progress as a decimal (0.0-1.0)")
    total: int = Field(default=0, description="Total number of items")
    completed: int = Field(default=0, description="Number of completed items")
    failed: int = Field(default=0, description="Number of failed items")
    

class StateData(StrictBaseModel):
    """State data model for cycle counting."""
    
    cycles: int = Field(default=0, description="Number of cycles executed")


class ExecutionStrategy(Enum):
    """Execution strategy for the agent."""
    AUTO = "auto"  # Automatically determine strategy
    SINGLE_CYCLE = "single_cycle"  # Traditional single-cycle execution
    TODO_DRIVEN = "todo_driven"  # Multi-step TODO-driven execution


class AgentEngine(AgentComponent, EngineInterface):
    """Engine that can execute both single flows and TODO-driven workflows."""
    
    def __init__(self, config: Optional[EngineConfig] = None, memory=None, planner=None, reflection=None, activity_stream=None, name: str = "agent_engine", agent_config: Optional[AgentConfig] = None):
        """Initialize the agent engine.
        
        Args:
            config: Engine configuration (respects parent interface)
            memory: Memory component
            planner: Planner component  
            reflection: Reflection component
            activity_stream: Activity stream
            name: Component name
            agent_config: Full agent config for advanced features (unified engine specific)
        """
        # Initialize as AgentComponent
        super().__init__(name)
        
        # Configuration
        self._config = config or EngineConfig()
        
        # Components (these should be provided by AgentCore)
        self._memory = memory
        self._planner = planner
        self._reflection = reflection
        self._activity_stream = activity_stream
        
        # TODO system  
        self._todo_manager = TodoManager("EngineToDoManager", activity_stream=activity_stream)
        
        # Execution state
        self._iteration = 0
        self._last_execution_result = None
        
        # Store agent config for unified features, derive engine config if needed
        if agent_config:
            self._agent_config = agent_config
            # If no engine config provided but agent config available, extract it
            if config is None:
                self._config = agent_config.engine_config
        elif config and hasattr(config, 'engine_config'):
            # Handle backward compatibility - if AgentConfig passed as config
            logger.warning("AgentConfig passed as config parameter, extracting engine_config. Use agent_config parameter instead.")
            self._agent_config = config
            self._config = config.engine_config
        else:
            # No agent config available, unified features will be limited
            self._agent_config = None
            
        self.todo_manager: Optional[TodoManager] = None
        self.unified_planner: Optional[AgentPlanner] = None
        
    async def _initialize_impl(self) -> None:
        """Initialize the unified engine components."""
        # Check for required components
        if not self._memory:
            raise ExecutionError("Memory component is required")
        
        if not self._planner:
            raise ExecutionError("Planner component is required")
        
        if not self._reflection:
            raise ExecutionError("Reflection component is required")
        
        # Initialize TODO manager
        await self._todo_manager.initialize()
        
        # Reset iteration counter
        self._iteration = 0
        
        logger.info(f"Initialized unified agent engine with {self._config}")
        
        # Initialize unified-specific TODO manager with activity stream
        self.todo_manager = TodoManager(name="unified_todo_manager", activity_stream=self._activity_stream)
        await self.todo_manager.initialize()
        
        # Replace planner with unified planner if not already done
        if not isinstance(self._planner, AgentPlanner):
            # Use the existing planner's config, which is guaranteed to exist and be properly typed
            planner_config = self._planner.config if self._planner else self._agent_config.planner_config
            
            self.unified_planner = AgentPlanner(planner_config, name="unified_planner", activity_stream=self._activity_stream)
            self.unified_planner.set_parent(self.parent)  # Set parent reference
            await self.unified_planner.initialize()
            
            # Swap out the original planner
            if self._planner:
                await self._planner.shutdown()
            self._planner = self.unified_planner
        else:
            self.unified_planner = self._planner
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the unified engine components."""
        if self.todo_manager:
            await self.todo_manager.shutdown()
        await self._todo_manager.shutdown()
        logger.info("Shutting down unified agent engine")
    
    async def execute(
        self,
        task_description: str,
        context: Optional[StrictBaseModel] = None,
        max_cycles: Optional[int] = None
    ) -> StrictBaseModel:
        """Execute a task using the agent engine.
        
        Args:
            task_description: Description of the task to execute
            context: Additional context for execution
            max_cycles: Maximum number of execution cycles (overrides config)
            
        Returns:
            Execution result with output and metadata
        """
        import time
        from datetime import datetime
        from ...models.state import AgentState, ExecutionResult
        
        self._check_initialized()
        
        # Create agent state for this task
        state = AgentState(
            task_id=f"task_{int(time.time())}",
            task_description=task_description
        )
        
        # Override max iterations if specified
        original_max = self._config.max_iterations
        if max_cycles is not None:
            self._config.max_iterations = max_cycles
        
        try:
            # Create task-specific memory context
            task_context = f"task_{state.task_id}"
            if self._memory:
                try:
                    await self._memory.create_context(
                        context_name=task_context,
                        metadata={
                            "task_id": state.task_id,
                            "task_description": state.task_description,
                            "created_at": datetime.now().isoformat()
                        }
                    )
                    logger.debug(f"Created task context: {task_context}")
                except Exception as e:
                    logger.debug(f"Could not create task context: {e}")
            
            # Execute cycles until completion or max iterations
            memory_context = "agent_execution"
            continue_execution = True
            
            while continue_execution and state.cycles < self._config.max_iterations:
                continue_execution = await self.execute_cycle(
                    state=state,
                    memory_context=memory_context,
                    no_flow_is_error=False
                )
            
            # Extract output from executed flows
            output = state.completion_reason or "Task completed"
            
            # Try to find meaningful output from execution history
            for entry in reversed(state.execution_history):
                result_data = entry.result
                if isinstance(result_data, dict) and "data" in result_data:
                    data = result_data["data"]
                    
                    # Use standardized user display extraction
                    from ...flows.user_display import format_flow_output_for_user
                    user_output = format_flow_output_for_user(
                        flow_name=entry.flow_name,
                        result_data=data,
                        success=entry.success
                    )
                    
                    # Only use non-generic outputs
                    if user_output and not user_output.endswith("completed successfully"):
                        output = user_output
                        break
            
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
            
        finally:
            # Restore original max iterations
            if max_cycles is not None:
                self._config.max_iterations = original_max
    
    async def execute_cycle(
        self,
        state,  # AgentState
        memory_context: str = "agent",
        no_flow_is_error: bool = False
    ) -> bool:
        """Execute a single execution cycle of the agent.
        
        An execution cycle involves:
        1. Planning the next action
        2. Executing the selected flow
        3. Reflecting on the results
        4. Updating the agent state
        
        Args:
            state: Agent state to use for this cycle
            memory_context: Parent memory context
            no_flow_is_error: Whether to treat no flow selection as an error
            
        Returns:
            True if execution should continue, False if execution is complete
        """
        try:
            # Increment iteration counter
            self._iteration += 1
            
            # Create memory context for this cycle
            cycle_context = await self._create_cycle_context(memory_context, self._iteration)
            
            # Update cycles counter in state
            state.increment_cycle()
            
            # Log cycle start
            logger.info(f"Starting execution cycle {state.cycles}")
            
            # Stream cycle start
            if self._activity_stream:
                self._activity_stream.start_section(f"Execution Cycle {state.cycles}")
                self._activity_stream.execution(f"Starting cycle {state.cycles} for task: {state.task_description[:50]}...", cycle=state.cycles)
            
            # Plan next action
            result = await self._plan_next_action(state, cycle_context)            
            # Get selected flow
            selected_flow = result.selected_flow
            
            # Check if a flow was selected
            if not selected_flow or selected_flow == "none":
                if no_flow_is_error:
                    state.add_error("No flow selected by planner")
                    logger.warning("No flow selected by planner")
                    await self._save_state(state)
                    return False
                else:
                    # No flow needed, just continue
                    logger.info("No flow selected by planner, continuing to next cycle")
                    return True
            
            # Generate inputs for the flow (as StrictBaseModel)
            inputs = await self._generate_inputs(state, cycle_context, selected_flow, result)
            
            # Execute the flow with the StrictBaseModel inputs
            flow_result = await self.execute_flow(selected_flow, inputs, state)
            
            # Reflect on results using StrictBaseModel inputs
            reflection_result = await self._reflect_on_results(state, cycle_context, selected_flow, inputs, flow_result)
            
            # Check if reflection indicated completion
            if reflection_result.is_complete:
                state.set_complete(reflection_result.completion_reason)
                state.progress = reflection_result.progress
                logger.info(f"Reflection indicated task completion: {state.completion_reason}")
                await self._save_state(state)
                return False
            
            # Update progress
            state.progress = reflection_result.progress
            
            # Check if we'll reach max iterations on the next cycle
            if self._iteration >= self._config.max_iterations:
                state.set_complete("Maximum iterations reached")
                logger.info(f"Reached max iterations ({self._config.max_iterations})")
                await self._save_state(state)
                return False
            
            # Save state after successful cycle
            await self._save_state(state)
            
            # Stream cycle end
            if self._activity_stream:
                self._activity_stream.execution(f"Cycle {state.cycles} completed - Progress: {state.progress}%", progress=state.progress)
                self._activity_stream.end_section()
            
            # Continue with more cycles
            return True
            
        except Exception as e:
            # Record the error in the state
            error_message = f"Error during execution cycle {state.cycles}: {str(e)}"
            state.add_error(error_message)
            logger.error(error_message, exc_info=True)
            
            # Stream error
            if self._activity_stream:
                self._activity_stream.error(f"Cycle {state.cycles} failed: {str(e)}")
                self._activity_stream.end_section()
            
            # Save state after error
            await self._save_state(state)
            
            # Determine whether to continue
            should_continue = not self._config.stop_on_error
            
            if should_continue:
                logger.info(f"Continuing execution despite error (stop_on_error={self._config.stop_on_error})")
            else:
                logger.info(f"Stopping execution due to error (stop_on_error={self._config.stop_on_error})")
                
            return should_continue
    
    def _check_initialized(self) -> None:
        """Check if the engine is initialized."""
        if not self._initialized:
            from ..core.errors import NotInitializedError
            raise NotInitializedError(
                component_name=self._name,
                operation="execute"
            )
    
    async def _create_cycle_context(self, memory_context: str, iteration: int) -> str:
        """Create context string for this execution cycle."""
        return f"{memory_context}_cycle_{iteration}"
    
    async def _plan_next_action(self, state, context: str):
        """Plan the next action using the planner."""
        if not self._planner:
            raise ExecutionError("Planner component is required")
        
        # Use simplified planning for now
        from ..planning.models import PlanningResult, PlanningExplanation
        return PlanningResult(
            selected_flow="none",
            reasoning=PlanningExplanation(
                explanation="No specific flow planning implemented",
                rationale="Using default fallback behavior",
                decision_factors=["No flows available", "Simplified execution mode"]
            )
        )
    
    async def _generate_inputs(self, state, context: str, selected_flow: str, planning_result):
        """Generate inputs for the selected flow."""
        return StrictBaseModel()
    
    async def execute_flow(self, flow_name: str, inputs, state):
        """Execute a flow with given inputs."""
        from flowlib.flows.models.results import FlowResult
        from flowlib.flows.models.constants import FlowStatus
        
        # Simplified flow execution
        return FlowResult(
            status=FlowStatus.SUCCESS,
            data={"output": f"Executed {flow_name}"},
            flow_name=flow_name
        )
    
    async def _reflect_on_results(self, state, context: str, flow_name: str, inputs, flow_result):
        """Reflect on the execution results."""
        if not self._reflection:
            raise ExecutionError("Reflection component is required")
        
        # Use simplified reflection for now
        from ..reflection.models import ReflectionResult
        return ReflectionResult(
            is_complete=True,
            progress=100.0,
            completion_reason="Task completed"
        )
    
    async def _save_state(self, state):
        """Save the agent state."""
        # Simplified state saving - in practice this would persist to storage
        logger.debug(f"Saving state for task {state.task_id}")
        pass
    
    async def execute_with_strategy(
        self,
        task_description: str,
        context: Dict[str, Any] = None,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.AUTO
    ) -> AgentResult:
        """Execute task using the most appropriate strategy.
        
        Args:
            task_description: Description of the task to execute
            context: Additional context for execution
            execution_strategy: Strategy to use for execution
            
        Returns:
            AgentResult containing execution results
        """
        context = context or {}
        
        # Determine execution strategy if auto
        if execution_strategy == ExecutionStrategy.AUTO:
            strategy_name = await self.unified_planner.determine_execution_strategy(task_description)
            execution_strategy = ExecutionStrategy(strategy_name)
        
        self._logger.info(f"Executing task with strategy: {execution_strategy.value}")
        
        if execution_strategy == ExecutionStrategy.TODO_DRIVEN:
            return await self._execute_todo_driven(task_description, context)
        else:
            return await self._execute_single_cycle(task_description, context)
    
    async def _execute_todo_driven(
        self, 
        task_description: str, 
        context: Dict[str, Any]
    ) -> AgentResult:
        """Execute task using TODO-driven approach.
        
        This method:
        1. Creates a multi-step plan and converts it to TODOs
        2. Executes TODOs in dependency order
        3. Tracks progress and handles failures
        4. Provides overall reflection on completion
        """
        # Phase 1: Planning and TODO Generation
        self._logger.info("Phase 1: Planning and TODO generation")
        
        state = await self._create_initial_state(task_description, context)
        memory_context = self._get_memory_context(state)
        available_flows = self._get_available_flows()
        
        # Generate plan and TODOs
        plan_result, todos = await self.unified_planner.plan_with_todos(
            state, available_flows, memory_context, auto_generate_todos=True
        )
        
        if not todos:
            # Fall back to single-cycle execution
            self._logger.info("No TODOs generated, falling back to single-cycle execution")
            return await self._execute_single_cycle(task_description, context, state)
        
        # Phase 2: TODO Management Setup
        self._logger.info(f"Phase 2: Setting up {len(todos)} TODOs for execution")
        
        # Create new TODO list for this task
        self.todo_manager.create_list(context={
            "task_description": task_description,
            "task_id": state.task_id,
            "strategy": "todo_driven"
        })
        
        # Add TODOs to manager
        for todo in todos:
            # Add TODO to the current list manually
            self.todo_manager.current_list.items[todo.id] = todo
        
        # Phase 3: TODO-Driven Execution Loop
        self._logger.info("Phase 3: TODO-driven execution loop")
        
        overall_result = AgentResult(success=True, results=[], state=state)
        execution_count = 0
        max_executions = len(todos) * 2  # Safety limit
        
        while self._has_executable_todos() and execution_count < max_executions:
            execution_count += 1
            
            # Get next executable TODO
            next_todo = self.todo_manager.get_next_todo()
            if not next_todo:
                self._logger.warning("No executable TODOs found, but TODOs remain. Possible dependency deadlock.")
                break
            
            self._logger.info(f"Executing TODO {execution_count}: {next_todo.content}")
            
            # Execute TODO
            try:
                cycle_result = await self._execute_todo_cycle(next_todo, state)
                overall_result.results.append(cycle_result)
                
                # Update TODO status
                if cycle_result.is_success():
                    next_todo.mark_completed({"result": cycle_result.model_dump()})
                    self._logger.info(f"TODO completed successfully: {next_todo.content}")
                else:
                    error_msg = cycle_result.error if hasattr(cycle_result, 'error') and cycle_result.error else 'Unknown error'
                    next_todo.mark_failed(str(error_msg))
                    self._logger.error(f"TODO failed: {next_todo.content} - {error_msg}")
                    
                    if self._config.stop_on_error:
                        overall_result.success = False
                        break
                
                # Update overall state with current execution
                overall_result.state = state
                
            except Exception as e:
                next_todo.mark_failed(str(e))
                self._logger.error(f"Exception during TODO execution: {e}", exc_info=True)
                
                if self._config.stop_on_error:
                    overall_result.success = False
                    break
        
        # Phase 4: Final Reflection and Completion
        self._logger.info("Phase 4: Final reflection and completion")
        
        completion_summary = self._get_completion_summary()
        overall_result = await self._reflect_on_overall_progress(
            task_description, overall_result, completion_summary
        )
        
        self._logger.info(f"TODO-driven execution completed. Summary: {completion_summary}")
        
        return overall_result
    
    async def _execute_single_cycle(
        self, 
        task_description: str, 
        context: Dict[str, Any],
        state: Optional[AgentState] = None
    ) -> AgentResult:
        """Execute task using traditional single-cycle approach."""
        self._logger.info("Executing task using single-cycle approach")
        
        # Use the original execute method from base class
        execution_result = await self.execute(task_description, context)
        
        # Convert ExecutionResult to AgentResult
        from flowlib.flows.models.results import AgentResult, FlowResult
        from flowlib.flows.models.constants import FlowStatus
        
        # Create a flow result from the execution result
        flow_result = FlowResult(
            data={"output": execution_result.output, "task_id": execution_result.task_id},
            flow_name="single_cycle_execution",
            status=FlowStatus.SUCCESS if execution_result.is_complete and not execution_result.errors else FlowStatus.ERROR,
            error="; ".join(execution_result.errors) if execution_result.errors else None,
            metadata={
                "cycles": execution_result.cycles,
                "progress": execution_result.progress,
                "stats": execution_result.stats
            }
        )
        
        return AgentResult(
            success=execution_result.is_complete and not execution_result.errors,
            results=[flow_result],
            state=state,  # Use provided state if available
            error="; ".join(execution_result.errors) if execution_result.errors else None,
            metadata={
                "execution_type": "single_cycle",
                "task_id": execution_result.task_id,
                "execution_history": execution_result.execution_history
            }
        )
    
    async def _execute_todo_cycle(
        self, 
        todo: TodoItem, 
        state: AgentState
    ) -> FlowResult:
        """Execute a single TODO item as a cycle.
        
        Args:
            todo: The TODO item to execute
            state: Current agent state
            
        Returns:
            FlowResult from executing the TODO
        """
        # Mark TODO as in progress
        todo.mark_in_progress()
        
        # Extract flow information from TODO
        flow_name = todo.assigned_tool or "conversation"  # Default fallback
        execution_context = todo.execution_context or {}
        
        # Create updated state for this TODO
        # Copy the current state metadata and add TODO-specific data
        updated_metadata = {**state.metadata, "todo_id": todo.id, "todo_context": execution_context}
        
        # Create new state by copying existing state data
        state_cycles = state.cycles if hasattr(state, 'cycles') and state.cycles else 0
        initial_state_data = {
            "task_id": state.task_id,
            "task_description": todo.content,  # Use TODO content as task description  
            "cycle": state_cycles + 1,
            "execution_history": state.execution_history.copy(),
            "metadata": updated_metadata
        }
        
        todo_state = AgentState(initial_state_data=initial_state_data)
        
        # Execute single cycle for this TODO
        try:
            should_continue = await self.execute_cycle(
                state=todo_state,
                memory_context=self._get_memory_context(todo_state),
                no_flow_is_error=False
            )
            
            # Create FlowResult from state 
            # Get the last execution from history
            if todo_state.execution_history:
                last_execution = todo_state.execution_history[-1]
                result_data = ExecutionResultData.model_validate(last_execution.result or {})
                return FlowResult(
                    data=result_data.data,
                    flow_name=last_execution.flow_name,
                    metadata={"todo_id": todo.id, "should_continue": should_continue}
                )
            else:
                # No execution happened
                return FlowResult(
                    data={"message": "No flow executed"},
                    flow_name="none",
                    metadata={"todo_id": todo.id, "should_continue": should_continue}
                )
            
        except Exception as e:
            # Create error result
            return FlowResult(
                data={},
                flow_name=flow_name or "unknown",
                status=FlowStatus.ERROR,
                error=str(e),
                metadata={"todo_id": todo.id, "error_type": type(e).__name__}
            )
    
    def _has_executable_todos(self) -> bool:
        """Check if there are any executable TODOs remaining."""
        if not self.todo_manager.current_list:
            return False
        
        executable_todos = self.todo_manager.current_list.get_executable_todos()
        return len(executable_todos) > 0
    
    def _get_completion_summary(self) -> Dict[str, Any]:
        """Get completion summary for the current TODO list."""
        if not self.todo_manager.current_list:
            return {"total": 0, "completed": 0, "failed": 0, "progress": 0}
        
        return self.todo_manager.current_list.get_progress_summary()
    
    def _get_memory_context(self, state: AgentState) -> str:
        """Get memory context identifier for the given state.
        
        Args:
            state: Agent state to get context for
            
        Returns:
            Memory context string
        """
        return f"task_{state.task_id}"
    
    def _get_available_flows(self) -> Dict[str, Any]:
        """Get available flows for execution.
        
        Returns:
            Dictionary of available flows
        """
        # Import here to avoid circular imports
        from ....flows.registry import flow_registry
        
        if flow_registry:
            return flow_registry.get_agent_selectable_flows()
        return {}
    
    async def _reflect_on_overall_progress(
        self,
        task_description: str,
        overall_result: AgentResult,
        completion_summary: Dict[str, Any]
    ) -> AgentResult:
        """Reflect on the overall progress of TODO-driven execution.
        
        This provides a final assessment of whether the original task was completed
        successfully based on the TODO execution results.
        """
        if not self._reflection:
            return overall_result
        
        try:
            # Use reflection to assess overall completion
            reflection_result = await self._reflection.reflect(
                state=overall_result.state,
                flow_name="todo_driven_execution",
                flow_inputs={"task_description": task_description, "completion_summary": completion_summary},
                flow_result=overall_result,
                memory_context=overall_result.state.task_id
            )
            
            # Create new metadata dict with reflection results
            updated_metadata = overall_result.metadata or {}
            updated_metadata.update({
                "completion_summary": completion_summary,
                "reflection": reflection_result.model_dump() if hasattr(reflection_result, 'model_dump') else str(reflection_result),
                "execution_strategy": "todo_driven"
            })
            
            # Consider task successful if most TODOs completed successfully
            summary_data = CompletionSummaryData.model_validate(completion_summary)
            
            # Create new AgentResult with updated metadata (since it's frozen)
            overall_result = AgentResult(
                success=summary_data.progress >= 0.8,  # 80% completion threshold
                results=overall_result.results,
                state=overall_result.state,
                error=overall_result.error,
                metadata=updated_metadata
            )
            
        except Exception as e:
            self._logger.warning(f"Failed to reflect on overall progress: {e}")
            # Don't fail the entire execution due to reflection issues
        
        return overall_result
    
    async def _create_initial_state(self, task_description: str, context: Dict[str, Any]) -> AgentState:
        """Create initial agent state for task execution."""
        import time
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
        return self.todo_manager