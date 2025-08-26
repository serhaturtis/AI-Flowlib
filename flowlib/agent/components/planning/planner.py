"""Agent planner that integrates planning with TODO generation."""

import logging
from typing import Dict, Any, Optional, Tuple, List
from flowlib.core.models import StrictBaseModel

from ...core.errors import PlanningError, NotInitializedError
from ...core.base import AgentComponent
from .interfaces import PlanningInterface
from .models import (
    PlanningResult,
    PlanningValidation,
    PlanningExplanation,
    Plan
)
from ....flows.registry import flow_registry
from ...models.state import AgentState
from ...models.config import PlannerConfig
from ....utils.pydantic.schema import model_to_simple_json_schema
from ....utils.formatting.conversation import format_execution_history

from .todo_generation import TodoGenerationFlow, TodoGenerationInput, TodoGenerationOutput
from .todo import TodoItem, TodoManager

logger = logging.getLogger(__name__)


class AgentPlanner(AgentComponent, PlanningInterface):
    """Enhanced planner that generates both flow selection and TODO management.
    
    This planner can operate in two modes:
    1. Single-cycle mode: Traditional planning for immediate execution
    2. TODO-driven mode: Multi-step planning with TODO generation for complex tasks
    """
    
    def __init__(self, config, name: str = "agent_planner", activity_stream=None):
        """Initialize the agent planner."""
        super().__init__(name)
        self.config = config
        self._activity_stream = activity_stream
        self.todo_generation_flow = TodoGenerationFlow()
        self.complexity_threshold = 3  # Steps threshold for TODO generation
        
        # Initialize planning templates and providers (will be set during initialization)
        self._planning_template = None
        self._input_generation_template = None
        self.llm_provider = None
    
    async def _initialize_impl(self) -> None:
        """Initialize the planning component."""
        # Component initialization - no parent dependencies needed
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the planning component."""
        # Clean up resources
        pass
    
    async def plan(
        self,
        context: AgentState
    ) -> PlanningResult:
        """Generate a plan based on the current context and available flows.
        
        Args:
            context: Current agent state
            
        Returns:
            PlanningResult containing the selected flow and inputs
            
        Raises:
            PlanningError: If planning fails
            NotInitializedError: If the planner is not properly initialized
        """
        try:
            self._logger.info(f"Planning with context task_id={context.task_id}")
            return await self._plan_impl(context)
        except Exception as e:
            self._logger.error(f"Failed to generate plan: {str(e)}")
            raise PlanningError(f"Failed to generate plan: {str(e)}") from e
    
    async def validate_plan(
        self,
        plan: PlanningResult
    ) -> PlanningValidation:
        """Validate a generated plan against available flows.
        
        Args:
            plan: The plan to validate
            
        Returns:
            PlanningValidation indicating if the plan is valid
            
        Raises:
            PlanningError: If validation fails
        """
        try:
            self._logger.info(f"Validating plan for flow '{plan.selected_flow}'")
            return await self._validate_plan_impl(plan)
        except Exception as e:
            self._logger.error(f"Failed to validate plan: {str(e)}")
            raise PlanningError(f"Failed to validate plan: {str(e)}") from e
    
    async def generate_inputs(
        self,
        state: AgentState,
        flow_name: str,
        planning_result: PlanningResult,
        memory_context: str,
        flow: Optional[Any] = None
    ) -> StrictBaseModel:
        """Generate inputs for a specific flow step based on its intent and rationale.
        
        Args:
            state: Agent state
            flow_name: Name of the flow to execute for this step
            planning_result: The planning result containing rationale
            memory_context: Context ID (e.g., task_id) to use for memory retrieval
            flow: Optional flow instance (currently unused)
            
        Returns:
            Input model instance for the flow
        """
        if not self._input_generation_template:
            self._logger.error("Input generation template not set")
            raise NotInitializedError("Input generation template not set")
        
        # Get flow metadata to determine input model
        flow_metadata = flow_registry.get_flow_metadata(flow_name)
        if not flow_metadata:
            raise PlanningError(f"Flow '{flow_name}' has no metadata in registry")
        
        input_model = flow_metadata.input_model
        if not input_model:
            raise PlanningError(f"Flow '{flow_name}' has no input model defined")
            
        # Get flow description
        flow_description = flow_metadata.description
        
        # Get input schema using the model's schema method
        input_schema = ""
        try:
            # Use the improved schema utility
            input_schema = model_to_simple_json_schema(input_model)
        except Exception as e:
            self._logger.warning(f"Failed to get schema for {input_model.__name__}: {str(e)}")
            # Fallback to string representation
            input_schema = str(input_model.__name__)
            
        # Format execution history
        execution_history_text = format_execution_history(state.execution_history)
        
        # Extract rationale from planning result
        planning_rationale = planning_result.reasoning.rationale
        
        # Get task description directly from AgentState
        task_description = state.task_description
            
        # Get relevant memories using the provided memory_context
        memory_context_summary = "No relevant memories available."
        if memory_context and self._registry:
            memory_manager = self.get_component("memory_manager")
            if memory_manager:
                try:
                    from ..memory.interfaces import MemoryInterface
                    memory: MemoryInterface = memory_manager
                    
                    relevant_memories = await memory.retrieve_relevant(
                        query=task_description, # Query based on task description
                        context=memory_context, # Use the passed context
                        limit=5
                    )
                    
                    if relevant_memories:
                        memory_context_summary = "Relevant Memories Found:\n" + "\n".join(
                            [f"- {memory}" for memory in relevant_memories]
                        )
                except Exception as e:
                    self._logger.warning(f"Error retrieving relevant memories during input generation: {str(e)}")
                    memory_context_summary = "Error retrieving memories."
        
        # Prepare variables for prompt
        prompt_variables = {
            "task_description": task_description,
            "flow_name": flow_name,
            "flow_description": flow_description,
            "input_schema": input_schema,
            "planning_rationale": planning_rationale,
            "step_intent": planning_result.reasoning.explanation,
            "execution_history_text": execution_history_text,
            "memory_context_summary": memory_context_summary
        }
        
        # Generate structured input
        inputs = await self.llm_provider.generate_structured(
            prompt=self._input_generation_template,
            output_type=input_model,
            model_name=self.config.model_name,
            prompt_variables=prompt_variables
        )
        
        # Special handling for conversation flows: inject agent persona
        if flow_name == "conversation" and hasattr(inputs, 'persona'):
            # Override the LLM-generated persona with the agent's actual persona
            config_manager = self.get_component("config_manager")
            if config_manager and hasattr(config_manager, 'config'):
                inputs.persona = config_manager.config.persona
                self._logger.debug(f"Injected agent persona into conversation flow input: {inputs.persona[:50]}...")
        
        self._logger.info(f"Generated inputs for flow '{flow_name}': {str(inputs.model_dump())[:100]}...")
        return inputs
    
    async def plan_with_todos(
        self,
        state: AgentState,
        available_flows: Dict[str, Any],
        memory_context: str,
        auto_generate_todos: bool = True
    ) -> Tuple[PlanningResult, Optional[List[TodoItem]]]:
        """Plan execution and optionally generate TODOs for complex tasks.
        
        Args:
            state: Current agent state
            available_flows: Available flows for execution
            memory_context: Memory context for planning
            auto_generate_todos: Whether to auto-generate TODOs for complex tasks
            
        Returns:
            Tuple of (planning_result, optional_todos)
        """
        # Generate multi-step plan using core planner - get raw Plan object
        raw_plan = await self._get_raw_plan_from_impl(state)
        
        # Determine if task needs TODO decomposition
        should_generate_todos = (
            auto_generate_todos and 
            raw_plan.steps and
            len(raw_plan.steps) >= self.complexity_threshold and
            self._is_complex_task(state.task_description, raw_plan)
        )
        
        todos = None
        if should_generate_todos:
            self._logger.info(f"Complex task detected ({len(raw_plan.steps)} steps). Generating TODOs...")
            
            # Stream TODO generation start
            if self._activity_stream:
                self._activity_stream.planning(f"Complex task detected ({len(raw_plan.steps)} steps) - generating TODOs")
            
            # Generate TODOs using the todo-generation flow
            try:
                todos = await self._generate_todos_from_plan(raw_plan, state)
                self._logger.info(f"Generated {len(todos)} TODO items for complex task")
                
                # Stream TODO generation success
                if self._activity_stream:
                    self._activity_stream.planning(f"Generated {len(todos)} TODO items for systematic execution")
                    for todo in todos:
                        self._activity_stream.todo_create(todo.content, todo.priority.value)
                        
            except Exception as e:
                self._logger.warning(f"Failed to generate TODOs: {e}. Falling back to single-cycle execution.")
                if self._activity_stream:
                    self._activity_stream.error(f"TODO generation failed: {str(e)} - using single-cycle execution")
                todos = None
        
        # Convert multi-step plan to single-step result for immediate execution
        planning_result = self._convert_plan_to_planning_result(raw_plan)
        
        return planning_result, todos
    
    async def _generate_todos_from_plan(
        self, 
        plan: Plan, 
        state: AgentState
    ) -> List[TodoItem]:
        """Generate TODO items from a multi-step plan.
        
        Args:
            plan: The multi-step plan to convert
            state: Current agent state for context
            
        Returns:
            List of generated TODO items
        """
        # Prepare input for TODO generation flow
        todo_input = TodoGenerationInput(
            plan=plan,
            task_description=state.task_description,
            context={
                "task_id": state.task_id,
                "execution_history": [str(result) for result in state.execution_history[-3:]],  # Last 3 results
                "current_cycle": getattr(state, 'current_cycle', 0)
            }
        )
        
        # Execute TODO generation flow
        todo_result = await self.todo_generation_flow.run_pipeline(todo_input)
        
        return todo_result.todos
    
    def _is_complex_task(self, task_description: str, plan: Plan) -> bool:
        """Determine if a task is complex enough to warrant TODO decomposition.
        
        Args:
            task_description: The original task description
            plan: The generated plan
            
        Returns:
            True if task is complex and should use TODO-driven execution
        """
        # Simple heuristics for complexity detection
        complexity_indicators = [
            len(plan.steps) >= self.complexity_threshold,
            len(task_description.split()) > 20,  # Long descriptions
            any(keyword in task_description.lower() for keyword in [
                'analyze', 'create', 'implement', 'develop', 'build', 'design',
                'refactor', 'optimize', 'integrate', 'setup', 'configure'
            ]),
            any(step.rationale and len(step.rationale) > 100 for step in plan.steps)  # Complex rationales
        ]
        
        # Task is complex if multiple indicators are true
        return sum(complexity_indicators) >= 2
    
    def _convert_plan_to_planning_result(self, plan: Plan) -> PlanningResult:
        """Convert a multi-step Plan to a single-step PlanningResult.
        
        This maintains compatibility with the existing execution engine
        while still capturing the multi-step nature in the reasoning.
        """
        if not plan.steps:
            from .models import PlanningExplanation
            return PlanningResult(
                selected_flow="none",
                reasoning=PlanningExplanation(
                    explanation="No steps were generated in the plan",
                    rationale="Task may be complete or no suitable flows found",
                    decision_factors=[]
                )
            )
        
        # Take the first step for immediate execution
        first_step = plan.steps[0]
        
        from .models import PlanningExplanation
        return PlanningResult(
            selected_flow=first_step.flow_name,
            reasoning=PlanningExplanation(
                explanation=first_step.rationale,
                rationale=plan.overall_rationale or first_step.step_intent,
                decision_factors=[step.step_intent for step in plan.steps]
            )
        )
    
    async def determine_execution_strategy(self, task_description: str) -> str:
        """Determine the best execution strategy for a task.
        
        Args:
            task_description: The task to analyze
            
        Returns:
            Execution strategy: 'single_cycle' or 'todo_driven'
        """
        # Quick analysis based on task characteristics
        complex_keywords = [
            'analyze', 'create', 'implement', 'develop', 'build', 'design',
            'refactor', 'optimize', 'integrate', 'setup', 'configure', 'plan',
            'organize', 'structure', 'comprehensive', 'detailed', 'multiple'
        ]
        
        task_lower = task_description.lower()
        complexity_score = sum(1 for keyword in complex_keywords if keyword in task_lower)
        
        # Use TODO-driven approach for complex tasks
        if complexity_score >= 2 or len(task_description.split()) > 20:
            return 'todo_driven'
        else:
            return 'single_cycle'
    
    async def _generate_raw_plan(self, state: AgentState) -> Plan:
        """Generate raw Plan object directly from LLM without conversion to PlanningResult.
        
        This method duplicates the core logic from base._plan_impl but returns
        the raw Plan object instead of converting it to PlanningResult.
        
        Args:
            state: Current agent state
            
        Returns:
            Raw Plan object with steps
        """
        # This is essentially the core of base._plan_impl but without the conversion
        # We duplicate the essential parts to get the raw Plan
        
        # Import here to avoid circular imports  
        from ...flows.registry import flow_registry
        from ..core.errors import PlanningError
        
        # Get flows from registry
        if not flow_registry:
            raise PlanningError("No flow registry available")
        
        flow_instances = flow_registry.get_agent_selectable_flows()
        if not flow_instances:
            raise PlanningError("No agent-selectable flows available for planning")
        
        # Build available flows text (simplified version)
        available_flows_text = ""
        for flow_name, flow_instance in flow_instances.items():
            flow_metadata = flow_registry.get_flow_metadata(flow_name)
            if flow_metadata:
                description = flow_metadata.description
                available_flows_text += f"- {flow_name}: {description}\n"
        
        # Prepare variables for prompt
        prompt_variables = {
            "task_description": state.task_description,
            "available_flows_text": available_flows_text,
            "execution_history_text": "",  # Simplified
            "memory_context_summary": "No relevant memories found.",  # Simplified
            "cycle": getattr(state, 'cycles', 0)
        }
        
        # Get model name
        model_name = getattr(self.config, "model_name", "agent-model-large")
        
        # Generate structured planning response - return raw Plan
        from .models import Plan
        plan_result: Plan = await self.llm_provider.generate_structured(
            prompt=self._planning_template,
            output_type=Plan,
            prompt_variables=prompt_variables,
            model_name=model_name
        )
        
        if plan_result is None:
            raise PlanningError("LLM returned None for the plan")
        
        if not isinstance(plan_result, Plan):
            raise PlanningError(f"LLM did not return a valid Plan object, got {type(plan_result)}")
        
        # Validate each step in the plan (simplified)
        for step in plan_result.steps:
            if step.flow_name not in flow_instances:
                raise PlanningError(f"Selected flow '{step.flow_name}' in plan step '{step.step_id}' not found in registry or is not agent-selectable.")
        
        return plan_result
    
    async def _plan_impl(
        self,
        context: AgentState
    ) -> PlanningResult:
        """Generate a multi-step plan based on the current context using LLM.
        
        This implementation requires:
        - A configured model provider (self.llm_provider)
        - Access to flow_registry for flow information
        - A planning template (_planning_template)
        
        Args:
            context: Current agent state
            
        Returns:
            PlanningResult with selected flow and reasoning
            
        Raises:
            PlanningError: If planning fails or no flows are available
            NotInitializedError: If required components are not initialized
        """
        # Stream planning start
        if self._activity_stream:
            self._activity_stream.planning(f"Starting planning for task: {context.task_description[:50]}...", cycle=context.cycles)
        
        # Get flows directly from flow_registry
        if not flow_registry:
            raise PlanningError("No flow registry available")
        
        # Get only agent-selectable flows (non-infrastructure)
        flow_instances = flow_registry.get_agent_selectable_flows()
        if not flow_instances:
            raise PlanningError("No agent-selectable flows available for planning")
        
        # Stream available flows
        if self._activity_stream:
            self._activity_stream.context(f"Found {len(flow_instances)} available flows", flows=list(flow_instances.keys()))
        
        # Format available flows text
        available_flows_text = ""
        for flow_name, flow_instance in flow_instances.items():
            # Get flow metadata
            flow_metadata = flow_registry.get_flow_metadata(flow_name)
            if not flow_metadata:
                raise PlanningError(f"Flow '{flow_name}' has no metadata in registry. Flows must be properly registered with metadata.")
            
            # Use the description from metadata directly
            description = flow_metadata.description
            available_flows_text += f"- {flow_name}: {description}\n"
        
        # Format execution history text
        execution_history_text = ""
        if context.user_messages:
            for i, user_msg in enumerate(context.user_messages):
                execution_history_text += f"User {i+1}: {user_msg}\n"
                # Add corresponding system message if available
                if i < len(context.system_messages):
                    execution_history_text += f"System {i+1}: {context.system_messages[i]}\n"
        
        # Get relevant memories and format as summary
        memory_context_summary = "No relevant memories found."
        memory_manager = self.get_component("memory_manager")
        if memory_manager:
            try:
                # Stream memory retrieval
                if self._activity_stream:
                    self._activity_stream.memory_retrieval(context.task_description, context=f"task_{context.task_id}")
                
                # Use the task context for scoping
                task_context = f"task_{context.task_id}"
                relevant_memories = await memory_manager.retrieve_relevant(
                    query=context.task_description,
                    context=task_context, # Use full task context name
                    limit=5
                )
                if relevant_memories:
                    memory_context_summary = "Relevant Memories Found:\n" + "\n".join(
                        [f"- {memory}" for memory in relevant_memories]
                    )
                    # Stream found memories
                    if self._activity_stream:
                        self._activity_stream.memory_retrieval(context.task_description, results=relevant_memories)
            except Exception as e:
                self._logger.warning(f"Error retrieving relevant memories during planning: {str(e)}")
                memory_context_summary = "Error retrieving memories."
                if self._activity_stream:
                    self._activity_stream.error(f"Memory retrieval failed: {str(e)}")
        
        # Prepare variables for prompt
        prompt_variables = {
            "task_description": context.task_description,
            "available_flows_text": available_flows_text,
            "execution_history_text": execution_history_text,
            "memory_context_summary": memory_context_summary, # Use the new key
            "cycle": context.cycles
        }
        
        # Assume config and model_name exist, or handle missing config appropriately
        model_name = getattr(self.config, "model_name", "agent-model-large")  # Use large model for complex planning
        
        # Stream prompt selection
        if self._activity_stream:
            self._activity_stream.prompt_selection("planning_default", prompt_variables)
            self._activity_stream.llm_call(model_name, f"Planning for: {context.task_description[:50]}...")
        
        # Generate structured planning response expecting a Plan object
        plan_result: Plan = await self.llm_provider.generate_structured(
            prompt=self._planning_template,
            output_type=Plan, # Expect the Plan model
            prompt_variables=prompt_variables,
            model_name=model_name
        )
        
        if plan_result is None:
            raise PlanningError("LLM returned None for the plan")
        
        # Optional: Add basic validation for the generated plan
        if not isinstance(plan_result, Plan):
            raise PlanningError(f"LLM did not return a valid Plan object, got {type(plan_result)}")
        
        # Validate each step in the plan
        for step in plan_result.steps:
            if step.flow_name not in flow_instances:
                raise PlanningError(f"Selected flow '{step.flow_name}' in plan step '{step.step_id}' not found in registry or is not agent-selectable.")
            # TODO: Add input validation for step.flow_inputs against the flow's schema? (More complex)
        
        # Convert Plan to PlanningResult by selecting the first step
        # In the future, we might want to handle multi-step plans differently
        if plan_result.steps:
            first_step = plan_result.steps[0]
            planning_result = PlanningResult(
                selected_flow=first_step.flow_name,
                reasoning=PlanningExplanation(
                    explanation=first_step.rationale,
                    rationale=plan_result.overall_rationale or first_step.step_intent,
                    decision_factors=[step.step_intent for step in plan_result.steps]
                )
            )
            
            # Stream decision
            if self._activity_stream:
                self._activity_stream.flow_selection(
                    first_step.flow_name,
                    first_step.rationale,
                    [step.flow_name for step in plan_result.steps[1:]] if len(plan_result.steps) > 1 else None
                )
                self._activity_stream.decision(
                    f"Execute flow: {first_step.flow_name}",
                    plan_result.overall_rationale or first_step.rationale
                )
            
            return planning_result
        else:
            # No steps in the plan, return "none" as selected flow
            return PlanningResult(
                selected_flow="none",
                reasoning=PlanningExplanation(
                    explanation="No action required for this task",
                    rationale=plan_result.overall_rationale or "Task does not require any flow execution",
                    decision_factors=[]
                )
            )
    
    async def _validate_plan_impl(
        self,
        plan: PlanningResult
    ) -> PlanningValidation:
        """Validate a generated multi-step plan.
        
        Args:
            plan: The plan to validate
            
        Returns:
            PlanningValidation indicating if the plan is valid
            
        Raises:
            PlanningError: If validation fails
        """
        errors = []
        
        # Check if selected flow exists in flow_registry and is selectable by the agent
        if not flow_registry:
            errors.append("Flow registry is not available for validation")
            return PlanningValidation(is_valid=False, errors=errors)
        
        # For PlanningResult, we validate the selected_flow field
        selected_flow = plan.selected_flow
        selectable_flows = flow_registry.get_agent_selectable_flows()
        
        # Validate the selected flow
        if selected_flow and selected_flow != "none":
            if selected_flow not in selectable_flows:
                # Check if it exists at all in the registry to give a better error
                if selected_flow in flow_registry.get_flow_instances():
                    errors.append(f"Selected flow '{selected_flow}' is an infrastructure flow and cannot be directly used by the agent")
                else:
                    errors.append(f"Selected flow '{selected_flow}' not found in registry")
        # "none" or empty flow is valid (indicates no action needed)
        
        # Create validation result
        validation = PlanningValidation(
            is_valid=len(errors) == 0,
            errors=errors
        )
        
        return validation
    
    async def _get_raw_plan_from_impl(self, state: AgentState) -> Plan:
        """Extract raw Plan object for TODO generation.
        
        This method uses the same logic as _plan_impl but returns the raw Plan
        before conversion to PlanningResult.
        
        Args:
            state: Current agent state
            
        Returns:
            Raw Plan object with steps
        """
        # Get flows directly from flow_registry
        if not flow_registry:
            raise PlanningError("No flow registry available")
        
        flow_instances = flow_registry.get_agent_selectable_flows()
        if not flow_instances:
            raise PlanningError("No agent-selectable flows available for planning")
        
        # Format available flows text (same as in _plan_impl)
        available_flows_text = ""
        for flow_name, flow_instance in flow_instances.items():
            flow_metadata = flow_registry.get_flow_metadata(flow_name)
            if not flow_metadata:
                raise PlanningError(f"Flow '{flow_name}' has no metadata in registry. Flows must be properly registered with metadata.")
            description = flow_metadata.description
            available_flows_text += f"- {flow_name}: {description}\n"
        
        # Simplified prompt variables for TODO generation
        prompt_variables = {
            "task_description": state.task_description,
            "available_flows_text": available_flows_text,
            "execution_history_text": "",  # Simplified for TODO mode
            "memory_context_summary": "No relevant memories found.",  # Simplified for TODO mode
            "cycle": getattr(state, 'cycles', 0)
        }
        
        model_name = getattr(self.config, "model_name", "agent-model-large")
        
        # Generate structured planning response expecting a Plan object
        plan_result: Plan = await self.llm_provider.generate_structured(
            prompt=self._planning_template,
            output_type=Plan,
            prompt_variables=prompt_variables,
            model_name=model_name
        )
        
        if plan_result is None:
            raise PlanningError("LLM returned None for the plan")
        
        if not isinstance(plan_result, Plan):
            raise PlanningError(f"LLM did not return a valid Plan object, got {type(plan_result)}")
        
        # Validate each step in the plan
        for step in plan_result.steps:
            if step.flow_name not in flow_instances:
                raise PlanningError(f"Selected flow '{step.flow_name}' in plan step '{step.step_id}' not found in registry or is not agent-selectable.")
        
        return plan_result