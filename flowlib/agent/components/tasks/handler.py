"""
Task execution handler for the Agent Architecture.

This module provides a handler for complex tasks that require planning and reflection.
"""

import logging
from typing import Dict, Any, Optional
from pydantic import Field
from flowlib.core.models import StrictBaseModel
from datetime import datetime

from flowlib.agent.models.state import AgentState
from flowlib.agent.components.planning.planner import AgentPlanner 
from flowlib.agent.components.reflection.base import AgentReflection
from flowlib.flows.registry.registry import flow_registry
from flowlib.flows.models.results import FlowResult
from flowlib.core.context.context import Context

logger = logging.getLogger(__name__)


class TaskExecutionResult(StrictBaseModel):
    """Result of task execution."""
    
    status: str = Field(..., description="Execution status")
    message: Optional[str] = Field(None, description="Optional status message")
    flow_result: Optional[Any] = Field(None, description="Result from flow execution")
    reflection: Optional[str] = Field(None, description="Reflection on execution")
    execution_time_seconds: Optional[float] = Field(None, description="Time taken to execute")


class TaskExecutionHandler:
    """Handles complex tasks requiring planning and reflection"""
    
    def __init__(self, 
                planner: AgentPlanner,
                reflection: AgentReflection,
                memory_context: str = "agent"):
        """Initialize the task execution handler.
        
        Args:
            planner: The planner component for selecting flows
            reflection: The reflection component for analyzing results
            memory_context: The memory context to use
        """
        self.planner = planner
        self.reflection = reflection
        self.memory_context = memory_context
        
    async def execute_task(self, 
                          state: AgentState) -> TaskExecutionResult:
        """Execute a complete task execution cycle with planning and reflection.
        
        Args:
            state: The agent state to use
            
        Returns:
            TaskExecutionResult containing execution results
        """
        # Plan next action directly using AgentState
        plan_result = await self.planner.plan(context=state)
        
        # Check if a flow was selected
        if not plan_result.selected_flow or plan_result.selected_flow == "none":
            return TaskExecutionResult(
                status="NO_FLOW_SELECTED", 
                message="No appropriate flow selected"
            )
        
        # Generate inputs for the flow
        flow_inputs = await self.planner.generate_inputs(
            state=state,
            flow_name=plan_result.selected_flow,
            planning_result=plan_result.model_dump(),
            memory_context_id=state.task_id
        )
        
        # Get the flow from registry
        flow = flow_registry.get_flow(plan_result.selected_flow)
        if not flow:
            return TaskExecutionResult(
                status="FLOW_NOT_FOUND", 
                message=f"Flow '{plan_result.selected_flow}' not found"
            )
        
        # --- Parse input into the Flow's specific Pydantic model ---
        try:
            # Get the flow's input model
            InputModel = flow.get_pipeline_input_model()
            if not InputModel or not issubclass(InputModel, StrictBaseModel):
                logger.error(f"Flow '{flow.name}' does not define a valid Pydantic InputModel. Found: {InputModel}")
                raise TypeError(f"Flow '{flow.name}' does not define a valid Pydantic InputModel.")
            
            # Ensure flow_inputs is a dict before parsing
            if not isinstance(flow_inputs, dict):
                 # If planner returned a StrictBaseModel, convert it back to dict for parsing
                 # (or ideally, ensure planner always returns dict for this stage)
                 if isinstance(flow_inputs, StrictBaseModel):
                     flow_inputs_dict = flow_inputs.model_dump()
                 else:
                    raise TypeError(f"Planner generate_inputs returned unexpected type: {type(flow_inputs)}. Expected dict.")
            else:
                 flow_inputs_dict = flow_inputs

            # Parse the dictionary into the specific Pydantic model
            parsed_input = InputModel(**flow_inputs_dict)
        except Exception as e:
            # Handle Pydantic validation errors or other parsing issues
            logger.error(f"Failed to parse inputs for flow '{flow.name}': {e}", exc_info=True)
            return TaskExecutionResult(
                status="INPUT_PARSING_ERROR", 
                message=f"Failed to parse inputs: {e}"
            )
        # ------------------------------------------------------------

        # Create a Context object containing the Pydantic input model instance
        input_context = Context(data=parsed_input) 
        
        # Execute the flow
        result = await flow.execute(input_context)
        
        # Reflect on the results
        # Convert inputs to dict for reflection, as it might expect serializable data
        input_data_dict = flow_inputs if isinstance(flow_inputs, dict) else flow_inputs.model_dump()
        reflection_result = await self.reflection.reflect(
            state=state,
            flow_name=plan_result.selected_flow,
            flow_inputs=input_data_dict, # Pass dict form to reflection
            flow_result=result,
            memory_context=state.task_id
        )
        
        # Return comprehensive result
        return TaskExecutionResult(
            status="SUCCESS",
            message=f"Successfully executed flow: {plan_result.selected_flow}",
            flow_result=result,
            reflection=str(reflection_result) if reflection_result else None
        ) 