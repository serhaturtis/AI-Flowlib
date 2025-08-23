from typing import List, Optional, Dict, Any
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel

from flowlib.flows.models.results import FlowResult


class ReflectionResult(StrictBaseModel):
    """Model for structured reflection results."""
    # Inherits strict configuration from StrictBaseModel
    
    reflection: str = Field(description="A detailed analysis of what happened and why")
    progress: int = Field(default=0, description="An estimate of overall task progress (0-100)")
    is_complete: bool = Field(default=False, description="Whether the task is complete")
    completion_reason: Optional[str] = Field(default=None, description="If is_complete is true, the reason the task is complete")
    insights: Optional[List[str]] = Field(default=None, description="Key insights or lessons learned from this execution")


class StepReflectionResult(StrictBaseModel):
    """Model for structured reflection results after a single plan step."""
    # Inherits strict configuration from StrictBaseModel
    
    step_id: str = Field(description="ID of the plan step being reflected upon")
    reflection: str = Field(description="A brief analysis of the step's outcome")
    step_success: bool = Field(description="Whether the step itself succeeded")
    key_observation: Optional[str] = Field(None, description="Most important observation from this step")


class ReflectionInput(StrictBaseModel):
    """Standardized input model for reflection process."""
    # Inherits strict configuration from StrictBaseModel
    
    task_description: str = Field(description="Description of the overall task")
    flow_name: str = Field(description="Name of the executed flow")
    flow_status: str = Field(description="Status of the flow execution")
    flow_result: FlowResult = Field(description="Result from the flow execution as a FlowResult model")
    flow_inputs: Optional[StrictBaseModel] = Field(None, description="Inputs provided to the flow as a Pydantic model, if applicable")
    state_summary: str = Field(description="Summary of current state")
    execution_history_text: str = Field(description="Formatted execution history")
    planning_rationale: str = Field(description="Rationale from the planning phase")
    cycle: int = Field(description="Current execution cycle number")
    progress: int = Field(default=0, description="Current progress percentage (0-100)")
    memory_context: Optional[str] = Field(default=None, description="Memory context for this reflection")


class StepReflectionInput(StrictBaseModel):
    """Input model for reflecting on a single plan step."""
    # Inherits strict configuration from StrictBaseModel
    
    task_description: str = Field(description="Description of the overall task")
    step_id: str = Field(description="ID of the plan step being reflected upon")
    step_intent: str = Field(description="The intent defined for this step")
    step_rationale: str = Field(description="The rationale defined for this step")
    flow_name: str = Field(description="Name of the executed flow for the step")
    flow_inputs: StrictBaseModel = Field(description="Inputs provided to the flow") # Use StrictBaseModel for consistency
    flow_result: FlowResult = Field(description="Result from the flow execution")
    current_progress: int = Field(description="Agent progress before this step's reflection")


# --- Model for Overall Plan Reflection Context ---

class PlanReflectionContext(StrictBaseModel):
    """Context provided for reflecting on the outcome of an entire plan."""
    # Inherits strict configuration from StrictBaseModel
    task_description: str = Field(description="Description of the overall task")
    plan_status: str = Field(description="The final status of the plan execution (e.g., SUCCESS, ERROR)")
    plan_error: Optional[str] = Field(None, description="Error message if the plan execution failed overall")
    step_reflections: List[StepReflectionResult] = Field(description="List of reflections from individual steps")
    state_summary: str = Field(description="Summary of agent state before final reflection")
    execution_history_text: str = Field(description="Formatted execution history covering the plan")
    current_progress: int = Field(description="Agent progress before final reflection")