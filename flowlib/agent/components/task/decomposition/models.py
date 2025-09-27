"""
Planning models for the agent system.

This module defines the data models used in planning operations, including:
- Planning results
- Plan validation  
- Planning explanations
- TODO models and management (unified from todo.py and todo_generation/models.py)
"""

from typing import List
from pydantic import Field
from flowlib.core.models import StrictBaseModel

# Import the original FlowMetadata to avoid duplication

# Use FlowRegistry for flow operations

# Import TODO models from single source of truth

# --- Imports needed for Plan/PlanStep ---
# ----------------------------------------

# --- Planning-Specific Models ---

class PlanningExplanation(StrictBaseModel):
    """Human-readable explanation of a plan.
    
    Attributes:
        explanation: Text explaining the planning decisions
        rationale: Optional rationale for the decisions
        decision_factors: Factors that influenced the decision
    """
    # Inherits strict configuration from StrictBaseModel
    
    explanation: str = Field(..., description="Text explaining the planning decisions")
    rationale: str = Field(default="", description="Rationale for the decisions")
    decision_factors: List[str] = Field(default_factory=list, description="Factors that influenced the decision")

class PlanningResult(StrictBaseModel):
    """Result of a planning operation.
    
    Attributes:
        selected_flow: Name of the selected flow
        inputs: Inputs for the selected flow
        metadata: Metadata about the planning decision
    """
    # Inherits strict configuration from StrictBaseModel
    
    selected_flow: str = Field(..., description="Name of the selected flow")
    reasoning: PlanningExplanation = Field(..., description="Reasoning behind the planning decision")

class PlanningValidation(StrictBaseModel):
    """Result of plan validation.
    
    Attributes:
        is_valid: Whether the plan is valid
        errors: List of validation errors if any
    """
    # Inherits strict configuration from StrictBaseModel
    
    is_valid: bool = Field(..., description="Whether the plan is valid")
    errors: List[str] = Field(default_factory=list, description="List of validation errors if any")


# Note: All TODO models (TodoStatus, TodoPriority, TodoItem, TodoList, etc.) 
# are imported from ..models following flowlib's single source of truth principle

