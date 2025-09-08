"""
Planning models for the agent system.

This module defines the data models used in planning operations, including:
- Planning results
- Plan validation  
- Planning explanations
- TODO models and management (unified from todo.py and todo_generation/models.py)
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type
from enum import Enum
from uuid import uuid4
from pydantic import Field, ConfigDict, BeforeValidator, field_validator
from typing_extensions import Annotated
from flowlib.core.models import StrictBaseModel, MutableStrictBaseModel

# Import the original FlowMetadata to avoid duplication
from flowlib.flows.models.metadata import FlowMetadata

# Use FlowRegistry for flow operations
from flowlib.flows.registry.registry import FlowRegistry

# Import TODO models from single source of truth
from ..models import (
    TodoStatus, TodoPriority, TodoStatusSummary, TodoItem, TodoList,
    validate_todo_status, validate_todo_priority
)

# --- Imports needed for Plan/PlanStep ---
import uuid
from pydantic import Field
from typing import List, Dict, Any # Added Dict, Any for PlanStep.flow_inputs if we revert
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
    rationale: str = Field(None, description="Rationale for the decisions")
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

