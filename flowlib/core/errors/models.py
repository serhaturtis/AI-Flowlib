"""Strict Pydantic models for error handling.

No fallbacks, no defaults, no optional fields unless explicitly required.
"""

from datetime import datetime
from typing import Any, Dict, List
from pydantic import Field, ConfigDict
from flowlib.core.models import StrictBaseModel


class ErrorContextData(StrictBaseModel):
    """Strict error context data model.
    
    All fields are required. No optionals, no defaults.
    """
    # Inherits strict configuration from StrictBaseModel
    flow_name: str = Field(..., description="Name of the flow where error occurred")
    error_type: str = Field(..., description="Type of error")
    error_location: str = Field(..., description="Location in code where error occurred")
    timestamp: datetime = Field(default_factory=datetime.now, description="When error occurred")
    
    # Additional context fields
    component: str = Field(..., description="Component that raised the error")
    operation: str = Field(..., description="Operation being performed")
    
    # Any additional data must be explicitly defined
    # Note: extra="forbid" is inherited from StrictBaseModel


class ValidationErrorDetail(StrictBaseModel):
    """Strict validation error detail."""
    # Inherits strict configuration from StrictBaseModel
    
    location: str = Field(..., description="Field or location of validation error")
    message: str = Field(..., description="Validation error message")
    error_type: str = Field(..., description="Type of validation error")


class ProviderErrorContext(StrictBaseModel):
    """Strict provider error context."""
    # Inherits strict configuration from StrictBaseModel
    
    provider_name: str = Field(..., description="Name of the provider")
    provider_type: str = Field(..., description="Type of provider")
    operation: str = Field(..., description="Operation that failed")
    retry_count: int = Field(..., description="Number of retries attempted")


class ResourceErrorContext(StrictBaseModel):
    """Strict resource error context."""
    # Inherits strict configuration from StrictBaseModel
    
    resource_id: str = Field(..., description="ID of the resource")
    resource_type: str = Field(..., description="Type of resource")
    operation: str = Field(..., description="Operation attempted on resource")


class StateErrorContext(StrictBaseModel):
    """Strict state error context."""
    # Inherits strict configuration from StrictBaseModel
    
    state_name: str = Field(..., description="Name of the state")
    state_type: str = Field(..., description="Type of state")
    transition_from: str = Field(..., description="Previous state")
    transition_to: str = Field(..., description="Target state")


class ConfigurationErrorContext(StrictBaseModel):
    """Strict configuration error context."""
    # Inherits strict configuration from StrictBaseModel
    
    config_key: str = Field(..., description="Configuration key that failed")
    config_section: str = Field(..., description="Configuration section")
    expected_type: str = Field(..., description="Expected type of configuration")
    actual_value: str = Field(..., description="Actual value provided")


class ExecutionErrorContext(StrictBaseModel):
    """Strict execution error context."""
    # Inherits strict configuration from StrictBaseModel
    
    flow_name: str = Field(..., description="Name of flow being executed")
    step_name: str = Field(..., description="Current step in execution")
    input_data: Dict[str, Any] = Field(..., description="Input data to the step")
    execution_id: str = Field(..., description="Unique execution ID")


class ErrorMetadata(StrictBaseModel):
    """Strict error metadata."""
    # Inherits strict configuration from StrictBaseModel
    
    error_id: str = Field(..., description="Unique error ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    severity: str = Field(..., description="Error severity level")
    category: str = Field(..., description="Error category")


class ErrorDetails(StrictBaseModel):
    """Complete error details with all required fields."""
    # Inherits strict configuration from StrictBaseModel
    
    metadata: ErrorMetadata = Field(..., description="Error metadata")
    message: str = Field(..., description="Error message")
    context: ErrorContextData = Field(..., description="Error context")
    traceback: str = Field(..., description="Stack trace")
    
    # Specific context based on error type
    validation_errors: List[ValidationErrorDetail] = Field(
        default_factory=list,
        description="Validation error details if applicable"
    )