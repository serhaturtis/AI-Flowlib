"""
Consolidated validation module using Pydantic as the foundation.

This module provides a unified, Pydantic-based approach to validation
that replaces the multiple validation mechanisms previously used.
"""

import inspect
from functools import wraps
from typing import Type, TypeVar, Dict, Any, Optional, get_type_hints, Callable, Literal, Union, get_origin, get_args
from pydantic import BaseModel, ValidationError as PydanticValidationError, create_model, Field

from flowlib.core.errors.errors import ValidationError, ErrorContext

T = TypeVar('T', bound=BaseModel)

def validate_data(
    data: Any,
    model_type: Type[T],
    location: str = "data",
    strict: bool = False
) -> T:
    """
    Validate data against a Pydantic model.
    
    Args:
        data: Data to validate
        model_type: Pydantic model class
        location: Location identifier for error reporting
        strict: Whether to use strict validation
        
    Returns:
        Validated model instance
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        if isinstance(data, model_type):
            return data
        
        if strict:
            return model_type.model_validate(data, strict=True)
        return model_type.model_validate(data)
            
    except PydanticValidationError as e:
        # Convert Pydantic validation errors to framework format
        from flowlib.core.errors.models import ValidationErrorDetail
        validation_errors = []
        for error in e.errors():
            # Format the location path
            loc_path = ".".join(str(loc) for loc in error["loc"])
            validation_errors.append(
                ValidationErrorDetail(
                    location=f"{location}.{loc_path}" if loc_path else location,
                    message=error["msg"],
                    error_type=error["type"]
                )
            )
        
        # Create and raise our validation error
        raise ValidationError(
            message=f"Validation failed for {model_type.__name__}",
            validation_errors=validation_errors,
            context=ErrorContext.create(
                flow_name="validation",
                error_type="validation",
                error_location=f"validation.{model_type.__name__}",
                component="data_validator",
                operation="validate_data"
            ),
            cause=e
        )

def create_dynamic_model(
    schema: Dict[str, Any],
    model_name: str = "DynamicModel"
) -> Type[BaseModel]:
    """
    Create a Pydantic model dynamically from a schema-like dictionary.
    
    Args:
        schema: Schema dictionary defining fields
        model_name: Name for the dynamic model
        
    Returns:
        Pydantic model class
    """
    field_definitions = {}
    
    # Schema must have properties to create a model
    if "properties" not in schema:
        raise ValueError("Schema must contain 'properties' field")
    
    properties = schema["properties"]
    required_fields = schema["required"] if "required" in schema else []
    
    for field_name, field_def in properties.items():
        # Determine if field is required
        is_required = field_name in required_fields
        
        # Handle nested objects recursively
        if ("type" in field_def and field_def["type"] == "object") and "properties" in field_def:
            # Create a nested model for this object
            nested_model = create_dynamic_model(field_def, f"{model_name}_{field_name}")
            field_type = nested_model
        else:
            # Get field type
            field_type = _convert_type(field_def["type"] if "type" in field_def else "any")
            
            # Handle enum specially in Pydantic v2
            if "enum" in field_def:
                enum_values = field_def["enum"]
                # Create Literal type dynamically from enum values
                import typing
                if hasattr(typing, '_GenericAlias'):
                    # For Python 3.9+
                    field_type = typing._GenericAlias(Literal, tuple(enum_values))
                else:
                    # Fallback: use Union type 
                    field_type = Union[tuple(enum_values)]
        
        # Extract constraints for Field (excluding enum)
        field_constraints = _extract_constraints(field_def)
        if "enum" in field_constraints:
            del field_constraints["enum"]  # Remove deprecated enum parameter
        
        # Set default if field is not required
        if not is_required:
            field_definitions[field_name] = (Optional[field_type], Field(default=None, **field_constraints))
        else:
            field_definitions[field_name] = (field_type, Field(**field_constraints))
    
    return create_model(model_name, **field_definitions)

def validate_with_schema(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    location: str = "data"
) -> Dict[str, Any]:
    """
    Validate data using a schema dictionary by converting it to a Pydantic model.
    
    Args:
        data: Data to validate
        schema: Schema dictionary
        location: Location identifier for error reporting
        
    Returns:
        Validated data
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Basic schema validation
        if not isinstance(schema, dict):
            raise ValueError("Schema must be a dictionary")
        
        # Check for required schema structure
        if "properties" not in schema and "type" not in schema:
            raise ValueError("Schema must contain 'properties' or 'type' field")
        
        # Create a dynamic model from the schema
        model = create_dynamic_model(schema)
        
        # Validate the data using the model
        validated = validate_data(data, model, location)
        
        # Return as dictionary
        return validated.model_dump()
    except Exception as e:
        if isinstance(e, ValidationError):
            # Re-raise validation errors
            raise
        
        # Handle other errors
        raise ValidationError(
            message="Schema validation failed",
            validation_errors=[],
            context=ErrorContext.create(
                flow_name="validation",
                error_type="schema_validation",
                error_location=f"schema_validation.{location}",
                component="schema_validator",
                operation="validate_with_schema"
            ),
            cause=e
        )

def validate_function(func: Callable) -> Callable:
    """
    Decorator to validate function arguments and return values using Pydantic models.
    
    This decorator validates:
    1. Any argument annotated with a Pydantic model
    2. The return value if it's annotated with a Pydantic model
    
    Args:
        func: Function to validate
        
    Returns:
        Decorated function with validation
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind arguments to parameters
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Validate arguments
        for param_name, param_value in bound_args.arguments.items():
            param_type = hints[param_name] if param_name in hints else None
            
            # Skip validation for parameters without type hints
            if not param_type:
                continue
                
            # Handle Optional types (Union[T, None])
            model_type = None
            if get_origin(param_type) is Union:
                # Check if it's Optional[SomeModel] 
                args = get_args(param_type)
                if len(args) == 2 and type(None) in args:
                    # It's Optional[T], extract T
                    model_type = next(arg for arg in args if arg is not type(None))
                    if param_value is None:
                        continue  # Skip validation for None values
            elif isinstance(param_type, type) and issubclass(param_type, BaseModel):
                model_type = param_type
            
            # Validate if parameter type is a Pydantic model
            if model_type and isinstance(model_type, type) and issubclass(model_type, BaseModel):
                try:
                    # Replace the argument with validated model
                    if not isinstance(param_value, model_type):
                        bound_args.arguments[param_name] = validate_data(
                            param_value, 
                            model_type,
                            location=f"args.{param_name}"
                        )
                except ValidationError as e:
                    # Re-raise with additional context in message
                    raise ValidationError(
                        message=f"Validation failed for parameter '{param_name}' in function '{func.__name__}': {e.message}",
                        validation_errors=e.validation_errors,
                        context=e.context,
                        cause=e.cause
                    )
        
        # Call the function with validated arguments
        result = func(*bound_args.args, **bound_args.kwargs)
        
        # Validate return value if needed
        return_type = hints['return'] if 'return' in hints else None
        if return_type and isinstance(return_type, type) and issubclass(return_type, BaseModel):
            try:
                return validate_data(result, return_type, location="return")
            except ValidationError as e:
                # Re-raise with additional context in message
                raise ValidationError(
                    message=f"Validation failed for return value in function '{func.__name__}': {e.message}",
                    validation_errors=e.validation_errors,
                    context=e.context,
                    cause=e.cause
                )
                
        return result
    
    return wrapper

# Helper functions for model creation
def _convert_type(type_name: str) -> Type:
    """Convert schema type to Python type."""
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
        "any": Any
    }
    if type_name not in type_mapping:
        return Any
    return type_mapping[type_name]

def _extract_constraints(field_def: Dict[str, Any]) -> Dict[str, Any]:
    """Extract validation constraints from field definition."""
    constraints = {}
    
    # Number validations
    if "minimum" in field_def:
        constraints["ge"] = field_def["minimum"]
    if "maximum" in field_def:
        constraints["le"] = field_def["maximum"]
    
    # String validations
    if "minLength" in field_def:
        constraints["min_length"] = field_def["minLength"]
    if "maxLength" in field_def:
        constraints["max_length"] = field_def["maxLength"]
    if "pattern" in field_def:
        constraints["pattern"] = field_def["pattern"]
    
    # Enum validation
    if "enum" in field_def:
        constraints["enum"] = field_def["enum"]
    
    # Title and description
    if "title" in field_def:
        constraints["title"] = field_def["title"]
    if "description" in field_def:
        constraints["description"] = field_def["description"]
        
    return constraints


# Compatibility functions for existing tests
class ValidationRule:
    """Simple validation rule for testing."""
    
    def __init__(self, field: str, validator: Callable[[Any], bool], message: str):
        self.field = field
        self.validator = validator
        self.message = message
    
    def validate(self, value: Any) -> bool:
        """Validate a value using this rule."""
        return self.validator(value)


def validate_input(func: Callable) -> Callable:
    """Decorator to validate function inputs using Pydantic models."""
    return validate_function(func)


def validate_output(func: Callable) -> Callable:
    """Decorator to validate function outputs using Pydantic models."""
    return validate_function(func)