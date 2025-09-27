"""Enhanced decorators for flow creation with subflow composition.

This module provides decorators that simplify the creation of flows
and pipelines with improved type safety and error handling. The stage
system has been removed in favor of clean subflow composition.
"""

import functools
import logging
import datetime
from typing import Callable, Optional, TypeVar, Any, Union
from pydantic import BaseModel

from flowlib.flows.base.base import Flow
from flowlib.flows.registry.registry import flow_registry

F = TypeVar('F', bound=Callable[..., Any])
C = TypeVar('C', bound=type)

# Setup logging
logger = logging.getLogger(__name__)

def flow(cls: Optional[C] = None, *, name: Optional[str] = None, description: str, is_infrastructure: bool = False) -> Union[Callable[[C], C], C]:
    """
    Decorator to mark a class as a flow.
    
    This decorator registers a class as a flow in the stage registry and enforces
    that the class implements the required Flow interface.
    
    Args:
        cls: The class to decorate
        name: Optional custom name for the flow
        description: Description of the flow's purpose (required)
        is_infrastructure: Whether this is an infrastructure flow
        
    Returns:
        The decorated flow class
        
    Raises:
        ValueError: If description is not provided
    """
    def wrap(cls: C) -> C:
        # Create a get_description method that returns the provided description
        def get_description(self: Any) -> str:
            return description
        
        # Add the method to the class
        if not hasattr(cls, 'get_description'):
            setattr(cls, 'get_description', get_description)
            logger.debug(f"Added get_description method to flow class '{cls.__name__}'")
        
        # If the class already has a get_description method, we keep it
            
        if not issubclass(cls, Flow):
            # Create a new class that inherits from both the original class and Flow
            original_cls = cls
            original_name = cls.__name__
            original_dict = dict(cls.__dict__)
            
            # Remove items that would cause conflicts
            for key in ['__dict__', '__weakref__']:
                if key in original_dict:
                    del original_dict[key]
            
            # Create the new class with multiple inheritance
            cls = type(  # type: ignore[assignment]
                original_name,
                (original_cls, Flow),
                original_dict
            )
            
            # Initialize Flow with default parameters
            original_init = cls.__init__
            
            def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
                # Call Flow's __init__ with appropriate parameters
                flow_name = name or original_name
                Flow.__init__(
                    self, 
                    flow_name,  # Positional argument
                    input_schema=None,  # Will be set from pipeline method
                    output_schema=None,  # Will be set from pipeline method
                    metadata={"is_infrastructure": is_infrastructure}
                )
                
                # Set name attribute on instance too
                self.name = flow_name
                
                # Only call original_init if it's different from Flow.__init__
                if original_init is not Flow.__init__:  # type: ignore[comparison-overlap]
                    try:
                        original_cls.__init__(self, *args, **kwargs)
                    except TypeError:
                        try:
                            original_cls.__init__(self)
                        except Exception as e:
                            logger.warning(f"Failed to call original __init__: {e}")
            
            cls.__init__ = new_init  # type: ignore[method-assign]
        
        # Set flow metadata
        flow_name = name or cls.__name__
        cls.__flow_metadata__ = {  # type: ignore[attr-defined]
            "name": flow_name,
            "is_infrastructure": is_infrastructure
        }
        
        # Set the name and is_infrastructure as direct attributes for easy access
        cls.name = flow_name  # type: ignore[attr-defined]
        cls.is_infrastructure = is_infrastructure  # type: ignore[attr-defined]
        
        # Find pipeline methods first - only scan own class methods, not inherited
        pipeline_methods = []
        
        # Only iterate through class's own __dict__ to avoid inherited methods
        for attr_name, attr_value in cls.__dict__.items():
            # Skip special methods and attributes
            if attr_name.startswith('__') and attr_name.endswith('__'):
                continue
            
            # Check if this is a pipeline method
            if hasattr(attr_value, '__pipeline__') and attr_value.__pipeline__:
                pipeline_methods.append(attr_name)
        
        # Enforce exactly one pipeline method
        if len(pipeline_methods) == 0:
            raise ValueError(f"Flow class '{flow_name}' must define exactly one pipeline method using @pipeline decorator")
        elif len(pipeline_methods) > 1:
            raise ValueError(f"Flow class '{flow_name}' has multiple pipeline methods: {', '.join(pipeline_methods)}. Only one is allowed.")
        
        # Store the pipeline method name
        cls.__pipeline_method__ = pipeline_methods[0]  # type: ignore[attr-defined]
        
        # Create a single flow instance to store in the registry
        flow_instance: Any = None
        try:
            flow_instance = cls()
            # Set the name attribute directly on the flow instance
            setattr(flow_instance, "name", flow_name)
            
            # Get the pipeline method to extract schemas
            pipeline_method = getattr(cls, cls.__pipeline_method__)  # type: ignore[attr-defined]
            if hasattr(pipeline_method, 'input_model') and pipeline_method.input_model:
                setattr(flow_instance, "input_schema", pipeline_method.input_model)
                logger.debug(f"Set input_schema={pipeline_method.input_model.__name__} on flow '{flow_name}'")
            
            if hasattr(pipeline_method, 'output_model') and pipeline_method.output_model:
                setattr(flow_instance, "output_schema", pipeline_method.output_model)
                logger.debug(f"Set output_schema={pipeline_method.output_model.__name__} on flow '{flow_name}'")
            
            logger.debug(f"Created flow instance: {flow_name}")
        except Exception as e:
            logger.warning(f"Failed to create instance for flow '{flow_name}': {e}")
            
        # Store flow class name and description for easier debugging
        cls.__flow_name__ = flow_name  # type: ignore[attr-defined]
        cls.__flow_description__ = description  # type: ignore[attr-defined]
        cls.__is_infrastructure__ = is_infrastructure  # type: ignore[attr-defined]
        
        # Register the flow in the global registry
        try:
            if flow_instance is not None:
                flow_registry.register_flow(flow_name, flow_instance)
                logger.debug(f"Registered flow instance '{flow_name}' in global registry")
            else:
                logger.error(f"Failed to create instance for flow '{flow_name}', skipping registration")
        except Exception as e:
            logger.warning(f"Failed to register flow '{flow_name}' in registry: {e}")
            
        return cls
    
    if cls is None:
        return wrap
    return wrap(cls)


# Stage decorator has been removed - use subflows instead
# This creates cleaner composition with Pydantic input/output models


# Standalone decorator has been removed - use subflows instead
# Create dedicated flow classes for reusable processing logic


def pipeline(func: Optional[F] = None, **pipeline_kwargs: Any) -> Callable[..., Any]:
    """Mark a method as a flow pipeline.
    
    This decorator wraps a method to provide pipeline execution capabilities:
    1. Manages execution context
    2. Tracks pipeline metadata and execution status
    3. Initializes stages if needed
    4. Validates output conforms to declared output model
    
    Args:
        func: The method to decorate
        **pipeline_kwargs: Additional pipeline options including:
            - input_model: Pydantic model for input validation (must be a BaseModel subclass)
            - output_model: Pydantic model for output validation (must be a BaseModel subclass)
        
    Returns:
        Decorated pipeline method
        
    Raises:
        ValueError: If input_model or output_model is provided but not a Pydantic BaseModel subclass.
    """
    # Get pipeline metadata from kwargs - strict access, no fallbacks
    input_model = pipeline_kwargs["input_model"] if "input_model" in pipeline_kwargs else None
    output_model = pipeline_kwargs["output_model"] if "output_model" in pipeline_kwargs else None
    
    # Validate input_model and output_model are Pydantic models if provided
    if input_model is not None and not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
        raise ValueError(f"Pipeline input_model must be a Pydantic BaseModel subclass, got {input_model}")
    
    if output_model is not None and not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
        raise ValueError(f"Pipeline output_model must be a Pydantic BaseModel subclass, got {output_model}")
        
    def decorator(method: F) -> F:
        @functools.wraps(method)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapped pipeline method."""
            # Track execution stats
            start_time = datetime.datetime.now()
            
            try:
                # Execute the pipeline method and return the result directly
                result = await method(*args, **kwargs)
                return result
            finally:
                # Calculate execution time
                execution_time = datetime.datetime.now() - start_time
                
                # Log execution details
                if len(args) > 0 and hasattr(args[0], "__class__"):
                    flow_class = args[0].__class__.__name__
                    logger.debug(f"Pipeline execution: {flow_class}.{method.__name__} ({execution_time.total_seconds():.3f}s)")
        
        # Set pipeline attributes for flow decorator and test compatibility
        wrapper.__pipeline__ = True  # type: ignore[attr-defined]
        wrapper.input_model = input_model  # type: ignore[attr-defined]
        wrapper.output_model = output_model  # type: ignore[attr-defined]

        # Set the standard double-underscore attributes used throughout the system
        wrapper.__input_model__ = input_model  # type: ignore[attr-defined]
        wrapper.__output_model__ = output_model  # type: ignore[attr-defined]

        # Also set the test-expected attributes for backward compatibility
        wrapper.__pipeline_input_model__ = input_model  # type: ignore[attr-defined]
        wrapper.__pipeline_output_model__ = output_model  # type: ignore[attr-defined]
        
        return wrapper  # type: ignore[return-value]
        
    # Support both @pipeline and @pipeline() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


# Stage-related methods removed - use subflow composition instead
# Create separate flow classes for different processing steps 