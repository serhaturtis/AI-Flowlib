"""Base flow implementation with enhanced execution model.

This module provides the foundation for all flow components with
enhanced result handling and error management.
"""

import inspect
from abc import ABC
from collections.abc import Callable
from datetime import datetime
from typing import (
    Any,
    Generic,
    TypeVar,
    Union,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator

from flowlib.core.context.context import Context
from flowlib.core.errors.errors import (
    BaseError,
    ErrorContext,
    ErrorManager,
    ExecutionError,
    ValidationError,
    default_manager,
)
from flowlib.core.errors.models import ValidationErrorDetail
from flowlib.flows.models.constants import FlowStatus
from flowlib.flows.models.models import FlowInstanceInfo, PipelineMethodInfo
from flowlib.flows.models.results import FlowResult

# Export FlowStatus for proper module access
__all__ = ["Flow", "FlowSettings", "FlowStatus", "FlowMetadataEntry", "StrictFlowMetadata"]


class FlowMetadataEntry(BaseModel):
    """Single metadata entry with strict typing."""
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True, strict=True)

    key: str = Field(description="Metadata key")
    value: Any = Field(description="Metadata value (any type)")


class StrictFlowMetadata(BaseModel):
    """Strict flow metadata model with no fallbacks."""
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True, strict=True)

    entries: list[FlowMetadataEntry] = Field(default_factory=list, description="Flow metadata entries")

    @classmethod
    def create(cls, metadata: dict[str, Any] | None = None) -> 'StrictFlowMetadata':
        """Create metadata from optional dict."""
        if metadata is None:
            return cls()

        entries = [
            FlowMetadataEntry(key=str(k), value=v)
            for k, v in metadata.items()
        ]
        return cls(entries=entries)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        return {entry.key: entry.value for entry in self.entries}


class StrictErrorManager(BaseModel):
    """Strict error manager model."""
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True, strict=True, arbitrary_types_allowed=True)

    manager: ErrorManager = Field(description="Error manager instance")

    @classmethod
    def create(cls, error_manager: ErrorManager | None = None) -> 'StrictErrorManager':
        """Create error manager wrapper."""
        if error_manager is None:
            return cls(manager=default_manager)
        return cls(manager=error_manager)


class PipelineMethodConfig(BaseModel):
    """Pipeline method configuration model."""
    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True, strict=True, arbitrary_types_allowed=True)

    method: Callable[..., Any] = Field(description="Pipeline method reference")
    name: str = Field(description="Method name")

    @classmethod
    def extract_from_instance(cls, instance: Any, method_info: PipelineMethodInfo) -> 'PipelineMethodConfig':
        """Extract pipeline method from instance with strict validation."""
        method_name = method_info.method_name

        if not hasattr(instance, method_name):
            raise ExecutionError(
                f"Instance {type(instance).__name__} missing required pipeline method '{method_name}'",
                ErrorContext.create(
                    flow_name="flow_validation",
                    error_type="MissingMethodError",
                    error_location="validate_pipeline_method",
                    component=type(instance).__name__,
                    operation="method_validation"
                )
            )

        method = getattr(instance, method_name)
        if not callable(method):
            raise ExecutionError(
                f"Pipeline method '{method_name}' on {type(instance).__name__} is not callable",
                ErrorContext.create(
                    flow_name="flow_validation",
                    error_type="NonCallableMethodError",
                    error_location="validate_pipeline_method",
                    component=type(instance).__name__,
                    operation="method_validation"
                )
            )

        return cls(method=method, name=method_name)

#T = TypeVar('T', bound='BaseModel')


class FlowSettings(BaseModel):
    """Settings for configuring flow execution behavior.

    This class provides:
    1. Timeout configuration for flow execution
    2. Retry behavior for handling transient errors
    3. Logging and debugging options
    4. Resource management settings
    """
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Execution settings
    timeout_seconds: float | None = None
    max_retries: int = 0
    retry_delay_seconds: float = 1.0

    # Validation settings
    validate_inputs: bool = True
    validate_outputs: bool = True

    # Logging settings
    log_level: str = "INFO"
    debug_mode: bool = False

    # Resource settings
    max_memory_mb: int | None = None
    max_cpu_percent: int | None = None

    # Advanced settings - using validated entries instead of Dict[str, Any]
    custom_settings: list[FlowMetadataEntry] = Field(default_factory=list, description="Custom flow settings")

    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @field_validator("max_retries")
    def validate_max_retries(cls, v: int) -> int:
        """Validate max retries."""
        if v < 0:
            raise ValueError("Max retries must be non-negative")
        return v

    @field_validator("retry_delay_seconds")
    def validate_retry_delay(cls, v: float) -> float:
        """Validate retry delay."""
        if v < 0:
            raise ValueError("Retry delay must be non-negative")
        return v

    def merge(self, other: Union['FlowSettings', dict[str, Any]]) -> 'FlowSettings':
        """Merge with another settings object or dictionary.

        Args:
            other: Settings object or dictionary to merge with

        Returns:
            New merged settings object
        """
        if isinstance(other, FlowSettings):
            # Convert to dictionary, excluding None values
            other_dict = other.model_dump(exclude_none=True)
        elif isinstance(other, dict):
            # Filter out None values from dict
            other_dict = {k: v for k, v in other.items() if v is not None}
        else:
            raise TypeError(f"Cannot merge with {type(other)}")

        # Create a copy of self as dictionary
        merged_dict = self.model_dump()

        # Update with other dict, handling custom_settings specially
        for key, value in other_dict.items():
            if key == "custom_settings" and value:
                # Handle custom_settings as list of entries
                existing_settings = merged_dict["custom_settings"] if "custom_settings" in merged_dict else []
                if isinstance(value, list):
                    # Merge lists of entries
                    merged_dict["custom_settings"] = existing_settings + value
                elif isinstance(value, dict):
                    # Convert dict to entries and merge
                    new_entries = [
                        {"key": str(k), "value": str(v)}
                        for k, v in value.items()
                    ]
                    merged_dict["custom_settings"] = existing_settings + new_entries
            else:
                merged_dict[key] = value

        return FlowSettings.model_validate(merged_dict)

    @classmethod
    def from_dict(cls, settings_dict: dict[str, Any]) -> 'FlowSettings':
        """Create settings from dictionary.

        Args:
            settings_dict: Dictionary of settings

        Returns:
            Settings object
        """
        return cls.model_validate(settings_dict)

    def update(self, **kwargs: Any) -> 'FlowSettings':
        """Create a new settings object with updated values.

        Args:
            **kwargs: New values to set

        Returns:
            New settings object
        """
        settings_dict = self.model_dump()
        settings_dict.update(kwargs)
        return FlowSettings.model_validate(settings_dict)

    def __str__(self) -> str:
        """String representation."""
        timeout_str = f"{self.timeout_seconds}s" if self.timeout_seconds else "None"
        return f"FlowSettings(timeout={timeout_str}, retries={self.max_retries}, debug={self.debug_mode})"

T = TypeVar('T')

class Flow(ABC, Generic[T]):
    """Base class for all flow components with enhanced result handling.

    This class provides:
    1. Consistent execution pattern with error handling
    2. Input and output validation with Pydantic models
    3. Attribute-based access to results
    """

    def __init__(
        self,
        name_or_instance: str | object,
        input_schema: type[BaseModel] | None = None,
        output_schema: type[BaseModel] | None = None,
        metadata: dict[str, Any] | None = None,
        error_manager: ErrorManager | None = None
    ):
        """Initialize flow.

        Args:
            name_or_instance: Either a name string or a flow class instance
            input_schema: Optional Pydantic model for input validation. Must be a Pydantic BaseModel subclass.
            output_schema: Optional Pydantic model for output validation. Must be a Pydantic BaseModel subclass.
            metadata: Optional metadata about the flow
            error_manager: Optional error manager instance

        Raises:
            ValueError: If input_schema or output_schema is provided but not a Pydantic BaseModel subclass.
        """

        if isinstance(name_or_instance, str):
            self.name = name_or_instance
            self.flow_instance = None
        else:
            # Handle when a flow class instance is passed in
            self.flow_instance = name_or_instance

            # Extract flow information strictly - no fallbacks
            flow_info = self._extract_flow_info_strict(name_or_instance)
            self.name = flow_info.name

            # Set pipeline method info with strict validation
            self.pipeline_method = None
            if flow_info.pipeline_method_info:
                # Extract pipeline method with strict validation
                method_config = PipelineMethodConfig.extract_from_instance(name_or_instance, flow_info.pipeline_method_info)
                self.pipeline_method = method_config.method

                # Override schemas if not provided and pipeline has them
                if not input_schema:
                    input_schema = flow_info.pipeline_method_info.input_model
                if not output_schema:
                    output_schema = flow_info.pipeline_method_info.output_model

        self.input_schema = input_schema
        self.output_schema = output_schema
        # Use strict models - no fallbacks
        metadata_config = StrictFlowMetadata.create(metadata)
        self.metadata = metadata_config.to_dict()
        # Use strict error manager
        error_config = StrictErrorManager.create(error_manager)
        self.error_manager = error_config.manager

    def _extract_flow_info_strict(self, flow_instance: object) -> FlowInstanceInfo:
        """Extract flow information without fallbacks.

        Args:
            flow_instance: Flow instance to inspect

        Returns:
            FlowInstanceInfo with complete information

        Raises:
            ValueError: If required information is missing
        """
        # Get flow name - explicit attribute check, no fallbacks
        if hasattr(flow_instance, "__flow_name__"):
            flow_name = flow_instance.__flow_name__
        else:
            flow_name = flow_instance.__class__.__name__

        class_name = flow_instance.__class__.__name__

        # Look for pipeline method - no fallbacks
        pipeline_method_info = None
        for name in dir(flow_instance):
            method = getattr(flow_instance, name)  # This getattr is necessary for reflection
            if hasattr(method, "__pipeline__") and method.__pipeline__:
                # Found pipeline method - extract info strictly
                if not (hasattr(method, "__input_model__") and method.__input_model__):
                    raise ValueError(f"Pipeline method {name} missing required __input_model__")
                if not (hasattr(method, "__output_model__") and method.__output_model__):
                    raise ValueError(f"Pipeline method {name} missing required __output_model__")

                pipeline_method_info = PipelineMethodInfo(
                    method_name=name,
                    input_model=method.__input_model__,
                    output_model=method.__output_model__,
                    is_pipeline=True
                )
                break

        return FlowInstanceInfo(
            name=flow_name,
            class_name=class_name,
            pipeline_method_info=pipeline_method_info
        )

    def _get_class_pipeline_method(self) -> object | None:
        """Get pipeline method from class without fallbacks.

        Returns:
            Pipeline method object or None if not found
        """
        # Check if class has pipeline method name defined
        if not hasattr(self.__class__, '__pipeline_method__'):
            return None

        pipeline_method_name = self.__class__.__pipeline_method__
        if not pipeline_method_name:
            return None

        # Get the method from instance with strict validation
        if not hasattr(self, pipeline_method_name):
            raise ExecutionError(
                f"Flow {type(self).__name__} missing required pipeline method '{pipeline_method_name}'",
                ErrorContext.create(
                    flow_name=type(self).__name__,
                    error_type="MissingPipelineMethodError",
                    error_location="_get_class_pipeline_method",
                    component=type(self).__name__,
                    operation="pipeline_method_lookup"
                )
            )

        method = getattr(self, pipeline_method_name)
        if not callable(method):
            raise ExecutionError(
                f"Pipeline method '{pipeline_method_name}' on {type(self).__name__} is not callable",
                ErrorContext.create(
                    flow_name=type(self).__name__,
                    error_type="NonCallablePipelineMethodError",
                    error_location="_get_class_pipeline_method",
                    component=type(self).__name__,
                    operation="pipeline_method_validation"
                )
            )

        return cast(object, method)  # method is validated as callable, safe to cast

    def get_pipeline_input_model(self) -> type[BaseModel] | None:
        """Get the input model for this flow's pipeline.

        Returns the pipeline method's input model or flow's input schema.
        No fallbacks - returns the first available model found.

        Returns:
            The input model class or None if not defined
        """
        # Check pipeline method first - strict contract
        pipeline_method = self._get_class_pipeline_method()
        if pipeline_method and hasattr(pipeline_method, '__input_model__'):
            input_model = pipeline_method.__input_model__
            if input_model is not None:
                try:
                    if issubclass(input_model, BaseModel):
                        return cast(type[BaseModel], input_model)
                except TypeError:
                    # input_model is not a class, skip it
                    pass
            return None

        # Return flow's schema if pipeline method unavailable
        return self.input_schema

    def get_pipeline_output_model(self) -> type[BaseModel] | None:
        """Get the output model for this flow's pipeline.

        Returns the pipeline method's output model or flow's output schema.
        No fallbacks - returns the first available model found.

        Returns:
            The output model class or None if not defined
        """
        # Check pipeline method first - strict contract
        pipeline_method = self._get_class_pipeline_method()
        if pipeline_method and hasattr(pipeline_method, '__output_model__'):
            output_model = pipeline_method.__output_model__
            if output_model is not None:
                try:
                    if issubclass(output_model, BaseModel):
                        return cast(type[BaseModel], output_model)
                except TypeError:
                    # output_model is not a class, skip it
                    pass
            return None

        # Return flow's schema if pipeline method unavailable
        return self.output_schema

    def get_pipeline_method(self) -> Any | None:
        """Get the pipeline method for this flow.

        Returns:
            The pipeline method or None if not found
        """
        # Find the pipeline method
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '__pipeline__') and attr.__pipeline__:
                return attr
        return None

    @classmethod
    def get_pipeline_method_cls(cls) -> Any | None:
        """Get the pipeline method for this flow class.

        Returns:
            The pipeline method or None if not found
        """
        # Find the pipeline method
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if hasattr(attr, '__pipeline__') and attr.__pipeline__:
                return attr
        return None

    async def execute(self, context: Context[Any]) -> FlowResult[Any]:
        """Execute the flow with given context.

        This is the ONLY method that should be called from outside the flow.
        It automatically executes the flow's pipeline method.

        Args:
            context: Execution context

        Returns:
            FlowResult containing execution outcome with attribute-based access

        Raises:
            BaseError: If execution fails
        """
        # Initialize execution
        execution_start = datetime.now()
        error_context = ErrorContext.create(
            flow_name=self.name,
            error_type="execution",
            error_location="Flow.execute",
            component="flow_execution",
            operation="execute_pipeline"
        )
        try:
            # Initialize prepared input data variable
            pipeline_input_arg = None # Renamed from prepared_input_data

            # Create error boundary
            async with self.error_manager.error_boundary(
                flow_name=self.name,
                component="flow_execution",
                operation="execute_pipeline"
            ):

                # === MODIFIED SECTION: Get pipeline method and its expected input model ===
                pipeline_method = self._get_class_pipeline_method()
                if not pipeline_method:
                    raise ExecutionError(
                        "No pipeline method found on flow. Each flow must have exactly one @pipeline method.",
                        error_context
                    )

                # Get expected input model without fallbacks
                expected_input_model = None
                if hasattr(pipeline_method, '__input_model__'):
                    expected_input_model = pipeline_method.__input_model__
                # ==========================================================================

                # Prepare and validate input based on the pipeline's expected model
                if expected_input_model:
                    if not context.data:
                         raise ValueError("Context data is missing for pipeline requiring input.")

                    # Ensure context.data matches or can be parsed into the expected model
                    if isinstance(context.data, expected_input_model):
                        pipeline_input_arg = context.data
                    elif isinstance(context.data, dict):
                        try:
                            pipeline_input_arg = expected_input_model(**context.data)
                        except Exception as validation_err:
                            raise ValidationError(
                                f"Input validation failed for {expected_input_model.__name__}: {validation_err}",
                                validation_errors=[ValidationErrorDetail(
                                    location="pipeline_input",
                                    message=str(validation_err),
                                    error_type="PydanticValidationError"
                                )],
                                context=ErrorContext.create(
                                    flow_name="pipeline_execution",
                                    error_type="InputValidationError",
                                    error_location="execute_pipeline",
                                    component="BaseFlow",
                                    operation="input_validation"
                                )
                            ) from validation_err
                    else:
                         raise TypeError(f"Context data type {type(context.data)} cannot be used for pipeline expecting {expected_input_model.__name__}")

                    # Input is validated and ready for execution

                    # Optional: Re-validate using the schema if needed (might be redundant)
                    # self._validate_input(pipeline_input_arg)
                else:
                     # Pipeline expects no input model
                     pipeline_input_arg = None

                # Input validation complete
                # ================================================================================

                # Execute the pipeline function
                try:
                    pipeline_args = {}
                    pipeline_sig = inspect.signature(cast(Callable[..., Any], pipeline_method))

                    # Check if the pipeline expects a context parameter
                    if 'context' in pipeline_sig.parameters:
                        pipeline_args['context'] = context
                    elif 'ctx' in pipeline_sig.parameters:
                        pipeline_args['ctx'] = context

                    # Handle passing the main input argument
                    if pipeline_sig.parameters:
                        param_names = list(pipeline_sig.parameters.keys())
                        first_param_name = param_names[0]

                        # If the first param isn't context/ctx, it expects the main input
                        if first_param_name not in ['context', 'ctx']:
                            pipeline_result = await cast(Callable[..., Any], pipeline_method)(pipeline_input_arg, **pipeline_args)
                        else:
                            # First param is context/ctx, only pass kwargs
                            pipeline_result = await cast(Callable[..., Any], pipeline_method)(**pipeline_args)
                    else:
                        # No parameters, call with kwargs only (which might be empty)
                        pipeline_result = await cast(Callable[..., Any], pipeline_method)(**pipeline_args)

                    # Validate result type against output_schema if defined - no fallbacks
                    output_model = None
                    if hasattr(pipeline_method, '__output_model__'):
                        output_model = pipeline_method.__output_model__

                    if output_model is not None:
                        if not isinstance(pipeline_result, output_model):
                            # Import validation error if needed
                            raise ValidationError(
                                f"Pipeline '{cast(Callable[..., Any], pipeline_method).__name__}' must return an instance of {output_model.__name__}, got {type(pipeline_result).__name__}",
                                validation_errors=[ValidationErrorDetail(
                                    location="pipeline_output",
                                    message=f"Expected {output_model.__name__}, got {type(pipeline_result).__name__}",
                                    error_type="TypeMismatchError"
                                )],
                                context=ErrorContext.create(
                                    flow_name="pipeline_execution",
                                    error_type="OutputValidationError",
                                    error_location="execute_pipeline",
                                    component="BaseFlow",
                                    operation="output_validation"
                                )
                            )

                    result: FlowResult[Any] = FlowResult(
                        data=pipeline_result,  # Store the model directly
                        original_type=type(pipeline_result),
                        flow_name=self.name,
                        status=FlowStatus.SUCCESS
                    )

                except ValidationError:
                    # Reraise validation errors
                    raise
                except Exception as e:
                    # Convert other errors to ExecutionError
                    raise ExecutionError(
                        message=f"Pipeline execution failed: {str(e)}",
                        context=error_context,
                        cause=e
                    ) from e

                # Record successful execution
                execution_time = (datetime.now() - execution_start).total_seconds()

                # Add execution info to result metadata
                # Get original type - strict attribute access
                original_type = result._original_type if hasattr(result, '_original_type') else None

                result_with_metadata: FlowResult[Any] = FlowResult(
                    data=result.data,
                    original_type=original_type,
                    flow_name=self.name,
                    status=FlowStatus.SUCCESS,
                    error=result.error,
                    error_details=result.error_details,
                    metadata={
                        **result.metadata,
                        "execution_time": execution_time,
                    },
                    timestamp=result.timestamp,
                    duration=execution_time
                )

                return result_with_metadata

        except Exception as e:
            execution_time = (datetime.now() - execution_start).total_seconds()

            # Record execution time for debugging

            # Convert to appropriate error type
            if isinstance(e, BaseError):
                error = e
            else:
                error = ExecutionError(
                    message=str(e),
                    context=error_context,
                    cause=e
                )

            # Create error result
            error_result: FlowResult[Any] = FlowResult(
                data={},
                flow_name=self.name,
                status=FlowStatus.ERROR,
                error=str(error),
                error_details={"error_type": type(error).__name__},
                metadata={"execution_time": execution_time},
                duration=execution_time
            )

            # Attach result to error
            error.result = error_result
            raise error from e

    def _validate_input(self, data: Any) -> None:
        """Validate input data against schema.

        Args:
            data: Input data to validate

        Raises:
            ValidationError: If validation fails or if data is not a Pydantic model
        """
        if self.input_schema:
            # Input data must be an instance of the expected model
            if not isinstance(data, self.input_schema):
                raise ValidationError(
                    f"Input must be an instance of {self.input_schema.__name__}, got {type(data).__name__}",
                    validation_errors=[ValidationErrorDetail(
                        location="input",
                        message=f"Expected {self.input_schema.__name__}, got {type(data).__name__}",
                        error_type="type_error"
                    )],
                    context=ErrorContext.create(
                        flow_name=self.name,
                        error_type="ValidationError",
                        error_location="_validate_input",
                        component="base_flow",
                        operation="input_validation"
                    )
                )

    def _validate_output(self, data: Any) -> None:
        """Validate output data against schema.

        Args:
            data: Output data to validate

        Raises:
            ValidationError: If validation fails or if data is not a Pydantic model
        """
        if self.output_schema:
            # For FlowResult, we validate its data dict against the schema
            if isinstance(data, dict) and self.output_schema:
                # Should never directly pass dicts in the new strict mode
                raise ValidationError(
                    f"Output must be an instance of {self.output_schema.__name__}, got dict",
                    validation_errors=[ValidationErrorDetail(
                        location="output",
                        message=f"Expected {self.output_schema.__name__}, got dict",
                        error_type="type_error"
                    )],
                    context=ErrorContext.create(
                        flow_name=self.name,
                        error_type="ValidationError",
                        error_location="_validate_output",
                        component="base_flow",
                        operation="output_validation"
                    )
                )
            # If not a dict, must be an instance of the output schema
            elif not isinstance(data, self.output_schema):
                raise ValidationError(
                    f"Output must be an instance of {self.output_schema.__name__}, got {type(data).__name__}",
                    validation_errors=[ValidationErrorDetail(
                        location="output",
                        message=f"Expected {self.output_schema.__name__}, got {type(data).__name__}",
                        error_type="type_error"
                    )],
                    context=ErrorContext.create(
                        flow_name=self.name,
                        error_type="ValidationError",
                        error_location="_validate_output",
                        component="base_flow",
                        operation="output_type_validation"
                    )
                )

    def _prepare_input(self, context: Context[Any]) -> Any:
        """Prepare input data from context.

        If input_schema is defined, retrieves the validated Pydantic model instance
        from the context using `context.as_model()`. Otherwise, returns raw context data.

        Args:
            context: Execution context

        Returns:
            Prepared input data (Pydantic model instance if schema defined, else dict)

        Raises:
            ExecutionError: If preparation fails
            ValidationError: If `context.as_model()` fails or returns wrong type
        """
        try:
            if self.input_schema:
                # Context stores model as dict; reconstruct instance using as_model()
                model_instance = context.as_model()

                if model_instance is None:
                     # This happens if context had no model_type initially
                     raise ValidationError(
                         f"Context does not contain a model of type {self.input_schema.__name__} needed by flow '{self.name}'",
                         validation_errors=[ValidationErrorDetail(
                             location="context",
                             message=f"Missing model of type {self.input_schema.__name__}",
                             error_type="missing_model_error"
                         )],
                         context=ErrorContext.create(
                             flow_name=self.name,
                             error_type="MissingModelError",
                             error_location="prepare_input",
                             component=self.name,
                             operation="model_reconstruction"
                         )
                     )

                # Validate the reconstructed model instance type
                if not isinstance(model_instance, self.input_schema):
                    raise ValidationError(
                        f"Expected input model of type {self.input_schema.__name__}, but context provided {type(model_instance).__name__}",
                        validation_errors=[ValidationErrorDetail(
                            location="input",
                            message=f"Expected {self.input_schema.__name__}, got {type(model_instance).__name__}",
                            error_type="type_error"
                        )],
                        context=ErrorContext.create(
                            flow_name=self.name,
                            error_type="ValidationError",
                            error_location="_prepare_input",
                            component="base_flow",
                            operation="input_reconstruction_validation"
                        )
                    )

                return model_instance
            else:
                # No input schema defined for this flow, return raw data dict
                return context.data

        except ValidationError:
            # Re-raise validation errors
            raise
        except ValueError as ve:
            # Catch specific errors from as_model() if internal data is inconsistent
            schema_name = self.input_schema.__name__ if self.input_schema else "unknown"
            raise ValidationError(
                f"Failed to reconstruct input model {schema_name} from context: {str(ve)}",
                validation_errors=[ValidationErrorDetail(
                    location="context",
                    message=f"Model reconstruction failed: {str(ve)}",
                    error_type="reconstruction_error"
                )],
                context=ErrorContext.create(
                    flow_name=self.name,
                    error_type="ModelReconstructionError",
                    error_location="prepare_input",
                    component=self.name,
                    operation="model_reconstruction"
                ),
                cause=ve
             ) from ve
        except Exception as e:
            # Wrap other exceptions
            # Get context data without fallbacks
            if hasattr(context, 'data'):
                pass

            error_context = ErrorContext.create(
                flow_name=self.name,
                error_type="ExecutionError",
                error_location=f"{self.__class__.__name__}._prepare_input",
                component=self.name,
                operation="prepare_input"
            )

            raise ExecutionError(
                message="Failed to prepare input",
                context=error_context,
                cause=e
            ) from e

    def add_error_handler(
        self,
        error_type: type[BaseError],
        handler: Any
    ) -> None:
        """Add error handler to flow.

        Args:
            error_type: Type of error to handle
            handler: Handler instance
        """
        self.error_manager.register(error_type, handler)

    @property
    def full_name(self) -> str:
        """Get fully qualified flow name."""
        return f"{self.__class__.__name__}.{self.name}"

    def __str__(self) -> str:
        """String representation."""
        return f"Flow(name='{self.name}')"

    def get_description(self) -> str:
        """
        Get the flow description.

        This method is automatically added by the @flow decorator.

        Returns:
            Flow description
        """
        # This method is always implemented by the decorator
        # No need to raise NotImplementedError
        return ""

