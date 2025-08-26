"""
Base reflection implementation for the agent system.

This module provides the reflection component for the agent system,
which is responsible for analyzing execution results and determining
task progress.
"""

import logging
from typing import Any, Dict, Optional, List, Union, Type
from flowlib.core.models import StrictBaseModel

# Import prompts to ensure registration
from .prompts import (
    DefaultReflectionPrompt,
    TaskCompletionReflectionPrompt
)
from .prompts.step_reflection import (
    DefaultStepReflectionPrompt
)

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ReflectionError, NotInitializedError
from flowlib.agent.models.config import ReflectionConfig
from flowlib.agent.models.state import AgentState
# Removed ProviderType import - using config-driven provider access
from flowlib.providers.core.registry import provider_registry
from flowlib.resources.registry.registry import resource_registry
from flowlib.flows.models.results import FlowResult, FlowStatus
from flowlib.agent.components.reflection.models import ReflectionResult, ReflectionInput, StepReflectionResult, StepReflectionInput
from flowlib.agent.components.reflection.interfaces import ReflectionInterface
from flowlib.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class AgentReflection(AgentComponent, ReflectionInterface):
    """Reflection component for the agent system.
    
    Responsibilities:
    1. Analyzing execution results
    2. Determining task progress and completion
    3. Extracting insights from execution results
    
    This class implements the ReflectionInterface protocol.
    """
    
    def __init__(
        self,
        config: Optional[ReflectionConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
        name: str = "reflection",
        activity_stream=None
    ):
        """Initialize the agent reflection.
        
        Args:
            config: Reflection configuration
            llm_provider: LLM provider component
            name: Component name
            activity_stream: Optional activity stream for real-time updates
        """
        super().__init__(name)
        
        # Configuration
        self._config = config or ReflectionConfig(model_name="agent-model-large")  # Use large model for deep reflection
        
        # Components
        self._llm_provider = llm_provider
        self._activity_stream = activity_stream
        
        # Execution state
        self._reflection_template = None
        self._step_reflection_template = None # Add template for step reflection
    
    def _check_initialized(self) -> None:
        """Check if the reflection component is initialized.
        
        Raises:
            NotInitializedError: If the reflection component is not initialized
        """
        if not self._initialized:
            raise NotInitializedError(
                message="Component must be initialized before use",
                component_name=self._name,
                operation="reflect"
            )
    
    async def _initialize_impl(self) -> None:
        """
        Initialize the reflection component.
        
        Raises:
            ReflectionError: If initialization fails
        """
        try:
            # First, check if we have a model provider already
            if not self._llm_provider:
                # Create our own model provider like the planner does
                provider_name = self._config.provider_name or "llamacpp"
                
                # Get model provider directly from provider registry
                try:
                    self._llm_provider = await provider_registry.get_by_config("default-llm")
                    logger.info(f"Created model provider '{provider_name}' for reflection")
                except Exception as e:
                    raise ReflectionError(
                        message=f"Failed to create model provider '{provider_name}' for reflection: {str(e)}",
                        agent=self.name,
                        cause=e
                    )
            
            # Ensure we have a model provider
            if not self._llm_provider:
                raise ReflectionError(
                    message="No model provider available for reflection",
                    agent=self.name
                )
            
            # Load template
            self._reflection_template = await self._load_reflection_template()
            self._step_reflection_template = await self._load_step_reflection_template() # Load step template
            
            logger.info(f"Initialized agent reflection with model {self._config.model_name}")
            
        except Exception as e:
            raise ReflectionError(
                message=f"Failed to initialize reflection component: {str(e)}",
                agent=self.name,
                cause=e
            ) from e
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the reflection component."""
        logger.info("Shutting down agent reflection")
    
    async def reflect(
        self,
        state: AgentState,
        flow_name: str,
        flow_inputs: StrictBaseModel,
        flow_result: FlowResult,
        memory_context: Optional[str] = None,
        **kwargs
    ) -> ReflectionResult:
        """
        Analyze execution results and update state.
        
        Args:
            state: Current agent state
            flow_name: Name of the flow that was executed  
            flow_inputs: Inputs provided to the flow as a Pydantic model
            flow_result: Result from the flow execution
            memory_context: Optional memory context for reflection
            **kwargs: Additional reflection arguments
            
        Returns:
            ReflectionResult with analysis, progress and completion status
            
        Raises:
            NotInitializedError: If reflection is not initialized
            ReflectionError: If reflection fails
        """
        self._check_initialized()
        
        try:
            # Stream reflection start
            if self._activity_stream:
                self._activity_stream.reflection(
                    f"Reflecting on execution of {flow_name}",
                    progress=state.progress,
                    cycle=state.cycles
                )
            
            if not self._reflection_template:
                raise NotInitializedError(
                    message="Reflection template not loaded",
                    component_name=self._name,
                    operation="reflect"
                )
            
            # Analyze the flow execution results
            execution_success = flow_result.status == FlowStatus.SUCCESS
            flow_error = str(flow_result.error) if flow_result.error else "No errors reported"
            
            # Format execution history for context
            execution_history_text = self._format_execution_history(state.execution_history)
            
            # Prepare variables for the reflection template
            template_variables = {
                "task_description": state.task_description,
                "state_summary": f"Task: {state.task_description}, Cycle: {state.cycles}, Progress: {state.progress}%",
                "current_progress": state.progress,
                "plan_status": "SUCCESS" if execution_success else "FAILED",
                "plan_error": flow_error,
                "step_reflections_summary": f"Executed flow '{flow_name}' with result: {execution_success}",
                "execution_history_text": execution_history_text
            }
            
            # Generate reflection using structured generation
            result = await self._llm_provider.generate_structured(
                prompt=self._reflection_template,
                output_type=ReflectionResult,
                model_name=self._config.model_name,
                prompt_variables=template_variables
            )

            # Ensure progress is valid
            clamped_progress = max(0, min(100, result.progress))
            if clamped_progress != result.progress:
                result = result.model_copy(update={"progress": clamped_progress})
            
            # Stream reflection results
            if self._activity_stream:
                self._activity_stream.reflection(
                    f"Task progress: {result.progress}% - {result.reflection[:100]}...",
                    progress=result.progress,
                    is_complete=result.is_complete,
                    completion_reason=result.completion_reason
                )
            
            logger.info("Overall plan reflection complete.")
            return result
            
        except NotInitializedError:
            # Re-raise NotInitializedError without wrapping
            raise
        except Exception as e:
            logger.error(f"Overall reflection failed: {str(e)}", exc_info=True)
            if self._activity_stream:
                self._activity_stream.error(f"Reflection failed: {str(e)}")
            raise ReflectionError(
                message=f"Reflection failed: {str(e)}",
                agent=self.name,
                cause=e
            ) from e
    
    def _format_execution_history(self, execution_history: List[Any]) -> str:
        """Format execution history for reflection prompt."""
        if not execution_history:
            return "No execution history available."
            
        lines = ["Recent Execution History:"]
        for i, entry in enumerate(execution_history[-5:]):  # Last 5 entries
            if hasattr(entry, 'flow_name'):
                lines.append(f"  {i+1}. Flow: {entry.flow_name}")
                if hasattr(entry, 'result'):
                    result_summary = str(entry.result)[:100] + "..." if len(str(entry.result)) > 100 else str(entry.result)
                    lines.append(f"     Result: {result_summary}")
            else:
                lines.append(f"  {i+1}. {str(entry)[:100]}...")
        
        return "\n".join(lines)
    
    def _format_step_reflections(self, step_reflections: List[StepReflectionResult]) -> str:
        """Formats a list of StepReflectionResult into a string for the prompt."""
        if not step_reflections:
            return "No step reflections were recorded for this plan execution."
        
        lines = ["Summary of Plan Step Reflections:"]
        for i, sr in enumerate(step_reflections):
            lines.append(f"  Step {i+1} (ID: {sr.step_id}):")
            lines.append(f"    - Success: {sr.step_success}")
            lines.append(f"    - Reflection: {sr.reflection}")
            if sr.key_observation:
                lines.append(f"    - Key Observation: {sr.key_observation}")
        return "\n".join(lines)
    
    async def _load_reflection_template(self) -> object:
        """Load the overall reflection prompt template.

        Prioritizes loading from resource_registry["reflection_default"],
        then falls back to instantiating DefaultReflectionPrompt.
        
        Returns:
            Reflection prompt template
        """
        # Try to get the standard template from registry
        template_name = "reflection_default"
        try:
            if resource_registry.contains(template_name):
                template = resource_registry.get(template_name)
                logger.info(f"Using '{template_name}' template from resource registry for overall reflection.")
                return template
        except Exception as e:
            logger.warning(f"Failed to get reflection template from registry: {str(e)}")
        
        # If we reach here, use the default implementation directly
        logger.info(f"'{template_name}' not found in registry. Using built-in DefaultReflectionPrompt for overall reflection.")
        return DefaultReflectionPrompt(name="reflection_default", type="prompt_config")
    
    async def _load_step_reflection_template(self) -> object:
        """Load the step reflection prompt template.

        Prioritizes loading from resource_registry["step_reflection_default"],
        then falls back to instantiating DefaultStepReflectionPrompt.

        Returns:
            Step reflection prompt template
        """
        # Try to get the standard template from registry
        template_name = "step_reflection_default"
        try:
            if resource_registry.contains(template_name):
                template = resource_registry.get(template_name)
                logger.info(f"Using '{template_name}' template from resource registry for step reflection.")
                return template
        except Exception as e:
            logger.warning(f"Failed to get step reflection template from registry: {str(e)}")

        # If we reach here, use the default implementation directly
        logger.info(f"'{template_name}' not found in registry. Using built-in DefaultStepReflectionPrompt for step reflection.")
        return DefaultStepReflectionPrompt(name="step_reflection_default", type="prompt_config")
    
    
    async def _execute_reflection(self, reflection_input: ReflectionInput) -> ReflectionResult:
        """
        Execute reflection using the LLM provider.
        
        Args:
            reflection_input: Reflection input with all required data
            
        Returns:
            ReflectionResult with analysis, progress and completion status
            
        Raises:
            ReflectionError: If reflection execution fails
        """
        try:
            # Check if we have the required components
            if not self._llm_provider:
                raise ReflectionError(
                    message="No LLM provider available for reflection",
                    agent=self.name
                )
                
            if not self._reflection_template:
                raise ReflectionError(
                    message="No reflection template available",
                    agent=self.name
                )
                
            # Format flow result for template
            flow_result_formatted = self._format_flow_result(reflection_input.flow_result)
            
            # Format flow inputs for template
            # Handle case where flow_inputs might be None for plan-level reflection
            if reflection_input.flow_inputs is not None:
                flow_inputs_formatted = self._format_flow_inputs(reflection_input.flow_inputs)
            else:
                flow_inputs_formatted = "N/A (Plan Level Reflection)"
                
            # Prepare variables for the template
            template_variables = {
                "task_description": reflection_input.task_description,
                "cycle": reflection_input.cycle,
                "flow_name": reflection_input.flow_name,
                "flow_status": reflection_input.flow_status,
                "flow_inputs": flow_inputs_formatted,
                "flow_result": flow_result_formatted,
                "execution_history_text": reflection_input.execution_history_text,
                "planning_rationale": reflection_input.planning_rationale,
                "state_summary": reflection_input.state_summary,
                "current_progress": reflection_input.progress
            }
            
            # Generate reflection using the structured generation interface
            result = await self._llm_provider.generate_structured(
                prompt=self._reflection_template,
                output_type=ReflectionResult,
                model_name=self._config.model_name,
                prompt_variables=template_variables
            )
            
            # Ensure progress is between 0 and 100
            result.progress = max(0, min(100, result.progress))
            
            return result
            
        except Exception as e:
            raise ReflectionError(
                message=f"Failed to execute reflection: {str(e)}",
                agent=self.name,
                cause=e
            ) from e
            
    def _format_flow_result(self, flow_result: FlowResult) -> str:
        """
        Format a FlowResult as human-readable text for the template.
        
        Args:
            flow_result: FlowResult to format
            
        Returns:
            Formatted string representation
        """
        # Start with basic info
        lines = [
            f"Status: {flow_result.status}",
            f"Timestamp: {flow_result.timestamp.isoformat()}",
        ]
        
        # Add duration if available
        if flow_result.duration is not None:
            lines.append(f"Duration: {flow_result.duration:.3f} seconds")
        
        # Add error info if present
        if flow_result.error:
            lines.append(f"Error: {flow_result.error}")
        
        # Format the data section
        lines.append("Data:")
        if hasattr(flow_result.data, "model_dump") and callable(flow_result.data.model_dump):
            data_dict = flow_result.data.model_dump()
            for key, value in data_dict.items():
                lines.append(f"  {key}: {value}")
        elif isinstance(flow_result.data, dict):
            for key, value in flow_result.data.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append(f"  {flow_result.data}")
        
        # Join all lines with newlines
        return "\n".join(lines)
    
    def _format_flow_inputs(self, flow_inputs: StrictBaseModel) -> str:
        """
        Format flow inputs as human-readable text for the template.
        
        Args:
            flow_inputs: Flow inputs (Pydantic model)
            
        Returns:
            Formatted string representation
        """
        # Get model data as dictionary
        if hasattr(flow_inputs, "model_dump") and callable(flow_inputs.model_dump):
            data_dict = flow_inputs.model_dump()
        else:
            data_dict = flow_inputs.__dict__
        
        # Format each field
        lines = [f"Model type: {flow_inputs.__class__.__name__}"]
        
        for key, value in data_dict.items():
            if key.startswith("_"):
                continue
                
            # Format the value based on its type
            if isinstance(value, dict):
                value_str = ", ".join(f"{k}: {v}" for k, v in value.items())
                lines.append(f"{key}: {{{value_str}}}")
            elif isinstance(value, list):
                if len(value) <= 3:
                    value_str = ", ".join(str(item) for item in value)
                    lines.append(f"{key}: [{value_str}]")
                else:
                    lines.append(f"{key}: [List with {len(value)} items]")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)

    async def step_reflect(
        self,
        step_input: StepReflectionInput # Use the dedicated input model
    ) -> StepReflectionResult:
        """Analyze the outcome of a single plan step."""
        self._check_initialized()

        if not self._step_reflection_template:
            raise NotInitializedError(
                message="Step reflection template not loaded",
                component_name=self._name,
                operation="step_reflect"
            )

        try:
            # Format inputs and results using existing helper methods
            flow_inputs_formatted = self._format_flow_inputs(step_input.flow_inputs)
            flow_result_formatted = self._format_flow_result(step_input.flow_result)

            # Prepare variables for the template
            template_variables = {
                "task_description": step_input.task_description,
                "step_id": step_input.step_id,
                "step_intent": step_input.step_intent,
                "step_rationale": step_input.step_rationale,
                "flow_name": step_input.flow_name,
                "flow_inputs_formatted": flow_inputs_formatted,
                "flow_result_formatted": flow_result_formatted,
                "current_progress": step_input.current_progress
            }

            # Generate reflection using structured generation
            result = await self._llm_provider.generate_structured(
                prompt=self._step_reflection_template,
                output_type=StepReflectionResult,
                model_name=self._config.model_name,
                prompt_variables=template_variables
            )

            # Ensure step_id is correctly carried over if LLM misses it
            if not result.step_id:
                result = StepReflectionResult(
                    step_id=step_input.step_id,
                    reflection=result.reflection,
                    step_success=result.step_success,
                    key_observation=result.key_observation if hasattr(result, 'key_observation') else None
                )

            logger.info(f"Step reflection complete for step ID: {result.step_id}")
            return result

        except Exception as e:
            logger.error(f"Step reflection failed for step ID {step_input.step_id}: {str(e)}", exc_info=True)
            # Return a default failure reflection result?
            return StepReflectionResult(
                step_id=step_input.step_id,
                reflection=f"Step reflection failed: {str(e)}",
                step_success=False, # Assume failure if reflection errors
                key_observation="Reflection process encountered an error."
            )
