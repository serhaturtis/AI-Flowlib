"""Flow for read tool parameter generation."""

from typing import cast, Dict, Any
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.core.models import StrictBaseModel
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry
from pydantic import Field
from .models import ReadParameters
from .prompts import ReadToolParameterGenerationPrompt


class ReadParameterGenerationInput(StrictBaseModel):
    """Input for read parameter generation flow."""
    
    task_content: str = Field(..., description="Task description to extract parameters from")
    working_directory: str = Field(default=".", description="Working directory context")


class ReadParameterGenerationOutput(StrictBaseModel):
    """Output from read parameter generation flow."""
    
    parameters: ReadParameters = Field(..., description="Generated read parameters")


@flow(name="read-parameter-generation", description="Generate parameters for read tool from task description")  # type: ignore[arg-type]
class ReadParameterGenerationFlow:
    """Flow for generating read tool parameters using LLM."""
    
    @pipeline(input_model=ReadParameterGenerationInput, output_model=ReadParameterGenerationOutput)
    async def run_pipeline(self, request: ReadParameterGenerationInput) -> ReadParameterGenerationOutput:
        """Generate read parameters from task content."""
        
        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))
        
        # Get prompt template
        prompt_template = resource_registry.get("read_tool_parameter_generation", ReadToolParameterGenerationPrompt)
        
        # Prepare prompt variables
        prompt_variables: Dict[str, Any] = {
            "task_content": request.task_content,
            "working_directory": request.working_directory
        }
        
        # Generate structured parameters
        parameters = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_template),
            output_type=ReadParameters,
            model_name="default-model",
            prompt_variables=prompt_variables
        )
        
        return ReadParameterGenerationOutput(parameters=parameters)