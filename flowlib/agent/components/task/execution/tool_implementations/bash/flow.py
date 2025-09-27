"""Flow for bash tool parameter generation."""

from typing import cast, Dict, Any
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.core.models import StrictBaseModel
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry
from pydantic import Field
from .models import BashParameters
from .prompts import BashToolParameterGenerationPrompt


class BashParameterGenerationInput(StrictBaseModel):
    """Input for bash parameter generation flow."""
    
    task_content: str = Field(..., description="Task description to extract parameters from")
    working_directory: str = Field(default=".", description="Working directory context")


class BashParameterGenerationOutput(StrictBaseModel):
    """Output from bash parameter generation flow."""
    
    parameters: BashParameters = Field(..., description="Generated bash parameters")


@flow(name="bash-parameter-generation", description="Generate parameters for bash tool from task description")  # type: ignore[arg-type]
class BashParameterGenerationFlow:
    """Flow for generating bash tool parameters using LLM."""
    
    @pipeline(input_model=BashParameterGenerationInput, output_model=BashParameterGenerationOutput)
    async def run_pipeline(self, request: BashParameterGenerationInput) -> BashParameterGenerationOutput:
        """Generate bash parameters from task content."""
        
        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))
        
        # Get prompt template
        prompt_template = resource_registry.get("bash_tool_parameter_generation", BashToolParameterGenerationPrompt)
        
        # Prepare prompt variables
        prompt_variables: Dict[str, Any] = {
            "task_content": request.task_content,
            "working_directory": request.working_directory
        }
        
        # Generate structured parameters
        parameters = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_template),
            output_type=BashParameters,
            model_name="default-model",
            prompt_variables=prompt_variables
        )
        
        return BashParameterGenerationOutput(parameters=parameters)