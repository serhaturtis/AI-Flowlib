"""Flow for write tool parameter generation."""

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.core.models import StrictBaseModel
from flowlib.providers.core.registry import provider_registry
from flowlib.resources.registry.registry import resource_registry
from pydantic import Field
from .models import WriteParameters
from .prompts import WriteToolParameterGenerationPrompt


class WriteParameterGenerationInput(StrictBaseModel):
    """Input for write parameter generation flow."""
    
    task_content: str = Field(..., description="Task description to extract parameters from")
    working_directory: str = Field(default=".", description="Working directory context")


class WriteParameterGenerationOutput(StrictBaseModel):
    """Output from write parameter generation flow."""
    
    parameters: WriteParameters = Field(..., description="Generated write parameters")


@flow(name="write-parameter-generation", description="Generate parameters for write tool from task description")
class WriteParameterGenerationFlow:
    """Flow for generating write tool parameters using LLM."""
    
    @pipeline(input_model=WriteParameterGenerationInput, output_model=WriteParameterGenerationOutput)
    async def run_pipeline(self, request: WriteParameterGenerationInput) -> WriteParameterGenerationOutput:
        """Generate write parameters from task content."""
        
        # Get LLM provider
        llm = await provider_registry.get_by_config("default-llm")
        
        # Get prompt template
        prompt_template = resource_registry.get("write_tool_parameter_generation", WriteToolParameterGenerationPrompt)
        
        # Prepare prompt variables
        prompt_variables = {
            "task_content": request.task_content,
            "working_directory": request.working_directory
        }
        
        # Generate structured parameters
        parameters = await llm.generate_structured(
            prompt=prompt_template,
            output_type=WriteParameters,
            model_name="default-model",
            prompt_variables=prompt_variables
        )
        
        return WriteParameterGenerationOutput(parameters=parameters)