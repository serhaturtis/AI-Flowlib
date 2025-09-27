"""Flow for edit tool parameter generation."""

from typing import cast, Dict, Any
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.core.models import StrictBaseModel
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry
from pydantic import Field
from .models import EditParameters
from .prompts import EditToolParameterGenerationPrompt


class EditParameterGenerationInput(StrictBaseModel):
    """Input for edit parameter generation flow."""
    
    task_content: str = Field(..., description="Task description to extract parameters from")
    working_directory: str = Field(default=".", description="Working directory context")


class EditParameterGenerationOutput(StrictBaseModel):
    """Output from edit parameter generation flow."""
    
    parameters: EditParameters = Field(..., description="Generated edit parameters")


@flow(name="edit-parameter-generation", description="Generate parameters for edit tool from task description")  # type: ignore[arg-type]
class EditParameterGenerationFlow:
    """Flow for generating edit tool parameters using LLM."""
    
    @pipeline(input_model=EditParameterGenerationInput, output_model=EditParameterGenerationOutput)
    async def run_pipeline(self, request: EditParameterGenerationInput) -> EditParameterGenerationOutput:
        """Generate edit parameters from task content."""
        
        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))
        
        # Get prompt template
        prompt_template = resource_registry.get("edit_tool_parameter_generation", EditToolParameterGenerationPrompt)
        
        # Prepare prompt variables
        prompt_variables: Dict[str, Any] = {
            "task_content": request.task_content,
            "working_directory": request.working_directory
        }
        
        # Generate structured parameters
        parameters = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_template),
            output_type=EditParameters,
            model_name="default-model",
            prompt_variables=prompt_variables
        )
        
        return EditParameterGenerationOutput(parameters=parameters)