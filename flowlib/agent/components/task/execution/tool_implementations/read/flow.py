"""Flow for read tool parameter generation."""

from typing import Any, cast

from pydantic import Field

from flowlib.agent.models.conversation import ConversationMessage
from flowlib.core.models import StrictBaseModel
from flowlib.flows.base import ContextBuildingFlow, ContextBuildingInput
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.flows.registry.registry import flow_registry
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry
from flowlib.config.required_resources import RequiredAlias

from .models import ReadParameters


class ReadParameterGenerationInput(StrictBaseModel):
    """Input for read parameter generation flow."""

    task_content: str = Field(..., description="Task description to extract parameters from")
    working_directory: str = Field(default=".", description="Working directory context")

    conversation_history: list[ConversationMessage] = Field(
        default_factory=list,
        description="Recent conversation history for context-aware parameter generation",
    )
    original_user_message: str | None = Field(
        default=None, description="Original user message that started this task"
    )


class ReadParameterGenerationOutput(StrictBaseModel):
    """Output from read parameter generation flow."""

    parameters: ReadParameters = Field(..., description="Generated read parameters")


@flow(
    name="read-parameter-generation",
    description="Generate parameters for read tool from task description",
)  # type: ignore[arg-type]
class ReadParameterGenerationFlow:
    """Flow for generating read tool parameters using LLM."""

    @pipeline(input_model=ReadParameterGenerationInput, output_model=ReadParameterGenerationOutput)
    async def run_pipeline(
        self, request: ReadParameterGenerationInput
    ) -> ReadParameterGenerationOutput:
        """Generate read parameters from task content."""

        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config(RequiredAlias.DEFAULT_LLM.value))

        # Get prompt template
        prompt_template = resource_registry.get("read_tool_parameter_generation")

        # Use context building flow to build comprehensive context
        context_flow = cast(ContextBuildingFlow, flow_registry.get("context-building"))
        context_result = await context_flow.run_pipeline(
            ContextBuildingInput(
                task_content=request.task_content,
                working_directory=request.working_directory,
                conversation_history=request.conversation_history,
                original_user_message=request.original_user_message,
            )
        )

        # Use the built prompt variables
        prompt_variables: dict[str, Any] = context_result.prompt_variables

        # Generate structured parameters
        parameters = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_template),
            output_type=ReadParameters,
            model_name=RequiredAlias.DEFAULT_MODEL.value,
            prompt_variables=prompt_variables,
        )

        return ReadParameterGenerationOutput(parameters=parameters)
