"""Task debriefing flows."""

from typing import cast, Dict, Any
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry
from .models import (
    IntentAnalysisInput, IntentAnalysisOutput,
    SuccessPresentationInput, SuccessPresentationOutput,
    CorrectiveTaskInput, CorrectiveTaskOutput,
    FailureExplanationInput, FailureExplanationOutput
)


@flow(  # type: ignore[arg-type]
    name="intent-analysis",
    description="Analyze if execution results fulfill user's original intent",
    is_infrastructure=False
)
class IntentAnalysisFlow:
    """Analyzes if user's intent was fulfilled by execution results."""
    
    @pipeline(input_model=IntentAnalysisInput, output_model=IntentAnalysisOutput)
    async def run_pipeline(self, input_data: IntentAnalysisInput) -> IntentAnalysisOutput:
        """Analyze intent fulfillment."""
        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))
        
        # Get prompt from registry
        prompt_instance = resource_registry.get("intent-analysis-prompt")
        
        # Prepare prompt variables
        prompt_vars: Dict[str, Any] = {
            "original_user_message": input_data.original_user_message,
            "generated_task": input_data.generated_task,
            "execution_results": input_data.execution_results,
            "agent_persona": input_data.agent_persona,
            "working_directory": input_data.working_directory
        }
        
        # Generate intent analysis using LLM
        result = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=IntentAnalysisOutput,
            model_name="default-model",
            prompt_variables=prompt_vars
        )

        # Type validation following flowlib's no-fallbacks principle
        if not isinstance(result, IntentAnalysisOutput):
            raise ValueError(f"Expected IntentAnalysisOutput from LLM, got {type(result)}")

        return result


@flow(  # type: ignore[arg-type]
    name="success-presentation",
    description="Generate user-friendly presentation for successful task completion",
    is_infrastructure=False
)
class SuccessPresentationFlow:
    """Generates user-friendly success presentation."""
    
    @pipeline(input_model=SuccessPresentationInput, output_model=SuccessPresentationOutput)
    async def run_pipeline(self, input_data: SuccessPresentationInput) -> SuccessPresentationOutput:
        """Generate success presentation."""
        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))
        
        # Get prompt from registry
        prompt_instance = resource_registry.get("success-presentation-prompt")
        
        # Prepare prompt variables
        prompt_vars: Dict[str, Any] = {
            "original_user_message": input_data.original_user_message,
            "execution_results": input_data.execution_results,
            "user_intent_summary": input_data.intent_analysis.user_intent_summary,
            "execution_summary": input_data.intent_analysis.execution_summary,
            "agent_persona": input_data.agent_persona
        }
        
        # Generate presentation using LLM
        result = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=SuccessPresentationOutput,
            model_name="default-model",
            prompt_variables=prompt_vars
        )

        # Type validation following flowlib's no-fallbacks principle
        if not isinstance(result, SuccessPresentationOutput):
            raise ValueError(f"Expected SuccessPresentationOutput from LLM, got {type(result)}")

        return result


@flow(  # type: ignore[arg-type]
    name="corrective-task-generation",
    description="Generate corrective task for retry based on failure analysis",
    is_infrastructure=False
)
class CorrectiveTaskFlow:
    """Generates corrective tasks for retry attempts."""
    
    @pipeline(input_model=CorrectiveTaskInput, output_model=CorrectiveTaskOutput)
    async def run_pipeline(self, input_data: CorrectiveTaskInput) -> CorrectiveTaskOutput:
        """Generate corrective task."""
        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))
        
        # Get prompt from registry
        prompt_instance = resource_registry.get("corrective-task-prompt")
        
        # Prepare prompt variables
        prompt_vars: Dict[str, Any] = {
            "original_user_message": input_data.original_user_message,
            "failed_task": input_data.failed_task,
            "execution_results": input_data.execution_results,
            "user_intent_summary": input_data.intent_analysis.user_intent_summary,
            "gap_analysis": input_data.intent_analysis.gap_analysis,
            "correction_suggestion": input_data.intent_analysis.correction_suggestion,
            "cycle_number": input_data.cycle_number,
            "working_directory": input_data.working_directory
        }
        
        # Generate corrective task using LLM
        result = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=CorrectiveTaskOutput,
            model_name="default-model",
            prompt_variables=prompt_vars
        )

        # Type validation following flowlib's no-fallbacks principle
        if not isinstance(result, CorrectiveTaskOutput):
            raise ValueError(f"Expected CorrectiveTaskOutput from LLM, got {type(result)}")

        return result


@flow(  # type: ignore[arg-type]
    name="failure-explanation",
    description="Generate helpful explanation for task failures after max retries",
    is_infrastructure=False
)
class FailureExplanationFlow:
    """Generates helpful failure explanations."""
    
    @pipeline(input_model=FailureExplanationInput, output_model=FailureExplanationOutput)
    async def run_pipeline(self, input_data: FailureExplanationInput) -> FailureExplanationOutput:
        """Generate failure explanation."""
        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))
        
        # Get prompt from registry
        prompt_instance = resource_registry.get("failure-explanation-prompt")
        
        # Prepare prompt variables
        prompt_vars: Dict[str, Any] = {
            "original_user_message": input_data.original_user_message,
            "execution_results": input_data.execution_results,
            "user_intent_summary": input_data.intent_analysis.user_intent_summary,
            "gap_analysis": input_data.intent_analysis.gap_analysis,
            "cycles_attempted": input_data.cycles_attempted,
            "agent_persona": input_data.agent_persona
        }
        
        # Generate failure explanation using LLM
        result = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=FailureExplanationOutput,
            model_name="default-model",
            prompt_variables=prompt_vars
        )

        # Type validation following flowlib's no-fallbacks principle
        if not isinstance(result, FailureExplanationOutput):
            raise ValueError(f"Expected FailureExplanationOutput from LLM, got {type(result)}")

        return result