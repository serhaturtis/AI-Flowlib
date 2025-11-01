"""Task completion evaluation flow."""

import time
from typing import cast

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry

from .models import (
    EvaluationInput,
    EvaluationOutput,
    EvaluationResult,
    LLMEvaluationResult,
)


@flow(  # type: ignore[arg-type]
    name="completion-evaluation",
    description="Evaluate whether a task has been completed successfully",
    is_infrastructure=False,
)
class CompletionEvaluationFlow:
    """Evaluates task completion in the Plan-and-Execute loop."""

    @pipeline(input_model=EvaluationInput, output_model=EvaluationOutput)
    async def run_pipeline(self, input_data: EvaluationInput) -> EvaluationOutput:
        """Evaluate whether the task is complete.

        Args:
            input_data: Contains goal, plan, and execution results

        Returns:
            EvaluationOutput with completion decision
        """
        start_time = time.time()

        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))

        # Get prompt from registry
        prompt_instance = resource_registry.get("completion-evaluation-prompt")

        # Format executed steps for prompt
        steps_text = self._format_executed_steps(input_data.executed_steps)

        # Prepare prompt variables
        prompt_vars = {
            "original_goal": input_data.original_goal,
            "plan_reasoning": input_data.plan_reasoning,
            "executed_steps": steps_text,
            "expected_outcome": input_data.expected_outcome,
        }

        # Get evaluation from LLM
        llm_result = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=LLMEvaluationResult,
            model_name="default-model",
            prompt_variables=cast(dict[str, object], prompt_vars),
        )

        # Convert confidence string to float
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        confidence = confidence_map.get(llm_result.completion_confidence.lower(), 0.6)

        # Validate consistency - fail fast if LLM contradicts itself
        if llm_result.is_complete and llm_result.next_action != "done":
            raise RuntimeError(
                f"LLM generated inconsistent evaluation: is_complete=True but next_action='{llm_result.next_action}'. "
                f"When task is complete, next_action must be 'done'. Reasoning: {llm_result.reasoning}"
            )

        # Create full evaluation result with programmatic fields
        evaluation_result = EvaluationResult(
            is_complete=llm_result.is_complete,
            completion_confidence=confidence,  # Programmatic conversion
            reasoning=llm_result.reasoning,
            next_action=llm_result.next_action,
            clarification_question=llm_result.clarification_question,
            evaluation_time_ms=(time.time() - start_time) * 1000,  # Programmatic
            llm_calls_made=1,  # Programmatic
        )

        processing_time = (time.time() - start_time) * 1000

        return EvaluationOutput(
            result=evaluation_result,
            success=True,  # Programmatic
            processing_time_ms=processing_time,  # Programmatic
        )

    def _format_executed_steps(self, steps: list[dict]) -> str:
        """Format executed steps for the prompt."""
        if not steps:
            return "No steps executed yet"

        formatted = []
        for idx, step in enumerate(steps, 1):
            tool = step.get("tool_name", "unknown")
            description = step.get("step_description", "")
            result = step.get("result", "No result")
            success = step.get("success", False)
            status_indicator = "✓" if success else "✗"
            formatted.append(
                f"{idx}. [{status_indicator}] {tool}: {description}\n   Result: {result}\n   Success: {success}"
            )

        return "\n".join(formatted)
