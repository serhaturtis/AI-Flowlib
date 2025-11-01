"""Task execution reflection flow."""

import time
from typing import cast

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry

from .models import (
    LLMReflectionResult,
    ReflectionInput,
    ReflectionOutput,
    ReflectionResult,
)


@flow(  # type: ignore[arg-type]
    name="execution-reflection",
    description="Reflect on execution results to determine if re-planning is needed",
    is_infrastructure=False,
)
class ExecutionReflectionFlow:
    """Reflects on execution results in the Plan-Execute-Reflect loop."""

    @pipeline(input_model=ReflectionInput, output_model=ReflectionOutput)
    async def run_pipeline(self, input_data: ReflectionInput) -> ReflectionOutput:
        """Reflect on execution results.

        Args:
            input_data: Contains goal, plan steps, and execution results

        Returns:
            ReflectionOutput with analysis and next action decision
        """
        start_time = time.time()

        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))

        # Get prompt from registry
        prompt_instance = resource_registry.get("execution-reflection-prompt")

        # Format plan steps for prompt
        plan_steps_text = self._format_plan_steps(input_data.plan_steps)

        # Format execution results for prompt
        execution_results_text = self._format_execution_results(input_data.execution_results)

        # Prepare prompt variables
        prompt_vars = {
            "original_goal": input_data.original_goal,
            "planned_steps": plan_steps_text,
            "execution_results": execution_results_text,
        }

        # Get reflection from LLM
        llm_result = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=LLMReflectionResult,
            model_name="default-model",
            prompt_variables=cast(dict[str, object], prompt_vars),
        )

        # Validate consistency - fail fast if LLM contradicts itself
        if llm_result.next_action == "replan" and not llm_result.replanning_guidance:
            raise RuntimeError(
                f"LLM generated inconsistent reflection: next_action='replan' but no replanning_guidance provided. "
                f"Reasoning: {llm_result.reasoning}"
            )

        if llm_result.next_action == "clarify" and not llm_result.clarification_question:
            raise RuntimeError(
                f"LLM generated inconsistent reflection: next_action='clarify' but no clarification_question provided. "
                f"Reasoning: {llm_result.reasoning}"
            )

        # Create full reflection result with programmatic fields
        reflection_result = ReflectionResult(
            execution_successful=llm_result.execution_successful,
            issues_found=llm_result.issues_found,
            reasoning=llm_result.reasoning,
            next_action=llm_result.next_action,
            replanning_guidance=llm_result.replanning_guidance,
            clarification_question=llm_result.clarification_question,
            reflection_time_ms=(time.time() - start_time) * 1000,  # Programmatic
            llm_calls_made=1,  # Programmatic
        )

        processing_time = (time.time() - start_time) * 1000

        return ReflectionOutput(
            result=reflection_result,
            success=True,  # Programmatic
            processing_time_ms=processing_time,  # Programmatic
        )

    def _format_plan_steps(self, steps: list[dict]) -> str:
        """Format plan steps for the prompt."""
        if not steps:
            return "No steps planned"

        formatted = []
        for idx, step in enumerate(steps, 1):
            tool = step.get("tool_name", "unknown")
            description = step.get("step_description", "")
            formatted.append(f"{idx}. {tool}: {description}")

        return "\n".join(formatted)

    def _format_execution_results(self, results: list[dict]) -> str:
        """Format execution results for the prompt."""
        if not results:
            return "No results available"

        formatted = []
        for idx, result in enumerate(results, 1):
            tool = result.get("tool_name", "unknown")
            success = result.get("success", False)
            result_text = result.get("result", "No result")
            error = result.get("error", None)
            status_indicator = "✓" if success else "✗"

            formatted.append(f"{idx}. [{status_indicator}] {tool}")
            formatted.append(f"   Result: {result_text}")
            if error:
                formatted.append(f"   Error: {error}")

        return "\n".join(formatted)
