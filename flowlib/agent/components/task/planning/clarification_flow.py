"""Clarification planning flow for gathering missing information.

This flow creates simple plans with just a conversation step to ask
the user for missing information identified by the context validator.
"""

import logging
import time
import uuid

from flowlib.flows.decorators.decorators import flow, pipeline

from .models import (
    PlanningInput,
    PlanningOutput,
    PlanStep,
    StructuredPlan,
)

logger = logging.getLogger(__name__)


@flow(  # type: ignore[arg-type]
    name="clarification-planning",
    description="Generate clarification plans to gather missing information",
    is_infrastructure=False,
)
class ClarificationPlanningFlow:
    """Generates simple clarification plans with conversation tool."""

    @pipeline(input_model=PlanningInput, output_model=PlanningOutput)
    async def run_pipeline(self, input_data: PlanningInput) -> PlanningOutput:
        """Generate clarification plan to ask user for missing information.

        Args:
            input_data: Contains user message, validation result with questions

        Returns:
            PlanningOutput with single-step clarification plan
        """
        start_time = time.time()

        # Validation result must have clarification questions
        if (
            not input_data.validation_result
            or not input_data.validation_result.clarification_questions
        ):
            raise ValueError(
                "Clarification planning requires validation_result with clarification_questions"
            )

        # Format the clarification message nicely
        questions = input_data.validation_result.clarification_questions
        missing_info = input_data.validation_result.missing_information or []

        # Build clarification message
        message_parts = ["I need some additional information to proceed:"]

        if missing_info:
            message_parts.append("\nMissing:")
            for info in missing_info:
                message_parts.append(f"  • {info}")

        message_parts.append("\nPlease provide:")
        for i, question in enumerate(questions, 1):
            message_parts.append(f"  {i}. {question}")

        clarification_message = "\n".join(message_parts)

        # Create simple plan with conversation tool
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        conversation_step = PlanStep(
            step_id=f"{plan_id}_step_0",
            tool_name="conversation",
            step_description="Ask user for missing information",
            parameters={"message": clarification_message},
            depends_on_step=None,
            executed=False,
            result=None,
        )

        plan = StructuredPlan(
            message_type="conversation",
            reasoning=f"Need to clarify: {', '.join(missing_info) if missing_info else 'missing information'}",
            steps=[conversation_step],
            expected_outcome="User provides the missing information",
            plan_id=plan_id,
            created_at=time.time(),
            execution_started=False,
            execution_complete=False,
        )

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Generated clarification plan in {processing_time:.2f}ms")

        return PlanningOutput(
            plan=plan,
            success=True,
            processing_time_ms=processing_time,
            llm_calls_made=0,  # No LLM calls for clarification
        )
