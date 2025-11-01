"""Context validation flow."""

import time
from typing import cast

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry

from .models import (
    ClarificationResponseParsingInput,
    ClarificationResponseParsingOutput,
    ClarificationResponseParsingResult,
    LLMClarificationResponseParsing,
    LLMClarifyGeneration,
    LLMContextClassification,
    LLMProceedGeneration,
    LLMValidationResult,
    ValidationInput,
    ValidationOutput,
    ValidationResult,
)


@flow(
    name="context-validation",
    description="Validate whether sufficient context exists to proceed with planning",
    is_infrastructure=False,
)
class ContextValidationFlow:
    """Validates information sufficiency before planning (CEP Framework)."""

    @pipeline(input_model=ValidationInput, output_model=ValidationOutput)
    async def run_pipeline(self, input_data: ValidationInput) -> ValidationOutput:
        """Validate context sufficiency using two-step approach.

        Step 1: Classification (simple boolean decision)
        Step 2: Generation (detailed output based on classification)

        This prevents the LLM from generating inconsistent structured output
        where has_sufficient_context and next_action contradict each other.

        Args:
            input_data: Contains user message, conversation history, domain state

        Returns:
            ValidationOutput with sufficiency assessment
        """
        start_time = time.time()

        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))

        # Format conversation history
        conversation_text = self._format_conversation(input_data.conversation_history)

        # Format domain state
        domain_state_text = self._format_domain_state(input_data.domain_state)

        # Prepare common prompt variables
        prompt_vars = {
            "user_message": input_data.user_message,
            "agent_role": input_data.agent_role,
            "conversation_history": conversation_text,
            "domain_state": domain_state_text,
        }

        # ====================================================================
        # STEP 1: CLASSIFICATION - Simple boolean decision
        # ====================================================================

        classification_prompt = resource_registry.get("context-classification-prompt")

        classification_result = await llm.generate_structured(
            prompt=cast(PromptTemplate, classification_prompt),
            output_type=LLMContextClassification,
            model_name="default-model",
            prompt_variables=cast(dict[str, object], prompt_vars),
        )

        # Convert confidence string to float
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        confidence = confidence_map.get(classification_result.confidence.lower(), 0.6)

        # ====================================================================
        # STEP 2: GENERATION - Detailed output based on classification
        # ====================================================================

        llm_calls = 1

        if classification_result.has_sufficient_context:
            # BRANCH A: Context is sufficient - generate detailed reasoning
            proceed_prompt = resource_registry.get("context-proceed-generation-prompt")

            proceed_result = await llm.generate_structured(
                prompt=cast(PromptTemplate, proceed_prompt),
                output_type=LLMProceedGeneration,
                model_name="default-model",
                prompt_variables=cast(dict[str, object], prompt_vars),
            )

            llm_calls += 1

            # Combine classification + generation
            validation_result = ValidationResult(
                has_sufficient_context=True,
                confidence=confidence,
                reasoning=f"{classification_result.reasoning} {proceed_result.detailed_reasoning}",
                next_action="proceed",  # Consistent with classification
                missing_information=[],
                clarification_questions=[],
                validation_time_ms=(time.time() - start_time) * 1000,
                llm_calls_made=llm_calls,
            )

        else:
            # BRANCH B: Context is insufficient - generate clarification questions
            clarify_prompt = resource_registry.get("context-clarify-generation-prompt")

            clarify_result = await llm.generate_structured(
                prompt=cast(PromptTemplate, clarify_prompt),
                output_type=LLMClarifyGeneration,
                model_name="default-model",
                prompt_variables=cast(dict[str, object], prompt_vars),
            )

            llm_calls += 1

            # Combine classification + generation
            validation_result = ValidationResult(
                has_sufficient_context=False,
                confidence=confidence,
                reasoning=f"{classification_result.reasoning} {clarify_result.detailed_reasoning}",
                next_action="clarify",  # Consistent with classification
                missing_information=clarify_result.missing_information,
                clarification_questions=clarify_result.clarification_questions,
                validation_time_ms=(time.time() - start_time) * 1000,
                llm_calls_made=llm_calls,
            )

        processing_time = (time.time() - start_time) * 1000

        return ValidationOutput(
            result=validation_result, success=True, processing_time_ms=processing_time
        )

    def _format_conversation(self, history: list[dict]) -> str:
        """Format conversation history for prompt."""
        if not history:
            return "No previous conversation"

        formatted = []
        for msg in history[-5:]:  # Last 5 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def _format_domain_state(self, state: dict) -> str:
        """Format domain state for prompt."""
        if not state:
            return "No domain state"

        formatted = []
        for key, value in state.items():
            formatted.append(f"- {key}: {value}")

        return "\n".join(formatted)


@flow(
    name="clarification-response-parsing",
    description="Parse user responses to clarification questions using LLM intelligence",
    is_infrastructure=False,
)
class ClarificationResponseParsingFlow:
    """Parses user responses to clarification questions using LLM.

    This replaces hardcoded regex patterns with LLM-based understanding
    of user intent, delegation, and specific information provided.
    """

    @pipeline(
        input_model=ClarificationResponseParsingInput,
        output_model=ClarificationResponseParsingOutput,
    )
    async def run_pipeline(
        self, input_data: ClarificationResponseParsingInput
    ) -> ClarificationResponseParsingOutput:
        """Parse clarification response.

        Args:
            input_data: Contains user response, questions asked, missing info, original request

        Returns:
            ClarificationResponseParsingOutput with structured parsing result
        """
        start_time = time.time()

        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))

        # Get prompt from registry
        prompt_instance = resource_registry.get("clarification-response-parsing-prompt")

        # Format questions and missing information
        questions_text = self._format_list(input_data.questions_asked)
        missing_info_text = self._format_list(input_data.missing_information)

        # Prepare prompt variables
        prompt_vars = {
            "original_request": input_data.original_request,
            "questions_asked": questions_text,
            "missing_information": missing_info_text,
            "user_response": input_data.user_response,
        }

        # Get parsing from LLM
        llm_result = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=LLMClarificationResponseParsing,
            model_name="default-model",
            prompt_variables=cast(dict[str, object], prompt_vars),
        )

        # Convert confidence string to float
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        confidence = confidence_map.get(llm_result.confidence.lower(), 0.6)

        # Validate consistency
        if llm_result.response_type == "delegation" and llm_result.provided_information:
            raise RuntimeError(
                f"LLM generated inconsistent parsing: response_type='delegation' but provided_information is not empty. "
                f"Delegation should not include specific information. Reasoning: {llm_result.reasoning}"
            )

        if llm_result.response_type == "specific_answers" and llm_result.delegation_items:
            raise RuntimeError(
                f"LLM generated inconsistent parsing: response_type='specific_answers' but delegation_items is not empty. "
                f"Specific answers should not include delegation. Reasoning: {llm_result.reasoning}"
            )

        if llm_result.response_type == "insufficient":
            # Insufficient responses should not have delegation or provided info
            if llm_result.delegation_items or llm_result.provided_information:
                raise RuntimeError(
                    f"LLM generated inconsistent parsing: response_type='insufficient' but has delegation_items or provided_information. "
                    f"Insufficient responses should have empty delegation and info. Reasoning: {llm_result.reasoning}"
                )
            # Insufficient responses MUST have follow-up questions
            if not llm_result.follow_up_questions:
                raise RuntimeError(
                    f"LLM generated inconsistent parsing: response_type='insufficient' but no follow_up_questions provided. "
                    f"Must generate follow-up questions for insufficient responses. Reasoning: {llm_result.reasoning}"
                )

        # Create full parsing result
        parsing_result = ClarificationResponseParsingResult(
            response_type=llm_result.response_type,
            confidence=confidence,
            reasoning=llm_result.reasoning,
            delegation_items=llm_result.delegation_items,
            provided_information=llm_result.provided_information,
            enrichment_note=llm_result.enrichment_note,
            follow_up_questions=llm_result.follow_up_questions,
            parsing_time_ms=(time.time() - start_time) * 1000,
            llm_calls_made=1,
        )

        processing_time = (time.time() - start_time) * 1000

        return ClarificationResponseParsingOutput(
            result=parsing_result, success=True, processing_time_ms=processing_time
        )

    def _format_list(self, items: list[str]) -> str:
        """Format list items for prompt."""
        if not items:
            return "None"

        return "\n".join(f"- {item}" for item in items)
