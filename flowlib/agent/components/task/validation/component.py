"""Context validation component."""

import logging
from typing import Any

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.context.manager import AgentContextManager
from flowlib.flows.registry.registry import flow_registry

from .models import (
    ClarificationResponseParsingInput,
    PendingClarification,
    ValidationInput,
    ValidationOutput,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ContextValidatorComponent(AgentComponent):
    """Validates information sufficiency before planning.

    This implements the Clarification step from CEP Framework (EMNLP 2024),
    ensuring agents gather sufficient context before taking action.

    The validator performs pure assessment - it does NOT choose tools or route
    execution. The planner uses the validation result to decide whether to
    create a clarification plan or a task execution plan.
    """

    def __init__(
        self, context_manager: AgentContextManager | None = None, name: str = "context_validator"
    ):
        """Initialize context validator.

        Args:
            context_manager: Agent context manager (required)
            name: Component name
        """
        super().__init__(name)

        if not context_manager:
            raise ValueError("ContextValidatorComponent requires AgentContextManager")

        self._context_manager = context_manager
        self._flow: Any | None = None
        self._parsing_flow: Any | None = None

        # Clarification state tracking
        self._pending_clarification: PendingClarification | None = None

    async def _initialize_impl(self) -> None:
        """Initialize the validator."""
        self._flow = flow_registry.get_flow("context-validation")
        self._parsing_flow = flow_registry.get_flow("clarification-response-parsing")
        logger.info("Context validator initialized")

    async def _shutdown_impl(self) -> None:
        """Shutdown the validator."""
        logger.info("Context validator shutdown")

    async def validate_context(
        self,
        user_message: str,
        conversation_history: list[dict] | None = None,
        domain_state: dict[str, Any] | None = None,
        agent_role: str = "assistant",
    ) -> ValidationOutput:
        """Validate whether there is sufficient context to proceed.

        This method manages the complete clarification lifecycle:
        1. If pending clarification exists, parse user's response
        2. Otherwise, perform normal validation
        3. If validation returns "clarify", store pending state

        Args:
            user_message: The user's current message
            conversation_history: Recent conversation context
            domain_state: Current domain-specific state
            agent_role: Agent's role

        Returns:
            ValidationOutput with sufficiency assessment
        """
        self._check_initialized()

        if not self._flow:
            raise RuntimeError("Validation flow not initialized")

        # ===================================================================
        # BRANCH A: Pending Clarification - User is responding to our questions
        # ===================================================================
        if self._pending_clarification:
            logger.info("Pending clarification detected - parsing user response")
            return await self._handle_clarification_response(
                user_message, conversation_history or [], domain_state or {}, agent_role
            )

        # ===================================================================
        # BRANCH B: Fresh Request - Normal validation
        # ===================================================================
        logger.info(f"Validating fresh request: {user_message[:50]}...")

        # Create validation input
        validation_input = ValidationInput(
            user_message=user_message,
            conversation_history=conversation_history or [],
            domain_state=domain_state or {},
            agent_role=agent_role,
        )

        # Run validation flow
        result = await self._flow.run_pipeline(validation_input)

        # Log result
        validation = result.result
        logger.info(
            f"Validation result: {validation.next_action} "
            f"(confidence: {validation.confidence:.2f}) - {validation.reasoning[:100]}..."
        )

        # If clarification needed, store pending state
        if validation.next_action == "clarify":
            logger.info(f"Missing information: {validation.missing_information}")
            logger.info(f"Clarification questions: {validation.clarification_questions}")

            # Store pending clarification
            self._pending_clarification = PendingClarification(
                original_message=user_message,
                questions_asked=validation.clarification_questions,
                missing_information=validation.missing_information,
            )
            logger.info(
                f"Stored pending clarification with {len(validation.clarification_questions)} questions"
            )

        return result

    async def _handle_clarification_response(
        self,
        user_message: str,
        conversation_history: list[dict],
        domain_state: dict[str, Any],
        agent_role: str,
    ) -> ValidationOutput:
        """Handle user response to clarification questions using LLM-based parsing.

        Uses LLM intelligence to understand:
        - Delegation ("you decide", "give recommendations")
        - Specific answers provided by user
        - Mixed responses (some delegation, some specific)

        Args:
            user_message: User's response to clarification
            conversation_history: Conversation context
            domain_state: Domain state
            agent_role: Agent role

        Returns:
            ValidationOutput with "proceed" action and enriched context
        """
        assert self._pending_clarification is not None
        assert self._parsing_flow is not None

        logger.info(
            f"Parsing clarification response for {len(self._pending_clarification.questions_asked)} questions"
        )

        # Create parsing input
        parsing_input = ClarificationResponseParsingInput(
            user_response=user_message,
            questions_asked=self._pending_clarification.questions_asked,
            missing_information=self._pending_clarification.missing_information,
            original_request=self._pending_clarification.original_message,
        )

        # Parse response using LLM
        parsing_output = await self._parsing_flow.run_pipeline(parsing_input)
        parsing_result = parsing_output.result

        logger.info(
            f"Response parsed as '{parsing_result.response_type}' "
            f"(confidence: {parsing_result.confidence:.2f})"
        )
        logger.info(f"Parsing reasoning: {parsing_result.reasoning}")

        # Log delegation and provided information
        if parsing_result.delegation_items:
            logger.info(f"Delegation items: {parsing_result.delegation_items}")
        if parsing_result.provided_information:
            logger.info(f"Provided information: {parsing_result.provided_information}")

        # ===================================================================
        # BRANCH A: Insufficient Response - Need to ask follow-up questions
        # ===================================================================
        if parsing_result.response_type == "insufficient":
            logger.info("User response is insufficient - generating follow-up questions")
            logger.info(f"Follow-up questions: {parsing_result.follow_up_questions}")

            # DO NOT clear pending state - we're still waiting for proper answers
            # Keep original message and missing information for next iteration
            logger.info(
                "Keeping pending clarification active - user needs to provide better response"
            )

            # Return CLARIFY with follow-up questions
            return ValidationOutput(
                result=ValidationResult(
                    has_sufficient_context=False,
                    confidence=parsing_result.confidence,
                    reasoning=(
                        f"User's response was insufficient: {parsing_result.enrichment_note} "
                        f"Need to ask follow-up questions for clarification."
                    ),
                    next_action="clarify",
                    missing_information=self._pending_clarification.missing_information,
                    clarification_questions=parsing_result.follow_up_questions,
                    validation_time_ms=parsing_result.parsing_time_ms,
                    llm_calls_made=parsing_result.llm_calls_made,
                ),
                success=True,
                processing_time_ms=parsing_output.processing_time_ms,
            )

        # ===================================================================
        # BRANCH B: Useful Response (delegation/specific_answers/mixed) - Proceed
        # ===================================================================
        # Create enriched context as natural language (no meta-instructions)
        # The planner should see a complete, natural user request

        if parsing_result.response_type == "delegation":
            if not parsing_result.delegation_items:
                raise RuntimeError(
                    "Parser returned delegation type but delegation_items is empty - consistency violation"
                )

            # Pure delegation: Transform to natural instruction
            delegated_items_str = ", ".join(parsing_result.delegation_items)
            enriched_task_context = (
                f"{self._pending_clarification.original_message} "
                f"Choose appropriate values for {delegated_items_str} based on best practices and domain expertise."
            )

        elif parsing_result.response_type == "specific_answers":
            if not parsing_result.provided_information:
                raise RuntimeError(
                    "Parser returned specific_answers type but provided_information is empty - consistency violation"
                )

            # Specific answers: Incorporate provided values naturally
            info_parts = [
                f"{key}={value}" for key, value in parsing_result.provided_information.items()
            ]
            enriched_task_context = (
                f"{self._pending_clarification.original_message} "
                f"Use the following specifications: {', '.join(info_parts)}."
            )

        elif parsing_result.response_type == "mixed":
            if not parsing_result.delegation_items or not parsing_result.provided_information:
                raise RuntimeError(
                    "Parser returned mixed type but missing delegation_items or provided_information - consistency violation"
                )

            # Mixed: Combine both delegation and specific values
            parts = [self._pending_clarification.original_message]

            info_parts = [
                f"{key}={value}" for key, value in parsing_result.provided_information.items()
            ]
            parts.append(f"Use the following specifications: {', '.join(info_parts)}.")

            delegated_items_str = ", ".join(parsing_result.delegation_items)
            parts.append(
                f"Choose appropriate values for {delegated_items_str} based on best practices."
            )

            enriched_task_context = " ".join(parts)

        else:
            raise RuntimeError(f"Unexpected response_type: {parsing_result.response_type}")

        # Clear pending state - we got useful information
        original_msg = self._pending_clarification.original_message
        self._pending_clarification = None
        logger.info(
            f"Cleared pending clarification - proceeding with enriched context for: {original_msg[:50]}..."
        )

        # Return PROCEED with enriched context
        # The enriched_task_context will be used by the engine/planner instead of current message
        return ValidationOutput(
            result=ValidationResult(
                has_sufficient_context=True,
                confidence=parsing_result.confidence,
                reasoning=(
                    f"User responded to clarification request. {parsing_result.enrichment_note} "
                    f"Context is now sufficient to proceed with planning."
                ),
                next_action="proceed",
                missing_information=[],
                clarification_questions=[],
                enriched_task_context=enriched_task_context,  # Pass enriched context
                validation_time_ms=parsing_result.parsing_time_ms,
                llm_calls_made=parsing_result.llm_calls_made,  # LLM was used for parsing
            ),
            success=True,
            processing_time_ms=parsing_output.processing_time_ms,
        )
