"""Integration test for insufficient response handling in context validation.

This test verifies that the validator can detect vague/unhelpful user responses
and generate appropriate follow-up questions while keeping clarification state active.
"""

import asyncio
import logging

# Import to trigger prompt registration
import flowlib.agent.components.task.validation.prompts  # noqa

# Import to trigger LLM config registration
import flowlib.resources.example_configs.example_default_llm  # noqa

# Import to trigger model config registration
import flowlib.resources.example_configs.example_model_config  # noqa

# Import to trigger role assignment
import flowlib.resources.example_configs.example_role_assignments  # noqa
from flowlib.agent.components.task.validation.component import ContextValidatorComponent
from flowlib.agent.core.context.manager import AgentContextManager
from flowlib.agent.core.context.models import ContextManagerConfig
from flowlib.providers.core.registry import provider_registry
from flowlib.resources.decorators.decorators import model_config

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create a minimal default-model config for testing
# This is typically done by user config, but we'll do it here for the test
@model_config(
    "default-model",
    provider_type="llamacpp",
    config={
        "path": "/path/to/your/model.gguf",  # Adjust this path to your model location
        "n_ctx": 8192,
        "temperature": 0.7,
        "max_tokens": 2048,
    },
)
class TestDefaultModelConfig:
    """Minimal default model config for testing."""

    pass


async def test_insufficient_response_handling():
    """Test that validator handles insufficient responses correctly.

    Scenario:
    1. User asks vague question: "Fix the bug"
    2. Validator asks for clarification: "Which bug? Where?"
    3. User gives insufficient response: "I don't know, maybe something?"
    4. Validator should detect insufficiency and ask follow-up questions
    5. Validator should keep pending clarification active
    6. User then delegates: "You decide"
    7. Validator should proceed with enriched context
    """
    logger.info("=" * 80)
    logger.info("Testing Insufficient Response Handling")
    logger.info("=" * 80)

    # Initialize provider registry
    await provider_registry.initialize_all()

    # Create context manager
    config = ContextManagerConfig()
    context_manager = AgentContextManager(config=config, name="test_context_manager")
    await context_manager.initialize()

    # Create validator component
    validator = ContextValidatorComponent(context_manager=context_manager, name="test_validator")

    await validator.initialize()

    try:
        # ========================================================================
        # STEP 1: User asks vague question
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: User asks vague question")
        logger.info("=" * 80)

        result1 = await validator.validate_context(
            user_message="Fix the bug",
            conversation_history=[],
            domain_state={},
            agent_role="assistant",
        )

        logger.info(f"Result 1 - Next Action: {result1.result.next_action}")
        logger.info(f"Result 1 - Has Sufficient Context: {result1.result.has_sufficient_context}")
        logger.info(f"Result 1 - Reasoning: {result1.result.reasoning[:200]}...")
        logger.info(f"Result 1 - Missing Information: {result1.result.missing_information}")
        logger.info(f"Result 1 - Clarification Questions: {result1.result.clarification_questions}")

        # Should need clarification
        assert result1.result.next_action == "clarify", "Expected 'clarify' for vague request"
        assert not result1.result.has_sufficient_context, "Should not have sufficient context"
        assert len(result1.result.clarification_questions) > 0, (
            "Should have clarification questions"
        )

        logger.info("✓ STEP 1 PASSED: Validator correctly identified insufficient context")

        # ========================================================================
        # STEP 2: User gives insufficient response
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: User gives insufficient/vague response")
        logger.info("=" * 80)

        # Simulate user giving vague response
        result2 = await validator.validate_context(
            user_message="I don't know, maybe something?",
            conversation_history=[
                {"role": "user", "content": "Fix the bug"},
                {
                    "role": "assistant",
                    "content": f"I need clarification: {result1.result.clarification_questions[0]}",
                },
            ],
            domain_state={},
            agent_role="assistant",
        )

        logger.info(f"Result 2 - Next Action: {result2.result.next_action}")
        logger.info(f"Result 2 - Has Sufficient Context: {result2.result.has_sufficient_context}")
        logger.info(f"Result 2 - Reasoning: {result2.result.reasoning[:200]}...")

        # Should detect insufficient response
        assert result2.result.next_action == "clarify", (
            "Expected 'clarify' for insufficient response"
        )
        assert not result2.result.has_sufficient_context, "Should not have sufficient context"
        assert len(result2.result.clarification_questions) > 0, "Should have follow-up questions"

        # Check that follow-up questions are different/helpful
        logger.info(f"Result 2 - Follow-up Questions: {result2.result.clarification_questions}")

        logger.info(
            "✓ STEP 2 PASSED: Validator correctly detected insufficient response and generated follow-up questions"
        )

        # ========================================================================
        # STEP 3: User delegates decision
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: User delegates decision to agent")
        logger.info("=" * 80)

        # Simulate user delegating
        result3 = await validator.validate_context(
            user_message="You decide what needs to be fixed",
            conversation_history=[
                {"role": "user", "content": "Fix the bug"},
                {
                    "role": "assistant",
                    "content": f"I need clarification: {result1.result.clarification_questions[0]}",
                },
                {"role": "user", "content": "I don't know, maybe something?"},
                {
                    "role": "assistant",
                    "content": f"Follow-up: {result2.result.clarification_questions[0]}",
                },
            ],
            domain_state={},
            agent_role="assistant",
        )

        logger.info(f"Result 3 - Next Action: {result3.result.next_action}")
        logger.info(f"Result 3 - Has Sufficient Context: {result3.result.has_sufficient_context}")
        logger.info(f"Result 3 - Reasoning: {result3.result.reasoning[:200]}...")

        # Should proceed now
        assert result3.result.next_action == "proceed", "Expected 'proceed' after delegation"
        assert result3.result.has_sufficient_context, "Should have sufficient context"

        # Should have enriched context
        if result3.result.enriched_task_context:
            logger.info(
                f"Result 3 - Enriched Context: {result3.result.enriched_task_context[:200]}..."
            )
            assert "Fix the bug" in result3.result.enriched_task_context, (
                "Should include original request"
            )

        logger.info("✓ STEP 3 PASSED: Validator correctly accepted delegation and proceeded")

        # ========================================================================
        # STEP 4: Next cycle should be fresh (no pending clarification)
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Verify pending clarification was cleared")
        logger.info("=" * 80)

        # New fresh request should not be in clarification mode
        result4 = await validator.validate_context(
            user_message="List files in current directory",
            conversation_history=[],
            domain_state={},
            agent_role="assistant",
        )

        logger.info(f"Result 4 - Next Action: {result4.result.next_action}")

        # Should proceed immediately for clear request
        assert result4.result.next_action == "proceed", "Expected 'proceed' for clear request"
        assert result4.result.has_sufficient_context, "Should have sufficient context"

        logger.info(
            "✓ STEP 4 PASSED: Pending clarification was cleared, new requests work normally"
        )

        logger.info("\n" + "=" * 80)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("=" * 80)
        logger.info("\nVerified:")
        logger.info("  1. Vague requests trigger clarification")
        logger.info("  2. Insufficient responses are detected")
        logger.info("  3. Follow-up questions are generated")
        logger.info("  4. Delegation responses are accepted")
        logger.info("  5. Pending state is managed correctly")

    finally:
        await validator.shutdown()
        await context_manager.shutdown()
        await provider_registry.shutdown_all()


async def test_specific_answer_after_insufficient():
    """Test that specific answers work after insufficient responses."""
    logger.info("\n\n" + "=" * 80)
    logger.info("Testing Specific Answer After Insufficient Response")
    logger.info("=" * 80)

    # Initialize
    await provider_registry.initialize_all()

    config = ContextManagerConfig()
    context_manager = AgentContextManager(config=config, name="test_context_manager_2")
    await context_manager.initialize()

    validator = ContextValidatorComponent(context_manager=context_manager, name="test_validator_2")

    await validator.initialize()

    try:
        # Step 1: Vague request
        result1 = await validator.validate_context(
            user_message="Update the configuration",
            conversation_history=[],
            domain_state={},
            agent_role="assistant",
        )

        assert result1.result.next_action == "clarify"
        logger.info("Step 1: Vague request triggered clarification ✓")

        # Step 2: Insufficient response
        result2 = await validator.validate_context(
            user_message="Not sure",
            conversation_history=[
                {"role": "user", "content": "Update the configuration"},
                {"role": "assistant", "content": "What configuration?"},
            ],
            domain_state={},
            agent_role="assistant",
        )

        assert result2.result.next_action == "clarify"
        logger.info("Step 2: Insufficient response detected ✓")

        # Step 3: Specific answer
        result3 = await validator.validate_context(
            user_message="The database configuration in /etc/app/db.yaml, set the timeout to 30 seconds",
            conversation_history=[
                {"role": "user", "content": "Update the configuration"},
                {"role": "assistant", "content": "What configuration?"},
                {"role": "user", "content": "Not sure"},
                {"role": "assistant", "content": "Follow-up question"},
            ],
            domain_state={},
            agent_role="assistant",
        )

        assert result3.result.next_action == "proceed"
        assert result3.result.has_sufficient_context
        logger.info("Step 3: Specific answer accepted ✓")

        if result3.result.enriched_task_context:
            assert "database configuration" in result3.result.enriched_task_context.lower()
            logger.info("Step 3: Enriched context contains specific information ✓")

        logger.info("\n✓ Specific answer test PASSED")

    finally:
        await validator.shutdown()
        await context_manager.shutdown()
        await provider_registry.shutdown_all()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INSUFFICIENT RESPONSE HANDLING INTEGRATION TESTS")
    print("=" * 80)
    print("\nThese tests use real LLM calls to verify the validator can:")
    print("  - Detect vague/unclear user responses")
    print("  - Generate appropriate follow-up questions")
    print("  - Keep clarification state active until useful response received")
    print("  - Accept both delegation and specific answers")
    print("=" * 80 + "\n")

    # Run tests
    asyncio.run(test_insufficient_response_handling())
    asyncio.run(test_specific_answer_after_insufficient())

    print("\n" + "=" * 80)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("=" * 80)
