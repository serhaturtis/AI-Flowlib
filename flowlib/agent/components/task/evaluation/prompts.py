"""Prompts for task completion evaluation."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt


@prompt(name="completion-evaluation-prompt")
class CompletionEvaluationPrompt:
    """Prompt for evaluating whether a task is complete.

    This is used in the Plan-and-Execute loop to determine if more planning/execution
    cycles are needed.
    """

    template: str = Field(
        default="""Evaluate if the task completed successfully.

Original Goal: {{original_goal}}
Plan Reasoning: {{plan_reasoning}}
Expected Outcome: {{expected_outcome}}

Executed Steps:
{{executed_steps}}

# Evaluation Rules

Note: This evaluation is ONLY called for multi-step tasks. Single-call tasks (conversation/single_tool) and clarification are handled separately.

1. **Check execution results**: ALL steps must have success=true for task to be complete

2. **Verify outcome**: Expected outcome must be achieved, not just steps executed

3. **Consider progress**: Even if incomplete, assess if meaningful progress was made

# Next Action

- **done**: All steps succeeded and expected outcome achieved
- **continue**: Need more execution cycles to complete the task
- **clarify**: Discovered missing information during execution that requires user input

Analyze the results and determine next action."""
    )
