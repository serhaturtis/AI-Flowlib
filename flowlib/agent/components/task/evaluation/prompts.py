"""Prompts for task completion evaluation."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt(name="completion-evaluation-prompt")
class CompletionEvaluationPrompt(ResourceBase):
    """Prompt for evaluating whether a task is complete.

    This is used in the Plan-and-Execute loop to determine if more planning/execution
    cycles are needed.
    """

    template: str = Field(default="""Evaluate if the task completed successfully.

Original Goal: {{original_goal}}
Expected Outcome: {{expected_outcome}}

Executed Steps:
{{executed_steps}}

# Evaluation Rules

1. **Check execution results**: ALL steps must have success=true for task to be complete
2. **Verify outcome**: Expected outcome must be achieved, not just steps executed
3. **Conversational tasks**: If only step is "conversation" and succeeded → task complete

# Next Action

- **done**: All steps succeeded and outcome achieved
- **continue**: Steps failed, retry needed (provide replanning_guidance)
- **clarify**: Missing info or unclear requirements (provide clarification_question with actual error message)

Analyze the results and determine next action.""")
