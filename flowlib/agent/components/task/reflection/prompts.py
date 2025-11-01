"""Prompts for execution reflection."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt


@prompt(name="execution-reflection-prompt")
class ExecutionReflectionPrompt:
    """Prompt for reflecting on execution results to determine if re-planning is needed."""

    template: str = Field(
        default="""Goal: {{original_goal}}

Planned:
{{planned_steps}}

Results:
{{execution_results}}

Analysis:
1. Did steps succeed?
2. Any errors?
3. Goal satisfied?

Next action:
- continue: Success, done
- replan: Failed, must provide replanning_guidance explaining what to do differently
- clarify: Need user input, must provide clarification_question

If replan: Include replanning_guidance
If clarify: Include clarification_question"""
    )
