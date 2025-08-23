from flowlib.resources.models.base import ResourceBase
from flowlib.agent.components.reflection.models import ReflectionResult
from typing import Any, ClassVar

# Import decorator
from flowlib.resources.decorators.decorators import prompt

DEFAULT_REFLECTION_TEMPLATE = '''
You are a reflection assistant for an autonomous agent.
Your task is to analyze the outcome of a completed multi-step plan execution, considering the reflections from each individual step, to determine the overall progress and status of the main task.

# **Overall Task:**
{{task_description}}

# **Current Agent State Summary:**
{{state_summary}}

# **Current Progress:** {{current_progress}}%

# **Plan Execution Outcome:**
- Overall Status: {{plan_status}}
- Error (if any): {{plan_error}}

# **Summary of Step Reflections:**
{{step_reflections_summary}}

# **Execution History Summary (optional context):**
{{execution_history_text}}

# **Analysis Task:**
Based on the overall task, the plan's outcome, the step reflections, and the agent's state/history:
1. Write a brief reflection on the overall success or failure of the executed plan in achieving the task goals.
2. Estimate the new overall task progress percentage (0-100).
3. Determine if the main task is now complete based on the plan outcome and step reflections. If yes, provide a brief completion reason.
4. Identify any key insights or lessons learned from this plan execution.

Please format your response as a JSON object adhering to the following schema:
```json
{
  "title": "ReflectionResult",
  "description": "Model for structured reflection results.",
  "properties": {
    "reflection": {
      "description": "A detailed analysis of what happened and why",
      "title": "Reflection",
      "type": "string"
    },
    "progress": {
      "default": 0,
      "description": "An estimate of overall task progress (0-100)",
      "title": "Progress",
      "type": "integer"
    },
    "is_complete": {
      "default": false,
      "description": "Whether the task is complete",
      "title": "Is Complete",
      "type": "boolean"
    },
    "completion_reason": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "If is_complete is true, the reason the task is complete",
      "title": "Completion Reason"
    },
    "insights": {
      "anyOf": [
        {
          "items": {
            "type": "string"
          },
          "type": "array"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Key insights or lessons learned from this execution",
      "title": "Insights"
    }
  },
  "required": [
    "reflection",
    "progress",
    "is_complete"
  ]
}
```
'''

# --- Reflection Prompt --- 

@prompt("reflection_default")
class DefaultReflectionPrompt(ResourceBase):
    """Default prompt for overall plan reflection."""
    
    template: ClassVar[str] = DEFAULT_REFLECTION_TEMPLATE
    output_model: ClassVar[type] = ReflectionResult

    def format(self, **kwargs: Any) -> str:
        # Basic check for required keys for the *new* template
        required_keys = [
            "task_description", "plan_status", "plan_error", 
            "step_reflections_summary", "execution_history_text",
            "state_summary", "current_progress"
        ]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required key for Overall Reflection Prompt: {key}")
        
        # Use the class attribute template
        return self.template.format(**kwargs) 

# --- Task Completion Prompt --- 

# TODO: Decide if this prompt needs updating for plan context or if it's used differently
@prompt("task_completion_reflection")
class TaskCompletionReflectionPrompt(ResourceBase):
    """Reflection prompt focused on determining task completion."""
    
    template: ClassVar[str] = """
    You are a task evaluation system for an autonomous agent.
    
    Your task is to determine if the current task has been completed based on execution results.
    
    Task description: {{task_description}}
    Flow name: {{flow_name}}
    
    Flow execution result:
    {{flow_result}}
    
    Execution history summary:
    {{execution_history_text}}
    
    Evaluate whether the task described has been completed.
    
    GUIDELINES:
    - Be strict in your evaluation - only mark a task complete if there's clear evidence
    - For multi-step tasks, ensure all steps have been addressed
    - For information requests, ensure the information has been provided accurately
    - For action tasks, ensure the actions have been executed successfully
    """
    config: ClassVar[dict] = {
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 30
    }