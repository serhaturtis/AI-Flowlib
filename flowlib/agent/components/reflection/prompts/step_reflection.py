from typing import Dict, Any, ClassVar
from flowlib.resources.models.base import ResourceBase

from flowlib.agent.components.reflection.models import StepReflectionResult

# Import decorator
from flowlib.resources.decorators.decorators import prompt

STEP_REFLECTION_TEMPLATE = '''
<|im_start|>user
You are a reflection assistant for an autonomous agent.
Your task is to analyze the outcome of a single step executed within a larger plan.

**Overall Task:**
{{task_description}}

**Current Step Details:**
- Step ID: {{step_id}}
- Intent: {{step_intent}}
- Rationale: {{step_rationale}}
- Flow Executed: {{flow_name}}

**Step Execution Inputs:**
```
{{flow_inputs_formatted}}
```

**Step Execution Result:**
```
{{flow_result_formatted}}
```

**Progress Before Step:** {{current_progress}}%

**Analysis Task:**
Based ONLY on the information provided about this specific step and its result:
1. Briefly reflect on whether the step achieved its intent. Was it successful?
2. Identify the single most important observation or outcome from this step (e.g., data obtained, error encountered, state change).

Please format your response as a JSON object adhering to the following schema:
```json
{
  "title": "StepReflectionResult",
  "description": "Model for structured reflection results after a single plan step.",
  "properties": {
    "step_id": {
      "description": "ID of the plan step being reflected upon",
      "title": "Step Id",
      "type": "string"
    },
    "reflection": {
      "description": "A brief analysis of the step's outcome",
      "title": "Reflection",
      "type": "string"
    },
    "step_success": {
      "description": "Whether the step itself succeeded",
      "title": "Step Success",
      "type": "boolean"
    },
    "key_observation": {
      "anyOf": [
        {
          "type": "string"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "description": "Most important observation from this step",
      "title": "Key Observation"
    }
  },
  "required": [
    "step_id",
    "reflection",
    "step_success"
  ]
}
```
Ensure your JSON output includes the `step_id` provided above.
<|im_end|>assistant
'''

@prompt("step_reflection_default")
class DefaultStepReflectionPrompt(ResourceBase):
    """Default prompt for single-step reflection."""
    
    template: ClassVar[str] = STEP_REFLECTION_TEMPLATE
    output_model: ClassVar[type] = StepReflectionResult

    def format(self, **kwargs: Any) -> str:
        # Basic check for required keys
        required_keys = [
            "task_description", "step_id", "step_intent", "step_rationale",
            "flow_name", "flow_inputs_formatted", "flow_result_formatted",
            "current_progress"
        ]
        for key in required_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required key for Step Reflection Prompt: {key}")
        
        # Use the class attribute template
        return self.template.format(**kwargs) 