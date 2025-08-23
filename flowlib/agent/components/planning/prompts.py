"""
Standard planning prompt templates for the agent system.

This module provides default prompt templates for planning and input generation
that can be used by agents out of the box.
"""

from typing import ClassVar
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import prompt


@prompt("planning_default")
class DefaultPlanningPrompt(ResourceBase):
    """Default planning prompt for the agent to select the appropriate flow."""
    
    template: ClassVar[str] = """
    You are a planning assistant for an autonomous agent.
    
    Your task is to generate a multi-step plan to achieve the given Task Description, utilizing the available flows.
    Analyze the task, execution history, and memory context to determine the necessary steps.
    Break down the task into logical steps if needed. A single step is acceptable if the task is simple.
    
    Task description: {{task_description}}
    
    Available flows:
    {{available_flows_text}}
    
    Current state:
    - Task: {{task_description}}
    - Cycle: {{cycle}}
    
    Execution history:
    {{execution_history_text}}
    
    ## Memory Context Summary
    {{memory_context_summary}}
    
    CRITICAL REQUIREMENTS:
    - For each step in your plan, you MUST select a flow from the Available Flows list. Do NOT invent flows.
    - Provide a clear rationale for each step, explaining why it's necessary.
    - Generate the required inputs (`flow_inputs`) for each step based on the task, history, and context.
    - Ensure the inputs match the expected schema for the selected flow (schemas provided in available_flows_text).
    - If the task is already complete or requires no action (e.g., simple acknowledgement), output an empty list for the `steps` field.
    - Output your final plan STRICTLY as a JSON object conforming to the following structure:
      ```json
      {
        "plan_id": "<generate_uuid>",
        "task_description": "{{task_description}}",
        "steps": [
          {
            "step_id": "<generate_uuid>",
            "flow_name": "<selected_flow_name>",
            "step_intent": "<intent_or_goal_for_this_step>",
            "rationale": "<rationale_for_this_step>",
            "expected_outcome": "<optional_expected_outcome>"
          }
          // ... more steps if needed ...
        ],
        "overall_rationale": "<optional_overall_plan_rationale>"
      }
      ```
    - Do NOT add any text before or after the JSON object.
    """
    config: ClassVar[dict] = {
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40
    }


@prompt("conversational_planning")
class ConversationalPlanningPrompt(ResourceBase):
    """Planning prompt optimized for conversational agents."""
    
    template: ClassVar[str] = """
    You are a planning assistant for a conversational agent.
    
    Your task is to select the most appropriate flow to execute next based on the user's message and available flows.
    
    User message: {{task_description}}
    Current cycle: {{cycle}}
    
    Available flows:
    {{available_flows_text}}
    
    Conversation history:
    {{execution_history_text}}
    
    {{memory_context}}
    
    CRITICAL REQUIREMENTS:
    - You MUST select a flow from the available flows listed above.
    - ONLY select from flows that are actually available and listed above.
    """
    config: ClassVar[dict] = {
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40
    }


@prompt("input_generation_default")
class DefaultInputGenerationPrompt(ResourceBase):
    """Default prompt template for generating inputs for flows."""
    
    template: ClassVar[str] = """
    You are an input generation assistant for an autonomous agent.
    
    Your task is to generate appropriate inputs for the selected flow based on the current state, task description, and relevant memories.
    
    Task description: {{task_description}}
    
    Selected flow: {{flow_name}}
    Step Intent: {{step_intent}}
    Flow description: {{flow_description}}
    Input schema: {{input_schema}}
    
    Step Rationale: {{planning_rationale}}
    
    Execution history: 
    {{execution_history_text}}
    
    ## Memory Context Summary:
    {{memory_context_summary}}
    
    GUIDELINES:
    - Generate inputs that strictly follow the specified input schema.
    - Ensure inputs are appropriate for the selected flow and the overall task description.
    - Use relevant context from execution history and the Memory Context Summary.
    - For shell command flows:
      * Extract EXACT content, filenames, and specifications from the task description
      * DO NOT use placeholders like "Your content here" or "example text"
      * Use the ACTUAL content requested by the user
      * Be precise with file paths, content, and operations
      * Example: If user says "create file X with content Y", use Y exactly, not a placeholder
    - For conversation flows, ensure the message is natural and appropriate.
    """
    config: ClassVar[dict] = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40
    }


@prompt("conversational_input_generation")
class ConversationalInputGenerationPrompt(ResourceBase):
    """Input generation prompt optimized for conversational flows."""
    
    template: ClassVar[str] = """
    You are an input generation assistant for a conversational agent.
    
    Your task is to generate appropriate inputs for the selected conversation flow.
    
    User message: {{task_description}}
    
    Selected flow: {{flow_name}}
    Flow description: {{flow_description}}
    Input schema: {{input_schema}}
    
    Planning rationale: {{planning_rationale}}
    
    Conversation history: 
    {{execution_history_text}}
    
    Relevant memories:
    {{relevant_memories_text}}
    
    GUIDELINES:
    - Add any relevant context from memories or conversation history
    - Format the inputs according to the input schema
    """
    config: ClassVar[dict] = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40
    } 