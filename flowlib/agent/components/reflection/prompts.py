"""
Standard reflection prompt templates for the agent system.

This module provides default prompt templates for reflection that can be
used by agents out of the box to evaluate execution results and improve performance.
"""

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase
from flowlib.providers.llm import PromptConfigOverride


@prompt("reflection_default")
class DefaultReflectionPrompt(ResourceBase):
    """Default reflection prompt for evaluating agent execution results."""
    
    template: str = """
    You are a reflection system for an autonomous agent.
    
    Your task is to analyze the execution results and provide insights to improve future planning.
    
    Task description: {{task_description}}
    Execution Type: {{flow_name}} # Indicates 'PlanExecution', 'PlanExecution_Failed', etc.
    
    Planning rationale: {{planning_rationale}}

    Execution status:
    {{flow_status}}
    
    Execution Result Summary (Could be the final step result or an error summary):
    {{flow_result}}
    
    Execution history:
    {{execution_history_text}}

    State summary:
    {{state_summary}}

    Current progress:
    {{current_progress}}
    
    Analyze the overall execution outcome based on the status, result summary, and history:
    1. Assess Success: Was the overall plan/execution successful in achieving the task or making progress? Explain why, considering the status and result.
    2. Analyze Plan/Steps: Review the execution history. Were the planned steps logical? Did they execute as expected?
    3. Identify Issues: If the status is ERROR, what caused the failure (planning error, step error)? Was it recoverable?
    4. Task Completion: Based on the execution, is the overall task now complete? If so, why? If not, what needs to happen next?
    5. Future Improvements: What lessons can be learned? How could planning or execution be improved for similar tasks?
    """
    config: PromptConfigOverride = PromptConfigOverride(
        max_tokens=1024,
        temperature=0.3,
        top_p=0.95,
        top_k=40
    )


@prompt("task_completion_reflection")
class TaskCompletionReflectionPrompt(ResourceBase):
    """Reflection prompt focused on determining task completion."""
    
    template: str = """
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
    config: PromptConfigOverride = PromptConfigOverride(
        max_tokens=1024,
        temperature=0.2,
        top_p=0.9,
        top_k=30
    ) 