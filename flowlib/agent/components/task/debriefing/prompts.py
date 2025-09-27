"""Prompts for task debriefing flows."""

from typing import ClassVar
from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt("intent-analysis-prompt")
class IntentAnalysisPrompt(ResourceBase):
    """Prompt for analyzing if user's intent was fulfilled."""
    
    template: ClassVar[str] = """You are an intent analysis assistant. Your job is to determine if the user's original intent was actually fulfilled by the task execution results.

ORIGINAL USER REQUEST: {{original_user_message}}

GENERATED TASK: {{generated_task}}

EXECUTION RESULTS:
{{execution_results}}

CONTEXT:
- Agent Persona: {{agent_persona}}
- Working Directory: {{working_directory}}

ANALYSIS INSTRUCTIONS:
1. Understand what the user ACTUALLY wanted (their real intent)
2. Analyze what was ACTUALLY accomplished by the execution
3. Determine if the user's intent was fulfilled (not just if tools succeeded)
4. Assess if any gaps can be corrected with a retry

IMPORTANT NOTES:
- Confidence should be a decimal between 0.0 and 1.0 (not percentage)
- Focus on USER INTENT, not technical success. A command might succeed technically but fail to fulfill what the user wanted

Intent fulfillment means the user got what they actually wanted from their request. For technical requests like checking disk space, this means getting the actual information even if commands initially fail. For conversational interactions like greetings or casual questions, appropriate responses that acknowledge and engage with the user fulfill the intent. Safety considerations may override fulfillment for potentially harmful requests.

Analyze the intent fulfillment thoroughly."""


@prompt("success-presentation-prompt") 
class SuccessPresentationPrompt(ResourceBase):
    """Prompt for generating user-friendly success presentations."""
    
    template: ClassVar[str] = """You are {{agent_persona}} presenting successful task completion to the user.

ORIGINAL USER REQUEST: {{original_user_message}}

EXECUTION RESULTS:
{{execution_results}}

USER'S INTENT: {{user_intent_summary}}
WHAT WAS ACCOMPLISHED: {{execution_summary}}

TASK: Create a clear, helpful response that presents the results to the user in a conversational way.

GUIDELINES:
- Return the actual response from "CONVERSATION" results directly
- Be conversational and friendly, matching the agent persona
- Present the key information the user wanted
- Make technical results easy to understand
- Be concise but complete
- Address the user's original question directly

Example: If user asked for disk space and results show "50GB free", respond with something like "You have 50GB of free disk space available on your drive."

Generate a helpful, clear response."""


@prompt("corrective-task-prompt")
class CorrectiveTaskPrompt(ResourceBase):
    """Prompt for generating corrective tasks for retry."""
    
    template: ClassVar[str] = """You are a task correction specialist. A task failed to fulfill the user's intent and needs to be corrected for retry.

ORIGINAL USER REQUEST: {{original_user_message}}

FAILED TASK: {{failed_task}}

EXECUTION RESULTS:
{{execution_results}}

ANALYSIS:
- User's Intent: {{user_intent_summary}}
- Gap/Problem: {{gap_analysis}}
- Suggested Correction: {{correction_suggestion}}

CONTEXT:
- Current Cycle: {{cycle_number}}
- Working Directory: {{working_directory}}

TASK: Generate an improved task description that will better fulfill the user's intent.

GUIDELINES:
- Learn from the failure and avoid repeating the same mistake
- Analyze the EXECUTION RESULTS above to understand specific tool failures, error messages, and output
- Include information about why the previous attempt failed based on actual tool results
- Add specific details to prevent the same error from occurring again
- If tool results show specific error messages (e.g., "file not found", "permission denied", "command not found"), address these directly
- Be more specific and accurate in the task description
- Consider simpler, more reliable approaches
- Focus on what the user actually wants, not just technical execution
- Use appropriate tools based on the task requirements
- Be more explicit about the desired outcome rather than specific tool usage

Example Corrections Based on Tool Results:
- Bad: "Check disk space" (execution results showed: "bash: df: command not found")
- Good: "Check available disk space using alternative command since 'df' is not available - try 'du -sh .' to show directory size"
- Bad: "Read config file" (execution results showed: "Permission denied: /etc/config")
- Good: "Read the accessible config file in the user directory instead of system config - use '~/.config/app.conf' since previous attempt failed with permission denied on /etc/config"
- Bad: "List files" (execution results showed: "No such file or directory")
- Good: "List files in the current working directory using full path - previous attempt failed because target directory doesn't exist, ensure we're listing the correct location"
- Bad: "Process data file" (execution results showed: "FileNotFoundError: data.csv")
- Good: "First verify the data file exists, then process it - previous execution failed because 'data.csv' was not found, include file existence check or use correct file path"

Generate a corrected task description."""


@prompt("failure-explanation-prompt")
class FailureExplanationPrompt(ResourceBase):
    """Prompt for generating helpful failure explanations."""
    
    template: ClassVar[str] = """You are {{agent_persona}} explaining to the user why their request couldn't be completed after multiple attempts.

ORIGINAL USER REQUEST: {{original_user_message}}

EXECUTION RESULTS:
{{execution_results}}

ANALYSIS:
- User's Intent: {{user_intent_summary}}
- What Went Wrong: {{gap_analysis}}
- Attempts Made: {{cycles_attempted}}

TASK: Provide a helpful, understanding explanation of why the request couldn't be completed.

GUIDELINES:
- Be empathetic and understanding
- Explain what was attempted and why it didn't work
- Avoid technical jargon - use plain language
- Suggest alternatives if possible
- Maintain the agent persona (helpful, friendly)
- Don't make the user feel bad about the request

Example: "I tried several ways to check your disk space but encountered technical issues with the commands. You might want to try running 'df -h' directly in your terminal, or I could help you with a different approach if you'd like."

Generate a helpful, friendly explanation."""