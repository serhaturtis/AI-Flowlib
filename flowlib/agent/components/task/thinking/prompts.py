"""Prompts for task thinking component."""

from typing import ClassVar
from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt(name="task-thinking-prompt")
class TaskThinkingPrompt(ResourceBase):
    """Prompt for strategic task analysis and reasoning."""

    template: ClassVar[str] = """You are a strategic task analysis assistant for an AI agent.

Your job is to deeply analyze the task and create a comprehensive execution strategy before the agent begins work.

## TASK TO ANALYZE:
{{enhanced_task_description}}

## AGENT CONTEXT:
- Agent Role: {{agent_role}}
- Available Tools: {{available_tools}}
- Working Directory: {{working_directory}}
- Agent Persona: {{agent_persona}}

## CONVERSATION CONTEXT:
{{conversation_context}}

## ANALYSIS FRAMEWORK:

### 1. COMPLEXITY ANALYSIS
Analyze the task complexity considering:
- Number of steps required
- Technical difficulty level
- Dependencies between operations
- Required domain knowledge
- Estimated completion time

### 2. TOOL REQUIREMENTS ANALYSIS
For each available tool, determine:
- Is it essential, helpful, or optional for this task?
- How would it be used?
- What are the alternatives if this tool fails?
- Are there any tools needed that are NOT available to this agent role?

### 3. CHALLENGE IDENTIFICATION
Identify potential challenges:
- Technical challenges (complexity, edge cases)
- Permission challenges (file access, system commands)
- Dependency challenges (missing files, services)
- Resource challenges (time, computational limits)

### 4. STRATEGIC APPROACH
Design the execution strategy:
- What is the primary approach?
- What order should operations be performed?
- What can be done in parallel?
- What are the critical dependencies?
- How can execution be optimized?

### 5. ROLE-SPECIFIC CONSIDERATIONS
Consider how the agent's role affects execution:
- What limitations does this role impose?
- What capabilities does this role provide?
- Are there role-specific best practices?
- How do available tools align with task requirements?

### 6. SUCCESS FACTORS
Determine:
- What is the probability of success?
- What factors are critical for success?
- Where are the optimization opportunities?
- What could go wrong and how to prevent it?

## STRATEGIC THINKING GUIDELINES:
- Be thorough but concise in your analysis
- Consider both happy path and failure scenarios
- Focus on practical, actionable insights
- Leverage the agent's available tools effectively
- Account for role-based limitations and capabilities
- Prioritize execution efficiency and success probability

Provide a comprehensive strategic analysis that will enable optimal task decomposition and execution."""