"""Prompts for structured task planning."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt


@prompt(name="structured-planning-prompt")
class StructuredPlanningPrompt:
    """Prompt for generating structured execution plans in a single call.

    This follows the Plan-and-Execute pattern optimized for local LLMs:
    - One planning call instead of multiple phases
    - Structured output with clear plan steps
    - Minimal cognitive load per decision
    """

    template: str = Field(
        default="""# User Request
{{user_message}}

# Available Tools
{{available_tools}}

# Context
Working directory: {{working_directory}}
Conversation: {{conversation_history}}
State: {{domain_state}}

# Planning Instructions

1. Check if any tool says "COMPLETE WORKFLOW" or "ONE SINGLE STEP" - use only that tool (single_tool)
2. For greetings/questions, use conversation tool
3. Otherwise, break into multiple steps (multi_step)

# Output

- message_type: "single_tool" (one tool does everything), "multi_step" (coordinate multiple tools), or "conversation"
- Use exact tool names from available tools
- Provide specific parameter values
- Expected outcome: what user will see

Generate the execution plan now."""
    )


@prompt(name="parameter-extraction-prompt")
class ParameterExtractionPrompt:
    """Prompt for extracting tool parameters using generate_structured.

    This prompt is used after planning to extract properly typed parameters
    for each tool in the plan. The LLM will see the parameter schema automatically
    via generate_structured's output_type.
    """

    template: str = Field(
        default="""Extract parameters for the '{{tool_name}}' tool.

# Tool Information

**Tool Name:** {{tool_name}}
**Tool Description:** {{tool_description}}

# Context

**User Request:** {{user_request}}

**Overall Plan Reasoning:** {{plan_reasoning}}

**This Step ({{step_number}}):** {{step_description}}

**Expected Outcome:** {{expected_outcome}}

**LLM Suggested Parameters:** {{suggested_params}}

**Working Directory:**
{{working_directory}}

**Conversation History:**
{{conversation_history}}

**Domain State:**
{{domain_state}}

# Your Task

Extract ALL required parameters and relevant optional parameters for the '{{tool_name}}' tool.
Use exact values from the user request and plan context above.

Guidelines:
- Use exact values from the user request where provided
- Infer reasonable values for optional parameters when context suggests them
- Leave optional parameters empty/null if not mentioned
- Match field names and types exactly as defined in the schema"""
    )


# Classification-Based Planning Prompts


@prompt(name="task-classification-prompt")
class TaskClassificationPrompt:
    """Prompt for classifying user requests into conversation/single_tool/multi_step."""

    template: str = Field(
        default="""You are an expert task classifier for an AI agent system.

Your task: Analyze the user's request and classify it into ONE of three categories:

1. **conversation**: Pure conversation with NO tools needed
   - Greetings, small talk, thank you messages
   - Questions about capabilities, clarifications
   - Requests for explanations or information that don't require tool execution
   - Examples: "Hello", "What can you do?", "Thanks!", "Explain what X means"

2. **single_tool**: Task that can be completed by EXACTLY ONE tool call
   - The request maps directly to a single tool's functionality
   - No chaining, sequencing, or multiple operations needed
   - Examples: "Read file X", "Create file Y", "Run command Z"

3. **multi_step**: Complex task requiring MULTIPLE tools in sequence
   - Multiple distinct operations needed
   - Tool outputs feed into subsequent tools
   - Requires coordination across different tools
   - Examples: "Read file X and summarize it", "Create Y based on Z", "Search for X then edit Y"

===== CONTEXT =====

User Message:
{{user_message}}

Available Tools:
{{available_tools}}

Conversation History:
{{conversation_history}}

Working Directory:
{{working_directory}}

Domain State:
{{domain_state}}

===== CLASSIFICATION TASK =====

Analyze the user's request and determine:
1. Does it require ANY tools at all? (If no → conversation)
2. If tools needed, can it be done with exactly one tool call? (If yes → single_tool)
3. If multiple operations needed, or chaining required? (If yes → multi_step)

Provide your classification with clear reasoning."""
    )


@prompt(name="conversation-planning-prompt")
class ConversationPlanningPrompt:
    """Prompt for planning pure conversational responses."""

    template: str = Field(
        default="""You are an expert conversation planner for an AI agent.

The user's request has been classified as CONVERSATION - no tools are needed.

Your task: Plan a conversational response that addresses the user's message appropriately.

===== CONTEXT =====

User Message:
{{user_message}}

Agent Role:
{{agent_role}}

Conversation History:
{{conversation_history}}

Domain State:
{{domain_state}}

===== PLANNING TASK =====

Create a conversation plan that includes:
1. **reasoning**: Why a conversational response is appropriate
2. **response_guidance**: What topics/information the response should cover
3. **expected_outcome**: What the user should expect

Be helpful, concise, and aligned with the agent's role."""
    )


@prompt(name="single-tool-planning-prompt")
class SingleToolPlanningPrompt:
    """Prompt for planning single-tool tasks."""

    template: str = Field(
        default="""You are an expert task planner for an AI agent system.

The user's request has been classified as SINGLE_TOOL - it can be completed with EXACTLY ONE tool call.

Your task: Plan the single tool execution that fulfills the user's request.

===== CONTEXT =====

User Message:
{{user_message}}

Available Tools:
{{available_tools}}

Working Directory:
{{working_directory}}

Conversation History:
{{conversation_history}}

Domain State:
{{domain_state}}

===== PLANNING TASK =====

Create a single-tool plan that includes:
1. **reasoning**: Why this single tool is sufficient
2. **step**: The tool to use, what it will do, and suggested parameters
3. **expected_outcome**: What the user should expect as the result

CRITICAL RULES:
- Select EXACTLY ONE tool that best matches the request
- Provide clear step_description explaining what the tool will accomplish
- Suggest parameters in the 'parameters' field (these will be validated later)
- Be specific about expected outcomes

Remember: You're planning a SINGLE tool call - if you need multiple tools, the classification was wrong."""
    )


@prompt(name="multi-step-planning-prompt")
class MultiStepPlanningPrompt:
    """Prompt for planning multi-step tasks."""

    template: str = Field(
        default="""You are an expert task planner for an AI agent system.

The user's request has been classified as MULTI_STEP - it requires MULTIPLE tool calls in sequence.

Your task: Plan a complete multi-step execution that fulfills the user's request.

===== CONTEXT =====

User Message:
{{user_message}}

Available Tools:
{{available_tools}}

Working Directory:
{{working_directory}}

Conversation History:
{{conversation_history}}

Domain State:
{{domain_state}}

===== PLANNING TASK =====

Create a multi-step plan that includes:
1. **reasoning**: Why multiple steps are needed and how they work together
2. **steps**: Ordered list of tool calls (MINIMUM 2 steps)
3. **expected_outcome**: What the user should expect as the final result

CRITICAL RULES FOR STEPS:
- MINIMUM 2 steps required (if only 1 step needed, classification was wrong)
- Each step must have:
  - tool_name: Exact tool name from available tools
  - step_description: What this step accomplishes
  - parameters: Suggested parameters (will be validated later)
  - depends_on_step: Index of previous step this depends on (0-based, or null if independent)

PLANNING STRATEGY:
1. Break down the request into atomic operations
2. Identify which tool handles each operation
3. Determine dependencies between steps
4. Order steps to respect dependencies
5. Ensure each step builds toward the expected outcome

DEPENDENCY EXAMPLES:
- Step 0: Read file → Step 1: Analyze content (depends_on_step: 0)
- Step 0: Search code → Step 1: Edit file (depends_on_step: 0)
- Step 0: Create file → Step 1: Write content (depends_on_step: 0)

Remember: You're planning MULTIPLE coordinated tool calls - break the task into logical steps."""
    )
