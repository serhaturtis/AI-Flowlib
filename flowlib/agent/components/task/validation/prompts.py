"""Prompts for context validation."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt


@prompt(name="context-classification-prompt")
class ContextClassificationPrompt:
    """Step 1: Simple classification - does the request have sufficient context?

    This is a lightweight boolean decision that smaller models can handle reliably.
    Detailed reasoning comes in step 2.
    """

    template: str = Field(
        default="""Does this request have sufficient context to proceed?

User Request: {{user_message}}

Agent Role: {{agent_role}}

Conversation History:
{{conversation_history}}

Domain State:
{{domain_state}}

# Task

Answer ONE question: Does the request have sufficient context to proceed?

**Default to YES unless critical information is genuinely missing.**

# When to answer YES (sufficient context)

✅ Answer YES if you understand:
- The user's goal or desired outcome
- The type of task (even if details are unspecified)
- What success looks like

Answer YES even if:
- Some parameters are missing (agent can choose defaults)
- Implementation details are vague (planner handles this)
- Request is brief but clear in intent

Examples that should be YES:
- "Write me a book about birds" → YES
- "Create a file in /tmp" → YES
- "Fix the authentication bug" → YES
- "Hello" → YES
- "Which books are in the current workspace?" → YES

# When to answer NO (insufficient context)

❌ Only answer NO if:
- Core intent is completely ambiguous
- Multiple contradictory interpretations exist
- Cannot determine what the user wants at all
- Request has undefined pronouns with no context

Examples that should be NO:
- "Fix it" (no context for "it") → NO
- "Do the thing" (completely vague) → NO
- Empty message → NO

# Output

Answer with:
- has_sufficient_context: true or false
- confidence: low, medium, or high
- reasoning: Brief 1-2 sentence explanation

That's it. Simple classification only."""
    )


@prompt(name="context-proceed-generation-prompt")
class ContextProceedGenerationPrompt:
    """Step 2a: Generate detailed reasoning when context is sufficient."""

    template: str = Field(
        default="""Generate detailed reasoning for why the context is sufficient.

User Request: {{user_message}}

Agent Role: {{agent_role}}

Conversation History:
{{conversation_history}}

Domain State:
{{domain_state}}

Classification: SUFFICIENT CONTEXT

# Task

Explain in detail why this request has sufficient context to proceed.

Include:
- What information is available
- What the user's goal appears to be
- Why the planner can work with this information
- Any assumptions that can reasonably be made

Be thorough but concise (2-4 sentences)."""
    )


@prompt(name="context-clarify-generation-prompt")
class ContextClarifyGenerationPrompt:
    """Step 2b: Generate clarification questions when context is insufficient."""

    template: str = Field(
        default="""Generate clarification questions for missing information.

User Request: {{user_message}}

Agent Role: {{agent_role}}

Conversation History:
{{conversation_history}}

Domain State:
{{domain_state}}

Classification: INSUFFICIENT CONTEXT

# Task

Identify what information is missing and generate clarification questions.

Provide:
1. missing_information: List specific information gaps (e.g., ["file path", "book genre"])
2. clarification_questions: Specific questions to ask (e.g., ["What file path should I use?", "What genre should the book be?"])
3. detailed_reasoning: Explain what's missing and why it's critical

Be specific and actionable. Ask only about truly critical gaps."""
    )


@prompt(name="context-validation-prompt")
class ContextValidationPrompt:
    """Prompt for validating information sufficiency before planning.

    This is the FIRST step in Plan-and-Execute, implementing proactive
    information gathering from CEP Framework (EMNLP 2024).

    The validator ONLY assesses context - it does NOT choose tools or create plans.
    That is the planner's job.
    """

    template: str = Field(
        default="""Assess whether there is sufficient information to proceed with this request.

User Request: {{user_message}}

Agent Role: {{agent_role}}

Conversation History:
{{conversation_history}}

Domain State:
{{domain_state}}

# Task

Determine if you understand WHAT the user wants. You only need the core intent - the planner will handle implementation details.

**Default to PROCEED unless critical information is genuinely missing.**

# When to PROCEED

✅ Proceed if you understand:
- The user's goal or desired outcome
- The type of task (even if details are unspecified)
- What success looks like

Proceed even if:
- Some parameters are missing (agent can choose reasonable defaults)
- Implementation details are vague (planner handles this)
- User says "create a book" without specifying every detail
- Request is brief but clear in intent

Examples of SUFFICIENT context:
- "Write me a book about birds" → PROCEED (clear goal, agent can decide details)
- "Create a file in /tmp" → PROCEED (agent can choose filename)
- "Fix the authentication bug" → PROCEED (planner will investigate)
- "Make the app faster" → PROCEED (planner will analyze and optimize)
- "Hello" → PROCEED (simple greeting)

# When to CLARIFY

❌ Only clarify if:
- Core intent is completely ambiguous
- Multiple contradictory interpretations exist
- Cannot determine what the user wants at all
- Request has undefined pronouns with no context ("fix it", "update that")

Examples of INSUFFICIENT context:
- "Fix it" → CLARIFY (no context for "it")
- "Do the thing" → CLARIFY (completely vague)
- Empty message → CLARIFY

# Guidelines

- Trust the planner to handle details
- Trust the agent to make reasonable choices
- Only ask about truly critical gaps
- Most detailed requests should PROCEED
- Brief clear requests should PROCEED

# Output

If proceeding: has_sufficient_context=true, explain why
If clarifying: has_sufficient_context=false, list critical gaps only

Analyze and decide."""
    )


@prompt(name="clarification-response-parsing-prompt")
class ClarificationResponseParsingPrompt:
    """Prompt for parsing user responses to clarification questions.

    This uses LLM intelligence to understand whether the user:
    - Delegated decisions to the agent ("you decide", "give recommendations")
    - Provided specific answers to questions
    - Mixed delegation and specific answers

    The LLM extracts structured information to enrich the original request.
    """

    template: str = Field(
        default="""Parse the user's response to clarification questions.

# Context

Original Request: {{original_request}}
Questions Asked: {{questions_asked}}
Missing Information: {{missing_information}}
User's Response: {{user_response}}

# Your Task

Classify the response type and extract information.

## Response Types

- **delegation**: User asks agent to decide (e.g., "you decide", "your choice")
- **specific_answers**: User provides explicit values (e.g., "use X", "set to 5")
- **mixed**: User provides some values, delegates others
- **insufficient**: Response is vague or irrelevant

## CRITICAL: Output Rules

**If delegation:**
- delegation_items = copy ALL items from "Missing Information" above
- provided_information = {}
- Example: Missing ["A", "B", "C"] → delegation_items: ["A", "B", "C"]

**If specific_answers:**
- delegation_items = []
- provided_information = {key: value pairs extracted from response}

**If mixed:**
- delegation_items = items user delegated
- provided_information = items user provided
- Both must be non-empty

**If insufficient:**
- delegation_items = []
- provided_information = {}
- follow_up_questions = [list of questions to ask]

## Examples

Delegation:
- Missing: ["A", "B", "C"]
- Response: "you decide"
- Output: delegation_items=["A", "B", "C"], provided_information={}

Specific:
- Missing: ["name", "size"]
- Response: "name is X, size is large"
- Output: delegation_items=[], provided_information={"name": "X", "size": "large"}

Parse the user's response now."""
    )
