"""Structured planning flow implementation.

This flow implements the Plan-and-Execute pattern with a single LLM call
that generates a complete structured plan.
"""

import logging
import time
import uuid
from typing import cast

from flowlib.agent.models.conversation import ConversationMessage
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry
from flowlib.config.required_resources import RequiredAlias

from .models import (
    LLMStructuredPlan,
    PlanningInput,
    PlanningOutput,
    PlanStep,
    StructuredPlan,
)

logger = logging.getLogger(__name__)


@flow(  # type: ignore[arg-type]
    name="execution-planning",
    description="Generate complete structured execution plans in a single LLM call",
    is_infrastructure=False,
)
class ExecutionPlanningFlow:
    """Generates structured execution plans following Plan-and-Execute pattern."""

    @pipeline(input_model=PlanningInput, output_model=PlanningOutput)
    async def run_pipeline(self, input_data: PlanningInput) -> PlanningOutput:
        """Generate structured execution plan from user message.

        Args:
            input_data: Contains user message, context, and available tools

        Returns:
            PlanningOutput with complete structured plan
        """
        start_time = time.time()

        # Get LLM provider using config-driven approach
        llm = cast(LLMProvider, await provider_registry.get_by_config(RequiredAlias.DEFAULT_LLM.value))

        # Get prompt from registry
        prompt_instance = resource_registry.get("structured-planning-prompt")

        # Format available tools for prompt
        tools_text = self._format_available_tools(input_data.available_tools)

        # Debug logging
        print(f"\n{'=' * 60}")
        print(f"PLANNING DEBUG: {len(input_data.available_tools)} tools available")
        print(f"{'=' * 60}")
        print(f"Tool names: {input_data.available_tools[:10]}...")  # First 10
        print(f"\nFormatted tools (first 500 chars):\n{tools_text[:500]}...")
        print(f"{'=' * 60}\n")

        logger.info(f"Planning with {len(input_data.available_tools)} tools")

        # Format conversation history
        history_text = self._format_conversation_history(input_data.conversation_history)

        # Format domain state for prompt
        domain_state_text = (
            self._format_domain_state(input_data.domain_state)
            if input_data.domain_state
            else "No domain state available"
        )

        # Prepare prompt variables
        prompt_vars = {
            "user_message": input_data.user_message,
            "available_tools": tools_text,
            "working_directory": input_data.working_directory,
            "conversation_history": history_text,
            "domain_state": domain_state_text,
        }

        # Generate complete plan in ONE LLM call - LLM only generates semantic content
        # Pydantic validation will enforce min_length=1 on steps field
        llm_result = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=LLMStructuredPlan,
            model_name=RequiredAlias.DEFAULT_MODEL.value,
            prompt_variables=cast(dict[str, object], prompt_vars),
        )

        # Convert LLM plan to full plan with programmatic metadata
        # Generate proper parameters for each step using generate_structured
        from flowlib.agent.components.task.execution.registry import tool_registry

        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        steps = []

        for idx, llm_step in enumerate(llm_result.steps):
            # Get tool metadata to access parameter_type
            metadata = tool_registry.get_metadata(llm_step.tool_name)

            generated_parameters = None
            if metadata and metadata.parameter_type:
                # Use generate_structured to extract parameters from step context
                try:
                    # Get extraction prompt
                    extraction_prompt = resource_registry.get("parameter-extraction-prompt")

                    # Format conversation history and domain state
                    history_text = self._format_conversation_history(input_data.conversation_history)
                    domain_state_text = (
                        self._format_domain_state(input_data.domain_state)
                        if input_data.domain_state
                        else "No domain state available"
                    )

                    # Prepare prompt variables
                    prompt_vars = {
                        "tool_name": llm_step.tool_name,
                        "tool_description": metadata.description,
                        "user_request": input_data.user_message,
                        "plan_reasoning": llm_result.reasoning,
                        "step_number": f"#{idx + 1}",
                        "step_description": llm_step.step_description,
                        "expected_outcome": llm_result.expected_outcome,
                        "suggested_params": str(llm_step.parameters or {}),
                        "working_directory": input_data.working_directory,
                        "conversation_history": history_text,
                        "domain_state": domain_state_text,
                    }

                    # Use LLM to extract parameters with proper schema
                    param_instance = await llm.generate_structured(
                        prompt=cast(PromptTemplate, extraction_prompt),
                        output_type=metadata.parameter_type,
                        model_name=RequiredAlias.DEFAULT_MODEL.value,
                        prompt_variables=prompt_vars,
                    )

                    # Convert Pydantic instance to dict for storage in plan
                    generated_parameters = (
                        param_instance.model_dump()
                        if hasattr(param_instance, "model_dump")
                        else dict(param_instance)
                    )

                    logger.info(
                        f"Generated parameters for {llm_step.tool_name}: {generated_parameters}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to generate parameters for {llm_step.tool_name}: {e}, using LLM suggestion"
                    )
                    generated_parameters = llm_step.parameters
            else:
                # No parameter type defined, use what LLM provided
                generated_parameters = llm_step.parameters

            step = PlanStep(
                step_id=f"{plan_id}_step_{idx}",
                tool_name=llm_step.tool_name,
                step_description=llm_step.step_description,
                parameters=generated_parameters,
                depends_on_step=llm_step.depends_on_step,
                executed=False,  # Programmatic field
                result=None,  # Programmatic field
            )
            steps.append(step)

        full_plan = StructuredPlan(
            message_type=llm_result.message_type,
            reasoning=llm_result.reasoning,
            steps=steps,
            expected_outcome=llm_result.expected_outcome,
            plan_id=plan_id,  # Programmatic field
            created_at=time.time(),  # Programmatic field
            execution_started=False,  # Programmatic field
            execution_complete=False,  # Programmatic field
        )

        processing_time = (time.time() - start_time) * 1000

        return PlanningOutput(
            plan=full_plan,
            success=True,  # Programmatic field
            processing_time_ms=processing_time,  # Programmatic field
            llm_calls_made=1,  # Programmatic field
        )

    def _format_available_tools(self, tools: list[str]) -> str:
        """Format available tools list for the prompt."""
        if not tools:
            return "No tools available"

        # Import here to avoid circular imports
        from flowlib.agent.components.task.execution.registry import tool_registry

        formatted_tools = []
        for tool_name in tools:
            tool_instance = tool_registry.get(tool_name)
            if not tool_instance:
                raise RuntimeError(f"Tool '{tool_name}' not found in registry")

            # Use planning description for concise prompts
            description = tool_instance.get_planning_description()
            formatted_tools.append(f"- {tool_name}: {description}")

        return "\n".join(formatted_tools)

    def _format_conversation_history(self, history: list[ConversationMessage]) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return "No previous conversation"

        formatted_messages = []
        for msg in history[-5:]:  # Last 5 messages for context
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = msg.content
            formatted_messages.append(f"{role}: {content}")

        return "\n".join(formatted_messages)

    def _format_domain_state(self, domain_state: dict) -> str:
        """Format domain state for the prompt - generic, no domain-specific knowledge."""
        if not domain_state:
            return "No domain state available - treat as fresh start with no existing artifacts."

        formatted_parts = ["=== CURRENT STATE (Source of Truth) ==="]

        # Show all keys equally in alphabetical order (no domain-specific prioritization)
        for key in sorted(domain_state.keys()):
            value = domain_state[key]
            if value:  # Only show non-empty values
                formatted_parts.append(f"{key}: {self._format_value(value)}")

        formatted_parts.append("=== End Current State ===")
        formatted_parts.append("")
        formatted_parts.append(
            "IMPORTANT: Use this state to make decisions. Don't create what already exists."
        )

        return "\n".join(formatted_parts)

    def _format_value(self, value) -> str:
        """Format a value for display in domain state."""
        if isinstance(value, dict):
            # Format nested dicts compactly
            items = [f"{k}={v}" for k, v in value.items()]
            return "{" + ", ".join(items[:5]) + ("..." if len(items) > 5 else "") + "}"
        elif isinstance(value, list):
            if not value:
                return "[]"
            # Show list items if short, otherwise show count
            if len(value) <= 3:
                return str(value)
            else:
                return f"[{len(value)} items: {value[0]}, {value[1]}, ...]"
        else:
            return str(value)
