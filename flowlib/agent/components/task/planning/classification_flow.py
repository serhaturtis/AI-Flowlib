"""Classification-based planning flows.

This module implements the classification-based planning architecture:
1. Classify request type (conversation/single_tool/multi_step)
2. Route to specialized generator with type-specific schema
3. Prevents contradictions through mutually exclusive schemas
"""

import logging
import time
import uuid
from typing import cast

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.flows.registry import flow_registry
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.llm.base import LLMProvider, PromptTemplate
from flowlib.resources.registry.registry import resource_registry

from .classification_models import (
    ConversationPlan,
    MultiStepPlan,
    SingleToolPlan,
    TaskClassification,
)
from .models import PlanningInput, PlanningOutput, PlanStep, StructuredPlan

logger = logging.getLogger(__name__)


@flow(  # type: ignore[arg-type]
    name="task-classification",
    description="Classify user request type (conversation/single_tool/multi_step)",
    is_infrastructure=False,
)
class TaskClassificationFlow:
    """Classifies user requests into conversation/single_tool/multi_step categories."""

    @pipeline(input_model=PlanningInput, output_model=TaskClassification)
    async def run_pipeline(self, input_data: PlanningInput) -> TaskClassification:
        """Classify the user's request type.

        Args:
            input_data: Contains user message, context, and available tools

        Returns:
            TaskClassification with task_type and reasoning
        """
        start_time = time.time()

        # Get LLM provider using config-driven approach
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))

        # Get prompt from registry
        prompt_instance = resource_registry.get("task-classification-prompt")

        # Format available tools for prompt
        tools_text = self._format_available_tools(input_data.available_tools)

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

        # Classify the request
        classification = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=TaskClassification,
            model_name="default-model",
            prompt_variables=cast(dict[str, object], prompt_vars),
        )

        processing_time = (time.time() - start_time) * 1000
        logger.info(
            f"Classified request as '{classification.task_type}' in {processing_time:.2f}ms"
        )

        return classification

    def _format_available_tools(self, tools: list[str]) -> str:
        """Format available tools list for the prompt."""
        if not tools:
            return "No tools available"

        from flowlib.agent.components.task.execution.registry import tool_registry

        formatted_tools = []
        for tool_name in tools:
            tool_instance = tool_registry.get(tool_name)
            if not tool_instance:
                raise RuntimeError(f"Tool '{tool_name}' not found in registry")

            description = tool_instance.get_planning_description()
            formatted_tools.append(f"- {tool_name}: {description}")

        return "\n".join(formatted_tools)

    def _format_conversation_history(self, history: list[dict]) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return "No previous conversation"

        formatted_messages = []
        for msg in history[-5:]:  # Last 5 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_messages.append(f"{role}: {content}")

        return "\n".join(formatted_messages)

    def _format_domain_state(self, domain_state: dict) -> str:
        """Format domain state for the prompt."""
        if not domain_state:
            return "No domain state available"

        formatted_parts = ["=== CURRENT STATE ==="]
        for key in sorted(domain_state.keys()):
            value = domain_state[key]
            if value:
                formatted_parts.append(f"{key}: {self._format_value(value)}")

        formatted_parts.append("=== End Current State ===")
        return "\n".join(formatted_parts)

    def _format_value(self, value) -> str:
        """Format a value for display."""
        if isinstance(value, dict):
            items = [f"{k}={v}" for k, v in value.items()]
            return "{" + ", ".join(items[:5]) + ("..." if len(items) > 5 else "") + "}"
        elif isinstance(value, list):
            if not value:
                return "[]"
            if len(value) <= 3:
                return str(value)
            else:
                return f"[{len(value)} items: {value[0]}, {value[1]}, ...]"
        else:
            return str(value)


@flow(  # type: ignore[arg-type]
    name="conversation-planning",
    description="Generate plan for pure conversational responses (no tools)",
    is_infrastructure=False,
)
class ConversationPlanningFlow:
    """Generates plans for conversational responses."""

    @pipeline(input_model=PlanningInput, output_model=ConversationPlan)
    async def run_pipeline(self, input_data: PlanningInput) -> ConversationPlan:
        """Generate conversational response plan.

        Args:
            input_data: Contains user message and context

        Returns:
            ConversationPlan with response guidance
        """
        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))

        # Get prompt from registry
        prompt_instance = resource_registry.get("conversation-planning-prompt")

        # Format conversation history
        history_text = self._format_conversation_history(input_data.conversation_history)

        # Format domain state
        domain_state_text = (
            self._format_domain_state(input_data.domain_state)
            if input_data.domain_state
            else "No domain state available"
        )

        # Prepare prompt variables
        prompt_vars = {
            "user_message": input_data.user_message,
            "agent_role": input_data.agent_role,
            "conversation_history": history_text,
            "domain_state": domain_state_text,
        }

        # Generate conversation plan
        plan = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=ConversationPlan,
            model_name="default-model",
            prompt_variables=cast(dict[str, object], prompt_vars),
        )

        logger.info("Generated conversation plan")
        return plan

    def _format_conversation_history(self, history: list[dict]) -> str:
        """Format conversation history."""
        if not history:
            return "No previous conversation"

        formatted_messages = []
        for msg in history[-5:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_messages.append(f"{role}: {content}")

        return "\n".join(formatted_messages)

    def _format_domain_state(self, domain_state: dict) -> str:
        """Format domain state."""
        if not domain_state:
            return "No domain state available"

        formatted_parts = []
        for key in sorted(domain_state.keys()):
            value = domain_state[key]
            if value:
                formatted_parts.append(f"{key}: {value}")

        return "\n".join(formatted_parts) if formatted_parts else "No domain state available"


@flow(  # type: ignore[arg-type]
    name="single-tool-planning",
    description="Generate plan for single-tool tasks",
    is_infrastructure=False,
)
class SingleToolPlanningFlow:
    """Generates plans for single-tool tasks."""

    @pipeline(input_model=PlanningInput, output_model=SingleToolPlan)
    async def run_pipeline(self, input_data: PlanningInput) -> SingleToolPlan:
        """Generate single-tool plan.

        Args:
            input_data: Contains user message, context, and available tools

        Returns:
            SingleToolPlan with single step
        """
        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))

        # Get prompt from registry
        prompt_instance = resource_registry.get("single-tool-planning-prompt")

        # Format available tools
        tools_text = self._format_available_tools(input_data.available_tools)

        # Format conversation history
        history_text = self._format_conversation_history(input_data.conversation_history)

        # Format domain state
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

        # Generate single-tool plan
        plan = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=SingleToolPlan,
            model_name="default-model",
            prompt_variables=cast(dict[str, object], prompt_vars),
        )

        logger.info(f"Generated single-tool plan using '{plan.step.tool_name}'")
        return plan

    def _format_available_tools(self, tools: list[str]) -> str:
        """Format available tools list."""
        if not tools:
            return "No tools available"

        from flowlib.agent.components.task.execution.registry import tool_registry

        formatted_tools = []
        for tool_name in tools:
            tool_instance = tool_registry.get(tool_name)
            if not tool_instance:
                raise RuntimeError(f"Tool '{tool_name}' not found in registry")

            description = tool_instance.get_planning_description()
            formatted_tools.append(f"- {tool_name}: {description}")

        return "\n".join(formatted_tools)

    def _format_conversation_history(self, history: list[dict]) -> str:
        """Format conversation history."""
        if not history:
            return "No previous conversation"

        formatted_messages = []
        for msg in history[-5:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_messages.append(f"{role}: {content}")

        return "\n".join(formatted_messages)

    def _format_domain_state(self, domain_state: dict) -> str:
        """Format domain state."""
        if not domain_state:
            return "No domain state available"

        formatted_parts = []
        for key in sorted(domain_state.keys()):
            value = domain_state[key]
            if value:
                formatted_parts.append(f"{key}: {value}")

        return "\n".join(formatted_parts) if formatted_parts else "No domain state available"


@flow(  # type: ignore[arg-type]
    name="multi-step-planning",
    description="Generate plan for multi-step tasks",
    is_infrastructure=False,
)
class MultiStepPlanningFlow:
    """Generates plans for multi-step tasks."""

    @pipeline(input_model=PlanningInput, output_model=MultiStepPlan)
    async def run_pipeline(self, input_data: PlanningInput) -> MultiStepPlan:
        """Generate multi-step plan.

        Args:
            input_data: Contains user message, context, and available tools

        Returns:
            MultiStepPlan with multiple steps
        """
        # Get LLM provider
        llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))

        # Get prompt from registry
        prompt_instance = resource_registry.get("multi-step-planning-prompt")

        # Format available tools
        tools_text = self._format_available_tools(input_data.available_tools)

        # Format conversation history
        history_text = self._format_conversation_history(input_data.conversation_history)

        # Format domain state
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

        # Generate multi-step plan
        plan = await llm.generate_structured(
            prompt=cast(PromptTemplate, prompt_instance),
            output_type=MultiStepPlan,
            model_name="default-model",
            prompt_variables=cast(dict[str, object], prompt_vars),
        )

        logger.info(f"Generated multi-step plan with {len(plan.steps)} steps")
        return plan

    def _format_available_tools(self, tools: list[str]) -> str:
        """Format available tools list."""
        if not tools:
            return "No tools available"

        from flowlib.agent.components.task.execution.registry import tool_registry

        formatted_tools = []
        for tool_name in tools:
            tool_instance = tool_registry.get(tool_name)
            if not tool_instance:
                raise RuntimeError(f"Tool '{tool_name}' not found in registry")

            description = tool_instance.get_planning_description()
            formatted_tools.append(f"- {tool_name}: {description}")

        return "\n".join(formatted_tools)

    def _format_conversation_history(self, history: list[dict]) -> str:
        """Format conversation history."""
        if not history:
            return "No previous conversation"

        formatted_messages = []
        for msg in history[-5:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_messages.append(f"{role}: {content}")

        return "\n".join(formatted_messages)

    def _format_domain_state(self, domain_state: dict) -> str:
        """Format domain state."""
        if not domain_state:
            return "No domain state available"

        formatted_parts = []
        for key in sorted(domain_state.keys()):
            value = domain_state[key]
            if value:
                formatted_parts.append(f"{key}: {value}")

        return "\n".join(formatted_parts) if formatted_parts else "No domain state available"


@flow(  # type: ignore[arg-type]
    name="classification-based-planning",
    description="Orchestrates classification → specialized planning pipeline",
    is_infrastructure=False,
)
class ClassificationBasedPlanningFlow:
    """Orchestrates the complete classification-based planning pipeline."""

    @pipeline(input_model=PlanningInput, output_model=PlanningOutput)
    async def run_pipeline(self, input_data: PlanningInput) -> PlanningOutput:
        """Execute classification-based planning pipeline.

        Args:
            input_data: Contains user message, context, and available tools

        Returns:
            PlanningOutput with unified plan
        """
        start_time = time.time()

        # Step 1: Classify the request
        classification_flow = flow_registry.get_flow("task-classification")
        assert classification_flow is not None, "task-classification flow not found"
        classification = await classification_flow.run_pipeline(input_data)

        logger.info(f"Task classified as: {classification.task_type}")

        # Step 2: Route to specialized planner based on classification
        if classification.task_type == "conversation":
            conversation_flow = flow_registry.get_flow("conversation-planning")
            assert conversation_flow is not None, "conversation-planning flow not found"
            specialized_plan = await conversation_flow.run_pipeline(input_data)
            structured_plan = self._convert_conversation_plan(specialized_plan)

        elif classification.task_type == "single_tool":
            single_tool_flow = flow_registry.get_flow("single-tool-planning")
            assert single_tool_flow is not None, "single-tool-planning flow not found"
            specialized_plan = await single_tool_flow.run_pipeline(input_data)
            structured_plan = await self._convert_single_tool_plan(specialized_plan, input_data)

        else:  # multi_step
            multi_step_flow = flow_registry.get_flow("multi-step-planning")
            assert multi_step_flow is not None, "multi-step-planning flow not found"
            specialized_plan = await multi_step_flow.run_pipeline(input_data)
            structured_plan = await self._convert_multi_step_plan(specialized_plan, input_data)

        processing_time = (time.time() - start_time) * 1000

        return PlanningOutput(
            plan=structured_plan,
            success=True,
            processing_time_ms=processing_time,
            llm_calls_made=2,  # Classification + specialized planning
        )

    def _convert_conversation_plan(self, conv_plan: ConversationPlan) -> StructuredPlan:
        """Convert ConversationPlan to StructuredPlan."""
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        return StructuredPlan(
            message_type="conversation",
            reasoning=conv_plan.reasoning,
            steps=[],  # No tool steps for conversation
            expected_outcome=conv_plan.expected_outcome,
            plan_id=plan_id,
            created_at=time.time(),
            execution_started=False,
            execution_complete=False,
        )

    async def _convert_single_tool_plan(
        self, single_plan: SingleToolPlan, input_data: PlanningInput
    ) -> StructuredPlan:
        """Convert SingleToolPlan to StructuredPlan with parameter extraction."""
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Extract parameters for the single step
        generated_params = await self._extract_parameters(
            tool_name=single_plan.step.tool_name,
            step_description=single_plan.step.step_description,
            suggested_params=single_plan.step.parameters,
            input_data=input_data,
            plan_reasoning=single_plan.reasoning,
            expected_outcome=single_plan.expected_outcome,
            step_number=1,
        )

        step = PlanStep(
            step_id=f"{plan_id}_step_0",
            tool_name=single_plan.step.tool_name,
            step_description=single_plan.step.step_description,
            parameters=generated_params,
            depends_on_step=None,
            executed=False,
            result=None,
        )

        return StructuredPlan(
            message_type="single_tool",
            reasoning=single_plan.reasoning,
            steps=[step],
            expected_outcome=single_plan.expected_outcome,
            plan_id=plan_id,
            created_at=time.time(),
            execution_started=False,
            execution_complete=False,
        )

    async def _convert_multi_step_plan(
        self, multi_plan: MultiStepPlan, input_data: PlanningInput
    ) -> StructuredPlan:
        """Convert MultiStepPlan to StructuredPlan with parameter extraction."""
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        steps = []
        for idx, llm_step in enumerate(multi_plan.steps):
            # Extract parameters for each step
            generated_params = await self._extract_parameters(
                tool_name=llm_step.tool_name,
                step_description=llm_step.step_description,
                suggested_params=llm_step.parameters,
                input_data=input_data,
                plan_reasoning=multi_plan.reasoning,
                expected_outcome=multi_plan.expected_outcome,
                step_number=idx + 1,
            )

            step = PlanStep(
                step_id=f"{plan_id}_step_{idx}",
                tool_name=llm_step.tool_name,
                step_description=llm_step.step_description,
                parameters=generated_params,
                depends_on_step=llm_step.depends_on_step,
                executed=False,
                result=None,
            )
            steps.append(step)

        return StructuredPlan(
            message_type="multi_step",
            reasoning=multi_plan.reasoning,
            steps=steps,
            expected_outcome=multi_plan.expected_outcome,
            plan_id=plan_id,
            created_at=time.time(),
            execution_started=False,
            execution_complete=False,
        )

    async def _extract_parameters(
        self,
        tool_name: str,
        step_description: str,
        suggested_params: dict,
        input_data: PlanningInput,
        plan_reasoning: str,
        expected_outcome: str,
        step_number: int,
    ) -> dict:
        """Extract properly typed parameters for a tool using LLM."""
        from flowlib.agent.components.task.execution.registry import tool_registry

        # Get tool metadata
        metadata = tool_registry.get_metadata(tool_name)

        if not metadata or not metadata.parameter_type:
            # No parameter type defined, use suggested params
            return suggested_params

        try:
            # Get LLM provider
            llm = cast(LLMProvider, await provider_registry.get_by_config("default-llm"))

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
                "tool_name": tool_name,
                "tool_description": metadata.description,
                "user_request": input_data.user_message,
                "plan_reasoning": plan_reasoning,
                "step_number": f"#{step_number}",
                "step_description": step_description,
                "expected_outcome": expected_outcome,
                "suggested_params": str(suggested_params or {}),
                "working_directory": input_data.working_directory,
                "conversation_history": history_text,
                "domain_state": domain_state_text,
            }

            # Use LLM to extract parameters with proper schema
            param_instance = await llm.generate_structured(
                prompt=cast(PromptTemplate, extraction_prompt),
                output_type=metadata.parameter_type,
                model_name="default-model",
                prompt_variables=prompt_vars,
            )

            # Convert Pydantic instance to dict
            generated_parameters = (
                param_instance.model_dump()
                if hasattr(param_instance, "model_dump")
                else dict(param_instance)
            )

            logger.info(f"Generated parameters for {tool_name}: {generated_parameters}")
            return generated_parameters

        except Exception as e:
            logger.warning(f"Failed to generate parameters for {tool_name}: {e}, using suggestion")
            return suggested_params

    def _format_conversation_history(self, history: list[dict]) -> str:
        """Format conversation history for the prompt."""
        if not history:
            return "No previous conversation"

        formatted_messages = []
        for msg in history[-5:]:  # Last 5 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_messages.append(f"{role}: {content}")

        return "\n".join(formatted_messages)

    def _format_domain_state(self, domain_state: dict) -> str:
        """Format domain state for the prompt."""
        if not domain_state:
            return "No domain state available"

        formatted_parts = []
        for key in sorted(domain_state.keys()):
            value = domain_state[key]
            if value:
                formatted_parts.append(f"{key}: {value}")

        return "\n".join(formatted_parts) if formatted_parts else "No domain state available"
