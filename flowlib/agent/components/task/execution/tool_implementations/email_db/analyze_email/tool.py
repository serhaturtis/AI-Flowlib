"""Analyze email tool - extracts sentiment, topics, intent via LLM."""

import json
import logging

from flowlib.agent.components.task.core.todo import TodoItem
from flowlib.agent.components.task.execution.decorators import tool
from flowlib.agent.components.task.execution.models import ToolExecutionContext, ToolStatus
from flowlib.agent.components.task.execution.tool_implementations.email_db.analyze_email.prompts import (
    EMAIL_ANALYSIS_PROMPT,
    EMAIL_ANALYSIS_SYSTEM_PROMPT,
)
from flowlib.agent.components.task.execution.tool_implementations.email_db.models import (
    AnalyzeEmailParameters,
    AnalyzeEmailResult,
    EmailAnalysis,
)
from flowlib.providers.core.registry import provider_registry

logger = logging.getLogger(__name__)


@tool(
    parameter_type=AnalyzeEmailParameters,
    name="analyze_email",
    tool_category="email_db",
    description="Analyze an email to extract sentiment, topics, intent, and urgency using LLM",
    planning_description="Extract sentiment/topics/intent from email content",
)
class AnalyzeEmailTool:
    """Tool for LLM-based email analysis.

    Extracts:
    - Sentiment (positive/negative/neutral) with score
    - Topics/themes discussed
    - Intent (inquiry, complaint, purchase, support, etc.)
    - Urgency level
    - Key entities mentioned
    - Suggested response action
    - Whether human escalation is needed
    """

    def get_name(self) -> str:
        """Return tool name."""
        return "analyze_email"

    def get_description(self) -> str:
        """Return tool description."""
        return "Analyze an email to extract sentiment, topics, intent, and urgency using LLM"

    async def execute(
        self,
        todo: TodoItem,
        params: AnalyzeEmailParameters,
        context: ToolExecutionContext,
    ) -> AnalyzeEmailResult:
        """Execute email analysis.

        Args:
            todo: The task description
            params: Validated AnalyzeEmailParameters
            context: Execution context

        Returns:
            AnalyzeEmailResult with analysis data
        """
        try:
            # Get LLM provider
            llm_provider = await provider_registry.get_by_config("default-llm")

            # Build prompt with optional sender context
            sender_context = ""
            if params.sender_email:
                sender_context = f"Sender: {params.sender_email}"

            prompt = EMAIL_ANALYSIS_PROMPT.format(
                subject=params.subject,
                body=params.body,
                sender_context=sender_context,
            )

            # Call LLM
            response = await llm_provider.generate(
                prompt=prompt,
                system_prompt=EMAIL_ANALYSIS_SYSTEM_PROMPT,
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=500,
            )

            # Parse JSON response
            analysis_data = self._parse_analysis_response(response.content)

            if analysis_data is None:
                return AnalyzeEmailResult(
                    status=ToolStatus.ERROR,
                    message="Failed to parse LLM analysis response",
                    analysis=None,
                )

            # Create EmailAnalysis model
            analysis = EmailAnalysis(
                sentiment=analysis_data.get("sentiment", "neutral"),
                sentiment_score=float(analysis_data.get("sentiment_score", 0.0)),
                topics=analysis_data.get("topics", []),
                intent=analysis_data.get("intent", "other"),
                urgency=analysis_data.get("urgency", "normal"),
                key_entities=analysis_data.get("key_entities", []),
                suggested_action=analysis_data.get("suggested_action", "")
                if params.include_suggestions
                else "",
                requires_human=analysis_data.get("requires_human", False),
            )

            return AnalyzeEmailResult(
                status=ToolStatus.SUCCESS,
                message=f"Email analyzed: {analysis.sentiment} sentiment, {analysis.intent} intent",
                analysis=analysis,
            )

        except Exception as e:
            logger.error(f"Error analyzing email: {e}", exc_info=True)
            return AnalyzeEmailResult(
                status=ToolStatus.ERROR,
                message=f"Failed to analyze email: {str(e)}",
                analysis=None,
            )

    def _parse_analysis_response(self, response: str) -> dict | None:
        """Parse LLM response into analysis dict.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed dict or None if parsing fails
        """
        try:
            # Try to find JSON in response
            response = response.strip()

            # Handle markdown code blocks
            if response.startswith("```"):
                lines = response.split("\n")
                # Remove first and last lines (```json and ```)
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith("```") and not in_json:
                        in_json = True
                        continue
                    elif line.startswith("```") and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                response = "\n".join(json_lines)

            # Parse JSON
            data = json.loads(response)

            # Validate required fields
            if "sentiment" not in data:
                data["sentiment"] = "neutral"
            if "sentiment_score" not in data:
                data["sentiment_score"] = 0.0
            if "intent" not in data:
                data["intent"] = "other"
            if "urgency" not in data:
                data["urgency"] = "normal"

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse analysis JSON: {e}")
            return None
