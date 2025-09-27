"""Task debriefing component."""

import logging
from typing import List, cast, Optional, Any

from flowlib.agent.core.base import AgentComponent
from flowlib.flows.registry import flow_registry
from ..execution.models import ToolResult
from .models import (
    DebriefingInput, DebriefingOutput, DebriefingDecision,
    IntentAnalysisInput, IntentAnalysisResult,
    SuccessPresentationInput, CorrectiveTaskInput, FailureExplanationInput
)

logger = logging.getLogger(__name__)


class TaskDebrieferComponent(AgentComponent):
    """Analyzes task execution results using LLM to determine if user intent was fulfilled.
    
    Responsibilities:
    - Analyze if execution results fulfill user's original intent
    - Generate user-friendly presentations for successful completions
    - Create corrective tasks for failed attempts
    - Provide helpful error explanations when retries are exhausted
    """
    
    def __init__(self, name: str = "task_debriefer", activity_stream: Optional[Any] = None) -> None:
        super().__init__(name)
        self._activity_stream = activity_stream
        self._intent_analysis_flow: Optional[Any] = None
        self._success_presentation_flow: Optional[Any] = None
        self._corrective_task_flow: Optional[Any] = None
        self._failure_explanation_flow: Optional[Any] = None
    
    async def _initialize_impl(self) -> None:
        """Initialize the task debriefer flows."""
        self._intent_analysis_flow = flow_registry.get_flow("intent-analysis")
        self._success_presentation_flow = flow_registry.get_flow("success-presentation")  
        self._corrective_task_flow = flow_registry.get_flow("corrective-task-generation")
        self._failure_explanation_flow = flow_registry.get_flow("failure-explanation")
        
        if not self._intent_analysis_flow:
            raise RuntimeError("IntentAnalysisFlow not found in registry")
        if not self._success_presentation_flow:
            raise RuntimeError("SuccessPresentationFlow not found in registry")
        if not self._corrective_task_flow:
            raise RuntimeError("CorrectiveTaskFlow not found in registry")
        if not self._failure_explanation_flow:
            raise RuntimeError("FailureExplanationFlow not found in registry")
        
        logger.info("TaskDebriefer initialized")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the task debriefer."""
        logger.info("TaskDebriefer shutdown")
    
    async def analyze_and_decide(self, input_data: DebriefingInput) -> DebriefingOutput:
        """Main method to analyze execution and decide next action."""
        self._check_initialized()
        
        # Analyze if user intent was fulfilled
        intent_analysis = await self._analyze_intent_fulfillment(input_data)
        
        # Make decision based on analysis
        if intent_analysis.intent_fulfilled:
            return await self._handle_success(input_data, intent_analysis)
        elif input_data.cycle_number < input_data.max_cycles and intent_analysis.is_correctable:
            return await self._handle_retry(input_data, intent_analysis)
        else:
            return await self._handle_failure(input_data, intent_analysis)
    
    async def _analyze_intent_fulfillment(self, input_data: DebriefingInput) -> IntentAnalysisResult:
        """Use LLM to analyze if user's intent was actually fulfilled."""
        analysis_input = IntentAnalysisInput(
            original_user_message=input_data.original_user_message,
            generated_task=input_data.generated_task_description,
            execution_results=self._format_execution_results(input_data.execution_results),
            todos_executed=input_data.todos_executed,
            agent_persona=input_data.agent_persona,
            working_directory=input_data.working_directory
        )
        
        assert self._intent_analysis_flow is not None, "Intent analysis flow not initialized"
        result = await self._intent_analysis_flow.run_pipeline(analysis_input)
        return cast(IntentAnalysisResult, result.intent_analysis)
    
    async def _handle_success(self, input_data: DebriefingInput, analysis: IntentAnalysisResult) -> DebriefingOutput:
        """Generate user-friendly success presentation."""
        presentation_input = SuccessPresentationInput(
            original_user_message=input_data.original_user_message,
            execution_results=self._format_execution_results(input_data.execution_results),
            intent_analysis=analysis,
            agent_persona=input_data.agent_persona
        )
        
        assert self._success_presentation_flow is not None, "Success presentation flow not initialized"
        result = await self._success_presentation_flow.run_pipeline(presentation_input)
        
        return DebriefingOutput(
            decision=DebriefingDecision.PRESENT_SUCCESS,
            user_response=result.presentation_response,
            reasoning=f"Task completed successfully: {analysis.user_intent_summary}",
            should_continue_cycle=False,
            intent_analysis=analysis
        )
    
    async def _handle_retry(self, input_data: DebriefingInput, analysis: IntentAnalysisResult) -> DebriefingOutput:
        """Generate corrective task for retry."""
        corrective_input = CorrectiveTaskInput(
            original_user_message=input_data.original_user_message,
            failed_task=input_data.generated_task_description,
            execution_results=self._format_execution_results(input_data.execution_results),
            intent_analysis=analysis,
            cycle_number=input_data.cycle_number,
            working_directory=input_data.working_directory
        )
        
        assert self._corrective_task_flow is not None, "Corrective task flow not initialized"
        result = await self._corrective_task_flow.run_pipeline(corrective_input)
        
        return DebriefingOutput(
            decision=DebriefingDecision.RETRY_WITH_CORRECTION,
            corrective_task=result.corrected_task,
            reasoning=f"Retry needed: {analysis.gap_analysis}",
            should_continue_cycle=True,
            intent_analysis=analysis
        )
    
    async def _handle_failure(self, input_data: DebriefingInput, analysis: IntentAnalysisResult) -> DebriefingOutput:
        """Generate helpful failure explanation."""
        failure_input = FailureExplanationInput(
            original_user_message=input_data.original_user_message,
            execution_results=self._format_execution_results(input_data.execution_results),
            intent_analysis=analysis,
            cycles_attempted=input_data.cycle_number,
            agent_persona=input_data.agent_persona
        )
        
        assert self._failure_explanation_flow is not None, "Failure explanation flow not initialized"
        result = await self._failure_explanation_flow.run_pipeline(failure_input)
        
        return DebriefingOutput(
            decision=DebriefingDecision.PRESENT_FAILURE,
            user_response=result.failure_explanation,
            reasoning=f"Max retries reached or uncorrectable: {analysis.gap_analysis}",
            should_continue_cycle=False,
            intent_analysis=analysis
        )
    
    def _format_execution_results(self, results: List[ToolResult]) -> str:
        """Format execution results for LLM consumption."""
        if not results:
            return "No execution results"
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            content = result.get_display_content()
            status = result.status.value
            formatted_results.append(f"Result {i} ({status}): {content}")
        
        return "\n".join(formatted_results)