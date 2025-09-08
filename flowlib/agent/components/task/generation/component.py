"""Task generation component for classifying and enriching user messages.

This component analyzes user messages, classifies them as conversations vs tasks,
and enriches them with context for optimal processing.
"""

import logging
import time
from typing import List, Optional, Dict, Any

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ExecutionError
from flowlib.flows.registry import flow_registry
from .models import (
    TaskGenerationInput, TaskGenerationOutput, GeneratedTask
)
from flowlib.agent.models.conversation import ConversationMessage

logger = logging.getLogger(__name__)


class TaskGeneratorComponent(AgentComponent):
    """Generates enriched task definitions from user messages."""
    
    def __init__(self, name: str = "task_generator"):
        """Initialize task generator component.
        
        Args:
            name: Component name
        """
        super().__init__(name)
        self._task_generation_flow = None
    
    async def _initialize_impl(self) -> None:
        """Initialize the task generator."""
        # Get TaskGenerationFlow from registry
        self._task_generation_flow = flow_registry.get_flow("task-generation")
        if not self._task_generation_flow:
            raise RuntimeError("TaskGenerationFlow not found in registry")
        
        logger.info("TaskGenerator initialized")
    
    async def _shutdown_impl(self) -> None:
        """Shutdown the task generator."""
        logger.info("TaskGenerator shutdown")
    
    async def convert_message_to_task(
        self,
        user_message: str,
        conversation_history: List[ConversationMessage],
        agent_persona: str,
        working_directory: str = "."
    ) -> TaskGenerationOutput:
        """Generate enriched task definition from user message.
        
        Args:
            user_message: The user's message to process
            conversation_history: Previous conversation messages for context
            agent_persona: Agent's persona/personality
            working_directory: Current working directory
            
        Returns:
            TaskGenerationOutput with generated task from user message
        """
        self._check_initialized()
        
        start_time = time.time()
        
        try:
            # Create input for task generation flow
            task_input = TaskGenerationInput(
                user_message=user_message,
                conversation_history=conversation_history,
                agent_persona=agent_persona,
                working_directory=working_directory
            )
            
            # Run task generation flow
            result = await self._task_generation_flow.run_pipeline(task_input)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Create successful output
            return TaskGenerationOutput(
                generated_task=result.generated_task,
                success=True,
                processing_time_ms=processing_time,
                llm_calls_made=1
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Task generation failed: {e}")
            
            # Fail fast - no fallbacks allowed in flowlib
            raise ExecutionError(f"Task generation failed: {str(e)}") from e
    
