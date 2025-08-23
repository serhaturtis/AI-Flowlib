"""
Base flow class with clean dependency injection.

This provides a standard pattern for flows that automatically handles
provider resolution and configuration management.
"""

import asyncio
import logging
import time
from typing import TypeVar, Generic, Optional
from abc import ABC, abstractmethod

from ...flows.decorators import flow
from .context import FlowContext, ProcessingOptions, create_flow_context

logger = logging.getLogger(__name__)

# Type variables for input and output
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class BaseAgentFlow(Generic[InputT, OutputT], ABC):
    """Base class for agent flows with clean dependency injection.
    
    This class provides:
    - Automatic provider resolution through FlowContext
    - Clean separation of data and configuration
    - Standardized error handling and logging
    - Performance monitoring
    """
    
    def __init__(self):
        self.context: Optional[FlowContext] = None
        self.start_time: Optional[float] = None
    
    async def execute(
        self, 
        input_data: InputT, 
        processing_options: Optional[ProcessingOptions] = None
    ) -> OutputT:
        """Execute the flow with clean dependency injection.
        
        Args:
            input_data: Pure data input (no configuration)
            processing_options: Optional processing configuration
            
        Returns:
            Flow output
            
        Raises:
            FlowExecutionError: If flow execution fails
        """
        self.start_time = time.time()
        
        try:
            # Create execution context
            self.context = await create_flow_context(processing_options)
            
            # Execute the flow
            logger.info(f"Starting {self.__class__.__name__} execution")
            result = await self.run_pipeline(input_data)
            
            # Add performance metadata
            execution_time = time.time() - self.start_time
            logger.info(f"{self.__class__.__name__} completed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - self.start_time if self.start_time else 0
            logger.error(f"{self.__class__.__name__} failed after {execution_time:.2f}s: {e}")
            raise FlowExecutionError(
                flow_name=self.__class__.__name__,
                execution_time=execution_time,
                original_error=e
            ) from e
    
    @abstractmethod
    async def run_pipeline(self, input_data: InputT) -> OutputT:
        """Implement the core flow logic.
        
        Subclasses should implement this method with their specific logic.
        The FlowContext is available as self.context for provider access.
        """
        pass
    
    async def get_llm(self):
        """Convenience method to get LLM provider from context."""
        if not self.context:
            raise RuntimeError("Flow context not initialized. Call execute() first.")
        return await self.context.llm()
    
    async def get_graph(self):
        """Convenience method to get graph provider from context.""" 
        if not self.context:
            raise RuntimeError("Flow context not initialized. Call execute() first.")
        return await self.context.graph()
    
    async def get_vector(self):
        """Convenience method to get vector provider from context."""
        if not self.context:
            raise RuntimeError("Flow context not initialized. Call execute() first.")
        return await self.context.vector()
    
    def get_confidence_threshold(self) -> float:
        """Get confidence threshold from context."""
        if not self.context:
            return 0.7  # Default fallback
        return self.context.confidence_threshold


class FlowExecutionError(Exception):
    """Exception raised when flow execution fails."""
    
    def __init__(self, flow_name: str, execution_time: float, original_error: Exception):
        self.flow_name = flow_name
        self.execution_time = execution_time
        self.original_error = original_error
        super().__init__(f"Flow {flow_name} failed after {execution_time:.2f}s: {original_error}")


# Utility functions for common flow patterns

async def execute_flow_with_timeout(
    flow: BaseAgentFlow,
    input_data: InputT,
    timeout_seconds: int = 60,
    processing_options: Optional[ProcessingOptions] = None
) -> OutputT:
    """Execute a flow with timeout protection.
    
    Args:
        flow: Flow instance to execute
        input_data: Input data for the flow
        timeout_seconds: Maximum execution time
        processing_options: Optional processing configuration
        
    Returns:
        Flow output
        
    Raises:
        asyncio.TimeoutError: If flow exceeds timeout
        FlowExecutionError: If flow execution fails
    """
    try:
        return await asyncio.wait_for(
            flow.execute(input_data, processing_options),
            timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        logger.error(f"Flow {flow.__class__.__name__} timed out after {timeout_seconds}s")
        raise


async def execute_flows_in_parallel(
    flows_and_inputs: list[tuple[BaseAgentFlow, InputT, Optional[ProcessingOptions]]],
    max_concurrent: int = 3
) -> list[OutputT]:
    """Execute multiple flows in parallel with concurrency control.
    
    Args:
        flows_and_inputs: List of (flow, input_data, processing_options) tuples
        max_concurrent: Maximum number of concurrent flow executions
        
    Returns:
        List of flow outputs in the same order as inputs
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_with_semaphore(flow_info):
        flow, input_data, processing_options = flow_info
        async with semaphore:
            return await flow.execute(input_data, processing_options)
    
    tasks = [execute_with_semaphore(flow_info) for flow_info in flows_and_inputs]
    return await asyncio.gather(*tasks)