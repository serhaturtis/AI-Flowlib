"""Tests for base agent flow class."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from flowlib.agent.core.base_flow import (
    BaseAgentFlow,
    FlowExecutionError,
    execute_flow_with_timeout,
    execute_flows_in_parallel
)
from flowlib.agent.core.context import ProcessingOptions


class MockFlowContext:
    """Mock FlowContext for testing."""
    
    def __init__(self):
        self.confidence_threshold = 0.8
        self._llm_provider = Mock()
        self._graph_provider = Mock()
        self._vector_provider = Mock()
    
    async def llm(self):
        return self._llm_provider
    
    async def graph(self):
        return self._graph_provider
    
    async def vector(self):
        return self._vector_provider


class ConcreteFlow(BaseAgentFlow[str, Dict[str, Any]]):
    """Concrete implementation for testing."""
    
    def __init__(self, should_fail: bool = False, processing_delay: float = 0.0):
        super().__init__()
        self.should_fail = should_fail
        self.processing_delay = processing_delay
        self.run_pipeline_called = False
    
    async def run_pipeline(self, input_data: str) -> Dict[str, Any]:
        """Mock pipeline implementation."""
        self.run_pipeline_called = True
        
        if self.processing_delay > 0:
            await asyncio.sleep(self.processing_delay)
        
        if self.should_fail:
            raise ValueError("Simulated pipeline failure")
        
        return {
            "input": input_data,
            "processed": f"processed_{input_data}",
            "timestamp": time.time()
        }


class TestBaseAgentFlow:
    """Test BaseAgentFlow functionality."""
    
    def test_initialization(self):
        """Test flow initialization."""
        flow = ConcreteFlow()
        
        assert flow.context is None
        assert flow.start_time is None
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test successful flow execution."""
        flow = ConcreteFlow()
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            result = await flow.execute("test_input")
            
            assert flow.run_pipeline_called is True
            assert flow.context == mock_context
            assert flow.start_time is not None
            assert result["input"] == "test_input"
            assert result["processed"] == "processed_test_input"
            assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_execution_with_processing_options(self):
        """Test execution with processing options."""
        flow = ConcreteFlow()
        processing_options = ProcessingOptions(
            model_preference="fast",
            confidence_threshold=0.9
        )
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            result = await flow.execute("test_input", processing_options)
            
            # Verify create_flow_context was called with options
            mock_create_context.assert_called_once_with(processing_options)
            assert result["input"] == "test_input"
    
    @pytest.mark.asyncio
    async def test_execution_failure(self):
        """Test flow execution failure handling."""
        flow = ConcreteFlow(should_fail=True)
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            with pytest.raises(FlowExecutionError) as exc_info:
                await flow.execute("test_input")
            
            error = exc_info.value
            assert error.flow_name == "ConcreteFlow"
            assert error.execution_time >= 0
            assert isinstance(error.original_error, ValueError)
            assert "Simulated pipeline failure" in str(error.original_error)
    
    @pytest.mark.asyncio
    async def test_get_llm_success(self):
        """Test successful LLM provider retrieval."""
        flow = ConcreteFlow()
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            # Execute to initialize context
            await flow.execute("test_input")
            
            # Test LLM retrieval
            llm_provider = await flow.get_llm()
            assert llm_provider == mock_context._llm_provider
    
    @pytest.mark.asyncio
    async def test_get_llm_without_context(self):
        """Test LLM provider retrieval without initialized context."""
        flow = ConcreteFlow()
        
        with pytest.raises(RuntimeError, match="Flow context not initialized"):
            await flow.get_llm()
    
    @pytest.mark.asyncio
    async def test_get_graph_success(self):
        """Test successful graph provider retrieval."""
        flow = ConcreteFlow()
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            # Execute to initialize context
            await flow.execute("test_input")
            
            # Test graph retrieval
            graph_provider = await flow.get_graph()
            assert graph_provider == mock_context._graph_provider
    
    @pytest.mark.asyncio
    async def test_get_graph_without_context(self):
        """Test graph provider retrieval without initialized context."""
        flow = ConcreteFlow()
        
        with pytest.raises(RuntimeError, match="Flow context not initialized"):
            await flow.get_graph()
    
    @pytest.mark.asyncio
    async def test_get_vector_success(self):
        """Test successful vector provider retrieval."""
        flow = ConcreteFlow()
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            # Execute to initialize context
            await flow.execute("test_input")
            
            # Test vector retrieval
            vector_provider = await flow.get_vector()
            assert vector_provider == mock_context._vector_provider
    
    @pytest.mark.asyncio
    async def test_get_vector_without_context(self):
        """Test vector provider retrieval without initialized context."""
        flow = ConcreteFlow()
        
        with pytest.raises(RuntimeError, match="Flow context not initialized"):
            await flow.get_vector()
    
    def test_get_confidence_threshold_with_context(self):
        """Test confidence threshold retrieval with context."""
        flow = ConcreteFlow()
        mock_context = MockFlowContext()
        mock_context.confidence_threshold = 0.95
        flow.context = mock_context
        
        threshold = flow.get_confidence_threshold()
        assert threshold == 0.95
    
    def test_get_confidence_threshold_without_context(self):
        """Test confidence threshold retrieval without context."""
        flow = ConcreteFlow()
        
        threshold = flow.get_confidence_threshold()
        assert threshold == 0.7  # Default fallback
    
    @pytest.mark.asyncio
    async def test_execution_timing(self):
        """Test that execution timing is properly recorded."""
        flow = ConcreteFlow(processing_delay=0.1)  # 100ms delay
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            start = time.time()
            await flow.execute("test_input")
            end = time.time()
            
            # Verify timing was recorded
            assert flow.start_time is not None
            assert flow.start_time >= start
            
            # Verify execution took at least the processing delay
            actual_duration = end - start
            assert actual_duration >= 0.1


class TestFlowExecutionError:
    """Test FlowExecutionError functionality."""
    
    def test_error_creation(self):
        """Test error creation with all parameters."""
        original_error = ValueError("Original error message")
        error = FlowExecutionError(
            flow_name="TestFlow",
            execution_time=1.5,
            original_error=original_error
        )
        
        assert error.flow_name == "TestFlow"
        assert error.execution_time == 1.5
        assert error.original_error == original_error
        assert "TestFlow" in str(error)
        assert "1.50s" in str(error)
        assert "Original error message" in str(error)
    
    def test_error_inheritance(self):
        """Test that FlowExecutionError is a proper Exception."""
        original_error = RuntimeError("Test error")
        error = FlowExecutionError("TestFlow", 0.5, original_error)
        
        assert isinstance(error, Exception)
        assert isinstance(error, FlowExecutionError)


class TestExecuteFlowWithTimeout:
    """Test execute_flow_with_timeout utility function."""
    
    @pytest.mark.asyncio
    async def test_successful_execution_within_timeout(self):
        """Test successful flow execution within timeout."""
        flow = ConcreteFlow(processing_delay=0.1)
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            result = await execute_flow_with_timeout(
                flow, "test_input", timeout_seconds=1
            )
            
            assert result["input"] == "test_input"
            assert result["processed"] == "processed_test_input"
    
    @pytest.mark.asyncio
    async def test_execution_timeout(self):
        """Test flow execution timeout."""
        flow = ConcreteFlow(processing_delay=2.0)  # 2 second delay
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            with pytest.raises(asyncio.TimeoutError):
                await execute_flow_with_timeout(
                    flow, "test_input", timeout_seconds=0.5  # 500ms timeout
                )
    
    @pytest.mark.asyncio
    async def test_execution_with_processing_options(self):
        """Test timeout execution with processing options."""
        flow = ConcreteFlow()
        processing_options = ProcessingOptions(model_preference="fast")
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            result = await execute_flow_with_timeout(
                flow, "test_input", timeout_seconds=1, processing_options=processing_options
            )
            
            assert result["input"] == "test_input"
    
    @pytest.mark.asyncio
    async def test_execution_failure_within_timeout(self):
        """Test flow execution failure within timeout."""
        flow = ConcreteFlow(should_fail=True)
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            with pytest.raises(FlowExecutionError):
                await execute_flow_with_timeout(flow, "test_input", timeout_seconds=1)


class TestExecuteFlowsInParallel:
    """Test execute_flows_in_parallel utility function."""
    
    @pytest.mark.asyncio
    async def test_parallel_execution_success(self):
        """Test successful parallel execution of multiple flows."""
        flows_and_inputs = [
            (ConcreteFlow(), "input1", None),
            (ConcreteFlow(), "input2", None),
            (ConcreteFlow(), "input3", None)
        ]
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            results = await execute_flows_in_parallel(flows_and_inputs, max_concurrent=2)
            
            assert len(results) == 3
            assert results[0]["input"] == "input1"
            assert results[1]["input"] == "input2"
            assert results[2]["input"] == "input3"
            assert results[0]["processed"] == "processed_input1"
            assert results[1]["processed"] == "processed_input2"
            assert results[2]["processed"] == "processed_input3"
    
    @pytest.mark.asyncio
    async def test_parallel_execution_with_processing_options(self):
        """Test parallel execution with processing options."""
        processing_options = ProcessingOptions(model_preference="accurate")
        flows_and_inputs = [
            (ConcreteFlow(), "input1", processing_options),
            (ConcreteFlow(), "input2", None),
        ]
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            results = await execute_flows_in_parallel(flows_and_inputs)
            
            assert len(results) == 2
            assert results[0]["input"] == "input1"
            assert results[1]["input"] == "input2"
    
    @pytest.mark.asyncio
    async def test_parallel_execution_concurrency_limit(self):
        """Test that concurrency limit is respected."""
        # Create flows with delays to test concurrency
        flows_and_inputs = [
            (ConcreteFlow(processing_delay=0.2), f"input{i}", None)
            for i in range(5)
        ]
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            start_time = time.time()
            results = await execute_flows_in_parallel(flows_and_inputs, max_concurrent=2)
            end_time = time.time()
            
            # With max_concurrent=2 and 5 flows of 0.2s each,
            # total time should be around 0.6s (3 batches: 2+2+1)
            execution_time = end_time - start_time
            assert execution_time >= 0.5  # At least 3 batches
            assert execution_time < 1.0   # Less than sequential execution
            
            assert len(results) == 5
            for i, result in enumerate(results):
                assert result["input"] == f"input{i}"
    
    @pytest.mark.asyncio
    async def test_parallel_execution_with_failure(self):
        """Test parallel execution when one flow fails."""
        flows_and_inputs = [
            (ConcreteFlow(), "input1", None),
            (ConcreteFlow(should_fail=True), "input2", None),
            (ConcreteFlow(), "input3", None)
        ]
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            # Should propagate the failure from the failing flow
            with pytest.raises(FlowExecutionError):
                await execute_flows_in_parallel(flows_and_inputs)
    
    @pytest.mark.asyncio
    async def test_parallel_execution_empty_list(self):
        """Test parallel execution with empty input list."""
        results = await execute_flows_in_parallel([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_parallel_execution_single_flow(self):
        """Test parallel execution with single flow."""
        flows_and_inputs = [(ConcreteFlow(), "single_input", None)]
        
        with patch('flowlib.agent.core.base_flow.create_flow_context') as mock_create_context:
            mock_context = MockFlowContext()
            mock_create_context.return_value = mock_context
            
            results = await execute_flows_in_parallel(flows_and_inputs)
            
            assert len(results) == 1
            assert results[0]["input"] == "single_input"
            assert results[0]["processed"] == "processed_single_input"