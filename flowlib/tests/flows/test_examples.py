"""Tests for flows examples."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from pydantic import BaseModel
from typing import Dict, Any

from flowlib.flows.examples.examples import (
    TextInput,
    TextOutput,
    ATypedFlow,
    AnotherTypedFlow,
    CombinedFlow,
    run_examples
)
from flowlib.core.context.context import Context


class TestTextModels:
    """Test the example text input/output models."""
    
    def test_text_input_creation(self):
        """Test TextInput model creation."""
        input_data = TextInput(text="hello world")
        
        assert input_data.text == "hello world"
        assert input_data.options == {}
    
    def test_text_input_with_options(self):
        """Test TextInput model with options."""
        options = {"key": "value", "setting": True}
        input_data = TextInput(text="test", options=options)
        
        assert input_data.text == "test"
        assert input_data.options == options
    
    def test_text_output_creation(self):
        """Test TextOutput model creation."""
        output_data = TextOutput(result="processed text")
        
        assert output_data.result == "processed text"
        assert output_data.metadata == {}
    
    def test_text_output_with_metadata(self):
        """Test TextOutput model with metadata."""
        metadata = {"process_time": 0.5, "method": "upper"}
        output_data = TextOutput(result="TEST", metadata=metadata)
        
        assert output_data.result == "TEST"
        assert output_data.metadata == metadata


class TestATypedFlow:
    """Test ATypedFlow example."""
    
    def test_flow_instantiation(self):
        """Test that ATypedFlow can be instantiated."""
        flow = ATypedFlow()
        assert flow is not None
        assert hasattr(flow, 'execute_pipeline')
    
    @pytest.mark.asyncio
    async def test_process_stage(self):
        """Test the process stage functionality."""
        flow = ATypedFlow()
        input_data = TextInput(text="hello world", options={"test": True})
        
        # Test the process stage directly (for testing purposes)
        result = await flow.process(input_data)
        
        assert isinstance(result, dict)
        assert result["processed"] == "HELLO WORLD"
        assert result["options_used"] == {"test": True}
    
    @pytest.mark.asyncio
    async def test_format_output_stage(self):
        """Test the format_output stage functionality."""
        flow = ATypedFlow()
        processed_data = {
            "processed": "HELLO WORLD",
            "options_used": {"test": True}
        }
        
        result = await flow.format_output(processed_data)
        
        assert isinstance(result, TextOutput)
        assert result.result == "HELLO WORLD"
        assert result.metadata == {"options": {"test": True}}
    
    @pytest.mark.asyncio
    async def test_execute_pipeline(self):
        """Test the complete pipeline execution."""
        flow = ATypedFlow()
        input_data = TextInput(text="hello world", options={"process": True})
        
        with patch.object(flow, 'process') as mock_process, \
             patch.object(flow, 'format_output') as mock_format:
            
            mock_process.return_value = {
                "processed": "HELLO WORLD",
                "options_used": {"process": True}
            }
            mock_format.return_value = TextOutput(
                result="HELLO WORLD",
                metadata={"options": {"process": True}}
            )
            
            result = await flow.execute_pipeline(input_data)
            
            assert isinstance(result, TextOutput)
            assert result.result == "HELLO WORLD"
            mock_process.assert_called_once_with(input_data)
            mock_format.assert_called_once()


class TestAnotherTypedFlow:
    """Test AnotherTypedFlow example."""
    
    def test_flow_instantiation(self):
        """Test that AnotherTypedFlow can be instantiated."""
        flow = AnotherTypedFlow()
        assert flow is not None
        assert hasattr(flow, 'execute_pipeline')
    
    @pytest.mark.asyncio
    async def test_process_stage(self):
        """Test the process stage functionality."""
        flow = AnotherTypedFlow()
        input_data = TextInput(text="test input", options={"flag": False})
        
        result = await flow.process(input_data)
        
        assert isinstance(result, dict)
        assert result["processed"] == "test input"
        assert result["options_used"] == {"flag": False}
    
    @pytest.mark.asyncio
    async def test_format_output_stage(self):
        """Test the format_output stage functionality."""
        flow = AnotherTypedFlow()
        processed_data = {
            "processed": "TEST INPUT",
            "options_used": {"flag": False}
        }
        
        result = await flow.format_output(processed_data)
        
        assert isinstance(result, TextOutput)
        assert result.result == "TEST INPUT"
        assert result.metadata == {"options": {"flag": False}}
    
    @pytest.mark.asyncio
    async def test_execute_pipeline(self):
        """Test the complete pipeline execution."""
        flow = AnotherTypedFlow()
        input_data = TextInput(text="test input", options={"setting": "value"})
        
        with patch.object(flow, 'process') as mock_process, \
             patch.object(flow, 'format_output') as mock_format:
            
            mock_process.return_value = {
                "processed": "TEST INPUT",
                "options_used": {"setting": "value"}
            }
            mock_format.return_value = TextOutput(
                result="TEST INPUT",
                metadata={"options": {"setting": "value"}}
            )
            
            result = await flow.execute_pipeline(input_data)
            
            assert isinstance(result, TextOutput)
            assert result.result == "TEST INPUT"
            mock_process.assert_called_once_with(input_data)
            mock_format.assert_called_once()


class TestCombinedFlow:
    """Test CombinedFlow example."""
    
    def test_flow_instantiation(self):
        """Test that CombinedFlow can be instantiated."""
        flow = CombinedFlow()
        assert flow is not None
        assert hasattr(flow, 'simple_flow')
        assert hasattr(flow, 'typed_flow')
        assert isinstance(flow.simple_flow, ATypedFlow)
        assert isinstance(flow.typed_flow, AnotherTypedFlow)
    
    @pytest.mark.asyncio
    async def test_run_all_pipeline(self):
        """Test the run_all pipeline functionality."""
        flow = CombinedFlow()
        
        # Mock the execute methods of the sub-flows
        simple_mock_result = Mock()
        simple_mock_result.data = {"result": "SIMPLE RESULT"}
        
        typed_mock_result = Mock()
        typed_mock_result.data = {"result": "TYPED RESULT"}
        
        with patch.object(flow.simple_flow, 'execute', return_value=simple_mock_result) as mock_simple, \
             patch.object(flow.typed_flow, 'execute', return_value=typed_mock_result) as mock_typed:
            
            input_data = {"text": "test input"}
            result = await flow.run_all(input_data)
            
            assert isinstance(result, dict)
            assert "simple_result" in result
            assert "typed_result" in result
            assert result["simple_result"] == {"result": "SIMPLE RESULT"}
            assert result["typed_result"] == {"result": "TYPED RESULT"}
            
            # Verify that execute was called on both flows
            mock_simple.assert_called_once()
            mock_typed.assert_called_once()
            
            # Verify the context passed to simple flow
            simple_call_args = mock_simple.call_args[0][0]
            assert isinstance(simple_call_args, Context)
            assert simple_call_args.data == input_data
            
            # Verify the context passed to typed flow
            typed_call_args = mock_typed.call_args[0][0]
            assert isinstance(typed_call_args, Context)
            # The TextInput gets converted to dict during context processing
            if isinstance(typed_call_args.data, dict):
                assert typed_call_args.data["text"] == "SIMPLE RESULT"
                assert typed_call_args.data["options"] == {"from_simple_flow": True}
            else:
                assert isinstance(typed_call_args.data, TextInput)
                assert typed_call_args.data.text == "SIMPLE RESULT"
                assert typed_call_args.data.options == {"from_simple_flow": True}


class TestExampleRunner:
    """Test the example runner function."""
    
    @pytest.mark.asyncio
    async def test_run_examples_function_exists(self):
        """Test that run_examples function exists and can be called."""
        # This is a basic test to ensure the function exists
        assert callable(run_examples)
    
    @pytest.mark.asyncio
    async def test_run_examples_with_mocks(self):
        """Test run_examples with mocked flow executions."""
        with patch('flowlib.flows.examples.examples.ATypedFlow') as MockATypedFlow, \
             patch('flowlib.flows.examples.examples.AnotherTypedFlow') as MockAnotherTypedFlow, \
             patch('flowlib.flows.examples.examples.CombinedFlow') as MockCombinedFlow, \
             patch('builtins.print') as mock_print:
            
            # Setup mock instances
            mock_a_flow = Mock()
            mock_a_flow.execute = AsyncMock()
            mock_a_flow.execute.return_value.data = "A Flow Result"
            mock_a_flow.data = "A Flow Result"
            MockATypedFlow.return_value = mock_a_flow
            
            mock_another_flow = Mock()
            mock_another_flow.execute = AsyncMock()
            mock_another_flow.execute.return_value.data = "Another Flow Result"
            MockAnotherTypedFlow.return_value = mock_another_flow
            
            mock_combined_flow = Mock()
            mock_combined_flow.execute = AsyncMock()
            mock_combined_flow.execute.return_value.data = "Combined Flow Result"
            MockCombinedFlow.return_value = mock_combined_flow
            
            # Mock process_text method to raise AttributeError
            mock_a_flow.process_text = Mock(side_effect=AttributeError("Private method"))
            
            # Run the examples
            await run_examples()
            
            # Verify that flows were instantiated and executed
            MockATypedFlow.assert_called_once()
            MockAnotherTypedFlow.assert_called_once()
            MockCombinedFlow.assert_called_once()
            
            mock_a_flow.execute.assert_called_once()
            mock_another_flow.execute.assert_called_once()
            mock_combined_flow.execute.assert_called_once()
            
            # Verify print statements were called
            assert mock_print.call_count >= 4  # At least 4 print statements
    
    @pytest.mark.asyncio
    async def test_run_examples_attribute_error_handling(self):
        """Test that run_examples properly handles AttributeError for private methods."""
        with patch('flowlib.flows.examples.examples.ATypedFlow') as MockATypedFlow, \
             patch('flowlib.flows.examples.examples.AnotherTypedFlow') as MockAnotherTypedFlow, \
             patch('flowlib.flows.examples.examples.CombinedFlow') as MockCombinedFlow, \
             patch('builtins.print') as mock_print:
            
            # Setup mock instances with minimal required attributes
            mock_a_flow = Mock()
            mock_a_flow.execute = AsyncMock()
            mock_a_flow.execute.return_value.data = "test"
            mock_a_flow.data = "test"
            mock_a_flow.process_text = Mock(side_effect=AttributeError("method not found"))
            MockATypedFlow.return_value = mock_a_flow
            
            mock_another_flow = Mock()
            mock_another_flow.execute = AsyncMock()
            mock_another_flow.execute.return_value.data = "test"
            MockAnotherTypedFlow.return_value = mock_another_flow
            
            mock_combined_flow = Mock()
            mock_combined_flow.execute = AsyncMock()
            mock_combined_flow.execute.return_value.data = "test"
            MockCombinedFlow.return_value = mock_combined_flow
            
            # Run the examples
            await run_examples()
            
            # Check that the success message was printed (AttributeError was caught)
            print_calls = [str(call) for call in mock_print.call_args_list]
            success_printed = any("SUCCESS: Private method access prevented" in call for call in print_calls)
            assert success_printed, f"Expected success message not found in print calls: {print_calls}"