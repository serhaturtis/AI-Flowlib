"""Tests for reflection interfaces."""

import pytest
from typing import Optional
from unittest.mock import Mock, AsyncMock
from pydantic import BaseModel

from flowlib.agent.components.reflection.interfaces import ReflectionInterface
from flowlib.agent.models.state import AgentState
from flowlib.flows.models.results import FlowResult
from flowlib.agent.components.reflection.models import ReflectionResult


class TestReflectionInterface:
    """Test ReflectionInterface protocol."""
    
    def test_reflection_interface_protocol(self):
        """Test that ReflectionInterface is a proper Protocol."""
        # ReflectionInterface should be a Protocol (runtime_checkable)
        from typing import Protocol
        assert isinstance(ReflectionInterface, type)
        assert issubclass(ReflectionInterface, Protocol)
        
        # Check that it has the required method signature
        assert hasattr(ReflectionInterface, 'reflect')
    
    @pytest.mark.asyncio
    async def test_reflection_interface_implementation(self):
        """Test implementing the ReflectionInterface protocol."""
        
        class TestReflection:
            """Test implementation of ReflectionInterface."""
            
            async def reflect(
                self,
                state: AgentState,
                flow_name: str,
                flow_inputs: BaseModel,
                flow_result: FlowResult,
                memory_context: Optional[str] = None,
                **kwargs
            ) -> ReflectionResult:
                """Reflect on execution results."""
                return ReflectionResult(
                    reflection="Test reflection",
                    progress=50,
                    is_complete=False
                )
        
        # Create instance
        reflection = TestReflection()
        
        # Test method exists and works
        mock_state = Mock(spec=AgentState)
        mock_inputs = Mock(spec=BaseModel)
        mock_result = Mock(spec=FlowResult)
        
        result = await reflection.reflect(
            state=mock_state,
            flow_name="test_flow",
            flow_inputs=mock_inputs,
            flow_result=mock_result
        )
        
        assert isinstance(result, ReflectionResult)
        assert result.reflection == "Test reflection"
        assert result.progress == 50
        assert result.is_complete is False
    
    @pytest.mark.asyncio
    async def test_reflection_with_memory_context(self):
        """Test reflection with memory context."""
        
        class MemoryReflection:
            """Reflection that uses memory context."""
            
            async def reflect(
                self,
                state: AgentState,
                flow_name: str,
                flow_inputs: BaseModel,
                flow_result: FlowResult,
                memory_context: Optional[str] = None,
                **kwargs
            ) -> ReflectionResult:
                """Reflect with memory context."""
                reflection_text = "Reflection"
                if memory_context:
                    reflection_text += f" with context: {memory_context}"
                
                return ReflectionResult(
                    reflection=reflection_text,
                    progress=25,
                    is_complete=False
                )
        
        reflection = MemoryReflection()
        result = await reflection.reflect(
            state=Mock(spec=AgentState),
            flow_name="test",
            flow_inputs=Mock(spec=BaseModel),
            flow_result=Mock(spec=FlowResult),
            memory_context="task_memory"
        )
        
        assert "with context: task_memory" in result.reflection
    
    @pytest.mark.asyncio
    async def test_reflection_with_kwargs(self):
        """Test reflection with additional kwargs."""
        
        class KwargsReflection:
            """Reflection that uses kwargs."""
            
            async def reflect(
                self,
                state: AgentState,
                flow_name: str,
                flow_inputs: BaseModel,
                flow_result: FlowResult,
                memory_context: Optional[str] = None,
                **kwargs
            ) -> ReflectionResult:
                """Reflect with kwargs."""
                insights = []
                if kwargs.get('include_insights', False):
                    insights = ["Insight 1", "Insight 2"]
                
                return ReflectionResult(
                    reflection="Reflection with kwargs",
                    progress=kwargs.get('progress', 0),
                    is_complete=kwargs.get('is_complete', False),
                    insights=insights
                )
        
        reflection = KwargsReflection()
        result = await reflection.reflect(
            state=Mock(spec=AgentState),
            flow_name="test",
            flow_inputs=Mock(spec=BaseModel),
            flow_result=Mock(spec=FlowResult),
            include_insights=True,
            progress=75,
            is_complete=True
        )
        
        assert result.progress == 75
        assert result.is_complete is True
        assert len(result.insights) == 2
    
    def test_mock_reflection_interface(self):
        """Test creating a mock that satisfies ReflectionInterface."""
        # Create a mock reflection
        mock_reflection = Mock()
        mock_reflection.reflect = AsyncMock(
            return_value=ReflectionResult(
                reflection="Mock reflection",
                progress=100,
                is_complete=True,
                completion_reason="Task completed successfully"
            )
        )
        
        # Verify the mock has the required method
        assert hasattr(mock_reflection, 'reflect')
        assert callable(mock_reflection.reflect)
    
    @pytest.mark.asyncio
    async def test_reflection_error_handling(self):
        """Test reflection with error handling."""
        
        class ErrorReflection:
            """Reflection that can raise errors."""
            
            async def reflect(
                self,
                state: AgentState,
                flow_name: str,
                flow_inputs: BaseModel,
                flow_result: FlowResult,
                memory_context: Optional[str] = None,
                **kwargs
            ) -> ReflectionResult:
                """Reflect with potential errors."""
                if flow_result and hasattr(flow_result, 'error') and flow_result.error:
                    raise ValueError(f"Flow error: {flow_result.error}")
                
                return ReflectionResult(
                    reflection="Success reflection",
                    progress=100,
                    is_complete=True
                )
        
        reflection = ErrorReflection()
        
        # Test normal execution
        mock_result = Mock(spec=FlowResult)
        mock_result.error = None
        
        result = await reflection.reflect(
            state=Mock(spec=AgentState),
            flow_name="test",
            flow_inputs=Mock(spec=BaseModel),
            flow_result=mock_result
        )
        assert result.is_complete is True
        
        # Test error execution
        mock_result.error = "Test error"
        with pytest.raises(ValueError, match="Flow error: Test error"):
            await reflection.reflect(
                state=Mock(spec=AgentState),
                flow_name="test",
                flow_inputs=Mock(spec=BaseModel),
                flow_result=mock_result
            )
    
    def test_type_annotations(self):
        """Test that ReflectionInterface has proper type annotations."""
        # Get reflect method annotations
        reflect = ReflectionInterface.reflect
        annotations = reflect.__annotations__
        
        # Check parameter types
        assert 'state' in annotations
        assert annotations['state'].__name__ == 'AgentState'
        
        assert 'flow_name' in annotations
        assert annotations['flow_name'] == str
        
        assert 'flow_inputs' in annotations
        assert annotations['flow_inputs'].__name__ == 'BaseModel'
        
        assert 'flow_result' in annotations
        assert annotations['flow_result'].__name__ == 'FlowResult'
        
        assert 'memory_context' in annotations
        assert annotations['memory_context'] == Optional[str]
        
        # Check return type
        assert 'return' in annotations
        assert annotations['return'].__name__ == 'ReflectionResult'