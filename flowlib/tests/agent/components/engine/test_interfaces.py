"""Tests for engine interfaces."""

import pytest
import pytest_asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock
from flowlib.agent.components.engine.interfaces import EngineInterface


class TestEngineInterface:
    """Test EngineInterface protocol."""
    
    def test_engine_interface_protocol(self):
        """Test that EngineInterface is a proper Protocol."""
        # EngineInterface should be a Protocol
        assert hasattr(EngineInterface, '__annotations__')
        
        # Check required methods exist
        assert hasattr(EngineInterface, 'execute_cycle')
        assert callable(getattr(EngineInterface, 'execute_cycle', None))
    
    @pytest.mark.asyncio
    async def test_engine_interface_implementation(self):
        """Test implementing the EngineInterface protocol."""
        
        class TestEngine:
            """Test implementation of EngineInterface."""
            
            async def execute_cycle(
                self,
                state: Any,
                memory_context: Optional[str] = None,
                **kwargs
            ) -> bool:
                """Execute test cycle."""
                return True
        
        # Create instance
        engine = TestEngine()
        
        # Test method exists and works
        mock_state = Mock()
        result = await engine.execute_cycle(mock_state)
        assert result is True
        
        # Test with optional parameters
        result = await engine.execute_cycle(
            mock_state,
            memory_context="test_context",
            extra_param="value"
        )
        assert result is True
    
    def test_mock_engine_interface(self):
        """Test creating a mock that satisfies EngineInterface."""
        # Create a mock engine
        mock_engine = Mock()
        mock_engine.execute_cycle = AsyncMock(return_value=True)
        
        # Verify the mock has the required method
        assert hasattr(mock_engine, 'execute_cycle')
        assert callable(mock_engine.execute_cycle)
    
    @pytest.mark.asyncio
    async def test_partial_implementation_detection(self):
        """Test that partial implementations are detectable."""
        
        class IncompleteEngine:
            """Incomplete implementation missing execute_cycle."""
            pass
        
        engine = IncompleteEngine()
        
        # This should not have the required method
        assert not hasattr(engine, 'execute_cycle')
    
    @pytest.mark.asyncio
    async def test_execute_cycle_signature(self):
        """Test execute_cycle method signature requirements."""
        
        class MinimalEngine:
            """Minimal valid implementation."""
            
            async def execute_cycle(self, state, memory_context=None, **kwargs):
                return False
        
        engine = MinimalEngine()
        
        # Test minimal call
        result = await engine.execute_cycle(Mock())
        assert result is False
        
        # Test with all parameters
        result = await engine.execute_cycle(
            Mock(),
            memory_context="context",
            param1="value1",
            param2="value2"
        )
        assert result is False
    
    @pytest.mark.asyncio
    async def test_engine_with_state_modification(self):
        """Test engine implementation that modifies state."""
        
        class StatefulEngine:
            """Engine that modifies state during execution."""
            
            async def execute_cycle(
                self,
                state: Any,
                memory_context: Optional[str] = None,
                **kwargs
            ) -> bool:
                """Execute cycle and modify state."""
                if hasattr(state, 'counter'):
                    state.counter += 1
                    return state.counter < 3
                return False
        
        engine = StatefulEngine()
        
        # Create mock state with counter
        state = Mock()
        state.counter = 0
        
        # First cycle
        result = await engine.execute_cycle(state)
        assert result is True
        assert state.counter == 1
        
        # Second cycle
        result = await engine.execute_cycle(state)
        assert result is True
        assert state.counter == 2
        
        # Third cycle - should stop
        result = await engine.execute_cycle(state)
        assert result is False
        assert state.counter == 3
    
    @pytest.mark.asyncio
    async def test_engine_error_handling(self):
        """Test engine implementation with error handling."""
        
        class ErrorEngine:
            """Engine that can raise errors."""
            
            async def execute_cycle(
                self,
                state: Any,
                memory_context: Optional[str] = None,
                **kwargs
            ) -> bool:
                """Execute cycle that might raise error."""
                if kwargs.get('should_error', False):
                    raise ValueError("Test error")
                return True
        
        engine = ErrorEngine()
        
        # Normal execution
        result = await engine.execute_cycle(Mock())
        assert result is True
        
        # Error execution
        with pytest.raises(ValueError, match="Test error"):
            await engine.execute_cycle(Mock(), should_error=True)
    
    def test_type_annotations(self):
        """Test that EngineInterface has proper type annotations."""
        # Get execute_cycle method annotations
        execute_cycle = EngineInterface.execute_cycle
        annotations = execute_cycle.__annotations__
        
        # Check parameter types
        assert 'state' in annotations
        assert annotations['state'] == Any
        
        assert 'memory_context' in annotations
        assert annotations['memory_context'] == Optional[str]
        
        # Check return type
        assert 'return' in annotations
        assert annotations['return'] == bool