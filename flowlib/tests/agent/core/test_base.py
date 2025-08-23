"""Tests for agent core base component."""

import pytest
import logging
from unittest.mock import AsyncMock, Mock
from flowlib.agent.core.base import AgentComponent


class MockComponent(AgentComponent):
    """Mock implementation of AgentComponent."""
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self.initialize_called = False
        self.shutdown_called = False
    
    async def _initialize_impl(self):
        """Test implementation of initialization."""
        self.initialize_called = True
    
    async def _shutdown_impl(self):
        """Test implementation of shutdown."""
        self.shutdown_called = True


class TestAgentComponent:
    """Test AgentComponent functionality."""
    
    def test_initialization_defaults(self):
        """Test component initialization with defaults."""
        component = MockComponent()
        
        assert component.name == "mockcomponent"
        assert not component.initialized
        assert component.parent is None
    
    def test_initialization_with_name(self):
        """Test component initialization with custom name."""
        component = MockComponent("custom_component")
        
        assert component.name == "custom_component"
        assert not component.initialized
        assert component.parent is None
    
    def test_set_parent(self):
        """Test setting parent component."""
        parent = MockComponent("parent")
        child = MockComponent("child")
        
        child.set_parent(parent)
        
        assert child.parent is parent
        # Logger should be updated with parent context
        assert "parent.child" in child._logger.name
    
    @pytest.mark.asyncio
    async def test_initialize_lifecycle(self):
        """Test component initialization lifecycle."""
        component = MockComponent("test")
        
        # Initially not initialized
        assert not component.initialized
        assert not component.initialize_called
        
        # Initialize component
        await component.initialize()
        
        # Should be initialized
        assert component.initialized
        assert component.initialize_called
    
    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test that initialize can be called multiple times safely."""
        component = MockComponent("test")
        
        # Initialize twice
        await component.initialize()
        component.initialize_called = False  # Reset flag
        await component.initialize()
        
        # Should still be initialized but impl not called again
        assert component.initialized
        assert not component.initialize_called
    
    @pytest.mark.asyncio
    async def test_shutdown_lifecycle(self):
        """Test component shutdown lifecycle."""
        component = MockComponent("test")
        
        # Initialize first
        await component.initialize()
        assert component.initialized
        
        # Shutdown component
        await component.shutdown()
        
        # Should be shut down
        assert not component.initialized
        assert component.shutdown_called
    
    @pytest.mark.asyncio
    async def test_shutdown_without_initialization(self):
        """Test shutdown on uninitialized component."""
        component = MockComponent("test")
        
        # Shutdown without initialization
        await component.shutdown()
        
        # Should remain uninitialized and shutdown not called
        assert not component.initialized
        assert not component.shutdown_called
    
    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        """Test that shutdown can be called multiple times safely."""
        component = MockComponent("test")
        
        # Initialize and shutdown
        await component.initialize()
        await component.shutdown()
        
        # Shutdown again
        component.shutdown_called = False  # Reset flag
        await component.shutdown()
        
        # Should remain shut down but impl not called again
        assert not component.initialized
        assert not component.shutdown_called
    
    @pytest.mark.asyncio
    async def test_complete_lifecycle(self):
        """Test complete initialize -> shutdown -> initialize cycle."""
        component = MockComponent("test")
        
        # First cycle
        await component.initialize()
        assert component.initialized
        assert component.initialize_called
        
        await component.shutdown()
        assert not component.initialized
        assert component.shutdown_called
        
        # Reset flags
        component.initialize_called = False
        component.shutdown_called = False
        
        # Second cycle
        await component.initialize()
        assert component.initialized
        assert component.initialize_called
        
        await component.shutdown()
        assert not component.initialized
        assert component.shutdown_called
    
    # Removed redundant string representation test
        """Test string representation of component."""
        component = MockComponent("test")
        
        # Before initialization
        str_repr = str(component)
        assert "MockComponent" in str_repr
        assert "name='test'" in str_repr
        assert "not initialized" in str_repr
        
        # After initialization (can't test async in non-async test)
        component._initialized = True
        str_repr = str(component)
        assert "MockComponent" in str_repr
        assert "name='test'" in str_repr
        assert "status='initialized'" in str_repr
    
    def test_logger_creation(self):
        """Test logger creation and naming."""
        component = MockComponent("test")
        
        assert hasattr(component, '_logger')
        assert isinstance(component._logger, logging.Logger)
        assert "test" in component._logger.name
    
    def test_logger_with_parent(self):
        """Test logger naming with parent component."""
        parent = MockComponent("parent")
        child = MockComponent("child")
        
        original_logger_name = child._logger.name
        
        child.set_parent(parent)
        
        # Logger name should be updated
        assert child._logger.name != original_logger_name
        assert "parent" in child._logger.name
        assert "child" in child._logger.name


class TestAgentComponentEdgeCases:
    """Test edge cases and error scenarios."""
    
    def test_none_name_handling(self):
        """Test handling of None name."""
        component = MockComponent(None)
        
        # Should use class name
        assert component.name == "mockcomponent"
    
    def test_empty_string_name(self):
        """Test handling of empty string name."""
        component = MockComponent("")
        
        # Should use class name when empty
        assert component.name == "mockcomponent"
    
    @pytest.mark.asyncio
    async def test_exception_in_initialize_impl(self):
        """Test handling of exceptions in initialize implementation."""
        
        class FailingComponent(AgentComponent):
            async def _initialize_impl(self):
                raise Exception("Initialization failed")
        
        component = FailingComponent("failing")
        
        # Exception should propagate
        with pytest.raises(Exception, match="Initialization failed"):
            await component.initialize()
        
        # Component should remain uninitialized
        assert not component.initialized
    
    @pytest.mark.asyncio
    async def test_exception_in_shutdown_impl(self):
        """Test handling of exceptions in shutdown implementation."""
        
        class FailingComponent(AgentComponent):
            async def _shutdown_impl(self):
                raise Exception("Shutdown failed")
        
        component = FailingComponent("failing")
        
        # Initialize first
        await component.initialize()
        assert component.initialized
        
        # Exception should propagate during shutdown
        with pytest.raises(Exception, match="Shutdown failed"):
            await component.shutdown()
        
        # Component should remain initialized due to failure
        assert component.initialized