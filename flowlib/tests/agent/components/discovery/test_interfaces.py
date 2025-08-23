"""Tests for discovery interface definitions."""

import pytest
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from unittest.mock import AsyncMock, Mock

from flowlib.agent.components.discovery.interfaces import FlowDiscoveryInterface


class MockFlowDiscovery:
    """Mock implementation of FlowDiscoveryInterface for testing."""
    
    def __init__(self):
        self.flows = {}
        self.refresh_called = False
        self.should_fail = False
        self.fail_operation = None
    
    async def refresh_flows(self) -> Dict[str, Any]:
        """Mock refresh_flows implementation."""
        self.refresh_called = True
        if self.should_fail and self.fail_operation == "refresh":
            raise RuntimeError("Mock refresh failure")
        return self.flows.copy()
    
    def register_flow(self, flow: Any) -> None:
        """Mock register_flow implementation."""
        if self.should_fail and self.fail_operation == "register":
            raise RuntimeError("Mock register failure")
        
        if hasattr(flow, 'name'):
            self.flows[flow.name] = flow
        else:
            # Use str representation if no name
            self.flows[str(flow)] = flow
    
    def get_flow(self, name: str) -> Optional[Any]:
        """Mock get_flow implementation."""
        if self.should_fail and self.fail_operation == "get":
            raise RuntimeError("Mock get failure")
        return self.flows.get(name)


class MockFlow:
    """Mock flow for testing."""
    
    def __init__(self, name: str = "test_flow", **kwargs):
        self.name = name
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestFlowDiscoveryInterface:
    """Test FlowDiscoveryInterface protocol."""
    
    @pytest.fixture
    def mock_discovery(self):
        """Create mock discovery instance."""
        return MockFlowDiscovery()
    
    @pytest.fixture
    def sample_flow(self):
        """Create sample flow."""
        return MockFlow(name="test_flow", description="Test flow")
    
    def test_interface_protocol_compliance(self, mock_discovery):
        """Test that mock implementation satisfies the protocol."""
        # Runtime check - this validates the protocol is properly defined
        assert isinstance(mock_discovery, FlowDiscoveryInterface)
    
    def test_interface_has_required_methods(self):
        """Test that interface defines required methods."""
        # Check protocol methods exist
        assert hasattr(FlowDiscoveryInterface, 'refresh_flows')
        assert hasattr(FlowDiscoveryInterface, 'register_flow')
        assert hasattr(FlowDiscoveryInterface, 'get_flow')
    
    @pytest.mark.asyncio
    async def test_refresh_flows_method_signature(self, mock_discovery):
        """Test refresh_flows method signature and return type."""
        result = await mock_discovery.refresh_flows()
        
        assert isinstance(result, dict)
        assert mock_discovery.refresh_called is True
    
    @pytest.mark.asyncio
    async def test_refresh_flows_returns_dict(self, mock_discovery, sample_flow):
        """Test that refresh_flows returns dictionary."""
        # Pre-populate some flows
        mock_discovery.flows = {"test": sample_flow}
        
        result = await mock_discovery.refresh_flows()
        
        assert isinstance(result, Dict)
        assert "test" in result
        assert result["test"] == sample_flow
    
    def test_register_flow_method_signature(self, mock_discovery, sample_flow):
        """Test register_flow method signature."""
        # Should not raise any errors
        mock_discovery.register_flow(sample_flow)
        
        # Verify flow was registered
        assert sample_flow.name in mock_discovery.flows
        assert mock_discovery.flows[sample_flow.name] == sample_flow
    
    def test_register_flow_accepts_any_type(self, mock_discovery):
        """Test that register_flow accepts Any type."""
        # Test with different types
        test_objects = [
            MockFlow("flow1"),
            {"name": "dict_flow"},
            "string_flow",
            42
        ]
        
        for obj in test_objects:
            # Should not raise type errors
            mock_discovery.register_flow(obj)
        
        # Verify all were registered in some form
        assert len(mock_discovery.flows) == len(test_objects)
    
    def test_get_flow_method_signature(self, mock_discovery, sample_flow):
        """Test get_flow method signature and return type."""
        # Register a flow first
        mock_discovery.register_flow(sample_flow)
        
        # Test retrieval
        result = mock_discovery.get_flow("test_flow")
        assert result == sample_flow
        
        # Test non-existent flow
        result = mock_discovery.get_flow("nonexistent")
        assert result is None
    
    def test_get_flow_return_type_optional(self, mock_discovery):
        """Test that get_flow returns Optional[Any]."""
        # Non-existent flow should return None
        result = mock_discovery.get_flow("nonexistent")
        assert result is None
        
        # Existing flow should return the flow
        flow = MockFlow("existing")
        mock_discovery.register_flow(flow)
        result = mock_discovery.get_flow("existing")
        assert result is not None
        assert result == flow
    
    @pytest.mark.asyncio
    async def test_interface_method_error_handling(self, mock_discovery):
        """Test that interface methods can handle errors appropriately."""
        mock_discovery.should_fail = True
        
        # Test refresh_flows error handling
        mock_discovery.fail_operation = "refresh"
        with pytest.raises(RuntimeError, match="Mock refresh failure"):
            await mock_discovery.refresh_flows()
        
        # Test register_flow error handling
        mock_discovery.fail_operation = "register"
        with pytest.raises(RuntimeError, match="Mock register failure"):
            mock_discovery.register_flow(MockFlow())
        
        # Test get_flow error handling
        mock_discovery.fail_operation = "get"
        with pytest.raises(RuntimeError, match="Mock get failure"):
            mock_discovery.get_flow("test")
    
    def test_interface_flow_lifecycle(self, mock_discovery):
        """Test complete flow lifecycle through interface."""
        flow1 = MockFlow("flow1", category="test")
        flow2 = MockFlow("flow2", category="production")
        
        # Register flows
        mock_discovery.register_flow(flow1)
        mock_discovery.register_flow(flow2)
        
        # Verify registration
        assert mock_discovery.get_flow("flow1") == flow1
        assert mock_discovery.get_flow("flow2") == flow2
        
        # Verify flows are accessible through refresh
        # (Mock implementation returns copy of internal flows)
        flows = mock_discovery.flows
        assert len(flows) == 2
        assert "flow1" in flows
        assert "flow2" in flows
    
    @pytest.mark.asyncio
    async def test_interface_concurrent_operations(self, mock_discovery):
        """Test that interface supports concurrent operations."""
        import asyncio
        
        flows = [MockFlow(f"flow_{i}") for i in range(5)]
        
        # Register flows concurrently
        register_tasks = [
            asyncio.create_task(asyncio.to_thread(mock_discovery.register_flow, flow))
            for flow in flows
        ]
        await asyncio.gather(*register_tasks)
        
        # Verify all flows were registered
        assert len(mock_discovery.flows) == 5
        
        # Retrieve flows concurrently  
        get_tasks = [
            asyncio.create_task(asyncio.to_thread(mock_discovery.get_flow, f"flow_{i}"))
            for i in range(5)
        ]
        results = await asyncio.gather(*get_tasks)
        
        # Verify all flows were retrieved
        assert len(results) == 5
        assert all(result is not None for result in results)
    
    def test_protocol_with_different_implementations(self):
        """Test that protocol works with different implementations."""
        
        class AlternativeDiscovery:
            """Alternative implementation using different internal structure."""
            
            def __init__(self):
                self._flow_store = []
                self._flow_lookup = {}
            
            async def refresh_flows(self) -> Dict[str, Any]:
                return {flow.name: flow for flow in self._flow_store if hasattr(flow, 'name')}
            
            def register_flow(self, flow: Any) -> None:
                self._flow_store.append(flow)
                if hasattr(flow, 'name'):
                    self._flow_lookup[flow.name] = flow
            
            def get_flow(self, name: str) -> Optional[Any]:
                return self._flow_lookup.get(name)
        
        alt_discovery = AlternativeDiscovery()
        
        # Should satisfy protocol
        assert isinstance(alt_discovery, FlowDiscoveryInterface)
        
        # Should work with same interface
        flow = MockFlow("alt_flow")
        alt_discovery.register_flow(flow)
        
        retrieved = alt_discovery.get_flow("alt_flow")
        assert retrieved == flow


class TestInterfaceTypeHints:
    """Test interface type hints and protocol structure."""
    
    def test_protocol_signature_annotations(self):
        """Test that protocol methods have proper type annotations."""
        import inspect
        
        # Get method signatures
        refresh_sig = inspect.signature(FlowDiscoveryInterface.refresh_flows)
        register_sig = inspect.signature(FlowDiscoveryInterface.register_flow)
        get_sig = inspect.signature(FlowDiscoveryInterface.get_flow)
        
        # Check refresh_flows signature
        assert refresh_sig.return_annotation == Dict[str, Any]
        
        # Check register_flow signature  
        assert list(register_sig.parameters.keys()) == ['self', 'flow']
        assert register_sig.parameters['flow'].annotation == Any
        assert refresh_sig.return_annotation == Dict[str, Any]
        
        # Check get_flow signature
        assert list(get_sig.parameters.keys()) == ['self', 'name']
        assert get_sig.parameters['name'].annotation == str
        assert get_sig.return_annotation == Optional[Any]
    
    def test_protocol_is_runtime_checkable(self):
        """Test that protocol can be used for runtime type checking."""
        # This should work without errors
        assert isinstance(MockFlowDiscovery(), FlowDiscoveryInterface)
        
        # This should fail
        class NotDiscovery:
            pass
        
        assert not isinstance(NotDiscovery(), FlowDiscoveryInterface)
    
    def test_protocol_method_count(self):
        """Test that protocol has expected number of methods."""
        # Get all methods defined in the protocol
        protocol_methods = [
            name for name in dir(FlowDiscoveryInterface)
            if not name.startswith('_') and callable(getattr(FlowDiscoveryInterface, name))
        ]
        
        # Should have exactly 3 methods
        assert len(protocol_methods) == 3
        assert 'refresh_flows' in protocol_methods
        assert 'register_flow' in protocol_methods
        assert 'get_flow' in protocol_methods