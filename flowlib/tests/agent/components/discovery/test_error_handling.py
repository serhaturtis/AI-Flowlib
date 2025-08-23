"""Tests for discovery error handling scenarios."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from flowlib.agent.components.discovery.flow_discovery import FlowDiscovery
from flowlib.agent.core.errors import FlowDiscoveryError, DiscoveryError


class MockFlow:
    """Mock flow for error testing."""
    
    def __init__(self, name: str = "test_flow", should_fail: bool = False, **kwargs):
        if should_fail:
            raise RuntimeError(f"Mock flow {name} construction failed")
        self.name = name
        self.__flow_metadata__ = kwargs.get('__flow_metadata__', {})
        
        # Set is_infrastructure based on either explicit kwarg or metadata
        if 'is_infrastructure' in kwargs:
            self.is_infrastructure = kwargs['is_infrastructure']
        elif 'is_infrastructure' in self.__flow_metadata__:
            self.is_infrastructure = self.__flow_metadata__['is_infrastructure']
        else:
            self.is_infrastructure = True  # Default to infrastructure flow
    
    @classmethod
    def create(cls, should_fail: bool = False):
        """Create method that can fail."""
        if should_fail:
            raise RuntimeError("Mock create method failed")
        return cls()


class FailingStageRegistry:
    """Mock registry that fails in various ways."""
    
    def __init__(self, fail_mode: str = None):
        self.fail_mode = fail_mode
        self.flows = {}
        self.flow_instances = {}
    
    def get_flow_instances(self):
        """Get flow instances with potential failure."""
        if self.fail_mode == "get_instances":
            raise RuntimeError("Failed to get flow instances")
        elif self.fail_mode == "get_instances_timeout":
            import time
            time.sleep(10)  # Simulate timeout
        elif self.fail_mode == "get_instances_empty":
            return {}
        return self.flow_instances
    
    def register_flow(self, name: str, flow):
        """Register flow with potential failure."""
        if self.fail_mode == "register":
            raise RuntimeError(f"Failed to register flow {name}")
        elif self.fail_mode == "register_permission":
            raise PermissionError(f"Permission denied for flow {name}")
        self.flows[name] = flow
    
    def get_flow_metadata(self, name: str):
        """Get metadata with potential failure."""
        if self.fail_mode == "metadata":
            raise RuntimeError(f"Failed to get metadata for {name}")
        return None


class TestFlowDiscoveryErrorHandling:
    """Test error handling in FlowDiscovery."""
    
    @pytest.fixture
    def discovery(self):
        """Create FlowDiscovery instance."""
        return FlowDiscovery()
    
    @pytest.mark.asyncio
    async def test_refresh_flows_registry_unavailable(self, discovery):
        """Test refresh_flows when flow_registry is None."""
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', None):
            with pytest.raises(FlowDiscoveryError) as exc_info:
                await discovery.refresh_flows()
            
            assert "Failed to refresh flows" in str(exc_info.value)
            assert isinstance(exc_info.value.__cause__, DiscoveryError)
            assert "Stage registry not available" in str(exc_info.value.__cause__)
    
    @pytest.mark.asyncio
    async def test_refresh_flows_registry_get_instances_failure(self, discovery):
        """Test refresh_flows when registry.get_flow_instances() fails."""
        failing_registry = FailingStageRegistry(fail_mode="get_instances")
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', failing_registry):
            with pytest.raises(FlowDiscoveryError) as exc_info:
                await discovery.refresh_flows()
            
            assert "Failed to refresh flows" in str(exc_info.value)
            assert "Failed to get flow instances" in str(exc_info.value.__cause__)
    
    @pytest.mark.asyncio
    async def test_refresh_flows_flow_registration_failure(self, discovery):
        """Test refresh_flows when flow registration fails."""
        # Create a flow instance that will cause registration to fail
        failing_flow = MockFlow("failing_flow", is_infrastructure=False)
        good_flow = MockFlow("good_flow", is_infrastructure=False)
        
        registry = Mock()
        registry.get_flow_instances.return_value = {
            "good_flow": good_flow,
            "failing_flow": failing_flow,
        }
        
        # Mock register_flow to fail for the failing flow
        original_register = discovery.register_flow
        def mock_register(flow):
            if flow.name == "failing_flow":
                raise ValueError("Registration failed")
            return original_register(flow)
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', registry):
            with patch.object(discovery, 'register_flow', side_effect=mock_register):
                # Should complete despite flow registration failure
                await discovery.refresh_flows()
                
                # Should have registered the good flow
                assert len(discovery._flows) == 1
                assert "good_flow" in discovery._flows
    
    # Note: test_refresh_flows_flow_create_method_failure removed as it's no longer
    # relevant with instance-based discovery approach
    
    @pytest.mark.asyncio
    async def test_refresh_flows_multiple_registration_failures(self, discovery):
        """Test refresh_flows with multiple flow registration failures."""
        # Create flow instances
        failing_flow1 = MockFlow("failing_flow1", is_infrastructure=False)
        failing_flow2 = MockFlow("failing_flow2", is_infrastructure=False)
        good_flow = MockFlow("good_flow", is_infrastructure=False)
        
        registry = Mock()
        registry.get_flow_instances.return_value = {
            "failing_flow1": failing_flow1,
            "failing_flow2": failing_flow2,
            "good_flow": good_flow,
        }
        
        # Mock register_flow to fail for multiple flows
        original_register = discovery.register_flow
        def mock_register(flow):
            if flow.name in ["failing_flow1", "failing_flow2"]:
                raise RuntimeError(f"Registration failed for {flow.name}")
            return original_register(flow)
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', registry):
            with patch.object(discovery, 'register_flow', side_effect=mock_register):
                # Should complete and register the good flow
                await discovery.refresh_flows()
                
                # Should have registered the one good flow
                assert len(discovery._flows) == 1
                assert "good_flow" in discovery._flows
    
    def test_register_flow_without_name_attribute(self, discovery):
        """Test registering flow without name attribute."""
        flow_without_name = Mock(spec=[])  # No name attribute
        
        # Should handle gracefully without error
        discovery.register_flow(flow_without_name)
        
        # Should not be registered
        assert len(discovery._flows) == 0
    
    def test_register_flow_registry_failure(self, discovery):
        """Test flow registration when flow_registry.register_flow fails."""
        flow = MockFlow("test_flow")
        failing_registry = FailingStageRegistry(fail_mode="register")
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', failing_registry):
            # Should not raise error but log warning
            discovery.register_flow(flow)
        
        # Should still store flow locally despite registry failure
        assert flow.name in discovery._flows
        assert discovery._flows[flow.name] == flow
    
    def test_register_flow_permission_error(self, discovery):
        """Test flow registration with permission error."""
        flow = MockFlow("test_flow")
        failing_registry = FailingStageRegistry(fail_mode="register_permission")
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', failing_registry):
            # Should not raise error but log warning
            discovery.register_flow(flow)
        
        # Should still store flow locally
        assert flow.name in discovery._flows
    
    def test_get_flow_metadata_registry_failure(self, discovery):
        """Test get_flow_metadata when registry operation fails."""
        failing_registry = FailingStageRegistry(fail_mode="metadata")
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', failing_registry):
            # Should return None without raising error
            result = discovery.get_flow_metadata("test_flow")
            assert result is None
    
    def test_discover_agent_flows_no_registry(self, discovery):
        """Test discover_agent_flows when registry is None."""
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', None):
            with pytest.raises(DiscoveryError) as exc_info:
                discovery.discover_agent_flows()
            
            assert exc_info.value.message == "Stage registry not available for flow discovery"
            assert exc_info.value.operation == "discover_agent_flows"
    
    def test_discover_agent_flows_registry_exception(self, discovery):
        """Test discover_agent_flows when registry raises exception."""
        failing_registry = FailingStageRegistry(fail_mode="get_instances")
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', failing_registry):
            # Should propagate the underlying exception
            with pytest.raises(RuntimeError, match="Failed to get flow instances"):
                discovery.discover_agent_flows()
    
    def test_discover_agent_flows_malformed_flow_instances(self, discovery):
        """Test discover_agent_flows with malformed flow instances."""
        registry = Mock()
        registry.get_flow_instances.return_value = {
            "good_flow": MockFlow("good_flow", is_infrastructure=False),
            "no_metadata": Mock(spec=[]),  # No __flow_metadata__ or is_infrastructure
            "none_flow": None,  # None value
            "broken_flow": Mock(side_effect=Exception("Broken")),  # Raises on access
        }
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', registry):
            # Should handle malformed flows gracefully
            result = discovery.discover_agent_flows()
            
            # Should only return well-formed flows
            assert len(result) == 1
            assert result[0] == MockFlow("good_flow", is_infrastructure=False).__class__
    
    @pytest.mark.asyncio
    async def test_initialization_failure_propagation(self, discovery):
        """Test that initialization failures are properly propagated."""
        with patch.object(discovery, 'refresh_flows', side_effect=FlowDiscoveryError("Refresh failed")):
            with pytest.raises(Exception):  # BaseComponent will wrap in ResourceError
                await discovery.initialize()
    
    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, discovery):
        """Test error handling under concurrent operations."""
        failing_registry = FailingStageRegistry(fail_mode="get_instances")
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', failing_registry):
            # Start multiple concurrent refresh operations
            tasks = [discovery.refresh_flows() for _ in range(5)]
            
            # All should fail with FlowDiscoveryError
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                assert isinstance(result, FlowDiscoveryError)
                assert "Failed to refresh flows" in str(result)
    
    def test_error_context_preservation(self, discovery):
        """Test that error context is preserved through error chain."""
        original_error = ValueError("Original error message")
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', None):
            with pytest.raises(FlowDiscoveryError) as exc_info:
                # This will trigger DiscoveryError which gets wrapped in FlowDiscoveryError
                asyncio.run(discovery.refresh_flows())
            
            # Check error chain
            flow_error = exc_info.value
            assert isinstance(flow_error, FlowDiscoveryError)
            assert "Failed to refresh flows" in str(flow_error)
            
            discovery_error = flow_error.__cause__
            assert isinstance(discovery_error, DiscoveryError)
            assert "Stage registry not available" in str(discovery_error)
            assert discovery_error.operation == "discover_agent_flow_instances"


class TestFlowDiscoveryEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def discovery(self):
        """Create FlowDiscovery instance."""
        return FlowDiscovery()
    
    @pytest.mark.asyncio
    async def test_refresh_flows_empty_registry(self, discovery):
        """Test refresh_flows with empty registry."""
        registry = Mock()
        registry.get_flow_instances.return_value = {}
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', registry):
            await discovery.refresh_flows()
        
        # Should complete successfully with no flows
        assert len(discovery._flows) == 0
    
    @pytest.mark.asyncio
    async def test_refresh_flows_all_infrastructure_flows(self, discovery):
        """Test refresh_flows when all flows are infrastructure flows."""
        # Clear any existing flows first
        discovery._flows.clear()
        
        registry = Mock()
        registry.get_flow_instances.return_value = {
            "infra1": MockFlow("infra1", is_infrastructure=True),
            "infra2": MockFlow("infra2", __flow_metadata__={"is_infrastructure": True}),
        }
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', registry):
            await discovery.refresh_flows()
        
        # Should complete successfully with no agent flows
        assert len(discovery._flows) == 0
    
    def test_register_flow_duplicate_names(self, discovery):
        """Test registering flows with duplicate names."""
        flow1 = MockFlow("duplicate_name", description="First flow")
        flow2 = MockFlow("duplicate_name", description="Second flow")
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry'):
            discovery.register_flow(flow1)
            discovery.register_flow(flow2)
        
        # Second flow should overwrite first
        assert len(discovery._flows) == 1
        assert discovery._flows["duplicate_name"] == flow2
    
    def test_register_flow_none_flow(self, discovery):
        """Test registering None as flow."""
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry'):
            # Should handle gracefully
            discovery.register_flow(None)
        
        assert len(discovery._flows) == 0
    
    def test_get_flow_empty_name(self, discovery):
        """Test getting flow with empty name."""
        result = discovery.get_flow("")
        assert result is None
        
        result = discovery.get_flow(None)
        assert result is None
    
    def test_discover_agent_flows_mixed_metadata_formats(self, discovery):
        """Test discovery with mixed metadata formats."""
        registry = Mock()
        registry.get_flow_instances.return_value = {
            "attr_false": MockFlow("attr_false", is_infrastructure=False),
            "attr_true": MockFlow("attr_true", is_infrastructure=True),
            "meta_false": MockFlow("meta_false", __flow_metadata__={"is_infrastructure": False}),
            "meta_true": MockFlow("meta_true", __flow_metadata__={"is_infrastructure": True}),
            "meta_missing": MockFlow("meta_missing", __flow_metadata__={}),
            "no_attr_no_meta": MockFlow("no_attr_no_meta"),
        }
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', registry):
            result = discovery.discover_agent_flows()
        
        # Should only include flows explicitly marked as non-infrastructure (deduplicated by class)
        # Since both flows are MockFlow instances, we get 1 unique class
        assert len(result) == 1
        # Verify that the expected flows are in the registry
        flow_instances = registry.get_flow_instances()
        assert "attr_false" in flow_instances
        assert "meta_false" in flow_instances
        assert not flow_instances["attr_false"].is_infrastructure
        assert not flow_instances["meta_false"].is_infrastructure
    
    @pytest.mark.asyncio
    async def test_shutdown_with_pending_operations(self, discovery):
        """Test shutdown while operations are pending."""
        # Start a long-running refresh operation
        slow_registry = Mock()
        slow_registry.get_flow_instances.side_effect = lambda: asyncio.sleep(5)
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', slow_registry):
            # Start refresh but don't wait
            refresh_task = asyncio.create_task(discovery.refresh_flows())
            
            # Shutdown immediately
            await discovery.shutdown()
            
            # Cancel the pending task
            refresh_task.cancel()
            
            # Should not raise errors
            assert not discovery.initialized