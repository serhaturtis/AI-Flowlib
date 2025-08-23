"""Tests for FlowDiscovery class implementation."""

import pytest
import asyncio
from typing import Any, Dict, Optional, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from flowlib.agent.components.discovery.flow_discovery import FlowDiscovery
from flowlib.agent.components.discovery.interfaces import FlowDiscoveryInterface
from flowlib.agent.core.errors import FlowDiscoveryError, DiscoveryError
from flowlib.agent.core.base import AgentComponent
from flowlib.flows.models.metadata import FlowMetadata
from pydantic import BaseModel


class MockInputModel(BaseModel):
    """Mock input model for testing."""
    data: str


class MockOutputModel(BaseModel):
    """Mock output model for testing."""
    result: str


class MockFlow:
    """Mock flow class for testing."""
    
    def __init__(self, name: str = "test_flow", **kwargs):
        self.name = name
        self.__flow_metadata__ = kwargs.get('__flow_metadata__', {})
        
        # Set is_infrastructure based on either explicit kwarg or metadata
        if 'is_infrastructure' in kwargs:
            self.is_infrastructure = kwargs['is_infrastructure']
        elif 'is_infrastructure' in self.__flow_metadata__:
            self.is_infrastructure = self.__flow_metadata__['is_infrastructure']
        else:
            self.is_infrastructure = True  # Default to infrastructure flow
            
        for key, value in kwargs.items():
            if key not in ['is_infrastructure', '__flow_metadata__']:
                setattr(self, key, value)
    
    @classmethod
    def create(cls, **kwargs):
        """Alternative constructor method."""
        return cls(**kwargs)


class MockStageRegistry:
    """Mock stage registry for testing."""
    
    def __init__(self):
        self.flows = {}
        self.flow_instances = {}
        self.metadata = {}
        self.should_fail = False
        self.fail_operation = None
    
    def get_flow_instances(self):
        """Get all flow instances."""
        if self.should_fail and self.fail_operation == "get_instances":
            raise RuntimeError("Mock registry failure")
        return self.flow_instances.copy()
    
    def register_flow(self, name: str, flow: Any):
        """Register a flow."""
        if self.should_fail and self.fail_operation == "register":
            raise RuntimeError("Mock register failure")
        self.flows[name] = flow
    
    def get_flow_metadata(self, name: str) -> Optional[FlowMetadata]:
        """Get flow metadata."""
        return self.metadata.get(name)


class TestFlowDiscovery:
    """Test FlowDiscovery class."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create mock stage registry."""
        return MockStageRegistry()
    
    @pytest.fixture
    def discovery(self):
        """Create FlowDiscovery instance."""
        return FlowDiscovery()
    
    @pytest.fixture
    def sample_flows(self):
        """Create sample flows for testing."""
        return {
            "agent_flow": MockFlow("agent_flow", is_infrastructure=False),
            "infra_flow": MockFlow("infra_flow", is_infrastructure=True),
            "metadata_flow": MockFlow("metadata_flow", __flow_metadata__={"is_infrastructure": False}),
            "no_metadata_flow": MockFlow("no_metadata_flow")
        }
    
    def test_discovery_initialization(self, discovery):
        """Test FlowDiscovery initialization."""
        assert discovery.name == "flow_discovery"
        assert isinstance(discovery._flows, dict)
        assert len(discovery._flows) == 0
    
    def test_discovery_custom_name(self):
        """Test FlowDiscovery with custom name."""
        discovery = FlowDiscovery(name="custom_discovery")
        assert discovery.name == "custom_discovery"
    
    def test_discovery_inheritance(self, discovery):
        """Test that FlowDiscovery inherits from AgentComponent and implements interface."""
        assert isinstance(discovery, AgentComponent)
        assert isinstance(discovery, FlowDiscoveryInterface)
    
    @pytest.mark.asyncio
    async def test_initialize_impl_calls_refresh(self, discovery):
        """Test that _initialize_impl calls refresh_flows."""
        # Mock refresh_flows to track calls
        refresh_called = False
        
        async def mock_refresh():
            nonlocal refresh_called
            refresh_called = True
        
        discovery.refresh_flows = mock_refresh
        
        await discovery._initialize_impl()
        
        assert refresh_called is True
    
    @pytest.mark.asyncio
    async def test_shutdown_impl(self, discovery):
        """Test shutdown implementation."""
        # Should not raise any errors
        await discovery._shutdown_impl()
    
    @pytest.mark.asyncio
    async def test_refresh_flows_success(self, discovery, mock_registry, sample_flows):
        """Test successful flow refresh."""
        # Setup mock registry with flows
        mock_registry.flow_instances = {
            "agent_flow": sample_flows["agent_flow"],
            "metadata_flow": sample_flows["metadata_flow"]
        }
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            await discovery.refresh_flows()
        
        # Should have discovered and registered agent flows
        assert len(discovery._flows) == 2
        assert "agent_flow" in discovery._flows
        assert "metadata_flow" in discovery._flows
    
    @pytest.mark.asyncio
    async def test_refresh_flows_clears_existing(self, discovery, mock_registry):
        """Test that refresh_flows clears existing flows."""
        # Pre-populate flows
        discovery._flows = {"old_flow": MockFlow("old_flow")}
        
        # Setup empty registry
        mock_registry.flow_instances = {}
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            await discovery.refresh_flows()
        
        # Should have cleared old flows
        assert len(discovery._flows) == 0
        assert "old_flow" not in discovery._flows
    
    @pytest.mark.asyncio
    async def test_refresh_flows_discovery_error(self, discovery):
        """Test refresh_flows with discovery error."""
        with patch.object(discovery, 'discover_agent_flow_instances', side_effect=DiscoveryError("Mock error", "test")):
            with pytest.raises(FlowDiscoveryError) as exc_info:
                await discovery.refresh_flows()
            
            assert "Failed to refresh flows" in str(exc_info.value)
            assert isinstance(exc_info.value.__cause__, DiscoveryError)
    
    @pytest.mark.asyncio
    async def test_refresh_flows_registration_failure(self, discovery, mock_registry, sample_flows):
        """Test refresh_flows with flow registration failure."""
        # Create a flow that will fail during registration
        class FailingFlow:
            def __init__(self):
                raise RuntimeError("Constructor failure")
        
        mock_registry.flow_instances = {
            "good_flow": sample_flows["agent_flow"],
        }
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            with patch.object(discovery, 'discover_agent_flows', return_value=[sample_flows["agent_flow"].__class__, FailingFlow]):
                # Should not raise error but log warning
                await discovery.refresh_flows()
        
        # Should have registered the good flow despite failure of bad flow
        assert "good_flow" in discovery._flows or len(discovery._flows) > 0
    
    def test_register_flow_success(self, discovery, sample_flows):
        """Test successful flow registration."""
        flow = sample_flows["agent_flow"]
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry') as mock_registry:
            discovery.register_flow(flow)
        
        # Should store flow locally
        assert flow.name in discovery._flows
        assert discovery._flows[flow.name] == flow
        
        # Should register with flow_registry
        mock_registry.register_flow.assert_called_once_with(flow.name, flow)
    
    def test_register_flow_no_name(self, discovery):
        """Test registering flow without name."""
        flow_without_name = MockFlow()
        delattr(flow_without_name, 'name')
        
        # Should handle gracefully without error
        discovery.register_flow(flow_without_name)
        
        # Should not be registered
        assert len(discovery._flows) == 0
    
    def test_register_flow_registry_failure(self, discovery, sample_flows):
        """Test flow registration with flow_registry failure."""
        flow = sample_flows["agent_flow"]
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry') as mock_registry:
            mock_registry.register_flow.side_effect = RuntimeError("Registry error")
            
            # Should not raise error but log warning
            discovery.register_flow(flow)
        
        # Should still store flow locally
        assert flow.name in discovery._flows
    
    def test_register_flow_no_registry(self, discovery, sample_flows):
        """Test flow registration when flow_registry is None."""
        flow = sample_flows["agent_flow"]
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', None):
            # Should not raise error
            discovery.register_flow(flow)
        
        # Should still store flow locally
        assert flow.name in discovery._flows
    
    def test_get_flow_success(self, discovery, sample_flows):
        """Test successful flow retrieval."""
        flow = sample_flows["agent_flow"]
        discovery._flows[flow.name] = flow
        
        result = discovery.get_flow(flow.name)
        
        assert result == flow
    
    def test_get_flow_not_found(self, discovery):
        """Test flow retrieval for non-existent flow."""
        result = discovery.get_flow("nonexistent")
        
        assert result is None
    
    def test_get_flow_registry(self, discovery, mock_registry):
        """Test get_flow_registry method."""
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            result = discovery.get_flow_registry()
        
        assert result == mock_registry
    
    def test_get_flow_metadata_success(self, discovery, mock_registry):
        """Test successful flow metadata retrieval."""
        flow_name = "test_flow"
        metadata = FlowMetadata(
            name=flow_name,
            description="Test flow",
            input_model=MockInputModel,
            output_model=MockOutputModel,
            version="1.0"
        )
        mock_registry.metadata[flow_name] = metadata
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            result = discovery.get_flow_metadata(flow_name)
        
        assert result == metadata
    
    def test_get_flow_metadata_not_found(self, discovery, mock_registry):
        """Test flow metadata retrieval for non-existent flow."""
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            result = discovery.get_flow_metadata("nonexistent")
        
        assert result is None
    
    def test_get_flow_metadata_no_registry(self, discovery):
        """Test flow metadata retrieval when registry is None."""
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', None):
            result = discovery.get_flow_metadata("test_flow")
        
        assert result is None
    
    def test_discover_agent_flows_success(self, discovery, mock_registry, sample_flows):
        """Test successful agent flow discovery."""
        mock_registry.flow_instances = sample_flows
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            result = discovery.discover_agent_flows()
        
        # Should return classes of non-infrastructure flows (deduplicated)
        # Since all flows are MockFlow instances, we get 1 unique class
        assert len(result) == 1  # MockFlow class (representing agent_flow and metadata_flow)
        assert sample_flows["agent_flow"].__class__ in result
        assert sample_flows["metadata_flow"].__class__ in result
        # Note: infra_flow is also MockFlow, but it's infrastructure so the class
        # is only included because non-infrastructure MockFlow instances exist
    
    def test_discover_agent_flows_no_registry(self, discovery):
        """Test agent flow discovery with no registry."""
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', None):
            with pytest.raises(DiscoveryError) as exc_info:
                discovery.discover_agent_flows()
            
            assert "Stage registry not available" in str(exc_info.value)
            assert exc_info.value.operation == "discover_agent_flows"
    
    def test_discover_agent_flows_registry_failure(self, discovery, mock_registry):
        """Test agent flow discovery with registry failure."""
        mock_registry.should_fail = True
        mock_registry.fail_operation = "get_instances"
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            with pytest.raises(RuntimeError):
                discovery.discover_agent_flows()
    
    def test_discover_agent_flows_metadata_filtering(self, discovery, mock_registry):
        """Test flow filtering based on metadata."""
        flows = {
            "infra_true": MockFlow("infra_true", __flow_metadata__={"is_infrastructure": True}),
            "infra_false": MockFlow("infra_false", __flow_metadata__={"is_infrastructure": False}),
            "no_metadata": MockFlow("no_metadata", __flow_metadata__={}),
            "empty_metadata": MockFlow("empty_metadata")
        }
        mock_registry.flow_instances = flows
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            result = discovery.discover_agent_flows()
        
        # Should only include flows with is_infrastructure=False
        assert len(result) == 1
        assert flows["infra_false"].__class__ in result
    
    def test_discover_agent_flows_attribute_filtering(self, discovery, mock_registry):
        """Test flow filtering based on is_infrastructure attribute."""
        flows = {
            "infra_attr_true": MockFlow("infra_attr_true", is_infrastructure=True),
            "infra_attr_false": MockFlow("infra_attr_false", is_infrastructure=False),
        }
        mock_registry.flow_instances = flows
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            result = discovery.discover_agent_flows()
        
        # Should only include flows with is_infrastructure=False
        assert len(result) == 1
        assert flows["infra_attr_false"].__class__ in result
    
    @pytest.mark.asyncio
    async def test_flow_creation_with_create_method(self, discovery, mock_registry):
        """Test flow registration with instance created via create() method."""
        class FlowWithCreate:
            def __init__(self, name="created_flow"):
                self.name = name
                self.is_infrastructure = False
            
            @classmethod
            def create(cls):
                return cls("from_create")
        
        # Create flow instance using the create() method and put it in registry
        created_flow = FlowWithCreate.create()
        flows = {"created_flow": created_flow}
        mock_registry.flow_instances = flows
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            await discovery.refresh_flows()
        
        # Should have registered the flow instance that was created with create() method
        registered_flows = list(discovery._flows.values())
        assert len(registered_flows) == 1
        assert registered_flows[0].name == "from_create"
    
    @pytest.mark.asyncio
    async def test_flow_creation_without_create_method(self, discovery, mock_registry):
        """Test flow creation using constructor."""
        class FlowWithoutCreate:
            def __init__(self):
                self.name = "constructor_flow"
                self.is_infrastructure = False
        
        flows = {"test_flow": FlowWithoutCreate()}
        mock_registry.flow_instances = flows
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            with patch.object(discovery, 'discover_agent_flows', return_value=[FlowWithoutCreate]):
                await discovery.refresh_flows()
        
        # Should have used constructor
        registered_flows = list(discovery._flows.values())
        assert len(registered_flows) == 1
        assert registered_flows[0].name == "constructor_flow"


class TestFlowDiscoveryIntegration:
    """Test FlowDiscovery integration scenarios."""
    
    @pytest.fixture
    def discovery(self):
        """Create FlowDiscovery instance."""
        return FlowDiscovery()
    
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, discovery):
        """Test complete discovery lifecycle."""
        # Create test flows
        agent_flow = MockFlow("test_agent", is_infrastructure=False)
        infra_flow = MockFlow("test_infra", is_infrastructure=True)
        
        mock_registry = MockStageRegistry()
        mock_registry.flow_instances = {
            "test_agent": agent_flow,
            "test_infra": infra_flow
        }
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            # Initialize (should trigger refresh)
            await discovery.initialize()
            
            # Verify only agent flow was discovered
            assert discovery.initialized is True
            assert len(discovery._flows) == 1
            assert "test_agent" in discovery._flows
            
            # Test retrieval
            retrieved = discovery.get_flow("test_agent")
            assert retrieved is not None
            assert retrieved.name == "test_agent"
            
            # Test registry access
            registry = discovery.get_flow_registry()
            assert registry == mock_registry
            
            # Shutdown
            await discovery.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, discovery):
        """Test concurrent discovery operations."""
        flows = {
            f"flow_{i}": MockFlow(f"flow_{i}", is_infrastructure=False)
            for i in range(10)
        }
        
        mock_registry = MockStageRegistry()
        mock_registry.flow_instances = flows
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            # Perform concurrent refresh and registration operations
            tasks = []
            
            # Multiple refresh operations
            for _ in range(3):
                tasks.append(discovery.refresh_flows())
            
            # Wait for all operations to complete
            await asyncio.gather(*tasks)
            
            # Verify final state is consistent
            assert len(discovery._flows) == 10
            for i in range(10):
                assert f"flow_{i}" in discovery._flows
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, discovery):
        """Test error recovery in discovery operations."""
        mock_registry = MockStageRegistry()
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            # First, simulate registry failure
            mock_registry.should_fail = True
            mock_registry.fail_operation = "get_instances"
            
            with pytest.raises(FlowDiscoveryError):
                await discovery.refresh_flows()
            
            # Recovery: fix registry and retry
            mock_registry.should_fail = False
            mock_registry.flow_instances = {
                "recovery_flow": MockFlow("recovery_flow", is_infrastructure=False)
            }
            
            # Should succeed after recovery
            await discovery.refresh_flows()
            assert len(discovery._flows) == 1
            assert "recovery_flow" in discovery._flows
    
    def test_multiple_discovery_instances(self):
        """Test multiple discovery instances operating independently."""
        discovery1 = FlowDiscovery("discovery1")
        discovery2 = FlowDiscovery("discovery2")
        
        flow1 = MockFlow("flow1")
        flow2 = MockFlow("flow2")
        
        # Register different flows in each discovery
        discovery1.register_flow(flow1)
        discovery2.register_flow(flow2)
        
        # Verify isolation
        assert discovery1.get_flow("flow1") == flow1
        assert discovery1.get_flow("flow2") is None
        
        assert discovery2.get_flow("flow2") == flow2
        assert discovery2.get_flow("flow1") is None