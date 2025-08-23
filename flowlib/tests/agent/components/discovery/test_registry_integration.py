"""Tests for FlowDiscovery integration with stage registry."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from flowlib.agent.components.discovery.flow_discovery import FlowDiscovery
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
            
        self.description = kwargs.get('description', f"Description for {name}")
        self.version = kwargs.get('version', "1.0")
    
    @classmethod
    def create(cls, **kwargs):
        """Alternative constructor."""
        return cls(**kwargs)


class MockStageRegistry:
    """Mock stage registry with comprehensive functionality."""
    
    def __init__(self):
        self.flows = {}  # name -> flow instance
        self.flow_instances = {}  # name -> flow instance  
        self.metadata = {}  # name -> FlowMetadata
        self.registration_calls = []
        self.get_instance_calls = []
        self.metadata_calls = []
    
    def get_flow_instances(self):
        """Get all flow instances."""
        self.get_instance_calls.append("get_flow_instances")
        return self.flow_instances.copy()
    
    def register_flow(self, name: str, flow):
        """Register a flow."""
        self.registration_calls.append((name, flow))
        self.flows[name] = flow
    
    def get_flow_metadata(self, name: str):
        """Get flow metadata."""
        self.metadata_calls.append(name)
        return self.metadata.get(name)
    
    def set_flow_metadata(self, name: str, metadata: FlowMetadata):
        """Set flow metadata for testing."""
        self.metadata[name] = metadata
    
    def add_flow_instance(self, name: str, flow):
        """Add flow instance for testing."""
        self.flow_instances[name] = flow


class TestFlowDiscoveryRegistryIntegration:
    """Test FlowDiscovery integration with stage registry."""
    
    @pytest.fixture
    def discovery(self):
        """Create FlowDiscovery instance."""
        return FlowDiscovery()
    
    @pytest.fixture
    def mock_registry(self):
        """Create mock stage registry."""
        return MockStageRegistry()
    
    @pytest.fixture
    def sample_flows(self):
        """Create sample flows with different metadata."""
        return {
            "agent_flow_1": MockFlow(
                "agent_flow_1", 
                is_infrastructure=False,
                description="Agent flow 1",
                version="1.0"
            ),
            "agent_flow_2": MockFlow(
                "agent_flow_2", 
                __flow_metadata__={"is_infrastructure": False, "category": "data"},
                description="Agent flow 2",
                version="1.1"
            ),
            "infra_flow": MockFlow(
                "infra_flow", 
                is_infrastructure=True,
                description="Infrastructure flow",
                version="2.0"
            ),
            "mixed_flow": MockFlow(
                "mixed_flow",
                is_infrastructure=False,
                __flow_metadata__={"is_infrastructure": False, "tags": ["processing"]},
                description="Mixed metadata flow",
                version="1.5"
            )
        }
    
    def test_get_flow_registry_returns_flow_registry(self, discovery, mock_registry):
        """Test that get_flow_registry returns the stage registry."""
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            result = discovery.get_flow_registry()
            
            assert result is mock_registry
    
    def test_get_flow_registry_returns_none_when_no_registry(self, discovery):
        """Test get_flow_registry when flow_registry is None."""
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', None):
            result = discovery.get_flow_registry()
            
            assert result is None
    
    def test_get_flow_metadata_success(self, discovery, mock_registry):
        """Test successful flow metadata retrieval."""
        flow_name = "test_flow"
        metadata = FlowMetadata(
            name=flow_name,
            description="Test flow description",
            input_model=MockInputModel,
            output_model=MockOutputModel,
            version="1.0"
        )
        mock_registry.set_flow_metadata(flow_name, metadata)
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            result = discovery.get_flow_metadata(flow_name)
            
            assert result == metadata
            assert flow_name in mock_registry.metadata_calls
    
    def test_get_flow_metadata_not_found(self, discovery, mock_registry):
        """Test get_flow_metadata for non-existent flow."""
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            result = discovery.get_flow_metadata("nonexistent_flow")
            
            assert result is None
            assert "nonexistent_flow" in mock_registry.metadata_calls
    
    def test_get_flow_metadata_no_registry(self, discovery):
        """Test get_flow_metadata when registry is None."""
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', None):
            result = discovery.get_flow_metadata("test_flow")
            
            assert result is None
    
    def test_register_flow_with_registry(self, discovery, mock_registry, sample_flows):
        """Test flow registration with stage registry."""
        flow = sample_flows["agent_flow_1"]
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            discovery.register_flow(flow)
        
        # Should register with both local storage and stage registry
        assert flow.name in discovery._flows
        assert discovery._flows[flow.name] == flow
        
        # Should have called registry.register_flow
        assert (flow.name, flow) in mock_registry.registration_calls
        assert flow.name in mock_registry.flows
    
    def test_register_flow_without_registry(self, discovery, sample_flows):
        """Test flow registration when stage registry is None."""
        flow = sample_flows["agent_flow_1"]
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', None):
            # Should not raise error
            discovery.register_flow(flow)
        
        # Should still store locally
        assert flow.name in discovery._flows
        assert discovery._flows[flow.name] == flow
    
    def test_discover_agent_flows_calls_registry(self, discovery, mock_registry, sample_flows):
        """Test that discover_agent_flows calls registry methods."""
        # Setup registry with flows
        for name, flow in sample_flows.items():
            mock_registry.add_flow_instance(name, flow)
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            result = discovery.discover_agent_flows()
        
        # Should have called get_flow_instances
        assert "get_flow_instances" in mock_registry.get_instance_calls
        
        # Should return non-infrastructure flows (deduplicated by class)
        # Since all flows are MockFlow instances, we get 1 unique class
        assert len(result) == 1  # MockFlow class (representing agent_flow_1, agent_flow_2, mixed_flow)
    
    @pytest.mark.asyncio
    async def test_refresh_flows_registry_interaction(self, discovery, mock_registry, sample_flows):
        """Test complete refresh_flows interaction with registry."""
        # Setup registry with flows
        for name, flow in sample_flows.items():
            mock_registry.add_flow_instance(name, flow)
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            await discovery.refresh_flows()
        
        # Should have discovered and registered agent flows
        expected_agent_flows = ["agent_flow_1", "agent_flow_2", "mixed_flow"]
        assert len(discovery._flows) == len(expected_agent_flows)
        
        for flow_name in expected_agent_flows:
            assert flow_name in discovery._flows
        
        # Should have registered each discovered flow with registry
        registered_names = [call[0] for call in mock_registry.registration_calls]
        for flow_name in expected_agent_flows:
            assert flow_name in registered_names
    
    def test_flow_metadata_extraction_from_instances(self, discovery, mock_registry):
        """Test metadata extraction from flow instances."""
        flows_with_metadata = {
            "metadata_flow": MockFlow(
                "metadata_flow",
                __flow_metadata__={
                    "is_infrastructure": False,
                    "category": "data_processing",
                    "tags": ["ETL", "transformation"],
                    "author": "test_author"
                }
            ),
            "attr_flow": MockFlow(
                "attr_flow",
                is_infrastructure=False,
                category="analysis"
            )
        }
        
        for name, flow in flows_with_metadata.items():
            mock_registry.add_flow_instance(name, flow)
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            result = discovery.discover_agent_flows()
        
        # Should extract flows based on metadata (deduplicated by class)
        # Since both flows are MockFlow instances, we get 1 unique class
        assert len(result) == 1
        
        # Verify that the expected flows are in the registry
        flow_instances = mock_registry.get_flow_instances()
        assert "metadata_flow" in flow_instances
        assert "attr_flow" in flow_instances
        assert not flow_instances["metadata_flow"].is_infrastructure
        assert not flow_instances["attr_flow"].is_infrastructure
    
    @pytest.mark.asyncio
    async def test_registry_state_consistency_after_operations(self, discovery, mock_registry, sample_flows):
        """Test that registry state remains consistent after discovery operations."""
        # Setup initial registry state
        initial_flows = {"existing_flow": MockFlow("existing_flow", is_infrastructure=True)}
        for name, flow in initial_flows.items():
            mock_registry.add_flow_instance(name, flow)
            mock_registry.flows[name] = flow
        
        # Add new flows for discovery
        for name, flow in sample_flows.items():
            mock_registry.add_flow_instance(name, flow)
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            # Perform refresh flows operation (which discovers instances and registers them)
            await discovery.refresh_flows()
        
        # Verify registry consistency
        # Original flows should still be there
        assert "existing_flow" in mock_registry.flows
        
        # New agent flows should be registered
        agent_flow_names = ["agent_flow_1", "agent_flow_2", "mixed_flow"]
        for flow_name in agent_flow_names:
            assert flow_name in mock_registry.flows
        
        # Infrastructure flows should not be registered by discovery
        assert "infra_flow" not in [call[0] for call in mock_registry.registration_calls]
    
    def test_flow_instance_creation_patterns(self, discovery, mock_registry):
        """Test different flow instance creation patterns."""
        class FlowWithCreate:
            def __init__(self, name="created_flow"):
                self.name = name
                self.is_infrastructure = False
            
            @classmethod
            def create(cls):
                return cls("from_create_method")
        
        class FlowWithoutCreate:
            def __init__(self):
                self.name = "constructor_flow" 
                self.is_infrastructure = False
        
        # Add flow instances to registry
        mock_registry.add_flow_instance("create_flow", FlowWithCreate())
        mock_registry.add_flow_instance("constructor_flow", FlowWithoutCreate())
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            # Discover flows
            flow_classes = discovery.discover_agent_flows()
            
            # Test flow creation during refresh
            with patch.object(discovery, 'discover_agent_flows', return_value=flow_classes):
                import asyncio
                asyncio.run(discovery.refresh_flows())
        
        # Verify flows were created and registered
        registered_names = [flow.name for flow in discovery._flows.values()]
        
        # Should include flows created via different methods
        assert len(registered_names) >= 2
    
    def test_registry_error_isolation(self, discovery, mock_registry, sample_flows):
        """Test that registry errors don't corrupt local state."""
        flow = sample_flows["agent_flow_1"]
        
        # Setup registry to fail on registration
        def failing_register(name, flow_obj):
            raise RuntimeError("Registry registration failed")
        
        mock_registry.register_flow = failing_register
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            # Should not raise error but still store locally
            discovery.register_flow(flow)
        
        # Local storage should still work
        assert flow.name in discovery._flows
        assert discovery._flows[flow.name] == flow
    
    @pytest.mark.asyncio
    async def test_concurrent_registry_access(self, discovery, mock_registry, sample_flows):
        """Test concurrent access to registry during discovery."""
        import asyncio
        
        # Setup registry with flows
        for name, flow in sample_flows.items():
            mock_registry.add_flow_instance(name, flow)
        
        # Track registry access
        call_count = {"get_instances": 0, "register": 0}
        
        original_get_instances = mock_registry.get_flow_instances
        original_register = mock_registry.register_flow
        
        def counting_get_instances():
            call_count["get_instances"] += 1
            return original_get_instances()
        
        def counting_register(name, flow):
            call_count["register"] += 1
            return original_register(name, flow)
        
        mock_registry.get_flow_instances = counting_get_instances
        mock_registry.register_flow = counting_register
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            # Perform concurrent refresh operations
            tasks = [discovery.refresh_flows() for _ in range(3)]
            await asyncio.gather(*tasks)
        
        # Verify registry was accessed correctly
        assert call_count["get_instances"] == 3  # One call per refresh
        assert call_count["register"] > 0  # Flows were registered
    
    def test_metadata_caching_behavior(self, discovery, mock_registry):
        """Test metadata retrieval and potential caching behavior."""
        flow_name = "test_flow"
        metadata = FlowMetadata(
            name=flow_name,
            description="Test description",
            input_model=MockInputModel,
            output_model=MockOutputModel,
            version="1.0"
        )
        mock_registry.set_flow_metadata(flow_name, metadata)
        
        with patch('flowlib.agent.components.discovery.flow_discovery.flow_registry', mock_registry):
            # Make multiple metadata requests
            result1 = discovery.get_flow_metadata(flow_name)
            result2 = discovery.get_flow_metadata(flow_name)
            result3 = discovery.get_flow_metadata(flow_name)
        
        # All should return the same metadata
        assert result1 == metadata
        assert result2 == metadata
        assert result3 == metadata
        
        # Should have made separate calls to registry (no caching in current implementation)
        assert mock_registry.metadata_calls.count(flow_name) == 3