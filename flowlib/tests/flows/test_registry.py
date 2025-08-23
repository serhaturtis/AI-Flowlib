"""
Tests for the FlowRegistry system.

This module tests the core flow registry functionality including
flow registration, metadata management, discovery, and agent-selectable flow filtering.
"""

import pytest
import logging
from typing import Any, Dict, Optional, List
from unittest.mock import Mock, patch
from pydantic import BaseModel

from flowlib.flows.registry.registry import FlowRegistry, stage_registry
from flowlib.flows.models.metadata import FlowMetadata


class MockInputModel(BaseModel):
    """Mock input model for testing."""
    value: str


class MockOutputModel(BaseModel):
    """Mock output model for testing."""
    result: str


class MockFlow:
    """Mock flow for testing."""
    
    def __init__(self, name: str = "test_flow", description: str = "Test flow", is_infrastructure: bool = False):
        self.name = name
        self.description = description
        self.is_infrastructure = is_infrastructure
        self.flow_type = "test"
        
        # Add flow metadata
        self.__flow_metadata__ = {
            "name": name,
            "description": description,
            "is_infrastructure": is_infrastructure,
            "version": "1.0.0",
            "author": "test",
            "tags": ["test"]
        }
    
    def get_description(self):
        return self.description
    
    def get_pipeline_method(self):
        """Return mock pipeline with input/output models."""
        pipeline = Mock()
        pipeline.__input_model__ = MockInputModel
        pipeline.__output_model__ = MockOutputModel
        return pipeline


class MockFlowClass:
    """Mock flow class for testing."""
    
    def __init__(self, name: str = "class_flow", should_fail_init: bool = False):
        self._name = name
        self._should_fail_init = should_fail_init
        
        self.__flow_metadata__ = {
            "name": name,
            "description": f"Class-based {name}",
            "is_infrastructure": False,
            "version": "1.0.0"
        }
    
    def get_description(self):
        return f"Instance of {self._name}"
    
    def get_pipeline_method(self):
        """Return mock pipeline with input/output models."""
        pipeline = Mock()
        pipeline.__input_model__ = MockInputModel
        pipeline.__output_model__ = MockOutputModel
        return pipeline


class TestFlowRegistry:
    """Test the FlowRegistry class."""
    
    def setup_method(self):
        """Set up each test with a fresh registry."""
        self.registry = FlowRegistry()
    
    def test_init(self):
        """Test registry initialization."""
        assert len(self.registry._flows) == 0
        assert len(self.registry._flow_instances) == 0
        assert len(self.registry._flow_metadata) == 0
    
    def test_register_flow_with_none(self):
        """Test registering a flow with None (name only)."""
        self.registry.register_flow("standalone_flow", None)
        
        assert "standalone_flow" in self.registry._flows
        flow_info = self.registry._flows["standalone_flow"]
        assert flow_info["name"] == "standalone_flow"
        assert flow_info["metadata"]["is_infrastructure"] is False
    
    def test_register_flow_with_instance(self):
        """Test registering a flow with an instance."""
        flow = MockFlow("instance_flow", "Instance-based flow")
        
        self.registry.register_flow("instance_flow", flow)
        
        assert "instance_flow" in self.registry._flows
        assert self.registry._flow_instances["instance_flow"] is flow
        
        flow_info = self.registry._flows["instance_flow"]
        assert flow_info["name"] == "instance_flow"
        assert flow_info["metadata"]["name"] == "instance_flow"
    
    def test_register_flow_with_class_success(self):
        """Test registering a flow with a class (successful instantiation)."""
        # Create a simple class for testing
        class SimpleFlowClass:
            __flow_metadata__ = {"name": "simple", "description": "Simple flow", "is_infrastructure": False}
            
            def __init__(self):
                self.name = "simple_flow"
                self.description = "Simple test flow"
                self.is_infrastructure = False
                self.__flow_metadata__ = SimpleFlowClass.__flow_metadata__.copy()
            
            def get_description(self):
                return self.description
            
            def get_pipeline_method(self):
                pipeline = Mock()
                pipeline.__input_model__ = MockInputModel
                pipeline.__output_model__ = MockOutputModel
                return pipeline
        
        self.registry.register_flow("simple_flow", SimpleFlowClass)
        
        assert "simple_flow" in self.registry._flows
        assert "simple_flow" in self.registry._flow_instances
        
        instance = self.registry._flow_instances["simple_flow"]
        assert instance.name == "simple_flow"
        assert instance.description == "Simple test flow"
    
    def test_register_flow_with_class_failure(self):
        """Test registering a flow with a class (failed instantiation)."""
        # Create a class that will fail on instantiation
        class FailingFlowClass:
            __flow_metadata__ = {"name": "failing", "description": "Failing flow", "is_infrastructure": False}
            
            def __init__(self):
                raise RuntimeError("Intentional failure")
            
            def get_description(self):
                return "Failing flow"
            
            def get_pipeline_method(self):
                pipeline = Mock()
                pipeline.__input_model__ = MockInputModel
                pipeline.__output_model__ = MockOutputModel
                return pipeline
        
        self.registry.register_flow("failing_flow", FailingFlowClass)
        
        assert "failing_flow" in self.registry._flows
        # Removed redundant context.get() test - strict validation
    
    def test_register_flow_with_metadata_creation(self):
        """Test flow registration with metadata creation."""
        flow = MockFlow("meta_flow", "Flow with metadata")
        
        # Don't patch - let real metadata creation happen
        self.registry.register_flow("meta_flow", flow)
        
        assert "meta_flow" in self.registry._flow_metadata
        metadata = self.registry._flow_metadata["meta_flow"]
        assert metadata is not None
        assert metadata.name == "meta_flow"
        assert metadata.description == "Flow with metadata"
    
    def test_register_flow_metadata_creation_failure(self):
        """Test flow registration when metadata creation fails."""
        # Create a flow that will cause metadata creation to fail
        class BadFlow:
            def __init__(self):
                self.name = "meta_fail_flow"
                self.description = "Flow that fails metadata creation"
                self.__flow_metadata__ = {"name": "meta_fail_flow", "description": "Bad flow"}
            
            def get_description(self):
                return self.description
            
            def get_pipeline_method(self):
                # Return None to cause metadata creation to fail
                return None
        
        flow = BadFlow()
        
        # Should not raise exception, just log warning
        self.registry.register_flow("meta_fail_flow", flow)
        
        assert "meta_fail_flow" in self.registry._flows
        assert "meta_fail_flow" not in self.registry._flow_metadata
    
    def test_register_flow_duplicate_skip(self):
        """Test registering a duplicate flow (should skip)."""
        flow1 = MockFlow("duplicate_flow", "First flow")
        flow2 = MockFlow("duplicate_flow", "Second flow")
        
        self.registry.register_flow("duplicate_flow", flow1)
        self.registry.register_flow("duplicate_flow", flow2)  # Should skip
        
        # Should still have the first instance
        assert self.registry._flow_instances["duplicate_flow"] is flow1
        assert self.registry._flow_instances["duplicate_flow"].description == "First flow"
    
    def test_contains_flow(self):
        """Test checking if a flow is registered."""
        flow = MockFlow("test_flow")
        self.registry.register_flow("test_flow", flow)
        
        assert self.registry.contains_flow("test_flow") is True
        assert self.registry.contains_flow("missing_flow") is False
    
    def test_get_flows(self):
        """Test getting all registered flow names."""
        flow1 = MockFlow("flow1")
        flow2 = MockFlow("flow2")
        flow3 = MockFlow("flow3")
        
        self.registry.register_flow("flow2", flow2)
        self.registry.register_flow("flow1", flow1)
        self.registry.register_flow("flow3", flow3)
        
        flows = self.registry.get_flows()
        assert flows == ["flow1", "flow2", "flow3"]  # Should be sorted
    
    def test_get_flow_metadata_exists(self):
        """Test getting metadata for existing flow."""
        flow = MockFlow("test_flow")
        self.registry.register_flow("test_flow", flow)
        
        metadata = self.registry.get_flow_metadata("test_flow")
        assert metadata is not None
        assert metadata.name == "test_flow"
    
    def test_get_flow_metadata_not_exists(self):
        """Test getting metadata for non-existent flow."""
        metadata = self.registry.get_flow_metadata("missing_flow")
        assert metadata is None
    
    def test_get_all_flow_metadata(self):
        """Test getting metadata for all flows."""
        flow1 = MockFlow("flow1")
        flow2 = MockFlow("flow2")
        
        self.registry.register_flow("flow1", flow1)
        self.registry.register_flow("flow2", flow2)
        
        all_metadata = self.registry.get_all_flow_metadata()
        assert len(all_metadata) == 2
        assert "flow1" in all_metadata
        assert "flow2" in all_metadata
    
    def test_get_flow_exists(self):
        """Test getting an existing flow instance."""
        flow = MockFlow("test_flow")
        self.registry.register_flow("test_flow", flow)
        
        retrieved = self.registry.get_flow("test_flow")
        assert retrieved is flow
    
    def test_get_flow_not_exists(self):
        """Test getting a non-existent flow instance."""
        retrieved = self.registry.get_flow("missing_flow")
        assert retrieved is None
    
    def test_get_flow_instances(self):
        """Test getting all flow instances."""
        flow1 = MockFlow("flow1")
        flow2 = MockFlow("flow2")
        
        self.registry.register_flow("flow1", flow1)
        self.registry.register_flow("flow2", flow2)
        
        instances = self.registry.get_flow_instances()
        assert len(instances) == 2
        assert instances["flow1"] is flow1
        assert instances["flow2"] is flow2
    
    def test_get_agent_selectable_flows(self):
        """Test getting agent-selectable flows (non-infrastructure)."""
        flow1 = MockFlow("flow1", is_infrastructure=False)
        flow2 = MockFlow("flow2", is_infrastructure=True)
        flow3 = MockFlow("flow3", is_infrastructure=False)
        
        self.registry.register_flow("flow1", flow1)
        self.registry.register_flow("flow2", flow2)
        self.registry.register_flow("flow3", flow3)
        
        selectable = self.registry.get_agent_selectable_flows()
        assert len(selectable) == 2
        assert "flow1" in selectable
        assert "flow3" in selectable
        assert "flow2" not in selectable  # Infrastructure flow excluded
    
    def test_clear(self):
        """Test clearing the registry."""
        flow = MockFlow("test_flow")
        self.registry.register_flow("test_flow", flow)
        
        self.registry.clear()
        
        assert len(self.registry._flows) == 0
        assert len(self.registry._flow_instances) == 0
        assert len(self.registry._flow_metadata) == 0


class TestGlobalRegistry:
    """Test the global registry singleton."""
    
    def test_global_registry_exists(self):
        """Test that the global registry exists."""
        assert stage_registry is not None
        assert isinstance(stage_registry, FlowRegistry)
    
    def test_global_registry_singleton(self):
        """Test that the global registry is a singleton."""
        from flowlib.flows.registry.registry import stage_registry as registry1
        from flowlib.flows.registry.registry import stage_registry as registry2
        
        assert registry1 is registry2


class TestFlowRegistryIntegration:
    """Integration tests for the flow registry."""
    
    def setup_method(self):
        """Set up each test with a fresh registry."""
        self.registry = FlowRegistry()
    
    def test_complex_flow_hierarchy(self):
        """Test registering a complex hierarchy of flows."""
        # Create infrastructure flows
        infra_flow1 = MockFlow("auth_flow", "Authentication flow", is_infrastructure=True)
        infra_flow2 = MockFlow("logging_flow", "Logging flow", is_infrastructure=True)
        
        # Create user-facing flows
        user_flow1 = MockFlow("chat_flow", "Chat conversation flow", is_infrastructure=False)
        user_flow2 = MockFlow("analysis_flow", "Data analysis flow", is_infrastructure=False)
        
        # Register all flows
        self.registry.register_flow("auth_flow", infra_flow1)
        self.registry.register_flow("logging_flow", infra_flow2)
        self.registry.register_flow("chat_flow", user_flow1)
        self.registry.register_flow("analysis_flow", user_flow2)
        
        # Test agent-selectable flows
        selectable = self.registry.get_agent_selectable_flows()
        assert len(selectable) == 2
        assert "chat_flow" in selectable
        assert "analysis_flow" in selectable
        
        # Test all flows
        all_flows = self.registry.get_flows()
        assert len(all_flows) == 4
    
    def test_agent_workflow_simulation(self):
        """Simulate an agent discovering and using flows."""
        # Create some flows
        flows = [
            MockFlow("memory_retrieval", "Retrieve memories", is_infrastructure=False),
            MockFlow("plan_generation", "Generate plans", is_infrastructure=False),
            MockFlow("task_execution", "Execute tasks", is_infrastructure=False),
            MockFlow("logging", "Log operations", is_infrastructure=True)
        ]
        
        # Register flows
        for flow in flows:
            self.registry.register_flow(flow.name, flow)
        
        # Agent discovers available flows
        available = self.registry.get_agent_selectable_flows()
        assert len(available) == 3  # Excludes infrastructure flow
        
        # Agent selects a flow to use
        selected_flow_name = "plan_generation"
        selected_flow = self.registry.get_flow(selected_flow_name)
        assert selected_flow is not None
        assert selected_flow.description == "Generate plans"
    
    def test_metadata_driven_discovery(self):
        """Test discovering flows based on metadata."""
        # Create flows with different metadata
        flow1 = MockFlow("flow1", "First flow")
        flow1.__flow_metadata__["tags"] = ["chat", "conversation"]
        
        flow2 = MockFlow("flow2", "Second flow")
        flow2.__flow_metadata__["tags"] = ["analysis", "data"]
        
        flow3 = MockFlow("flow3", "Third flow")
        flow3.__flow_metadata__["tags"] = ["chat", "analysis"]
        
        # Register flows
        self.registry.register_flow("flow1", flow1)
        self.registry.register_flow("flow2", flow2)
        self.registry.register_flow("flow3", flow3)
        
        # Get all metadata
        all_metadata = self.registry.get_all_flow_metadata()
        
        # Find flows with specific tags
        chat_flows = []
        for name, metadata in all_metadata.items():
            flow = self.registry.get_flow(name)
            if flow and hasattr(flow, "__flow_metadata__"):
                tags = flow.__flow_metadata__.get("tags", [])
                if "chat" in tags:
                    chat_flows.append(name)
        
        assert len(chat_flows) == 2
        assert "flow1" in chat_flows
        assert "flow3" in chat_flows
    
    def test_error_recovery_scenarios(self):
        """Test various error recovery scenarios."""
        # Test registering flow that fails metadata creation
        class PartiallyBrokenFlow:
            def __init__(self):
                self.name = "broken"
                # Missing required attributes
            
            def get_pipeline_method(self):
                return None  # Will cause metadata creation to fail
        
        # Should not crash
        self.registry.register_flow("broken", PartiallyBrokenFlow())
        assert "broken" in self.registry._flows
        assert self.registry.get_flow_metadata("broken") is None
        
        # Test retrieving non-existent flow
        assert self.registry.get_flow("non_existent") is None
        assert self.registry.get_flow_metadata("non_existent") is None
        
        # Test clearing and re-registering
        self.registry.clear()
        assert len(self.registry.get_flows()) == 0
        
        # Re-register after clear
        flow = MockFlow("new_flow")
        self.registry.register_flow("new_flow", flow)
        assert "new_flow" in self.registry.get_flows()