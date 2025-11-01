"""Flow registry for tracking and accessing flows.

This module provides a registry for flows defined with the @flow decorator,
enabling easy access to flow instances and metadata.
"""

import builtins
import logging
from typing import Any, Union, cast

from flowlib.core.registry.registry import BaseRegistry
from flowlib.flows.base.base import Flow
from flowlib.flows.models.metadata import FlowMetadata

logger = logging.getLogger(__name__)


class FlowRegistry(BaseRegistry[Flow]):
    """
    Registry for flows.

    This registry stores information about flows, including their input/output models,
    metadata, and instances. The stage system has been removed in favor of subflow composition.
    """

    def __init__(self) -> None:
        """Initialize the stage registry with empty collections."""
        # Dictionary mapping flow names to flow info
        self._flows: dict[str, dict[str, Any]] = {}

        # Store flow instances
        self._flow_instances: dict[str, object] = {}

        # Store flow metadata
        self._flow_metadata: dict[str, FlowMetadata] = {}

    def register_flow(
        self, flow_name: str, flow_class_or_instance: Union[type["Flow"], "Flow"] | None = None
    ) -> None:
        """Register a flow with the registry.

        Args:
            flow_name: Flow name
            flow_class_or_instance: Flow class or instance (optional)
        """
        if flow_name in self._flows:
            logger.debug(f"Flow '{flow_name}' already exists in registry. Skipping...")
            return

        if flow_class_or_instance is None:
            # Register flow name only
            self._flows[flow_name] = {"name": flow_name, "metadata": {"is_infrastructure": False}}
            logger.debug(f"Registered flow name: {flow_name}")
        else:
            # Determine if this is a class or instance
            if isinstance(flow_class_or_instance, type):
                flow_class = flow_class_or_instance
                try:
                    flow_instance = flow_class(name_or_instance=flow_name)
                except Exception as e:
                    logger.warning(f"Could not create instance of flow '{flow_name}': {e}")
                    flow_instance = None
            else:
                flow_instance = flow_class_or_instance
                flow_class = flow_instance.__class__

            # Get metadata from the instance if possible
            metadata = {}
            if flow_instance is not None and hasattr(flow_instance, "__flow_metadata__"):
                metadata = flow_instance.__flow_metadata__
            elif hasattr(flow_class, "__flow_metadata__"):
                metadata = flow_class.__flow_metadata__

            # Register the flow
            self._flows[flow_name] = {"name": flow_name, "metadata": metadata}
            self._flow_instances[flow_name] = flow_instance

            # Create and store FlowMetadata for the flow
            if flow_instance:
                try:
                    from flowlib.flows.models.metadata import FlowMetadata

                    flow_metadata = FlowMetadata.from_flow(flow_instance, flow_name)
                    self._flow_metadata[flow_name] = flow_metadata
                    logger.debug(f"Created and stored metadata for flow: {flow_name}")
                except Exception as e:
                    logger.warning(f"Failed to create metadata for flow '{flow_name}': {e}")

            logger.debug(f"Registered flow: {flow_name}")

    # BaseRegistry interface implementation
    def register(self, name: str, obj: object, **metadata: object) -> None:
        """Register an object with the registry (BaseRegistry interface).

        Args:
            name: Unique name for the flow
            obj: The flow object to register
            **metadata: Additional metadata about the flow
        """
        self.register_flow(name, cast(type[Flow] | Flow | None, obj))

    def get(self, name: str, expected_type: type[Any] | None = None) -> Flow[Any]:
        """Get a flow by name with optional type checking (BaseRegistry interface).

        Args:
            name: Name of the flow to retrieve
            expected_type: Optional type for type checking

        Returns:
            The registered flow

        Raises:
            KeyError: If the flow doesn't exist
            TypeError: If the flow doesn't match the expected type
        """
        flow = self.get_flow(name)
        if flow is None:
            raise KeyError(f"Flow '{name}' not found in registry")

        if expected_type is not None and not isinstance(flow, expected_type):
            raise TypeError(f"Flow '{name}' is not of expected type {expected_type}")

        return cast(Flow[Any], flow)

    def contains(self, name: str) -> bool:
        """Check if a flow exists in the registry (BaseRegistry interface).

        Args:
            name: Name to check

        Returns:
            True if the flow exists, False otherwise
        """
        return self.contains_flow(name)

    def list(self, filter_criteria: dict[str, Any] | None = None) -> list[str]:
        """List registered flows matching criteria (BaseRegistry interface).

        Args:
            filter_criteria: Optional criteria to filter results

        Returns:
            List of flow names matching the criteria
        """
        flows = self.get_flows()

        if filter_criteria is None:
            return flows

        # Apply filtering logic
        filtered_flows = []
        for flow_name in flows:
            if self._matches_criteria(flow_name, filter_criteria):
                filtered_flows.append(flow_name)

        return filtered_flows

    def _matches_criteria(self, flow_name: str, criteria: dict[str, Any]) -> bool:
        """Check if a flow matches the given criteria.

        Args:
            flow_name: Name of the flow to check
            criteria: Criteria to match against

        Returns:
            True if the flow matches all criteria
        """
        if flow_name not in self._flows:
            return False

        flow_info = self._flows[flow_name]
        flow_metadata = flow_info["metadata"] if "metadata" in flow_info else {}

        for key, value in criteria.items():
            if key == "agent_selectable":
                # Special criteria for agent-selectable flows
                is_infrastructure = (
                    flow_metadata["is_infrastructure"]
                    if "is_infrastructure" in flow_metadata
                    else False
                )
                if value and is_infrastructure:
                    return False
            elif key in flow_metadata:
                if flow_metadata[key] != value:
                    return False
            else:
                return False

        return True

    # Stage registration removed - use subflow composition instead

    # Stage retrieval removed - use subflow composition instead

    # Stage listing removed - use subflow composition instead

    def contains_flow(self, flow_name: str) -> bool:
        """
        Check if a flow is registered.

        Args:
            flow_name: Name of the flow to check

        Returns:
            bool: True if the flow is registered, False otherwise
        """
        return flow_name in self._flows

    def get_flows(self) -> builtins.list[str]:
        """
        Get all registered flow names.

        Returns:
            List[str]: The list of flow names.
        """
        return sorted(self._flows.keys())

    def get_flow_metadata(self, flow_name: str) -> FlowMetadata | None:
        """
        Get flow metadata by name.

        Args:
            flow_name: Name of the flow

        Returns:
            Flow metadata or None if not found
        """
        return self._flow_metadata[flow_name] if flow_name in self._flow_metadata else None

    def get_all_flow_metadata(self) -> dict[str, FlowMetadata]:
        """
        Get metadata for all registered flows.

        Returns:
            Dictionary mapping flow names to their metadata
        """
        return self._flow_metadata.copy()

    def get_flow(self, flow_name: str) -> Any | None:
        """
        Get a flow instance by name.

        Args:
            flow_name: Name of the flow to retrieve.

        Returns:
            The flow instance if found, None otherwise.
        """
        if flow_name in self._flow_instances:
            flow_instance = self._flow_instances[flow_name]
            logger.debug(f"Retrieved flow instance for: {flow_name}")
            return flow_instance
        else:
            logger.debug(f"No flow instance found for: {flow_name}")
            return None

    def get_flow_instances(self) -> dict[str, object]:
        """
        Get all registered flow instances.

        Returns:
            Dict[str, Any]: Dictionary mapping flow names to flow instances.
        """
        return self._flow_instances.copy()

    def get_agent_selectable_flows(self) -> dict[str, object]:
        """
        Get flow instances that can be selected by the agent.

        This method filters out infrastructure flows that shouldn't be directly
        selectable by the agent during planning.

        Returns:
            Dict[str, Any]: Dictionary mapping flow names to flow instances where is_infrastructure=False.
        """
        selectable_flows = {}
        for flow_name, flow_instance in self._flow_instances.items():
            # Check if the flow_instance has is_infrastructure attribute or metadata
            is_infrastructure = False

            # Check the direct attribute first
            if hasattr(flow_instance, "is_infrastructure"):
                is_infrastructure = flow_instance.is_infrastructure
            # Then check in flow metadata if available
            elif hasattr(flow_instance, "__flow_metadata__"):
                metadata = flow_instance.__flow_metadata__
                # Fail-fast approach - explicit check for infrastructure flag
                is_infrastructure = (
                    metadata["is_infrastructure"] if "is_infrastructure" in metadata else False
                )

            # Add to result if not an infrastructure flow
            if not is_infrastructure:
                selectable_flows[flow_name] = flow_instance

        logger.debug(f"Found {len(selectable_flows)} agent-selectable flows")
        return selectable_flows

    def clear(self) -> None:
        """Clear all registered flows."""
        self._flows.clear()
        self._flow_instances.clear()
        self._flow_metadata.clear()
        logger.debug("Cleared flow registry")

    def remove(self, name: str) -> bool:
        """Remove a specific flow registration from the registry.

        Args:
            name: Name of the flow to remove

        Returns:
            True if the flow was found and removed, False if not found
        """
        removed = False

        if name in self._flows:
            del self._flows[name]
            removed = True

        if name in self._flow_instances:
            del self._flow_instances[name]
            removed = True

        if name in self._flow_metadata:
            del self._flow_metadata[name]
            removed = True

        if removed:
            logger.debug(f"Removed flow '{name}' from registry")

        return removed

    def update(self, name: str, obj: object, **metadata: object) -> bool:
        """Update or replace an existing flow registration.

        Args:
            name: Name of the flow to update
            obj: New flow object to register
            **metadata: Additional metadata about the flow

        Returns:
            True if an existing flow was updated, False if this was a new registration
        """
        existing_found = self.contains(name)

        if existing_found:
            # Remove existing
            self.remove(name)

            # Re-register
            self.register(name, obj, **metadata)
            logger.debug(f"Updated existing flow '{name}' in registry")
            return True
        else:
            # New registration
            self.register(name, obj, **metadata)
            logger.debug(f"Registered new flow '{name}' in registry")
            return False


# Global flow registry instance
flow_registry = FlowRegistry()

# Backward compatibility alias
stage_registry = flow_registry
