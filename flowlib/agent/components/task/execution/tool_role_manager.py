"""Tool Role Manager for agent role to tool access mapping.

Simple string-based role system with tool categories for flexible access control.
"""

import logging

from flowlib.resources.models.constants import ResourceType
from flowlib.resources.models.role_config_resource import (
    RoleConfigResource,
    ToolCategoryConfigResource,
)

from .models import ToolMetadata
from .registry import tool_registry

logger = logging.getLogger(__name__)


class ToolPermissionError(Exception):
    """Raised when an agent lacks permission to use a tool."""

    pass


class ToolRoleManager:
    """Dynamic tool access manager based on resource registry.

    Uses RoleConfigResource and ToolCategoryConfigResource for flexible configuration.
    No hardcoded mappings - all configuration comes from the resource registry.
    """

    def __init__(self) -> None:
        """Initialize tool role manager."""
        self._resource_registry = None

    def _get_resource_registry(self):
        """Lazy import of resource registry to avoid circular dependencies."""
        if self._resource_registry is None:
            from flowlib.resources.registry.registry import resource_registry

            self._resource_registry = resource_registry
        return self._resource_registry

    def get_allowed_tools(self, agent_role: str | None) -> list[str]:
        """Get list of tools allowed for an agent role.

        Args:
            agent_role: Agent role string (e.g., "software_engineer")

        Returns:
            List of tool names the agent role can access
        """
        if not agent_role:
            raise ValueError("Agent role must be specified")

        # Get tool categories this role can access from resource registry
        allowed_categories = self._get_role_tool_categories(agent_role)

        allowed_tools = []
        all_tools = tool_registry.list()

        for tool_name in all_tools:
            try:
                metadata = tool_registry.get_metadata(tool_name)
                if metadata and self._is_tool_allowed_for_role(
                    agent_role, metadata, allowed_categories
                ):
                    allowed_tools.append(tool_name)
            except KeyError:
                continue

        return allowed_tools

    def validate_tool_access(self, agent_role: str | None, tool_name: str) -> bool:
        """Validate if an agent role can access a specific tool.

        Args:
            agent_role: Agent role string to check
            tool_name: Name of the tool to check access for

        Returns:
            True if access is allowed

        Raises:
            ToolPermissionError: If access is denied
        """
        if not agent_role:
            raise ValueError("Agent role must be specified")

        try:
            metadata = tool_registry.get_metadata(tool_name)
            if not metadata:
                raise ToolPermissionError(f"No metadata found for tool '{tool_name}'")

            allowed_categories = self._get_role_tool_categories(agent_role)

            if self._is_tool_allowed_for_role(agent_role, metadata, allowed_categories):
                return True
            else:
                raise ToolPermissionError(
                    f"Agent role '{agent_role}' cannot access tool '{tool_name}' "
                    f"(tool category: {metadata.tool_category}, role access: {allowed_categories})"
                )

        except KeyError:
            raise ToolPermissionError(f"Tool '{tool_name}' not found in registry") from None

    def _is_tool_allowed_for_role(
        self, agent_role: str, metadata: ToolMetadata, allowed_categories: list[str]
    ) -> bool:
        """Check if a tool is allowed for an agent role.

        Args:
            agent_role: Agent role string
            metadata: Tool metadata
            allowed_categories: Categories this role can access

        Returns:
            True if the tool is allowed
        """
        # Check explicit denials first
        if agent_role in metadata.denied_roles:
            return False

        # Check explicit allowances
        if metadata.allowed_roles and agent_role in metadata.allowed_roles:
            return True

        # Default: check if tool category is allowed for this role
        return metadata.tool_category in allowed_categories

    def get_role_capabilities(self, agent_role: str) -> dict[str, list[str]]:
        """Get capabilities for an agent role.

        Args:
            agent_role: Agent role string

        Returns:
            Dictionary with role info and allowed tool categories
        """
        role_config = self._get_role_config(agent_role)
        if not role_config:
            raise ValueError(f"Role '{agent_role}' not found in resource registry")

        return {
            "role": [agent_role],
            "description": [role_config.description],
            "allowed_categories": role_config.tool_categories,
            "allowed_tools": self.get_allowed_tools(agent_role),
        }

    def list_all_roles(self) -> dict[str, str]:
        """List all available agent roles.

        Returns:
            Dictionary mapping role names to descriptions
        """
        registry = self._get_resource_registry()
        roles = {}
        resources_by_type = registry.get_by_type(ResourceType.ROLE_CONFIG)
        for _resource_name, role_config in resources_by_type.items():
            if isinstance(role_config, RoleConfigResource):
                roles[role_config.agent_role] = role_config.description
        return roles

    def list_all_categories(self) -> dict[str, str]:
        """List all available tool categories.

        Returns:
            Dictionary mapping category names to descriptions
        """
        registry = self._get_resource_registry()
        categories = {}
        resources_by_type = registry.get_by_type(ResourceType.TOOL_CATEGORY_CONFIG)
        for _resource_name, category_config in resources_by_type.items():
            if isinstance(category_config, ToolCategoryConfigResource):
                categories[category_config.category_name] = category_config.description
        return categories

    def get_tools_by_category(self, category: str) -> list[str]:
        """Get all tools in a specific category.

        Args:
            category: Tool category name

        Returns:
            List of tool names in that category
        """
        tools = []
        all_tools = tool_registry.list()

        for tool_name in all_tools:
            try:
                metadata = tool_registry.get_metadata(tool_name)
                if metadata and metadata.tool_category == category:
                    tools.append(tool_name)
            except KeyError:
                continue

        return tools

    def _get_role_config(self, agent_role: str) -> RoleConfigResource | None:
        """Get role configuration from resource registry.

        Args:
            agent_role: Agent role string

        Returns:
            RoleConfigResource instance or None if not found
        """
        registry = self._get_resource_registry()
        resources_by_type = registry.get_by_type(ResourceType.ROLE_CONFIG)
        for _resource_name, role_config in resources_by_type.items():
            if isinstance(role_config, RoleConfigResource) and role_config.agent_role == agent_role:
                return role_config
        return None

    def _get_role_tool_categories(self, agent_role: str) -> list[str]:
        """Get tool categories allowed for a role.

        Args:
            agent_role: Agent role string

        Returns:
            List of tool category names
        """
        role_config = self._get_role_config(agent_role)
        if not role_config:
            raise ValueError(f"Role '{agent_role}' not found in resource registry")
        return role_config.tool_categories


# Global tool role manager instance
tool_role_manager = ToolRoleManager()
