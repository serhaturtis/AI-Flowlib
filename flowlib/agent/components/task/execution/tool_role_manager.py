"""Tool Role Manager for agent role to tool access mapping.

Simple string-based role system with tool categories for flexible access control.
"""

import logging
from typing import Dict, List, Optional
from .models import AGENT_ROLES, TOOL_CATEGORIES, ROLE_TOOL_ACCESS, ToolMetadata
from .registry import tool_registry

logger = logging.getLogger(__name__)


class ToolPermissionError(Exception):
    """Raised when an agent lacks permission to use a tool."""
    pass


class ToolRoleManager:
    """Simple tool access manager based on agent roles and tool categories.

    Maps agent roles (strings) to tool categories they can access.
    Much simpler than the previous enum/security level system.
    """

    def __init__(self) -> None:
        """Initialize tool role manager."""
        pass

    def get_allowed_tools(self, agent_role: Optional[str]) -> List[str]:
        """Get list of tools allowed for an agent role.

        Args:
            agent_role: Agent role string (e.g., "software_engineer")

        Returns:
            List of tool names the agent role can access
        """
        if not agent_role:
            agent_role = "general_purpose"

        # Get tool categories this role can access
        allowed_categories = ROLE_TOOL_ACCESS.get(agent_role, ["generic"])

        allowed_tools = []
        all_tools = tool_registry.list()

        for tool_name in all_tools:
            try:
                metadata = tool_registry.get_metadata(tool_name)
                if metadata and self._is_tool_allowed_for_role(agent_role, metadata, allowed_categories):
                    allowed_tools.append(tool_name)
            except KeyError:
                continue

        return allowed_tools

    def validate_tool_access(self, agent_role: Optional[str], tool_name: str) -> bool:
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
            agent_role = "general_purpose"

        try:
            metadata = tool_registry.get_metadata(tool_name)
            if not metadata:
                raise ToolPermissionError(f"No metadata found for tool '{tool_name}'")

            allowed_categories = ROLE_TOOL_ACCESS.get(agent_role, ["generic"])

            if self._is_tool_allowed_for_role(agent_role, metadata, allowed_categories):
                return True
            else:
                raise ToolPermissionError(
                    f"Agent role '{agent_role}' cannot access tool '{tool_name}' "
                    f"(tool category: {metadata.tool_category}, role access: {allowed_categories})"
                )

        except KeyError:
            raise ToolPermissionError(f"Tool '{tool_name}' not found in registry")

    def _is_tool_allowed_for_role(self, agent_role: str, metadata: ToolMetadata, allowed_categories: List[str]) -> bool:
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

    def get_role_capabilities(self, agent_role: str) -> Dict[str, List[str]]:
        """Get capabilities for an agent role.

        Args:
            agent_role: Agent role string

        Returns:
            Dictionary with role info and allowed tool categories
        """
        if agent_role not in AGENT_ROLES:
            agent_role = "general_purpose"

        return {
            "role": [agent_role],
            "description": [AGENT_ROLES.get(agent_role, "Unknown role")],
            "allowed_categories": ROLE_TOOL_ACCESS.get(agent_role, ["generic"]),
            "allowed_tools": self.get_allowed_tools(agent_role)
        }

    def list_all_roles(self) -> Dict[str, str]:
        """List all available agent roles.

        Returns:
            Dictionary mapping role names to descriptions
        """
        return AGENT_ROLES.copy()

    def list_all_categories(self) -> Dict[str, str]:
        """List all available tool categories.

        Returns:
            Dictionary mapping category names to descriptions
        """
        return TOOL_CATEGORIES.copy()

    def get_tools_by_category(self, category: str) -> List[str]:
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


# Global tool role manager instance
tool_role_manager = ToolRoleManager()