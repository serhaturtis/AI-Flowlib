"""Tool access manager for category-based permissions."""

from __future__ import annotations

from collections.abc import Iterable

from flowlib.agent.components.task.execution.registry import tool_registry


class ToolPermissionError(Exception):
    """Raised when an agent lacks permission to use a tool."""

    pass


class ToolAccessManager:
    """Category-based tool access enforcement."""

    def get_allowed_tools(self, allowed_categories: Iterable[str]) -> list[str]:
        """Return tool names permitted for the provided categories."""
        categories = list(allowed_categories)
        self._validate_categories(categories)

        allowed_tools: list[str] = []
        for tool_name in tool_registry.list():
            try:
                metadata = tool_registry.get_metadata(tool_name)
            except KeyError:
                continue

            if not metadata:
                continue

            if metadata.tool_category in categories:
                allowed_tools.append(tool_name)

        return allowed_tools

    def validate_tool_access(
        self,
        allowed_categories: Iterable[str],
        tool_name: str,
    ) -> bool:
        """Validate that the requested tool is available for the agent."""
        categories = list(allowed_categories)
        self._validate_categories(categories)

        try:
            metadata = tool_registry.get_metadata(tool_name)
        except KeyError as exc:
            raise ToolPermissionError(f"Tool '{tool_name}' not found in registry") from exc

        if not metadata:
            raise ToolPermissionError(f"Tool '{tool_name}' is missing metadata")

        if metadata.tool_category not in categories:
            raise ToolPermissionError(
                f"Tool '{tool_name}' not permitted (category {metadata.tool_category}, "
                f"allowed categories: {categories})"
            )

        return True

    def _validate_categories(self, categories: list[str]) -> None:
        if not categories:
            raise ValueError("Agent has no allowed tool categories configured")


# Global instance
tool_access_manager = ToolAccessManager()

