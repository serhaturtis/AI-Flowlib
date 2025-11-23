"""Generate aliases.py file content for projects."""

from __future__ import annotations


class AliasFileGenerator:
    """Generate aliases.py file content based on setup type."""

    @staticmethod
    def generate_empty_aliases(project_name: str) -> str:
        """Generate aliases.py for empty projects with commented examples."""
        return f'''"""Alias bindings for {project_name}."""

from flowlib.config.alias_manager import alias_manager
from flowlib.config.required_resources import RequiredAlias

# TODO: Create provider and resource configs first, then uncomment and update the aliases below.
#
# Example alias assignments:
# alias_manager.assign_alias(RequiredAlias.DEFAULT_LLM.value, "your-llm-provider-name")
# alias_manager.assign_alias(RequiredAlias.DEFAULT_MODEL.value, "your-model-config-name")
# alias_manager.assign_alias(RequiredAlias.DEFAULT_EMBEDDING.value, "your-embedding-provider-name")
# alias_manager.assign_alias(
#     RequiredAlias.DEFAULT_EMBEDDING_MODEL.value, "your-embedding-model-name"
# )
# alias_manager.assign_alias(RequiredAlias.DEFAULT_VECTOR_DB.value, "your-vector-db-name")
# alias_manager.assign_alias(RequiredAlias.DEFAULT_GRAPH_DB.value, "your-graph-db-name")
'''

    @staticmethod
    def generate_configured_aliases(project_name: str, aliases: dict[str, str]) -> str:
        """Generate aliases.py with actual alias assignments.

        Args:
            project_name: Project name for docstring
            aliases: Dictionary mapping alias names to canonical config names

        Returns:
            Complete aliases.py file content
        """
        # Generate alias assignments
        assignments = []
        for alias_name, config_name in sorted(aliases.items()):
            # Check if it's a RequiredAlias enum value
            if alias_name.startswith("default-"):
                # Convert default-llm -> DEFAULT_LLM
                enum_name = alias_name.replace("-", "_").upper()
                assignments.append(
                    f'alias_manager.assign_alias(RequiredAlias.{enum_name}.value, "{config_name}")'
                )
            else:
                # Custom alias, use string directly
                assignments.append(f'alias_manager.assign_alias("{alias_name}", "{config_name}")')

        assignments_str = "\n".join(assignments)

        return f'''"""Alias bindings for {project_name}."""

from flowlib.config.alias_manager import alias_manager
from flowlib.config.required_resources import RequiredAlias

# Alias assignments generated during project setup
{assignments_str}
'''
