"""
Alias Manager for canonical configuration indirection.

This module provides the high-level interface for managing strict alias bindings
between semantic role names (e.g., RequiredAlias.DEFAULT_LLM.value) and canonical configuration names.
All operations fail fast and keep the resource registry as the single source of truth.
"""

import logging

from flowlib.resources.registry.registry import resource_registry
from flowlib.config.required_resources import RequiredAlias

logger = logging.getLogger(__name__)


class AliasManager:
    """Manage strict alias assignments for configuration resources."""

    def assign_alias(self, alias_name: str, canonical_name: str) -> None:
        """Bind an alias to a canonical configuration. Fails fast if invalid."""
        if not resource_registry.contains(canonical_name):
            raise ValueError(
                f"Cannot assign alias '{alias_name}' -> '{canonical_name}': canonical config not found"
            )

        existing_target = self.get_alias_target(alias_name)
        if existing_target:
            if existing_target != canonical_name:
                raise ValueError(
                    f"Alias '{alias_name}' already points to '{existing_target}' "
                    f"and cannot be rebound without explicit removal"
                )
            logger.debug(f"Alias '{alias_name}' already bound to '{canonical_name}', skipping")
            return

        success = resource_registry.create_alias(alias_name, canonical_name)
        if not success:
            raise ValueError(
                f"Failed to assign alias '{alias_name}' -> '{canonical_name}'. "
                "Resource registry rejected the binding."
            )

        logger.info(f"Alias '{alias_name}' assigned to canonical config '{canonical_name}'")

    def reassign_alias(self, alias_name: str, canonical_name: str) -> None:
        """Rebind an alias to a different canonical configuration."""
        if not resource_registry.contains(canonical_name):
            raise ValueError(
                f"Cannot reassign alias '{alias_name}' -> '{canonical_name}': canonical config not found"
            )

        resource_registry.remove_alias(alias_name)
        success = resource_registry.create_alias(alias_name, canonical_name)
        if not success:
            raise ValueError(
                f"Failed to reassign alias '{alias_name}' -> '{canonical_name}'. "
                "Resource registry rejected the binding."
            )

        logger.info(f"Alias '{alias_name}' reassigned to canonical config '{canonical_name}'")

    def remove_alias(self, alias_name: str) -> None:
        """Remove an alias binding."""
        removed = resource_registry.remove_alias(alias_name)
        if not removed:
            raise KeyError(f"Alias '{alias_name}' is not defined")
        logger.info(f"Alias '{alias_name}' removed")

    def get_alias_target(self, alias_name: str) -> str | None:
        """Return the canonical configuration name for an alias."""
        try:
            config = resource_registry.get(alias_name)
            return config.name
        except KeyError:
            return None

    def list_aliases_for(self, canonical_name: str) -> list[str]:
        """List every alias bound to a canonical configuration."""
        return resource_registry.list_aliases(canonical_name)

    def list_all_aliases(self) -> dict[str, str]:
        """Return {alias: canonical} for every registered alias."""
        aliases: dict[str, str] = {}

        for resource_type in resource_registry.list_types():
            configs_of_type = resource_registry.get_by_type(resource_type)
            for config_name, _ in configs_of_type.items():
                for alias in resource_registry.list_aliases(config_name):
                    aliases[alias] = config_name

        return aliases

    def validate_aliases(self) -> list[str]:
        """Validate that every alias resolves to an existing canonical configuration."""
        issues: list[str] = []
        for alias, canonical in self.list_all_aliases().items():
            if not resource_registry.contains(canonical):
                issues.append(f"Alias '{alias}' targets missing config '{canonical}'")
                continue

            resolved = self.get_alias_target(alias)
            if resolved != canonical:
                issues.append(
                    f"Alias '{alias}' resolves to '{resolved}' but expected '{canonical}'"
                )

        return issues

    def get_stats(self) -> dict[str, float]:
        """Return alias statistics for observability."""
        aliases = self.list_all_aliases()
        canonical_configs = set(aliases.values())
        total_aliases = len(aliases)
        unique_configs = len(canonical_configs)

        return {
            "total_aliases": total_aliases,
            "unique_configs": unique_configs,
            "avg_aliases_per_config": total_aliases / max(unique_configs, 1),
        }


# Global alias manager instance
alias_manager = AliasManager()
