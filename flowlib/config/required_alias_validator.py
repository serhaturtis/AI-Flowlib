"""
Required Alias Validation System.

This module provides validation functionality to ensure all required aliases
are properly configured before agent execution. Validates both presence and
correctness of alias bindings.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from flowlib.config.alias_manager import alias_manager
from flowlib.config.required_resources import (
    REQUIRED_RESOURCE_SPECS,
    RequiredAlias,
    get_always_required,
)
from flowlib.resources.registry.registry import resource_registry

if TYPE_CHECKING:
    from flowlib.agent.models.agent_config import AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of alias validation."""

    valid: bool
    missing_always: list[RequiredAlias]
    missing_conditional: list[RequiredAlias]
    invalid_bindings: dict[RequiredAlias, str]  # Alias -> error message
    warnings: list[str]

    def get_error_message(self) -> str:
        """Get human-readable error message."""
        if self.valid and not self.warnings:
            return "All required aliases are properly configured."

        errors = []

        if self.missing_always:
            errors.append("Missing REQUIRED aliases:")
            for alias in self.missing_always:
                spec = REQUIRED_RESOURCE_SPECS[alias]
                errors.append(f"  - {alias.value}")
                errors.append(f"    Type: {spec.resource_type.value}")
                errors.append(f"    Needed by: {', '.join(spec.required_by)}")
                errors.append(f"    Hint: {spec.validation_hints}")

        if self.invalid_bindings:
            if errors:
                errors.append("")
            errors.append("Invalid alias bindings:")
            for alias, error in self.invalid_bindings.items():
                errors.append(f"  - {alias.value}: {error}")

        if self.warnings:
            if errors:
                errors.append("")
            errors.append("Warnings:")
            for warning in self.warnings:
                errors.append(f"  - {warning}")

        return "\n".join(errors)


class RequiredAliasValidator:
    """Validates that all required aliases are properly configured."""

    @staticmethod
    def validate_project(
        required_aliases: list[RequiredAlias] | None = None,
    ) -> ValidationResult:
        """Validate project has all required aliases.

        Args:
            required_aliases: Specific aliases to validate. If None, validates all always-required.

        Returns:
            ValidationResult with details of missing/invalid aliases.
        """
        if required_aliases is None:
            required_aliases = get_always_required()

        missing_always = []
        missing_conditional = []
        invalid_bindings = {}
        warnings = []

        for alias in required_aliases:
            spec = REQUIRED_RESOURCE_SPECS.get(alias)
            if not spec:
                warnings.append(f"Unknown required alias: {alias.value}")
                continue

            # Check if alias exists
            target = alias_manager.get_alias_target(alias.value)
            if target is None:
                if spec.required_always:
                    missing_always.append(alias)
                else:
                    missing_conditional.append(alias)
                continue

            # Validate alias target exists and has correct type
            try:
                if not resource_registry.contains(target):
                    invalid_bindings[alias] = f"Target '{target}' does not exist"
                    continue

                # Validate resource type matches expectation
                resource = resource_registry.get(target)
                if hasattr(resource, "resource_type"):
                    actual_type = resource.resource_type
                    # Convert enum to string if needed
                    expected_type = spec.resource_type.value
                    actual_type_str = actual_type.value if hasattr(actual_type, "value") else str(actual_type)

                    if actual_type_str != expected_type:
                        warnings.append(
                            f"{alias.value} → {target}: Expected type {expected_type}, "
                            f"got {actual_type_str}"
                        )

            except Exception as e:
                invalid_bindings[alias] = str(e)

        valid = len(missing_always) == 0 and len(invalid_bindings) == 0

        return ValidationResult(
            valid=valid,
            missing_always=missing_always,
            missing_conditional=missing_conditional,
            invalid_bindings=invalid_bindings,
            warnings=warnings,
        )

    @staticmethod
    def validate_agent_config(agent_config: "AgentConfig") -> ValidationResult:
        """Validate agent configuration has all required aliases.

        Args:
            agent_config: Agent configuration to validate.

        Returns:
            ValidationResult with details of missing/invalid aliases.
        """
        required = agent_config.get_all_required_aliases()
        return RequiredAliasValidator.validate_project(required)
