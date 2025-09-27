"""
Role Manager for dynamic role assignment and management.

This module provides a high-level interface for managing role assignments
between semantic role names and canonical configuration names.
"""

import logging
from typing import Dict, List, Optional
from flowlib.resources.registry.registry import resource_registry

logger = logging.getLogger(__name__)


class RoleManager:
    """Manages dynamic role assignments for configurations.
    
    This class provides a high-level interface for:
    - Assigning roles to existing configurations
    - Reassigning roles to different configurations
    - Removing role assignments
    - Querying role assignments
    
    All operations work through the resource registry's alias system.
    """
    
    def __init__(self) -> None:
        """Initialize the role manager."""
        # We don't need separate storage - the resource registry handles everything
        # This manager is just a high-level interface
        pass
    
    def assign_role(self, role_name: str, canonical_name: str) -> bool:
        """Assign a role to an existing configuration.
        
        Args:
            role_name: Role name to assign (e.g., "knowledge-extraction")
            canonical_name: Existing canonical configuration name
            
        Returns:
            True if role was assigned successfully, False if canonical config doesn't exist
            
        Raises:
            ValueError: If role already exists and points to a different config
            
        Example:
            # User creates "my-phi4-model" config
            # GUI calls: role_manager.assign_role("knowledge-extraction", "my-phi4-model")
        """
        if not resource_registry.contains(canonical_name):
            logger.warning(f"Cannot assign role '{role_name}' - canonical config '{canonical_name}' does not exist")
            return False
        
        # Check if role already exists
        try:
            existing_config = resource_registry.get(role_name)
            if existing_config.name != canonical_name:
                raise ValueError(
                    f"Role '{role_name}' already assigned to '{existing_config.name}'. "
                    f"Use reassign_role() to change assignment."
                )
            # Role already points to this config - nothing to do
            logger.debug(f"Role '{role_name}' already assigned to '{canonical_name}'")
            return True
        except KeyError:
            # Role doesn't exist - create it
            pass
        
        # Create the role alias
        success = resource_registry.create_alias(role_name, canonical_name)
        if success:
            logger.info(f"Assigned role '{role_name}' to configuration '{canonical_name}'")
        else:
            logger.error(f"Failed to assign role '{role_name}' to '{canonical_name}'")
        
        return success
    
    def reassign_role(self, role_name: str, new_canonical_name: str) -> bool:
        """Reassign an existing role to a different configuration.
        
        Args:
            role_name: Role name to reassign
            new_canonical_name: New canonical configuration name
            
        Returns:
            True if role was reassigned successfully
            
        Example:
            # User wants to swap knowledge-extraction from phi4 to claude
            role_manager.reassign_role("knowledge-extraction", "claude-config")
        """
        if not resource_registry.contains(new_canonical_name):
            logger.warning(f"Cannot reassign role '{role_name}' - target config '{new_canonical_name}' does not exist")
            return False
        
        # Remove existing role assignment
        old_canonical_name = None
        try:
            old_config = resource_registry.get(role_name)
            old_canonical_name = old_config.name
            resource_registry.remove_alias(role_name)
            logger.debug(f"Removed old assignment: '{role_name}' -> '{old_canonical_name}'")
        except KeyError:
            # Role doesn't exist - that's fine
            pass
        
        # Create new assignment
        success = resource_registry.create_alias(role_name, new_canonical_name)
        if success:
            if old_canonical_name:
                logger.info(f"Reassigned role '{role_name}' from '{old_canonical_name}' to '{new_canonical_name}'")
            else:
                logger.info(f"Assigned new role '{role_name}' to '{new_canonical_name}'")
        else:
            logger.error(f"Failed to reassign role '{role_name}' to '{new_canonical_name}'")
        
        return success
    
    def unassign_role(self, role_name: str) -> bool:
        """Remove a role assignment.
        
        Args:
            role_name: Role name to remove
            
        Returns:
            True if role was removed, False if it didn't exist
            
        Example:
            role_manager.unassign_role("old-experiment-llm")
        """
        success = resource_registry.remove_alias(role_name)
        if success:
            logger.info(f"Removed role assignment '{role_name}'")
        else:
            logger.debug(f"Role '{role_name}' was not assigned")
        
        return success
    
    def get_role_assignment(self, role_name: str) -> Optional[str]:
        """Get the canonical name assigned to a role.
        
        Args:
            role_name: Role name to query
            
        Returns:
            Canonical configuration name, or None if role is not assigned
        """
        try:
            config = resource_registry.get(role_name)
            return config.name
        except KeyError:
            return None
    
    def get_canonical_roles(self, canonical_name: str) -> List[str]:
        """Get all roles assigned to a canonical configuration.
        
        Args:
            canonical_name: Canonical configuration name
            
        Returns:
            List of role names assigned to this configuration
        """
        return resource_registry.list_aliases(canonical_name)
    
    def list_all_roles(self) -> Dict[str, str]:
        """List all active role assignments.
        
        Returns:
            Dictionary mapping role names to canonical names
        """
        roles = {}
        
        # Get all configurations
        try:
            # This is a bit hacky but we need to iterate through all configs
            # to find their aliases
            for resource_type in resource_registry.list_types():
                configs_of_type = resource_registry.get_by_type(resource_type)
                for config_name, config in configs_of_type.items():
                    aliases = resource_registry.list_aliases(config_name)
                    for alias in aliases:
                        roles[alias] = config_name
        except Exception as e:
            logger.warning(f"Error listing roles: {e}")
        
        return roles
    
    def validate_role_assignments(self) -> List[str]:
        """Validate all role assignments and return any issues found.
        
        Returns:
            List of validation error messages (empty if all valid)
        """
        issues = []
        
        try:
            role_assignments = self.list_all_roles()
            
            for role_name, canonical_name in role_assignments.items():
                # Check that canonical config still exists
                if not resource_registry.contains(canonical_name):
                    issues.append(f"Role '{role_name}' points to missing config '{canonical_name}'")
                
                # Check that role actually resolves to the canonical config
                try:
                    resolved_config = resource_registry.get(role_name)
                    if resolved_config.name != canonical_name:
                        issues.append(f"Role '{role_name}' resolves to '{resolved_config.name}' but should be '{canonical_name}'")
                except KeyError:
                    issues.append(f"Role '{role_name}' cannot be resolved")
                    
        except Exception as e:
            issues.append(f"Validation failed with error: {e}")
        
        return issues
    
    def get_stats(self) -> Dict[str, float]:
        """Get role assignment statistics.
        
        Returns:
            Dictionary with role assignment statistics
        """
        try:
            role_assignments = self.list_all_roles()
            canonical_configs = set(role_assignments.values())
            
            return {
                'total_roles': len(role_assignments),
                'unique_configs': len(canonical_configs),
                'avg_roles_per_config': len(role_assignments) / max(len(canonical_configs), 1)
            }
        except Exception as e:
            logger.warning(f"Error getting stats: {e}")
            return {'total_roles': 0, 'unique_configs': 0, 'avg_roles_per_config': 0.0}


# Global role manager instance
role_manager = RoleManager()