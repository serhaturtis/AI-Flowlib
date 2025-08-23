"""
Role Validation and Dependency Management System

Advanced role validation system with dependency checking, conflict resolution,
and policy enforcement following CLAUDE.md principles.
"""

import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pydantic import Field
from flowlib.core.models import StrictBaseModel

# Add compatibility for the validation issues return type
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RoleType(Enum):
    """Types of roles in the system."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    BACKUP = "backup"
    MONITORING = "monitoring"
    UTILITY = "utility"


@dataclass
class ValidationIssue:
    """Represents a validation issue with context."""
    severity: ValidationSeverity
    message: str
    provider_name: str
    role_name: str
    context: Dict[str, Any] = None
    suggested_fix: Optional[str] = None
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.provider_name}/{self.role_name}: {self.message}"


class RoleDefinition(StrictBaseModel):
    """Definition of a role with its constraints and dependencies."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., description="Role name")
    type: RoleType = Field(..., description="Role type")
    description: str = Field(..., description="Role description")
    
    # Dependencies and constraints
    required_dependencies: Set[str] = Field(default_factory=set, description="Required role dependencies")
    optional_dependencies: Set[str] = Field(default_factory=set, description="Optional role dependencies")
    conflicts_with: Set[str] = Field(default_factory=set, description="Roles that conflict with this one")
    mutually_exclusive: Set[str] = Field(default_factory=set, description="Mutually exclusive roles")
    
    # Provider constraints
    compatible_provider_types: Set[str] = Field(default_factory=set, description="Compatible provider types")
    incompatible_provider_types: Set[str] = Field(default_factory=set, description="Incompatible provider types")
    minimum_provider_version: Optional[str] = Field(default=None, description="Minimum provider version")
    
    # Resource requirements
    minimum_memory_mb: Optional[int] = Field(default=None, description="Minimum memory requirement")
    requires_gpu: bool = Field(default=False, description="Whether GPU is required")
    network_access_required: bool = Field(default=False, description="Whether network access is required")
    
    # Cardinality constraints
    max_instances_per_provider: int = Field(default=1, description="Maximum instances per provider")
    max_instances_global: Optional[int] = Field(default=None, description="Maximum instances across all providers")
    
    # Priority and ordering
    priority: int = Field(default=100, description="Role priority (lower = higher priority)")
    load_order: int = Field(default=100, description="Load order (lower = earlier)")


class ProviderCapabilities(StrictBaseModel):
    """Capabilities and constraints of a provider."""
    # Inherits strict configuration from StrictBaseModel
    
    name: str = Field(..., description="Provider name")
    type: str = Field(..., description="Provider type")
    version: str = Field(default="1.0.0", description="Provider version")
    
    # Resource capabilities
    available_memory_mb: Optional[int] = Field(default=None, description="Available memory")
    has_gpu: bool = Field(default=False, description="GPU availability")
    has_network_access: bool = Field(default=True, description="Network access availability")
    
    # Current state
    current_roles: Set[str] = Field(default_factory=set, description="Currently assigned roles")
    max_concurrent_roles: int = Field(default=10, description="Maximum concurrent roles")
    health_status: str = Field(default="healthy", description="Current health status")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")


class RoleValidationEngine:
    """Advanced role validation engine with comprehensive checking."""
    
    def __init__(self):
        self.role_definitions: Dict[str, RoleDefinition] = {}
        self.provider_capabilities: Dict[str, ProviderCapabilities] = {}
        self.validation_policies: Dict[str, Any] = {}
        self.load_default_configurations()
    
    def load_default_configurations(self):
        """Load default role definitions and validation policies."""
        # Default role definitions
        default_roles = [
            RoleDefinition(
                name="primary-llm",
                type=RoleType.PRIMARY,
                description="Primary language model provider",
                compatible_provider_types={"llm"},
                requires_gpu=True,
                minimum_memory_mb=4096,
                max_instances_global=1,
                priority=10,
                conflicts_with={"backup-llm"},
                network_access_required=True
            ),
            RoleDefinition(
                name="backup-llm", 
                type=RoleType.BACKUP,
                description="Backup language model provider",
                compatible_provider_types={"llm"},
                required_dependencies={"primary-llm"},
                minimum_memory_mb=2048,
                max_instances_global=2,
                priority=20,
                mutually_exclusive={"primary-llm"}
            ),
            RoleDefinition(
                name="vector-store",
                type=RoleType.PRIMARY,
                description="Primary vector storage provider",
                compatible_provider_types={"vector_db", "database"},
                minimum_memory_mb=1024,
                max_instances_global=1,
                priority=15,
                network_access_required=True
            ),
            RoleDefinition(
                name="cache-layer",
                type=RoleType.UTILITY,
                description="Caching layer provider",
                compatible_provider_types={"cache", "database"},
                minimum_memory_mb=512,
                max_instances_global=3,
                priority=30,
                optional_dependencies={"vector-store"}
            ),
            RoleDefinition(
                name="monitoring",
                type=RoleType.MONITORING,
                description="System monitoring provider",
                compatible_provider_types={"*"},  # Compatible with all types
                minimum_memory_mb=256,
                priority=50,
                network_access_required=True
            )
        ]
        
        for role_def in default_roles:
            self.role_definitions[role_def.name] = role_def
        
        # Default validation policies
        self.validation_policies = {
            "enforce_dependencies": True,
            "allow_conflicts": False,
            "require_primary_roles": True,
            "max_roles_per_provider": 5,
            "enable_resource_checking": True,
            "strict_compatibility": True
        }
    
    def register_role_definition(self, role_def: RoleDefinition):
        """Register a new role definition."""
        self.role_definitions[role_def.name] = role_def
        logger.info(f"Registered role definition: {role_def.name}")
    
    def register_provider_capabilities(self, provider_caps: ProviderCapabilities):
        """Register provider capabilities."""
        self.provider_capabilities[provider_caps.name] = provider_caps
        logger.info(f"Registered provider capabilities: {provider_caps.name}")
    
    def validate_role_assignment(
        self, 
        provider_name: str, 
        role_name: str, 
        current_assignments: Dict[str, Set[str]] = None
    ) -> List[ValidationIssue]:
        """
        Validate a single role assignment with comprehensive checking.
        
        Args:
            provider_name: Name of the provider
            role_name: Name of the role to assign
            current_assignments: Current role assignments {provider_name: {role_names}}
            
        Returns:
            List of validation issues
        """
        issues = []
        current_assignments = current_assignments or {}
        
        # Get role definition and provider capabilities
        role_def = self.role_definitions[role_name] if role_name in self.role_definitions else None
        provider_caps = self.provider_capabilities[provider_name] if provider_name in self.provider_capabilities else None
        
        if not role_def:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Unknown role definition: {role_name}",
                provider_name=provider_name,
                role_name=role_name,
                suggested_fix="Register the role definition first"
            ))
            return issues
        
        if not provider_caps:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Provider capabilities not registered: {provider_name}",
                provider_name=provider_name,
                role_name=role_name,
                suggested_fix="Register provider capabilities for better validation"
            ))
        
        # Check provider type compatibility
        if provider_caps:
            issues.extend(self._validate_provider_compatibility(role_def, provider_caps))
        
        # Check resource requirements
        if provider_caps and self.validation_policies["enable_resource_checking"]:
            issues.extend(self._validate_resource_requirements(role_def, provider_caps))
        
        # Check role cardinality constraints
        issues.extend(self._validate_cardinality_constraints(role_def, provider_name, current_assignments))
        
        # Check dependencies
        if self.validation_policies["enforce_dependencies"]:
            issues.extend(self._validate_dependencies(role_def, current_assignments))
        
        # Check conflicts
        if not self.validation_policies["allow_conflicts"]:
            issues.extend(self._validate_conflicts(role_def, provider_name, current_assignments))
        
        # Check mutual exclusivity
        issues.extend(self._validate_mutual_exclusivity(role_def, provider_name, current_assignments))
        
        # Check provider role limits - fail-fast approach
        if "max_roles_per_provider" not in self.validation_policies:
            raise ValueError("Validation policy 'max_roles_per_provider' is required")
        max_roles = self.validation_policies["max_roles_per_provider"]
        
        if provider_name not in current_assignments:
            raise ValueError(f"Provider '{provider_name}' not found in current assignments")
        current_roles = current_assignments[provider_name]
        if len(current_roles) >= max_roles:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Provider already has maximum roles ({max_roles})",
                provider_name=provider_name,
                role_name=role_name,
                suggested_fix="Remove existing roles or increase the limit"
            ))
        
        return issues
    
    def validate_complete_assignment(
        self, 
        assignments: Dict[str, Set[str]]
    ) -> List[ValidationIssue]:
        """
        Validate complete role assignment configuration.
        
        Args:
            assignments: Complete assignments {provider_name: {role_names}}
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Validate each individual assignment
        for provider_name, role_names in assignments.items():
            for role_name in role_names:
                issues.extend(self.validate_role_assignment(provider_name, role_name, assignments))
        
        # System-level validations
        issues.extend(self._validate_system_requirements(assignments))
        issues.extend(self._validate_global_constraints(assignments))
        issues.extend(self._validate_load_order(assignments))
        
        return issues
    
    def _validate_provider_compatibility(
        self, 
        role_def: RoleDefinition, 
        provider_caps: ProviderCapabilities
    ) -> List[ValidationIssue]:
        """Validate provider type compatibility."""
        issues = []
        
        # Check compatible types
        if role_def.compatible_provider_types:
            if "*" not in role_def.compatible_provider_types:
                if provider_caps.type not in role_def.compatible_provider_types:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Provider type '{provider_caps.type}' not compatible with role",
                        provider_name=provider_caps.name,
                        role_name=role_def.name,
                        context={"compatible_types": list(role_def.compatible_provider_types)},
                        suggested_fix=f"Use provider of type: {', '.join(role_def.compatible_provider_types)}"
                    ))
        
        # Check incompatible types
        if provider_caps.type in role_def.incompatible_provider_types:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Provider type '{provider_caps.type}' is explicitly incompatible",
                provider_name=provider_caps.name,
                role_name=role_def.name,
                suggested_fix="Use a different provider type"
            ))
        
        # Check version requirements
        if role_def.minimum_provider_version:
            # Simple version comparison (can be enhanced)
            if provider_caps.version < role_def.minimum_provider_version:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Provider version {provider_caps.version} < required {role_def.minimum_provider_version}",
                    provider_name=provider_caps.name,
                    role_name=role_def.name,
                    suggested_fix="Upgrade the provider version"
                ))
        
        return issues
    
    def _validate_resource_requirements(
        self, 
        role_def: RoleDefinition, 
        provider_caps: ProviderCapabilities
    ) -> List[ValidationIssue]:
        """Validate resource requirements."""
        issues = []
        
        # Check memory requirements
        if role_def.minimum_memory_mb and provider_caps.available_memory_mb:
            if provider_caps.available_memory_mb < role_def.minimum_memory_mb:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Insufficient memory: {provider_caps.available_memory_mb}MB < {role_def.minimum_memory_mb}MB required",
                    provider_name=provider_caps.name,
                    role_name=role_def.name,
                    suggested_fix="Increase provider memory allocation"
                ))
        
        # Check GPU requirements
        if role_def.requires_gpu and not provider_caps.has_gpu:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Role requires GPU but provider doesn't have GPU access",
                provider_name=provider_caps.name,
                role_name=role_def.name,
                suggested_fix="Use a GPU-enabled provider"
            ))
        
        # Check network requirements
        if role_def.network_access_required and not provider_caps.has_network_access:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Role requires network access but provider is offline",
                provider_name=provider_caps.name,
                role_name=role_def.name,
                suggested_fix="Enable network access for the provider"
            ))
        
        return issues
    
    def _validate_cardinality_constraints(
        self, 
        role_def: RoleDefinition, 
        provider_name: str,
        current_assignments: Dict[str, Set[str]]
    ) -> List[ValidationIssue]:
        """Validate role cardinality constraints."""
        issues = []
        
        # Check per-provider limits - fail-fast approach
        if provider_name not in current_assignments:
            raise ValueError(f"Provider '{provider_name}' not found in current assignments")
        current_roles = current_assignments[provider_name]
        role_count = sum(1 for role in current_roles if role == role_def.name)
        
        if role_count >= role_def.max_instances_per_provider:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Maximum instances per provider exceeded: {role_count}/{role_def.max_instances_per_provider}",
                provider_name=provider_name,
                role_name=role_def.name,
                suggested_fix="Remove existing instances or increase the limit"
            ))
        
        # Check global limits
        if role_def.max_instances_global:
            global_count = sum(
                sum(1 for role in roles if role == role_def.name)
                for roles in current_assignments.values()
            )
            
            if global_count >= role_def.max_instances_global:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Maximum global instances exceeded: {global_count}/{role_def.max_instances_global}",
                    provider_name=provider_name,
                    role_name=role_def.name,
                    suggested_fix="Remove existing instances from other providers"
                ))
        
        return issues
    
    def _validate_dependencies(
        self, 
        role_def: RoleDefinition, 
        current_assignments: Dict[str, Set[str]]
    ) -> List[ValidationIssue]:
        """Validate role dependencies."""
        issues = []
        
        # Get all currently assigned roles across all providers
        all_assigned_roles = set()
        for roles in current_assignments.values():
            all_assigned_roles.update(roles)
        
        # Check required dependencies
        for required_dep in role_def.required_dependencies:
            if required_dep not in all_assigned_roles:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Required dependency '{required_dep}' is not assigned to any provider",
                    provider_name="",  # System-level issue
                    role_name=role_def.name,
                    suggested_fix=f"Assign role '{required_dep}' to a provider first"
                ))
        
        # Check optional dependencies (warnings only)
        for optional_dep in role_def.optional_dependencies:
            if optional_dep not in all_assigned_roles:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Optional dependency '{optional_dep}' is not assigned",
                    provider_name="",
                    role_name=role_def.name,
                    suggested_fix=f"Consider assigning role '{optional_dep}' for enhanced functionality"
                ))
        
        return issues
    
    def _validate_conflicts(
        self, 
        role_def: RoleDefinition, 
        provider_name: str,
        current_assignments: Dict[str, Set[str]]
    ) -> List[ValidationIssue]:
        """Validate role conflicts."""
        issues = []
        
        # Check conflicts with roles on the same provider - fail-fast approach
        if provider_name not in current_assignments:
            raise ValueError(f"Provider '{provider_name}' not found in current assignments")
        current_roles = current_assignments[provider_name]
        
        for conflicting_role in role_def.conflicts_with:
            if conflicting_role in current_roles:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Role conflicts with '{conflicting_role}' on the same provider",
                    provider_name=provider_name,
                    role_name=role_def.name,
                    suggested_fix=f"Remove '{conflicting_role}' from this provider or assign to different provider"
                ))
        
        return issues
    
    def _validate_mutual_exclusivity(
        self, 
        role_def: RoleDefinition, 
        provider_name: str,
        current_assignments: Dict[str, Set[str]]
    ) -> List[ValidationIssue]:
        """Validate mutual exclusivity constraints."""
        issues = []
        
        # Get all currently assigned roles across all providers
        all_assigned_roles = set()
        for roles in current_assignments.values():
            all_assigned_roles.update(roles)
        
        for exclusive_role in role_def.mutually_exclusive:
            if exclusive_role in all_assigned_roles:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Role is mutually exclusive with '{exclusive_role}' (assigned elsewhere)",
                    provider_name=provider_name,
                    role_name=role_def.name,
                    suggested_fix=f"Remove '{exclusive_role}' from all providers first"
                ))
        
        return issues
    
    def _validate_system_requirements(
        self, 
        assignments: Dict[str, Set[str]]
    ) -> List[ValidationIssue]:
        """Validate system-level requirements."""
        issues = []
        
        # Fail-fast approach - check for required policy
        if "require_primary_roles" not in self.validation_policies:
            raise ValueError("Validation policy 'require_primary_roles' is required")
        
        if self.validation_policies["require_primary_roles"]:
            # Check that all primary roles have at least one assignment
            primary_roles = [
                name for name, role_def in self.role_definitions.items()
                if role_def.type == RoleType.PRIMARY
            ]
            
            all_assigned_roles = set()
            for roles in assignments.values():
                all_assigned_roles.update(roles)
            
            for primary_role in primary_roles:
                if primary_role not in all_assigned_roles:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Primary role '{primary_role}' is not assigned to any provider",
                        provider_name="",
                        role_name=primary_role,
                        suggested_fix=f"Assign '{primary_role}' to ensure system functionality"
                    ))
        
        return issues
    
    def _validate_global_constraints(
        self, 
        assignments: Dict[str, Set[str]]
    ) -> List[ValidationIssue]:
        """Validate global constraints."""
        issues = []
        
        # Check for duplicate critical roles
        critical_role_assignments = {}
        
        for provider_name, roles in assignments.items():
            for role_name in roles:
                role_def = self.role_definitions[role_name] if role_name in self.role_definitions else None
                if role_def and role_def.type == RoleType.PRIMARY:
                    if role_name not in critical_role_assignments:
                        critical_role_assignments[role_name] = []
                    critical_role_assignments[role_name].append(provider_name)
        
        # Warn about multiple assignments of critical roles
        for role_name, providers in critical_role_assignments.items():
            if len(providers) > 1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Primary role '{role_name}' assigned to multiple providers: {', '.join(providers)}",
                    provider_name="",
                    role_name=role_name,
                    suggested_fix="Consider using primary/backup pattern instead"
                ))
        
        return issues
    
    def _validate_load_order(
        self, 
        assignments: Dict[str, Set[str]]
    ) -> List[ValidationIssue]:
        """Validate role load order constraints."""
        issues = []
        
        # This is a simplified check - in practice, you'd want more sophisticated
        # dependency graph analysis
        
        all_assigned_roles = set()
        for roles in assignments.values():
            all_assigned_roles.update(roles)
        
        # Check that dependencies are loaded before dependents
        for role_name in all_assigned_roles:
            role_def = self.role_definitions[role_name] if role_name in self.role_definitions else None
            if role_def:
                for dep_name in role_def.required_dependencies:
                    if dep_name in all_assigned_roles:
                        dep_def = self.role_definitions[dep_name] if dep_name in self.role_definitions else None
                        if dep_def and dep_def.load_order > role_def.load_order:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message=f"Dependency '{dep_name}' has higher load order than dependent '{role_name}'",
                                provider_name="",
                                role_name=role_name,
                                suggested_fix="Adjust load order values to ensure proper initialization sequence"
                            ))
        
        return issues
    
    def suggest_role_assignment(
        self, 
        role_name: str, 
        available_providers: List[str],
        current_assignments: Dict[str, Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Suggest optimal provider assignments for a role.
        
        Args:
            role_name: Name of the role to assign
            available_providers: List of available provider names
            current_assignments: Current role assignments
            
        Returns:
            List of (provider_name, score) tuples sorted by score (higher = better)
        """
        current_assignments = current_assignments or {}
        suggestions = []
        
        if role_name not in self.role_definitions:
            return suggestions
        role_def = self.role_definitions[role_name]
        
        for provider_name in available_providers:
            # Validate assignment and calculate score
            issues = self.validate_role_assignment(provider_name, role_name, current_assignments)
            
            # Calculate compatibility score
            score = 100.0
            
            # Penalize based on validation issues
            for issue in issues:
                if issue.severity == ValidationSeverity.CRITICAL:
                    score -= 50
                elif issue.severity == ValidationSeverity.ERROR:
                    score -= 30
                elif issue.severity == ValidationSeverity.WARNING:
                    score -= 10
                elif issue.severity == ValidationSeverity.INFO:
                    score -= 2
            
            # Bonus for exact type matches
            provider_caps = self.provider_capabilities[provider_name] if provider_name in self.provider_capabilities else None
            if provider_caps and provider_caps.type in role_def.compatible_provider_types:
                score += 20
            
            # Bonus for resource availability
            if provider_caps:
                if role_def.requires_gpu and provider_caps.has_gpu:
                    score += 15
                if role_def.network_access_required and provider_caps.has_network_access:
                    score += 10
                
                # Bonus for sufficient resources
                if (role_def.minimum_memory_mb and provider_caps.available_memory_mb and
                    provider_caps.available_memory_mb >= role_def.minimum_memory_mb * 2):
                    score += 10
            
            # Penalty for high current role count - fail-fast approach
            if provider_name not in current_assignments:
                raise ValueError(f"Provider '{provider_name}' not found in current assignments")
            current_roles = len(current_assignments[provider_name])
            score -= current_roles * 5
            
            if score > 0:  # Only suggest viable assignments
                suggestions.append((provider_name, score))
        
        # Sort by score (descending)
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions
    
    def generate_assignment_report(
        self, 
        assignments: Dict[str, Set[str]]
    ) -> str:
        """Generate a comprehensive assignment validation report."""
        issues = self.validate_complete_assignment(assignments)
        
        report = []
        report.append("=== Role Assignment Validation Report ===")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        critical_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)
        error_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.INFO)
        
        report.append("=== Summary ===")
        report.append(f"Critical Issues: {critical_count}")
        report.append(f"Errors: {error_count}")
        report.append(f"Warnings: {warning_count}")
        report.append(f"Info: {info_count}")
        report.append("")
        
        # Overall status
        if critical_count > 0 or error_count > 0:
            report.append("❌ VALIDATION FAILED - Critical issues must be resolved")
        elif warning_count > 0:
            report.append("⚠️  VALIDATION PASSED WITH WARNINGS")
        else:
            report.append("✅ VALIDATION PASSED")
        report.append("")
        
        # Current assignments
        report.append("=== Current Assignments ===")
        for provider_name, roles in assignments.items():
            report.append(f"{provider_name}: {', '.join(sorted(roles)) if roles else 'No roles assigned'}")
        report.append("")
        
        # Issues by severity
        for severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR, 
                        ValidationSeverity.WARNING, ValidationSeverity.INFO]:
            severity_issues = [issue for issue in issues if issue.severity == severity]
            if severity_issues:
                report.append(f"=== {severity.value.upper()} ISSUES ===")
                for issue in severity_issues:
                    report.append(f"• {issue}")
                    if issue.suggested_fix:
                        report.append(f"  Fix: {issue.suggested_fix}")
                report.append("")
        
        return "\n".join(report)