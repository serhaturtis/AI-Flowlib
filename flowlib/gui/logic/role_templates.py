"""
Role Templates and Presets System

Advanced template system for role configurations with presets, customization,
and automated deployment following CLAUDE.md principles.
"""

import logging
import json
from typing import Dict, List, Set, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from pydantic import Field
from flowlib.core.models import StrictBaseModel

from .role_validation import RoleDefinition, ProviderCapabilities, RoleValidationEngine

logger = logging.getLogger(__name__)


class TemplateCategory(Enum):
    """Categories for role templates."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    MONITORING = "monitoring"
    RESEARCH = "research"
    CUSTOM = "custom"


class DeploymentStrategy(Enum):
    """Deployment strategies for role templates."""
    SINGLE_PROVIDER = "single_provider"
    HIGH_AVAILABILITY = "high_availability"
    DISTRIBUTED = "distributed"
    BACKUP_REDUNDANCY = "backup_redundancy"
    LOAD_BALANCED = "load_balanced"


@dataclass
class RoleAssignment:
    """Represents a single role assignment in a template."""
    role_name: str
    provider_name: str
    priority: int = 100
    configuration: Dict[str, Any] = None
    conditions: Dict[str, Any] = None  # Conditional assignment rules
    
    def __post_init__(self):
        if self.configuration is None:
            self.configuration = {}
        if self.conditions is None:
            self.conditions = {}


class RoleTemplate(StrictBaseModel):
    """A template defining a complete role assignment configuration."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    category: TemplateCategory = Field(..., description="Template category")
    version: str = Field(default="1.0.0", description="Template version")
    
    # Assignment configuration
    role_assignments: List[Dict[str, Any]] = Field(default_factory=list, description="Role assignments")
    deployment_strategy: DeploymentStrategy = Field(default=DeploymentStrategy.SINGLE_PROVIDER, description="Deployment strategy")
    
    # Requirements and constraints
    minimum_providers: int = Field(default=1, description="Minimum required providers")
    maximum_providers: Optional[int] = Field(default=None, description="Maximum allowed providers")
    required_provider_types: Set[str] = Field(default_factory=set, description="Required provider types")
    
    # Resource requirements
    total_memory_mb: Optional[int] = Field(default=None, description="Total memory requirement")
    requires_gpu: bool = Field(default=False, description="GPU requirement")
    network_access_required: bool = Field(default=False, description="Network access requirement")
    
    # Metadata
    author: str = Field(default="System", description="Template author")
    tags: Set[str] = Field(default_factory=set, description="Template tags")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    # Customization
    is_customizable: bool = Field(default=True, description="Whether template can be customized")
    customizable_fields: Set[str] = Field(default_factory=set, description="Fields that can be customized")
    
    def to_assignments_dict(self) -> Dict[str, Set[str]]:
        """Convert template to standard assignments dictionary."""
        assignments = {}
        for assignment_data in self.role_assignments:
            if "provider_name" not in assignment_data:
                raise ValueError("Assignment data missing required 'provider_name' field")
            if "role_name" not in assignment_data:
                raise ValueError("Assignment data missing required 'role_name' field")
                
            provider_name = assignment_data["provider_name"]
            role_name = assignment_data["role_name"]
            
            if provider_name and role_name:
                if provider_name not in assignments:
                    assignments[provider_name] = set()
                assignments[provider_name].add(role_name)
        
        return assignments


class TemplatePreset(StrictBaseModel):
    """Predefined template preset for common scenarios."""
    # Inherits strict configuration from StrictBaseModel with frozen=True
    
    name: str = Field(..., description="Preset name")
    description: str = Field(..., description="Preset description")
    template_name: str = Field(..., description="Associated template name")
    
    # Preset configuration
    provider_mappings: Dict[str, str] = Field(default_factory=dict, description="Role to provider mappings")
    configuration_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Configuration overrides")
    
    # Conditions for auto-application
    auto_apply_conditions: Dict[str, Any] = Field(default_factory=dict, description="Conditions for automatic application")
    
    # Metadata
    use_count: int = Field(default=0, description="Number of times used")
    last_used: Optional[datetime] = Field(default=None, description="Last usage timestamp")


class RoleTemplateManager:
    """Advanced role template and preset management system."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        self.templates_dir = templates_dir or Path.home() / ".flowlib" / "role_templates"
        self.templates: Dict[str, RoleTemplate] = {}
        self.presets: Dict[str, TemplatePreset] = {}
        self.validation_engine = RoleValidationEngine()
        
        # Ensure templates directory exists
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Load default templates and presets
        self.load_default_templates()
        self.load_templates_from_disk()
    
    def load_default_templates(self):
        """Load default role templates for common scenarios."""
        
        # Development Template
        dev_template = RoleTemplate(
            name="development-setup",
            description="Basic development environment with LLM, vector storage, and caching",
            category=TemplateCategory.DEVELOPMENT,
            deployment_strategy=DeploymentStrategy.SINGLE_PROVIDER,
            role_assignments=[
                {
                    "role_name": "primary-llm",
                    "provider_name": "dev-llm",
                    "priority": 10,
                    "configuration": {"temperature": 0.7, "max_tokens": 2048}
                },
                {
                    "role_name": "vector-store", 
                    "provider_name": "dev-vectors",
                    "priority": 20,
                    "configuration": {"collection_name": "dev_collection"}
                },
                {
                    "role_name": "cache-layer",
                    "provider_name": "dev-cache", 
                    "priority": 30,
                    "configuration": {"ttl": 3600}
                }
            ],
            required_provider_types={"llm", "vector_db", "cache"},
            tags={"development", "basic", "local"},
            is_customizable=True,
            customizable_fields={"role_assignments", "configuration"}
        )
        
        # Production Template
        prod_template = RoleTemplate(
            name="production-ha",
            description="High-availability production setup with redundancy",
            category=TemplateCategory.PRODUCTION,
            deployment_strategy=DeploymentStrategy.HIGH_AVAILABILITY,
            role_assignments=[
                {
                    "role_name": "primary-llm",
                    "provider_name": "prod-llm-primary",
                    "priority": 10,
                    "configuration": {"temperature": 0.3, "max_tokens": 4096}
                },
                {
                    "role_name": "backup-llm",
                    "provider_name": "prod-llm-backup",
                    "priority": 15,
                    "configuration": {"temperature": 0.3, "max_tokens": 4096}
                },
                {
                    "role_name": "vector-store",
                    "provider_name": "prod-vectors-primary",
                    "priority": 20,
                    "configuration": {"replication_factor": 3}
                },
                {
                    "role_name": "cache-layer",
                    "provider_name": "prod-cache-cluster",
                    "priority": 25,
                    "configuration": {"cluster_mode": True}
                },
                {
                    "role_name": "monitoring",
                    "provider_name": "prod-monitoring",
                    "priority": 40,
                    "configuration": {"metrics_interval": 30}
                }
            ],
            minimum_providers=4,
            required_provider_types={"llm", "vector_db", "cache", "monitoring"},
            total_memory_mb=16384,
            requires_gpu=True,
            network_access_required=True,
            tags={"production", "high-availability", "enterprise"},
            is_customizable=False  # Production templates should be stable
        )
        
        # Research Template
        research_template = RoleTemplate(
            name="research-experimental",
            description="Experimental research setup with multiple LLM providers",
            category=TemplateCategory.RESEARCH,
            deployment_strategy=DeploymentStrategy.DISTRIBUTED,
            role_assignments=[
                {
                    "role_name": "primary-llm",
                    "provider_name": "research-llm-main",
                    "priority": 10,
                    "configuration": {"temperature": 0.9, "top_k": 50}
                },
                {
                    "role_name": "backup-llm",
                    "provider_name": "research-llm-alt",
                    "priority": 15,
                    "configuration": {"temperature": 0.1, "top_p": 0.9}
                },
                {
                    "role_name": "vector-store",
                    "provider_name": "research-vectors",
                    "priority": 20,
                    "configuration": {"experimental_features": True}
                },
                {
                    "role_name": "monitoring",
                    "provider_name": "research-metrics",
                    "priority": 30,
                    "configuration": {"detailed_logging": True}
                }
            ],
            required_provider_types={"llm", "vector_db"},
            tags={"research", "experimental", "flexible"},
            is_customizable=True,
            customizable_fields={"role_assignments", "configuration", "deployment_strategy"}
        )
        
        # Testing Template
        test_template = RoleTemplate(
            name="testing-minimal",
            description="Minimal testing setup with mock providers",
            category=TemplateCategory.TESTING,
            deployment_strategy=DeploymentStrategy.SINGLE_PROVIDER,
            role_assignments=[
                {
                    "role_name": "primary-llm",
                    "provider_name": "test-llm-mock",
                    "priority": 10,
                    "configuration": {"mock_mode": True, "response_delay": 0.1}
                },
                {
                    "role_name": "vector-store",
                    "provider_name": "test-vectors-memory",
                    "priority": 20,
                    "configuration": {"in_memory": True}
                }
            ],
            minimum_providers=1,
            required_provider_types={"llm"},
            tags={"testing", "mock", "minimal"},
            is_customizable=True,
            customizable_fields={"*"}  # All fields customizable for testing
        )
        
        # Store templates
        templates = [dev_template, prod_template, research_template, test_template]
        for template in templates:
            self.templates[template.name] = template
        
        # Create presets for templates
        self._create_default_presets()
    
    def _create_default_presets(self):
        """Create default presets for templates."""
        
        # Local development preset
        local_dev_preset = TemplatePreset(
            name="local-development",
            description="Local development with LlamaCpp and ChromaDB",
            template_name="development-setup",
            provider_mappings={
                "dev-llm": "llamacpp-local",
                "dev-vectors": "chroma-local", 
                "dev-cache": "redis-local"
            },
            configuration_overrides={
                "dev-llm": {"model_path": "/models/dev-model.gguf"},
                "dev-vectors": {"persist_directory": "./chroma_data"},
                "dev-cache": {"database": 1}
            }
        )
        
        # Cloud development preset
        cloud_dev_preset = TemplatePreset(
            name="cloud-development",
            description="Cloud development with OpenAI and Pinecone",
            template_name="development-setup",
            provider_mappings={
                "dev-llm": "openai-gpt4",
                "dev-vectors": "pinecone-dev",
                "dev-cache": "redis-cloud"
            },
            configuration_overrides={
                "dev-llm": {"model": "gpt-4", "temperature": 0.5},
                "dev-vectors": {"environment": "dev", "index_name": "dev-index"},
                "dev-cache": {"ssl": True}
            }
        )
        
        # Production AWS preset
        prod_aws_preset = TemplatePreset(
            name="production-aws",
            description="Production deployment on AWS infrastructure",
            template_name="production-ha",
            provider_mappings={
                "prod-llm-primary": "bedrock-claude",
                "prod-llm-backup": "bedrock-claude-backup",
                "prod-vectors-primary": "opensearch-vectors",
                "prod-cache-cluster": "elasticache-redis",
                "prod-monitoring": "cloudwatch-metrics"
            },
            configuration_overrides={
                "prod-llm-primary": {"region": "us-east-1", "model_id": "claude-3"},
                "prod-vectors-primary": {"cluster_name": "prod-search", "node_count": 3},
                "prod-cache-cluster": {"cluster_mode": True, "num_shards": 3}
            }
        )
        
        presets = [local_dev_preset, cloud_dev_preset, prod_aws_preset]
        for preset in presets:
            self.presets[preset.name] = preset
    
    def create_template(
        self, 
        name: str,
        description: str,
        role_assignments: List[RoleAssignment],
        category: TemplateCategory = TemplateCategory.CUSTOM,
        **kwargs
    ) -> RoleTemplate:
        """Create a new role template."""
        
        # Convert RoleAssignment objects to dictionaries
        assignments_data = []
        for assignment in role_assignments:
            assignments_data.append(asdict(assignment))
        
        template = RoleTemplate(
            name=name,
            description=description,
            category=category,
            role_assignments=assignments_data,
            **kwargs
        )
        
        # Validate template
        validation_issues = self.validate_template(template)
        if any(issue.severity.value in ['error', 'critical'] for issue in validation_issues):
            error_messages = [str(issue) for issue in validation_issues 
                            if issue.severity.value in ['error', 'critical']]
            raise ValueError(f"Template validation failed:\n" + "\n".join(error_messages))
        
        self.templates[name] = template
        self.save_template(template)
        
        logger.info(f"Created role template: {name}")
        return template
    
    def create_preset(
        self,
        name: str,
        description: str,
        template_name: str,
        provider_mappings: Dict[str, str],
        configuration_overrides: Dict[str, Dict[str, Any]] = None
    ) -> TemplatePreset:
        """Create a new template preset."""
        
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        preset = TemplatePreset(
            name=name,
            description=description,
            template_name=template_name,
            provider_mappings=provider_mappings,
            configuration_overrides=configuration_overrides or {}
        )
        
        self.presets[name] = preset
        self.save_preset(preset)
        
        logger.info(f"Created template preset: {name}")
        return preset
    
    def apply_template(
        self, 
        template_name: str,
        available_providers: List[str],
        preset_name: Optional[str] = None,
        custom_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Set[str]]:
        """
        Apply a template to generate role assignments.
        
        Args:
            template_name: Name of the template to apply
            available_providers: List of available provider names
            preset_name: Optional preset to use for provider mappings
            custom_overrides: Custom configuration overrides
            
        Returns:
            Dictionary of role assignments {provider_name: {role_names}}
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        template = self.templates[template_name]
        
        preset = None
        if preset_name:
            if preset_name not in self.presets:
                raise ValueError(f"Preset '{preset_name}' not found")
            preset = self.presets[preset_name]
            if not preset:
                raise ValueError(f"Preset '{preset_name}' not found")
        
        # Start with template assignments
        assignments = {}
        provider_mappings = preset.provider_mappings if preset else {}
        
        for assignment_data in template.role_assignments:
            if "role_name" not in assignment_data:
                raise ValueError("Role assignment missing required 'role_name' field")
            if "provider_name" not in assignment_data:
                raise ValueError("Role assignment missing required 'provider_name' field")
            role_name = assignment_data["role_name"]
            template_provider = assignment_data["provider_name"]
            
            # Resolve actual provider name - fail-fast approach
            if template_provider in provider_mappings:
                actual_provider = provider_mappings[template_provider]
            else:
                actual_provider = template_provider
            
            # Check if provider is available
            if actual_provider not in available_providers:
                # Try to find a suitable alternative
                suitable_providers = self._find_suitable_providers(
                    role_name, available_providers, assignments
                )
                if suitable_providers:
                    actual_provider = suitable_providers[0][0]  # Best match
                else:
                    logger.warning(f"No suitable provider found for role '{role_name}'")
                    continue
            
            # Add assignment
            if actual_provider not in assignments:
                assignments[actual_provider] = set()
            assignments[actual_provider].add(role_name)
        
        # Validate final assignments
        validation_issues = self.validation_engine.validate_complete_assignment(assignments)
        critical_issues = [issue for issue in validation_issues 
                         if issue.severity.value in ['error', 'critical']]
        
        if critical_issues:
            logger.error(f"Template application failed validation:")
            for issue in critical_issues:
                logger.error(f"  {issue}")
            raise ValueError("Template application resulted in invalid role assignments")
        
        # Update preset usage if used
        if preset:
            preset.use_count += 1
            preset.last_used = datetime.now()
            self.save_preset(preset)
        
        logger.info(f"Applied template '{template_name}' with {len(assignments)} providers")
        return assignments
    
    def _find_suitable_providers(
        self, 
        role_name: str, 
        available_providers: List[str],
        current_assignments: Dict[str, Set[str]]
    ) -> List[Tuple[str, float]]:
        """Find suitable providers for a role using the validation engine."""
        return self.validation_engine.suggest_role_assignment(
            role_name, available_providers, current_assignments
        )
    
    def customize_template(
        self, 
        template_name: str,
        customizations: Dict[str, Any],
        new_name: Optional[str] = None
    ) -> RoleTemplate:
        """
        Create a customized version of an existing template.
        
        Args:
            template_name: Name of the template to customize
            customizations: Dictionary of customizations to apply
            new_name: Name for the new customized template
            
        Returns:
            New customized template
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        base_template = self.templates[template_name]
        
        if not base_template.is_customizable:
            raise ValueError(f"Template '{template_name}' is not customizable")
        
        # Create new template based on the original
        template_data = base_template.model_dump()
        template_data["name"] = new_name or f"{template_name}-custom"
        template_data["category"] = TemplateCategory.CUSTOM
        template_data["created_at"] = datetime.now()
        template_data["updated_at"] = datetime.now()
        
        # Apply customizations
        for field, value in customizations.items():
            if (base_template.customizable_fields and 
                "*" not in base_template.customizable_fields and
                field not in base_template.customizable_fields):
                logger.warning(f"Field '{field}' is not customizable in template '{template_name}'")
                continue
            
            if field in template_data:
                template_data[field] = value
        
        # Create new template
        customized_template = RoleTemplate(**template_data)
        
        # Validate
        validation_issues = self.validate_template(customized_template)
        if any(issue.severity.value in ['error', 'critical'] for issue in validation_issues):
            error_messages = [str(issue) for issue in validation_issues 
                            if issue.severity.value in ['error', 'critical']]
            raise ValueError(f"Customized template validation failed:\n" + "\n".join(error_messages))
        
        self.templates[customized_template.name] = customized_template
        self.save_template(customized_template)
        
        logger.info(f"Created customized template: {customized_template.name}")
        return customized_template
    
    def validate_template(self, template: RoleTemplate) -> List[Any]:
        """Validate a role template."""
        # Convert template to assignments format
        assignments = template.to_assignments_dict()
        
        # Use validation engine
        return self.validation_engine.validate_complete_assignment(assignments)
    
    def list_templates(
        self, 
        category: Optional[TemplateCategory] = None,
        tags: Optional[Set[str]] = None
    ) -> List[RoleTemplate]:
        """List available templates with optional filtering."""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if tags:
            templates = [t for t in templates if tags.intersection(t.tags)]
        
        return sorted(templates, key=lambda t: t.name)
    
    def list_presets(self, template_name: Optional[str] = None) -> List[TemplatePreset]:
        """List available presets with optional template filtering."""
        presets = list(self.presets.values())
        
        if template_name:
            presets = [p for p in presets if p.template_name == template_name]
        
        return sorted(presets, key=lambda p: p.name)
    
    def save_template(self, template: RoleTemplate):
        """Save template to disk."""
        try:
            template_file = self.templates_dir / f"{template.name}.json"
            with open(template_file, 'w') as f:
                json.dump(template.model_dump(mode='json'), f, indent=2, default=str)
            logger.debug(f"Saved template to {template_file}")
        except Exception as e:
            logger.error(f"Failed to save template {template.name}: {e}")
    
    def save_preset(self, preset: TemplatePreset):
        """Save preset to disk."""
        try:
            presets_dir = self.templates_dir / "presets"
            presets_dir.mkdir(exist_ok=True)
            preset_file = presets_dir / f"{preset.name}.json"
            with open(preset_file, 'w') as f:
                json.dump(preset.model_dump(mode='json'), f, indent=2, default=str)
            logger.debug(f"Saved preset to {preset_file}")
        except Exception as e:
            logger.error(f"Failed to save preset {preset.name}: {e}")
    
    def load_templates_from_disk(self):
        """Load templates and presets from disk."""
        try:
            # Load templates
            for template_file in self.templates_dir.glob("*.json"):
                try:
                    with open(template_file, 'r') as f:
                        template_data = json.load(f)
                    
                    template = RoleTemplate(**template_data)
                    self.templates[template.name] = template
                    logger.debug(f"Loaded template: {template.name}")
                except Exception as e:
                    logger.error(f"Failed to load template from {template_file}: {e}")
            
            # Load presets
            presets_dir = self.templates_dir / "presets"
            if presets_dir.exists():
                for preset_file in presets_dir.glob("*.json"):
                    try:
                        with open(preset_file, 'r') as f:
                            preset_data = json.load(f)
                        
                        preset = TemplatePreset(**preset_data)
                        self.presets[preset.name] = preset
                        logger.debug(f"Loaded preset: {preset.name}")
                    except Exception as e:
                        logger.error(f"Failed to load preset from {preset_file}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load templates from disk: {e}")
    
    def delete_template(self, template_name: str):
        """Delete a template and its associated files."""
        if template_name in self.templates:
            # Remove from memory
            del self.templates[template_name]
            
            # Remove file
            template_file = self.templates_dir / f"{template_name}.json"
            if template_file.exists():
                template_file.unlink()
            
            # Remove associated presets
            presets_to_remove = [name for name, preset in self.presets.items() 
                               if preset.template_name == template_name]
            for preset_name in presets_to_remove:
                self.delete_preset(preset_name)
            
            logger.info(f"Deleted template: {template_name}")
    
    def delete_preset(self, preset_name: str):
        """Delete a preset and its associated files."""
        if preset_name in self.presets:
            # Remove from memory
            del self.presets[preset_name]
            
            # Remove file
            preset_file = self.templates_dir / "presets" / f"{preset_name}.json"
            if preset_file.exists():
                preset_file.unlink()
            
            logger.info(f"Deleted preset: {preset_name}")
    
    def export_template(self, template_name: str, export_path: Path) -> Path:
        """Export a template to a specified location."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        template = self.templates[template_name]
        
        export_data = {
            "template": template.model_dump(mode='json'),
            "presets": [
                preset.model_dump(mode='json') 
                for preset in self.presets.values()
                if preset.template_name == template_name
            ],
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported template '{template_name}' to {export_path}")
        return export_path
    
    def import_template(self, import_path: Path) -> RoleTemplate:
        """Import a template from a file."""
        with open(import_path, 'r') as f:
            import_data = json.load(f)
        
        # Import template
        if "template" not in import_data:
            raise ValueError("Invalid template export file: missing required 'template' field")
        template_data = import_data["template"]
        
        template = RoleTemplate(**template_data)
        
        # Validate before importing
        validation_issues = self.validate_template(template)
        if any(issue.severity.value in ['error', 'critical'] for issue in validation_issues):
            error_messages = [str(issue) for issue in validation_issues 
                            if issue.severity.value in ['error', 'critical']]
            raise ValueError(f"Imported template validation failed:\n" + "\n".join(error_messages))
        
        # Store template
        self.templates[template.name] = template
        self.save_template(template)
        
        # Import presets - fail-fast approach
        presets_data = import_data["presets"] if "presets" in import_data else []
        for preset_data in presets_data:
            preset = TemplatePreset(**preset_data)
            self.presets[preset.name] = preset
            self.save_preset(preset)
        
        logger.info(f"Imported template '{template.name}' with {len(presets_data)} presets")
        return template