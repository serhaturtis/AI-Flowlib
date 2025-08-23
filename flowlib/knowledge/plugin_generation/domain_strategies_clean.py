"""Domain-specific generation strategies for knowledge plugins.

This module provides domain strategies without heavy dependencies, 
making it testable in isolation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

from .models import DomainStrategy, DomainStrategyConfig


class DomainExtractionConfig(BaseModel):
    """Configuration for domain-specific extraction."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    
    entity_types: List[str] = Field(description="Expected entity types for this domain")
    relationship_types: List[str] = Field(description="Expected relationship types")
    extraction_focus: List[str] = Field(description="Key areas to focus extraction on")
    chunking_strategy: str = Field(default="default", description="Chunking strategy to use")
    validation_rules: Dict[str, Any] = Field(default_factory=dict, description="Validation rules")


class BaseDomainStrategy(ABC):
    """Base class for domain-specific generation strategies."""
    
    def __init__(self, config: Optional[DomainStrategyConfig] = None):
        self.config = config or DomainStrategyConfig(strategy=DomainStrategy.GENERIC)
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable name for this strategy."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this strategy specializes in."""
        pass
    
    @abstractmethod
    def get_extraction_config(self) -> DomainExtractionConfig:
        """Get domain-specific extraction configuration."""
        pass
    
    @abstractmethod
    def get_entity_extraction_prompt(self) -> str:
        """Get domain-specific entity extraction prompt name."""
        pass
    
    @abstractmethod
    def get_relationship_extraction_prompt(self) -> str:
        """Get domain-specific relationship extraction prompt name."""
        pass
    
    @abstractmethod
    def get_concept_extraction_prompt(self) -> str:
        """Get domain-specific concept extraction prompt name."""
        pass
    
    def validate_extracted_data(self, data: Dict[str, Any]) -> List[str]:
        """Validate extracted data according to domain rules."""
        errors = []
        config = self.get_extraction_config()
        
        # Basic validation - can be overridden by subclasses
        if "entities" in data:
            for entity in data["entities"]:
                if "type" in entity and entity["type"] not in config.entity_types:
                    errors.append(f"Unknown entity type: {entity['type']}")
        
        if "relationships" in data:
            for rel in data["relationships"]:
                if "type" in rel and rel["type"] not in config.relationship_types:
                    errors.append(f"Unknown relationship type: {rel['type']}")
        
        return errors


class GenericDomainStrategy(BaseDomainStrategy):
    """Default generic domain strategy (current behavior)."""
    
    @property
    def strategy_name(self) -> str:
        return "Generic"
    
    @property
    def description(self) -> str:
        return "General-purpose extraction suitable for any document type"
    
    def get_extraction_config(self) -> DomainExtractionConfig:
        return DomainExtractionConfig(
            entity_types=[
                "system", "component", "standard", "protocol", "framework", 
                "tool", "concept", "organization", "product", "technology"
            ],
            relationship_types=[
                "implements", "extends", "uses", "depends_on", "interfaces_with",
                "complies_with", "replaces", "contains", "part_of", "related_to"
            ],
            extraction_focus=[
                "Technical standards and specifications",
                "Systems, components, and tools", 
                "Protocols and interfaces",
                "Frameworks and methodologies",
                "Important concepts and patterns",
                "Organizations and companies",
                "Products and technologies"
            ]
        )
    
    def get_entity_extraction_prompt(self) -> str:
        return "entity-extraction-llm"
    
    def get_relationship_extraction_prompt(self) -> str:
        return "relationship-extraction-llm"
    
    def get_concept_extraction_prompt(self) -> str:
        return "concept-extraction-llm"


class SoftwareEngineeringDomainStrategy(BaseDomainStrategy):
    """Software Engineering domain strategy with specialized prompts."""
    
    @property
    def strategy_name(self) -> str:
        return "Software Engineering"
    
    @property
    def description(self) -> str:
        return "Specialized for code, APIs, architectures, and software development documentation"
    
    def get_extraction_config(self) -> DomainExtractionConfig:
        return DomainExtractionConfig(
            entity_types=[
                "class", "function", "method", "api", "library", "framework", 
                "service", "database", "interface", "protocol", "architecture_pattern",
                "design_pattern", "tool", "language", "platform", "endpoint"
            ],
            relationship_types=[
                "inherits", "implements", "extends", "uses", "depends_on", "calls",
                "imports", "deploys_to", "communicates_with", "configures", "manages",
                "exposes", "consumes", "stores_in", "retrieves_from", "compiles_to"
            ],
            extraction_focus=[
                "Code structures (classes, functions, methods, interfaces)",
                "APIs and endpoints (REST, GraphQL, gRPC)",
                "Software libraries and frameworks",
                "Services and microservices",
                "Databases and data stores",
                "Development tools and platforms",
                "Programming languages and runtime environments",
                "Architecture and design patterns",
                "Protocols and communication standards",
                "Software engineering methodologies"
            ],
            chunking_strategy="code_aware",
            validation_rules={
                "require_technical_attributes": True,
                "validate_api_endpoints": True,
                "check_code_references": True
            }
        )
    
    def get_entity_extraction_prompt(self) -> str:
        return "se-entity-extraction-llm"
    
    def get_relationship_extraction_prompt(self) -> str:
        return "se-relationship-extraction-llm"
    
    def get_concept_extraction_prompt(self) -> str:
        return "se-concept-extraction-llm"
    
    def validate_extracted_data(self, data: Dict[str, Any]) -> List[str]:
        """Enhanced validation for software engineering domain."""
        errors = super().validate_extracted_data(data)
        
        # Add software-specific validation
        if "entities" in data:
            for entity in data["entities"]:
                # Check for required technical attributes
                entity_type = entity["type"] if "type" in entity else None
                entity_name = entity["name"] if "name" in entity else "unknown"
                
                if entity_type in ["api", "service", "database"]:
                    attributes = entity["attributes"] if "attributes" in entity else {}
                    if "technical_details" not in attributes or not attributes["technical_details"]:
                        errors.append(f"Missing technical details for {entity_type}: {entity_name}")
                
                # Validate API endpoints
                if entity_type == "api":
                    attributes = entity["attributes"] if "attributes" in entity else {}
                    if "endpoint" not in attributes or not attributes["endpoint"]:
                        errors.append(f"API entity missing endpoint: {entity_name}")
        
        return errors


class ScientificResearchDomainStrategy(BaseDomainStrategy):
    """Scientific Research domain strategy (placeholder)."""
    
    @property
    def strategy_name(self) -> str:
        return "Scientific Research"
    
    @property
    def description(self) -> str:
        return "Optimized for research papers, citations, and scientific methodologies"
    
    def get_extraction_config(self) -> DomainExtractionConfig:
        return DomainExtractionConfig(
            entity_types=[
                "paper", "author", "institution", "methodology", "dataset", "experiment",
                "hypothesis", "finding", "theory", "model", "algorithm", "metric"
            ],
            relationship_types=[
                "cites", "authored_by", "affiliated_with", "based_on", "validates",
                "contradicts", "extends", "applies", "measures", "compares_to"
            ],
            extraction_focus=[
                "Research papers and publications",
                "Authors and institutions",
                "Methodologies and experiments",
                "Datasets and data sources",
                "Findings and conclusions",
                "Citations and references"
            ]
        )
    
    def get_entity_extraction_prompt(self) -> str:
        return "sr-entity-extraction-llm"
    
    def get_relationship_extraction_prompt(self) -> str:
        return "sr-relationship-extraction-llm"
    
    def get_concept_extraction_prompt(self) -> str:
        return "sr-concept-extraction-llm"


class BusinessProcessDomainStrategy(BaseDomainStrategy):
    """Business Process domain strategy (placeholder)."""
    
    @property
    def strategy_name(self) -> str:
        return "Business Process"
    
    @property
    def description(self) -> str:
        return "Focused on workflows, decisions, policies, and business operations"
    
    def get_extraction_config(self) -> DomainExtractionConfig:
        return DomainExtractionConfig(
            entity_types=[
                "process", "workflow", "decision", "policy", "role", "department",
                "system", "document", "procedure", "milestone", "deliverable", "stakeholder"
            ],
            relationship_types=[
                "depends_on", "triggers", "approves", "manages", "reports_to",
                "uses", "produces", "requires", "follows", "complies_with"
            ],
            extraction_focus=[
                "Business processes and workflows",
                "Organizational roles and responsibilities",
                "Policies and procedures",
                "Decision points and approvals",
                "Systems and tools",
                "Stakeholders and departments"
            ]
        )
    
    def get_entity_extraction_prompt(self) -> str:
        return "bp-entity-extraction-llm"
    
    def get_relationship_extraction_prompt(self) -> str:
        return "bp-relationship-extraction-llm"
    
    def get_concept_extraction_prompt(self) -> str:
        return "bp-concept-extraction-llm"


class LegalComplianceDomainStrategy(BaseDomainStrategy):
    """Legal/Compliance domain strategy (placeholder)."""
    
    @property
    def strategy_name(self) -> str:
        return "Legal/Compliance"
    
    @property
    def description(self) -> str:
        return "Specialized for regulations, clauses, legal precedents, and compliance documentation"
    
    def get_extraction_config(self) -> DomainExtractionConfig:
        return DomainExtractionConfig(
            entity_types=[
                "regulation", "clause", "precedent", "court", "case", "statute",
                "compliance_requirement", "jurisdiction", "authority", "penalty", "exception", "definition"
            ],
            relationship_types=[
                "regulates", "requires", "prohibits", "supersedes", "references",
                "applies_to", "enforced_by", "violates", "complies_with", "exempts"
            ],
            extraction_focus=[
                "Legal regulations and statutes",
                "Compliance requirements",
                "Legal precedents and cases",
                "Jurisdictions and authorities",
                "Penalties and enforcement",
                "Definitions and interpretations"
            ]
        )
    
    def get_entity_extraction_prompt(self) -> str:
        return "lc-entity-extraction-llm"
    
    def get_relationship_extraction_prompt(self) -> str:
        return "lc-relationship-extraction-llm"
    
    def get_concept_extraction_prompt(self) -> str:
        return "lc-concept-extraction-llm"


class DomainStrategyRegistry:
    """Registry for domain-specific generation strategies."""
    
    def __init__(self):
        self._strategies: Dict[DomainStrategy, BaseDomainStrategy] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default domain strategies."""
        self._strategies[DomainStrategy.GENERIC] = GenericDomainStrategy()
        self._strategies[DomainStrategy.SOFTWARE_ENGINEERING] = SoftwareEngineeringDomainStrategy()
        self._strategies[DomainStrategy.SCIENTIFIC_RESEARCH] = ScientificResearchDomainStrategy()
        self._strategies[DomainStrategy.BUSINESS_PROCESS] = BusinessProcessDomainStrategy()
        self._strategies[DomainStrategy.LEGAL_COMPLIANCE] = LegalComplianceDomainStrategy()
    
    def register_strategy(self, domain: DomainStrategy, strategy: BaseDomainStrategy):
        """Register a custom domain strategy."""
        self._strategies[domain] = strategy
    
    def get_strategy(self, domain: DomainStrategy, config: Optional[DomainStrategyConfig] = None) -> BaseDomainStrategy:
        """Get a domain strategy instance."""
        if domain not in self._strategies:
            raise ValueError(f"Unknown domain strategy: {domain}")
        
        strategy = self._strategies[domain]
        
        # If config is provided, create a new instance with config
        if config:
            strategy_class = type(strategy)
            return strategy_class(config)
        
        return strategy
    
    def list_available_strategies(self) -> Dict[DomainStrategy, Dict[str, str]]:
        """List all available domain strategies."""
        return {
            domain: {
                "name": strategy.strategy_name,
                "description": strategy.description
            }
            for domain, strategy in self._strategies.items()
        }


# Global registry instance
domain_strategy_registry = DomainStrategyRegistry()