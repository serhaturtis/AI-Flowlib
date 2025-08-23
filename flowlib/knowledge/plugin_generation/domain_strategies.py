"""Domain-specific generation strategies for knowledge plugins."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

from .models import DomainStrategy, DomainStrategyConfig
from ..analysis.prompts import EntityExtractionLLMPrompt, RelationshipExtractionLLMPrompt, ConceptExtractionLLMPrompt
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import prompt
from flowlib.providers.llm import PromptConfigOverride
from typing import ClassVar


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


# Software Engineering Domain Strategy with specialized prompts
@prompt("se-entity-extraction-llm")
class SoftwareEngineeringEntityPrompt(ResourceBase):
    template: ClassVar[str] = """You are analyzing software engineering documentation.
        
Extract all software-related entities from the following text. For each entity, provide:
- name: The entity name
- type: The type (class, function, method, api, library, framework, service, database, interface, protocol, architecture_pattern, design_pattern, tool, language, platform)
- description: Brief technical description
- importance: critical/high/medium/low
- attributes: Technical attributes (version, language, protocol, etc.)

Focus specifically on:
1. Code structures (classes, functions, methods, interfaces)
2. APIs and endpoints (REST, GraphQL, gRPC)
3. Software libraries and frameworks
4. Services and microservices
5. Databases and data stores
6. Development tools and platforms
7. Programming languages and runtime environments
8. Architecture and design patterns
9. Protocols and communication standards
10. Software engineering methodologies

Context: {{context}}

Text to analyze:
{{text}}

Extract entities that would be valuable for a software engineer or developer.
Output a JSON object with an 'entities' array."""
    
    config: ClassVar[PromptConfigOverride] = PromptConfigOverride(
        temperature=0.2,  # Lower temperature for precise technical extraction
        max_tokens=1200   # More tokens for detailed technical entities
    )


@prompt("se-relationship-extraction-llm")
class SoftwareEngineeringRelationshipPrompt(ResourceBase):
    template: ClassVar[str] = """You are analyzing relationships in software engineering documentation.

Given these software entities:
{{entity_list}}

Extract all technical relationships between these entities from the text. For each relationship:
- source: The source entity
- target: The target entity  
- type: Relationship type (inherits, implements, extends, uses, depends_on, calls, imports, deploys_to, communicates_with, configures, manages, exposes, consumes, stores_in, retrieves_from, compiles_to, runs_on, interfaces_with, version_of)
- description: Technical explanation of the relationship
- confidence: 0.0-1.0 confidence score

Focus on software engineering relationships that show:
- Code dependencies and inheritance
- API interactions and data flows
- Deployment and infrastructure relationships
- Development and build dependencies
- Runtime and execution relationships

Text to analyze:
{{text}}

Output a JSON object with a 'relationships' array."""
    
    config: ClassVar[PromptConfigOverride] = PromptConfigOverride(
        temperature=0.2,  # Lower temperature for precise technical relationships
        max_tokens=1000   # Sufficient for technical relationships
    )


@prompt("se-concept-extraction-llm") 
class SoftwareEngineeringConceptPrompt(ResourceBase):
    template: ClassVar[str] = """You are analyzing software engineering documentation.

Extract the {{max_concepts}} most important software engineering concepts. For each:
- concept: The concept name
- abbreviation: Any common abbreviation (API, REST, OOP, etc.)
- explanation: Clear technical explanation for developers
- importance: high/medium/low
- related_concepts: List of related software concepts

Focus on concepts that are crucial for software development:
1. Programming paradigms and methodologies
2. Architecture patterns and design principles
3. Development practices and workflows
4. Technical standards and protocols
5. Software engineering principles

Text to analyze:
{{text}}

Output a JSON object with a 'concepts' array."""
    
    config: ClassVar[PromptConfigOverride] = PromptConfigOverride(
        temperature=0.2,  # Lower temperature for precise technical concepts
        max_tokens=1400   # More tokens for detailed technical explanations
    )


class SoftwareEngineeringDomainStrategy(BaseDomainStrategy):
    """Domain strategy specialized for software engineering documentation."""
    
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
                "design_pattern", "tool", "language", "platform", "endpoint",
                "repository", "module", "package", "dependency", "environment"
            ],
            relationship_types=[
                "inherits", "implements", "extends", "uses", "depends_on", "calls",
                "imports", "deploys_to", "communicates_with", "configures", "manages",
                "exposes", "consumes", "stores_in", "retrieves_from", "compiles_to",
                "runs_on", "interfaces_with", "version_of", "builds", "tests"
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
            chunking_strategy="code_aware",  # Future: implement code-aware chunking
            validation_rules={
                "require_technical_attributes": True,
                "validate_api_endpoints": True,
                "check_dependency_consistency": True
            }
        )
    
    def get_entity_extraction_prompt(self) -> str:
        return "se-entity-extraction-llm"
    
    def get_relationship_extraction_prompt(self) -> str:
        return "se-relationship-extraction-llm"
    
    def get_concept_extraction_prompt(self) -> str:
        return "se-concept-extraction-llm"


# Domain Strategy Registry
class DomainStrategyRegistry:
    """Registry for domain-specific generation strategies."""
    
    def __init__(self):
        self._strategies: Dict[DomainStrategy, BaseDomainStrategy] = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register built-in domain strategies."""
        self._strategies[DomainStrategy.GENERIC] = GenericDomainStrategy()
        self._strategies[DomainStrategy.SOFTWARE_ENGINEERING] = SoftwareEngineeringDomainStrategy()
        # Future strategies will be added here
    
    def get_strategy(self, domain: DomainStrategy, config: Optional[DomainStrategyConfig] = None) -> BaseDomainStrategy:
        """Get a domain strategy instance."""
        if domain not in self._strategies:
            raise ValueError(f"Unknown domain strategy: {domain}")
        
        strategy_class = type(self._strategies[domain])
        return strategy_class(config)
    
    def list_available_strategies(self) -> Dict[DomainStrategy, Dict[str, str]]:
        """List all available strategies with their metadata."""
        return {
            domain: {
                "name": strategy.strategy_name,
                "description": strategy.description
            }
            for domain, strategy in self._strategies.items()
        }
    
    def register_strategy(self, domain: DomainStrategy, strategy: BaseDomainStrategy):
        """Register a custom domain strategy."""
        self._strategies[domain] = strategy


# Global registry instance
domain_strategy_registry = DomainStrategyRegistry()