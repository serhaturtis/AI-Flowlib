"""Prompts for entity analysis flow."""

from pydantic import Field

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt("entity-extraction-llm")
class EntityExtractionLLMPrompt(ResourceBase):
    template: str = Field(default="""You are analyzing a technical document in the {{domain}} domain.
        
Extract all important entities from the following text. For each entity, provide:
- name: The entity name
- type: The type (e.g., system, component, standard, protocol, framework, tool, concept, organization)
- description: Brief description of what it is
- importance: critical/high/medium/low
- attributes: Any relevant attributes (version, category, etc.)

Focus on:
1. Technical standards and specifications
2. Systems, components, and tools
3. Protocols and interfaces
4. Frameworks and methodologies
5. Important concepts and patterns
6. Organizations and companies
7. Products and technologies

Context: {{context}}

Text to analyze:
{{text}}

Extract entities that would be valuable for an AI agent working in the {{domain}} field.
Provide an 'entities' array with the extracted entities.""")

    config: dict = Field(default={
        "temperature": 0.3,  # Lower temperature for precise extraction
        "max_tokens": 3000   # Increased for comprehensive entity extraction
    })


@prompt("relationship-extraction-llm")
class RelationshipExtractionLLMPrompt(ResourceBase):
    template: str = Field(default="""You are analyzing relationships in a {{domain}} technical document.

Given these entities:
{{entity_list}}

Extract all relationships between these entities from the text. For each relationship:
- source: The source entity
- target: The target entity  
- type: Relationship type (e.g., implements, extends, uses, depends_on, interfaces_with, complies_with, replaces, etc.)
- description: Brief explanation of the relationship
- confidence: 0.0-1.0 confidence score

Focus on technical relationships that would help an AI understand how these entities interact.

Text to analyze:
{{text}}

Provide a 'relationships' array with the identified relationships.""")

    config: dict = Field(default={
        "temperature": 0.3,  # Lower temperature for precise extraction
        "max_tokens": 2500   # Increased for comprehensive relationship extraction
    })


@prompt("concept-extraction-llm")
class ConceptExtractionLLMPrompt(ResourceBase):
    template: str = Field(default="""You are analyzing a {{domain}} technical document.

Extract the {{max_concepts}} most important technical concepts. For each:
- concept: The concept name
- abbreviation: Any common abbreviation (if applicable)
- explanation: Clear, concise explanation
- importance: high/medium/low
- related_concepts: List of related concepts

Focus on concepts that are crucial for understanding {{domain}}.

Text to analyze:
{{text}}

Provide a 'concepts' array with the most important concepts.""")

    config: dict = Field(default={
        "temperature": 0.3,  # Lower temperature for precise extraction
        "max_tokens": 3500   # Increased for comprehensive concept extraction
    })
