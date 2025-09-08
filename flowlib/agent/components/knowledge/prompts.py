"""Prompts for the unified knowledge component."""

from typing import ClassVar

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt("knowledge-extraction-prompt")
class KnowledgeExtractionPrompt(ResourceBase):
    """Prompt for structured knowledge extraction using the KnowledgeSet model."""
    template: ClassVar[str] = """
You are a Knowledge Extraction Assistant. Extract structured knowledge from the given content.

Content: {{content}}
Context: {{context}}
Focus Areas: {{focus_areas}}
Domain Hint: {{domain_hint}}

Extract knowledge in these categories:

1. ENTITIES: Named entities (people, places, organizations, products, etc.)
   - Extract the entity name, type, description, and properties
   - Provide confidence score (0.0 to 1.0)

2. CONCEPTS: Abstract concepts, ideas, theories, definitions
   - Extract concept name, definition, category, and related terms
   - Provide confidence score (0.0 to 1.0)

3. RELATIONSHIPS: Connections between entities or concepts
   - Extract source, target, relationship type, and description
   - Provide confidence score (0.0 to 1.0)

4. PATTERNS: Recurring themes, behaviors, or structures
   - Extract pattern name, description, frequency, and examples
   - Provide confidence score (0.0 to 1.0)

Extract all available knowledge while focusing on accuracy and relevance to the domain and context provided.
"""


@prompt("knowledge-retrieval-prompt")
class KnowledgeRetrievalPrompt(ResourceBase):
    """Prompt for intelligent knowledge retrieval and ranking."""
    template: ClassVar[str] = """
You are a Knowledge Retrieval Assistant. Analyze the search query and rank the provided knowledge items by relevance.

Search Query: {{query}}
Context Filter: {{context_filter}}
Knowledge Types Requested: {{knowledge_types}}

Knowledge Items to Rank:
{{knowledge_items}}

Rank the knowledge items by relevance to the query. Consider:
1. Direct relevance to the query terms
2. Contextual relevance 
3. Knowledge type matching (if specified)
4. Confidence scores of the knowledge items

Return a ranked list with relevance scores (0.0 to 1.0) and brief explanations.
"""