"""Prompts for semantic recall flow."""

from typing import ClassVar
from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import prompt


@prompt("semantic_analysis")
class SemanticAnalysisPrompt(ResourceBase):
    """Prompt for analyzing semantic aspects of a query."""
    
    template: ClassVar[str] = """
Analyze the following query for semantic understanding:

Query: {{query}}
Context: {{context}}

Please analyze this query and identify:
1. Key concepts and topics mentioned
2. Semantic relationships between concepts
3. Contextual meaning and intent
4. Topic categories this query relates to
5. Your confidence in this analysis

Return your analysis as a JSON object:
{
    "key_concepts": ["concept1", "concept2", "concept3"],
    "semantic_relationships": ["relationship1", "relationship2"],
    "contextual_meaning": "detailed explanation of what the query is asking for",
    "topic_categories": ["category1", "category2"],
    "confidence": 0.85
}
"""
    
    config: ClassVar[dict] = {
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.95
    }