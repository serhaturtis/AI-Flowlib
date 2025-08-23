"""Prompts for knowledge extraction and retrieval flows."""

from typing import ClassVar

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt("knowledge_extraction")
class KnowledgeExtractionPrompt(ResourceBase):
    """Prompt for extracting knowledge from text."""
    template: ClassVar[str] = """
You are a Knowledge Extraction Assistant. Your job is to identify and extract useful knowledge from the given text.

Extract knowledge that falls into these categories:
- FACTUAL: Concrete facts, data points, definitions
- PROCEDURAL: How-to information, processes, steps
- CONCEPTUAL: Ideas, theories, explanations
- PERSONAL: User preferences, appointments, personal information
- TECHNICAL: Domain-specific technical knowledge

Text to analyze: "{{text}}"
Context: {{context}}
Domain hint: {{domain_hint}}
Extract personal info: {{extract_personal}}

For each piece of knowledge, identify:
1. The content (what is the knowledge)
2. The type (factual, procedural, conceptual, personal, technical)
3. The domain (chemistry, personal, technology, etc.)
4. Confidence level (0.0 to 1.0)
5. Key entities mentioned
6. Relevant metadata

Focus on extracting knowledge that would be useful for future reference. Avoid extracting trivial conversational elements.

Return your analysis as JSON with this structure:
{
  "extracted_knowledge": [
    {
      "content": "Knowledge content here",
      "knowledge_type": "factual|procedural|conceptual|personal|technical",
      "domain": "domain name",
      "confidence": 0.95,
      "source_context": "Original context snippet",
      "entities": ["entity1", "entity2"],
      "metadata": {"key": "value"}
    }
  ],
  "processing_notes": "Notes about extraction process",
  "domains_detected": ["domain1", "domain2"]
}
"""


@prompt("domain_detection")
class DomainDetectionPrompt(ResourceBase):
    """Prompt for detecting knowledge domains in text."""
    template: ClassVar[str] = """
You are a Domain Detection Assistant. Analyze the given text and identify the primary knowledge domains present.

Common domains include:
- chemistry, physics, biology, medicine
- technology, programming, engineering
- business, finance, economics
- personal, lifestyle, health
- education, research, academic
- entertainment, sports, arts

Text: "{{text}}"

Return the top 3 most relevant domains as a JSON list:
["domain1", "domain2", "domain3"]

Only return domains that are clearly present in the text. If fewer than 3 domains are relevant, return fewer.
"""


@prompt("knowledge_synthesis")
class KnowledgeSynthesisPrompt(ResourceBase):
    """Prompt for synthesizing retrieved knowledge."""
    template: ClassVar[str] = """
You are a Knowledge Synthesis Assistant. Combine and summarize the retrieved knowledge to answer the user's query.

User Query: "{{query}}"

Retrieved Knowledge:
{{knowledge_items}}

Sources Searched: {{sources_searched}}

Synthesize the information to provide a comprehensive answer. Include:
1. Direct answers to the query
2. Additional relevant context
3. Source attribution
4. Confidence assessment

Format your response as JSON:
{
  "search_summary": "Brief summary of search process and what was found",
  "synthesized_answer": "Comprehensive answer based on retrieved knowledge",
  "confidence": 0.85,
  "sources_used": ["source1", "source2"],
  "gaps": "Any information gaps or limitations"
}
"""