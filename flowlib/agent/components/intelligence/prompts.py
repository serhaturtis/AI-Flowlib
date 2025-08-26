"""Prompts for intelligent learning flow."""

from flowlib.resources.models.base import ResourceBase
from flowlib.resources.decorators.decorators import prompt
from flowlib.providers.llm import PromptConfigOverride
from typing import ClassVar


@prompt("content-analysis")
class ContentAnalysisPrompt(ResourceBase):
    """Prompt template for analyzing content to guide knowledge extraction."""
    
    template: ClassVar[str] = """Analyze this content and provide a structured analysis:

Content: {{content}}

{{#context}}
Additional context: {{context}}
{{/context}}

Determine:
1. content_type: What type of content is this? (technical, narrative, structured, conversational, etc.)
2. key_topics: What are the main topics or themes? (list of strings)
3. complexity_level: How complex is the content? (simple, medium, complex)
4. language: What language is this in? (language code)
5. structure_type: How is the content structured? (structured, semi-structured, unstructured)
6. suggested_focus: What should knowledge extraction focus on? (entities, concepts, relationships, patterns)
7. confidence: How confident are you in this analysis? (0.0 to 1.0)

Respond with valid JSON."""
    
    config: ClassVar[PromptConfigOverride] = PromptConfigOverride(
        temperature=0.3,  # Lower temperature for consistent analysis
        max_tokens=1000   # Sufficient for analysis output
    )


@prompt("learning-worthiness-evaluation")
class LearningWorthinessPrompt(ResourceBase):
    """Prompt template for evaluating if content is worth learning from."""
    
    template: ClassVar[str] = """Evaluate if this conversation exchange contains information worth learning or remembering for future interactions.

Content: {{content}}

{{#context}}
Context: {{context}}
{{/context}}

Examples of WORTH LEARNING:
- Factual information or knowledge
- User preferences or personal details  
- Important context or background
- Tasks, goals, or decisions
- Problems and solutions
- Technical information
- Names, dates, or specific details

Examples of NOT WORTH LEARNING:
- Simple greetings (hello, hi, bye)
- Basic pleasantries or small talk
- Repetitive questions already answered
- Generic conversational responses
- Simple acknowledgments"""
    
    config: ClassVar[PromptConfigOverride] = PromptConfigOverride(
        temperature=0.1,  # Very low temperature for consistent evaluation
        max_tokens=200    # Short response needed
    )


@prompt("knowledge-extraction-prompt")
class KnowledgeExtractionPrompt(ResourceBase):
    """Prompt template for extracting knowledge from content."""
    
    template: ClassVar[str] = """Extract knowledge from this {{content_type}} content:

Content: {{content}}

Extract the following (respond with valid JSON):

{{#extract_entities}}
1. entities: List of distinct entities (people, places, things, organizations)
   Each entity should have: name, type, description, confidence

{{/extract_entities}}
{{#extract_concepts}}
2. concepts: List of key concepts and ideas  
   Each concept should have: name, description, category, examples, confidence

{{/extract_concepts}}
{{#extract_relationships}}
3. relationships: List of relationships between entities/concepts
   Each relationship should have: source, target, type, description, confidence

{{/extract_relationships}}
{{#extract_patterns}}
4. patterns: List of patterns or structures identified
   Each pattern should have: name, description, pattern_type, examples, frequency, confidence
   frequency must be one of: "unknown", "rare", "occasional", "common", "frequent"

{{/extract_patterns}}
5. summary: Brief summary of the content's main points
6. confidence: Overall confidence in the extraction (0.0 to 1.0)
7. notes: Any processing notes or observations

Focus on accuracy and relevance. Only include items you're confident about."""
    
    config: ClassVar[PromptConfigOverride] = PromptConfigOverride(
        temperature=0.4,  # Slightly higher for creative extraction
        max_tokens=4000   # More tokens for comprehensive extraction
    )