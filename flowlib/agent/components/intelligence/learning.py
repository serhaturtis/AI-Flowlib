"""Intelligent Learning Flow - Unified Knowledge Extraction.

This module provides a single, intelligent learning flow that replaces the
complex hierarchy of separate learning flows. It intelligently extracts all
types of knowledge in one operation, eliminating orchestration complexity.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from flowlib.core.decorators.decorators import flow, inject
from flowlib.core.interfaces.interfaces import LLMProvider
from .knowledge import (
    Entity, Concept, Relationship, Pattern, KnowledgeSet,
    ContentAnalysis, LearningResult
)

logger = logging.getLogger(__name__)


@flow('intelligent-learning', description='Unified learning from any content')
class IntelligentLearningFlow:
    """Single learning flow that intelligently processes any content.
    
    This flow replaces the complex hierarchy of entity extraction, concept formation,
    pattern recognition, relationship learning, and knowledge integration flows
    with one intelligent operation that adapts based on content.
    
    Key Benefits:
    - Single entry point for all learning operations
    - Intelligent adaptation based on content type
    - No complex strategy determination
    - Unified knowledge extraction in one pass
    - Simple memory storage with graceful degradation
    """
    
    @inject(llm='default-llm')
    async def learn_from_content(
        self, 
        content: str,
        focus_areas: Optional[List[str]] = None,
        context: Optional[str] = None,
        llm: LLMProvider = None
    ) -> LearningResult:
        """Learn from content with intelligent operation selection.
        
        Args:
            content: Text content to learn from
            focus_areas: Optional areas to focus extraction on
            context: Optional context to guide extraction
            llm: LLM provider (injected automatically)
            
        Returns:
            Learning result with extracted knowledge
        """
        start_time = datetime.now()
        
        try:
            # Validate input
            if not content or not content.strip():
                return LearningResult(
                    success=False,
                    knowledge=KnowledgeSet(),
                    message="No content provided for learning",
                    errors=["Empty or missing content"]
                )
            
            logger.info(f"Starting intelligent learning from {len(content)} characters of content")
            
            # Step 1: Intelligent content analysis (no complex strategy determination)
            analysis = await self._analyze_content(content, context, llm)
            logger.debug(f"Content analysis: {analysis.content_type}, complexity: {analysis.complexity_level}")
            
            # Step 2: Unified knowledge extraction (single operation, no orchestration)
            knowledge = await self._extract_knowledge(content, analysis, focus_areas, llm)
            logger.info(f"Extracted {knowledge.total_items} knowledge items")
            
            # Step 3: Store in memory (handled separately, not part of learning flow)
            # This follows separation of concerns - learning extracts, memory stores
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LearningResult(
                success=True,
                knowledge=knowledge,
                processing_time_seconds=processing_time,
                message=f"Successfully learned {knowledge.total_items} items from content"
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Learning failed: {e}")
            
            return LearningResult(
                success=False,
                knowledge=KnowledgeSet(),
                processing_time_seconds=processing_time,
                message="Learning operation failed",
                errors=[str(e)]
            )
    
    async def _analyze_content(
        self, 
        content: str, 
        context: Optional[str], 
        llm: LLMProvider
    ) -> ContentAnalysis:
        """Analyze content to guide knowledge extraction.
        
        Simple analysis that determines content characteristics without
        complex strategy determination logic.
        """
        # Build analysis prompt
        prompt = self._build_analysis_prompt(content, context)
        
        try:
            # Get structured analysis from LLM
            analysis_data = await llm.generate_structured(prompt, dict)
            
            # Convert to ContentAnalysis object with strict validation
            return ContentAnalysis(
                content_type=analysis_data['content_type'] if 'content_type' in analysis_data else 'general',
                key_topics=analysis_data['key_topics'] if 'key_topics' in analysis_data else [],
                complexity_level=analysis_data['complexity_level'] if 'complexity_level' in analysis_data else 'medium',
                language=analysis_data['language'] if 'language' in analysis_data else 'en',
                length_category=self._determine_length_category(content),
                structure_type=analysis_data['structure_type'] if 'structure_type' in analysis_data else 'unstructured',
                suggested_focus=analysis_data['suggested_focus'] if 'suggested_focus' in analysis_data else [],
                confidence=analysis_data['confidence'] if 'confidence' in analysis_data else 0.8
            )
            
        except Exception as e:
            logger.warning(f"Content analysis failed, using defaults: {e}")
            
            # Fallback to simple analysis
            return ContentAnalysis(
                content_type='general',
                key_topics=[],
                complexity_level='medium',
                suggested_focus=['entities', 'concepts'],
                confidence=0.5
            )
    
    async def _extract_knowledge(
        self,
        content: str,
        analysis: ContentAnalysis,
        focus_areas: Optional[List[str]],
        llm: LLMProvider
    ) -> KnowledgeSet:
        """Extract all knowledge types in one intelligent operation.
        
        This replaces the complex orchestration of separate flows with
        a single, focused extraction operation.
        """
        # Build extraction prompt based on analysis
        prompt = self._build_extraction_prompt(content, analysis, focus_areas)
        
        try:
            # Single extraction call for all knowledge types
            extraction_data = await llm.generate_structured(prompt, dict)
            
            # Process extracted data into knowledge objects
            entities = self._process_entities(extraction_data['entities'] if 'entities' in extraction_data else [])
            concepts = self._process_concepts(extraction_data['concepts'] if 'concepts' in extraction_data else [])
            relationships = self._process_relationships(extraction_data['relationships'] if 'relationships' in extraction_data else [])
            patterns = self._process_patterns(extraction_data['patterns'] if 'patterns' in extraction_data else [])
            
            # Create unified knowledge set
            knowledge = KnowledgeSet(
                entities=entities,
                concepts=concepts,
                relationships=relationships,
                patterns=patterns,
                summary=extraction_data['summary'] if 'summary' in extraction_data else '',
                confidence=extraction_data['confidence'] if 'confidence' in extraction_data else 0.8,
                source_content=content[:200] + "..." if len(content) > 200 else content,
                processing_notes=extraction_data['notes'] if 'notes' in extraction_data else []
            )
            
            logger.debug(f"Knowledge extraction complete: {knowledge.get_stats()}")
            return knowledge
            
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            
            # Return empty knowledge set on failure
            return KnowledgeSet(
                summary="Knowledge extraction failed",
                confidence=0.0,
                source_content=content[:100] + "..." if len(content) > 100 else content,
                processing_notes=[f"Extraction error: {str(e)}"]
            )
    
    def _build_analysis_prompt(self, content: str, context: Optional[str]) -> str:
        """Build prompt for content analysis."""
        base_prompt = f"""Analyze this content and provide a structured analysis:

Content: {content}

Determine:
1. content_type: What type of content is this? (technical, narrative, structured, conversational, etc.)
2. key_topics: What are the main topics or themes? (list of strings)
3. complexity_level: How complex is the content? (simple, medium, complex)
4. language: What language is this in? (language code)
5. structure_type: How is the content structured? (structured, semi-structured, unstructured)
6. suggested_focus: What should knowledge extraction focus on? (entities, concepts, relationships, patterns)
7. confidence: How confident are you in this analysis? (0.0 to 1.0)

Respond with valid JSON."""
        
        if context:
            base_prompt += f"\n\nAdditional context: {context}"
        
        return base_prompt
    
    def _build_extraction_prompt(
        self, 
        content: str, 
        analysis: ContentAnalysis, 
        focus_areas: Optional[List[str]]
    ) -> str:
        """Build prompt for knowledge extraction."""
        # Determine what to extract based on analysis and focus areas
        extract_entities = not focus_areas or 'entities' in focus_areas
        extract_concepts = not focus_areas or 'concepts' in focus_areas
        extract_relationships = not focus_areas or 'relationships' in focus_areas
        extract_patterns = not focus_areas or 'patterns' in focus_areas
        
        prompt = f"""Extract knowledge from this {analysis.content_type} content:

Content: {content}

Extract the following (respond with valid JSON):

"""
        
        if extract_entities:
            prompt += """1. entities: List of distinct entities (people, places, things, organizations)
   Each entity should have: name, type, description, confidence

"""
        
        if extract_concepts:
            prompt += """2. concepts: List of key concepts and ideas  
   Each concept should have: name, description, category, examples, confidence

"""
        
        if extract_relationships:
            prompt += """3. relationships: List of relationships between entities/concepts
   Each relationship should have: source, target, type, description, confidence

"""
        
        if extract_patterns:
            prompt += """4. patterns: List of patterns or structures identified
   Each pattern should have: name, description, pattern_type, examples, confidence

"""
        
        prompt += """5. summary: Brief summary of the content's main points
6. confidence: Overall confidence in the extraction (0.0 to 1.0)
7. notes: Any processing notes or observations

Focus on accuracy and relevance. Only include items you're confident about."""
        
        return prompt
    
    def _determine_length_category(self, content: str) -> str:
        """Determine content length category."""
        char_count = len(content)
        if char_count < 500:
            return "short"
        elif char_count < 2000:
            return "medium"
        else:
            return "long"
    
    def _process_entities(self, entities_data: List[Dict[str, Any]]) -> List[Entity]:
        """Process extracted entity data into Entity objects."""
        entities = []
        for data in entities_data:
            try:
                if 'name' not in data:
                    logger.warning("Entity data missing required 'name' field, skipping")
                    continue
                entity = Entity(
                    name=data['name'],
                    type=data['type'] if 'type' in data else 'unknown',
                    description=data['description'] if 'description' in data else None,
                    confidence=float(data['confidence'] if 'confidence' in data else 0.8),
                    properties=data['properties'] if 'properties' in data else {},
                    aliases=data['aliases'] if 'aliases' in data else []
                )
                entities.append(entity)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid entity data: {e}")
        
        return entities
    
    def _process_concepts(self, concepts_data: List[Dict[str, Any]]) -> List[Concept]:
        """Process extracted concept data into Concept objects."""
        concepts = []
        for data in concepts_data:
            try:
                if 'name' not in data:
                    logger.warning("Concept data missing required 'name' field, skipping")
                    continue
                concept = Concept(
                    name=data['name'],
                    description=data['description'] if 'description' in data else '',
                    category=data['category'] if 'category' in data else None,
                    examples=data['examples'] if 'examples' in data else [],
                    related_concepts=data['related_concepts'] if 'related_concepts' in data else [],
                    confidence=float(data['confidence'] if 'confidence' in data else 0.8),
                    abstraction_level=int(data['abstraction_level'] if 'abstraction_level' in data else 1)
                )
                concepts.append(concept)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid concept data: {e}")
        
        return concepts
    
    def _process_relationships(self, relationships_data: List[Dict[str, Any]]) -> List[Relationship]:
        """Process extracted relationship data into Relationship objects."""
        relationships = []
        for data in relationships_data:
            try:
                if 'source' not in data or 'target' not in data:
                    logger.warning("Relationship data missing required 'source' or 'target' field, skipping")
                    continue
                relationship = Relationship(
                    source=data['source'],
                    target=data['target'],
                    type=data['type'] if 'type' in data else 'related_to',
                    description=data['description'] if 'description' in data else None,
                    confidence=float(data['confidence'] if 'confidence' in data else 0.8),
                    bidirectional=bool(data['bidirectional'] if 'bidirectional' in data else False),
                    strength=data['strength'] if 'strength' in data else 'medium'
                )
                relationships.append(relationship)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid relationship data: {e}")
        
        return relationships
    
    def _process_patterns(self, patterns_data: List[Dict[str, Any]]) -> List[Pattern]:
        """Process extracted pattern data into Pattern objects."""
        patterns = []
        for data in patterns_data:
            try:
                if 'name' not in data:
                    logger.warning("Pattern data missing required 'name' field, skipping")
                    continue
                pattern = Pattern(
                    name=data['name'],
                    description=data['description'] if 'description' in data else '',
                    pattern_type=data['pattern_type'] if 'pattern_type' in data else 'general',
                    examples=data['examples'] if 'examples' in data else [],
                    variables=data['variables'] if 'variables' in data else [],
                    confidence=float(data['confidence'] if 'confidence' in data else 0.8),
                    frequency=data['frequency'] if 'frequency' in data else 'unknown'
                )
                patterns.append(pattern)
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid pattern data: {e}")
        
        return patterns


# Simple API function for easy usage
async def learn_from_text(
    content: str,
    focus_areas: Optional[List[str]] = None,
    context: Optional[str] = None
) -> LearningResult:
    """Simple API for learning from text content.
    
    This function provides a dead-simple interface for learning operations,
    hiding the complexity of flow instantiation and execution.
    
    Args:
        content: Text content to learn from
        focus_areas: Optional areas to focus on ('entities', 'concepts', 'relationships', 'patterns')
        context: Optional context to guide learning
        
    Returns:
        Learning result with extracted knowledge
        
    Example:
        result = await learn_from_text("AI is transforming software development...")
        if result.success:
            print(f"Learned {result.knowledge.total_items} items")
    """
    flow = IntelligentLearningFlow()
    return await flow.learn_from_content(content, focus_areas, context)