"""Intelligent Learning Flow - Unified Knowledge Extraction.

This module provides a single, intelligent learning flow that replaces the
complex hierarchy of separate learning flows. It intelligently extracts all
types of knowledge in one operation, eliminating orchestration complexity.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import Field

from flowlib.core.decorators.decorators import flow
from flowlib.flows.decorators.decorators import pipeline
from flowlib.core.interfaces.interfaces import LLMProvider
from flowlib.core.models import StrictBaseModel
from flowlib.providers.core.registry import provider_registry
from flowlib.resources.registry.registry import resource_registry
from .knowledge import (
    Entity, Concept, Relationship, Pattern, KnowledgeSet,
    ContentAnalysis, LearningResult
)


class LearningContentAnalysisOutput(StrictBaseModel):
    """Structured output for learning-specific content analysis."""
    content_type: str = Field(description="Type of content: technical, narrative, structured, conversational")
    key_topics: List[str] = Field(description="Main topics identified in the content")
    complexity_level: str = Field(description="Complexity level: simple, medium, complex")
    language: str = Field(default="en", description="Language of the content")
    length_category: str = Field(description="Length category: short, medium, long")
    structure_type: str = Field(description="Structure type: structured, semi-structured, unstructured")
    suggested_focus: List[str] = Field(description="Suggested areas to focus extraction on")
    confidence: float = Field(description="Confidence score for the analysis")


class LearningKnowledgeExtractionOutput(StrictBaseModel):
    """Structured output for learning-specific knowledge extraction."""
    entities: List[Dict[str, Any]] = Field(description="Extracted entities with name, type, description, confidence")
    concepts: List[Dict[str, Any]] = Field(description="Extracted concepts with name, definition, category, confidence")  
    relationships: List[Dict[str, Any]] = Field(description="Extracted relationships with source, target, type, description, confidence")
    patterns: List[Dict[str, Any]] = Field(description="Extracted patterns with name, description, frequency, confidence")

logger = logging.getLogger(__name__)


class LearningInput(StrictBaseModel):
    """Input model for intelligent learning flow."""
    content: str = Field(..., min_length=1, description="Text content to learn from")
    focus_areas: Optional[List[str]] = Field(default=None, description="Optional areas to focus extraction on")
    context: Optional[str] = Field(default=None, description="Optional context to guide extraction")


class LearningOutput(StrictBaseModel):
    """Output model for intelligent learning flow."""
    success: bool = Field(..., description="Whether learning was successful")
    knowledge: Dict[str, Any] = Field(default_factory=dict, description="Extracted knowledge as dict")
    processing_time_seconds: float = Field(default=0.0, description="Processing time in seconds")
    message: str = Field(default="", description="Result message")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


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
    
    @pipeline(input_model=LearningInput, output_model=LearningOutput)
    async def run_pipeline(
        self, 
        input_data: LearningInput
    ) -> LearningOutput:
        """Learn from content with intelligent operation selection.
        
        Args:
            input_data: Learning input with content and options
            
        Returns:
            Learning output with extracted knowledge
        """
        start_time = datetime.now()
        
        try:
            # Get LLM provider using config-driven approach
            llm = await provider_registry.get_by_config("default-llm")
            
            # Extract inputs
            content = input_data.content
            focus_areas = input_data.focus_areas
            context = input_data.context
            
            # Validate input
            if not content or not content.strip():
                return LearningOutput(
                    success=False,
                    knowledge={},
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
            
            return LearningOutput(
                success=True,
                knowledge=knowledge.to_dict() if hasattr(knowledge, 'to_dict') else knowledge.__dict__,
                processing_time_seconds=processing_time,
                message=f"Successfully learned {knowledge.total_items} items from content"
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Learning failed: {e}")
            
            return LearningOutput(
                success=False,
                knowledge={},
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
        # Get analysis prompt from registry
        prompt = resource_registry.get("content-analysis")
        
        # Prepare prompt variables
        prompt_vars = {
            "content": content,
            "context": context
        }
        
        try:
            # Debug: Log what we're about to do
            logger.error(f"DEBUG: About to call generate_structured with model_name=default-model, prompt_vars={prompt_vars}")
            
            # Get structured analysis from LLM
            analysis_data = await llm.generate_structured(
                prompt=prompt, 
                output_type=LearningContentAnalysisOutput, 
                model_name="default-model", 
                prompt_variables=prompt_vars
            )
            
            # Debug: Log what we got back
            logger.error(f"DEBUG: Structured analysis succeeded: {analysis_data}")
            
            # Convert to ContentAnalysis object with strict validation
            return ContentAnalysis(
                content_type=analysis_data.content_type,
                key_topics=analysis_data.key_topics,
                complexity_level=analysis_data.complexity_level,
                language=analysis_data.language,
                length_category=self._determine_length_category(content),
                structure_type=analysis_data.structure_type,
                suggested_focus=analysis_data.suggested_focus,
                confidence=analysis_data.confidence
            )
            
        except Exception as e:
            logger.error(f"DEBUG: Content analysis failed with error: {e}")
            import traceback
            logger.error(f"DEBUG: Full traceback: {traceback.format_exc()}")
            logger.error(f"Content analysis failed: {e}")
            raise
    
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
        # Get extraction prompt from registry
        prompt = resource_registry.get("knowledge-extraction-prompt")
        
        # Determine what to extract based on focus areas
        prompt_vars = {
            "content": content,
            "content_type": analysis.content_type,
            "extract_entities": not focus_areas or 'entities' in focus_areas,
            "extract_concepts": not focus_areas or 'concepts' in focus_areas,
            "extract_relationships": not focus_areas or 'relationships' in focus_areas,
            "extract_patterns": not focus_areas or 'patterns' in focus_areas
        }
        
        try:
            # Single extraction call for all knowledge types
            extraction_data = await llm.generate_structured(
                prompt=prompt, 
                output_type=LearningKnowledgeExtractionOutput, 
                model_name="default-model", 
                prompt_variables=prompt_vars
            )
            
            # Process extracted data into knowledge objects
            entities = self._process_entities(extraction_data.entities)
            concepts = self._process_concepts(extraction_data.concepts)
            relationships = self._process_relationships(extraction_data.relationships)
            patterns = self._process_patterns(extraction_data.patterns)
            
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
            raise
    
    
    
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
) -> LearningOutput:
    """Simple API for learning from text content.
    
    This function provides a dead-simple interface for learning operations,
    hiding the complexity of flow instantiation and execution.
    
    Args:
        content: Text content to learn from
        focus_areas: Optional areas to focus on ('entities', 'concepts', 'relationships', 'patterns')
        context: Optional context to guide learning
        
    Returns:
        Learning output with extracted knowledge
        
    Example:
        result = await learn_from_text("AI is transforming software development...")
        if result.success:
            print(f"Learned items successfully")
    """
    flow = IntelligentLearningFlow()
    input_data = LearningInput(
        content=content,
        focus_areas=focus_areas,
        context=context
    )
    return await flow.run_pipeline(input_data)