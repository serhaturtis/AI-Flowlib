"""Tests for intelligent learning flow."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List

from flowlib.agent.components.intelligence.learning import (
    IntelligentLearningFlow,
    learn_from_text
)
from flowlib.agent.components.intelligence.knowledge import (
    Entity, Concept, Relationship, Pattern, KnowledgeSet,
    ContentAnalysis, LearningResult, ConfidenceLevel
)


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self):
        self.generate_structured_calls = []
        self.should_fail = False
        self.fail_operation = None
        self.responses = {}
        self.default_analysis = {
            'content_type': 'technical',
            'key_topics': ['testing', 'software'],
            'complexity_level': 'medium',
            'language': 'en',
            'structure_type': 'unstructured',
            'suggested_focus': ['entities', 'concepts'],
            'confidence': 0.8
        }
        self.default_extraction = {
            'entities': [
                {'name': 'Python', 'type': 'technology', 'description': 'Programming language', 'confidence': 0.9}
            ],
            'concepts': [
                {'name': 'Testing', 'description': 'Software quality assurance', 'confidence': 0.8}
            ],
            'relationships': [
                {'source': 'Python', 'target': 'Testing', 'type': 'used_for', 'confidence': 0.7}
            ],
            'patterns': [
                {'name': 'Test Pattern', 'description': 'Common testing approach', 'pattern_type': 'behavioral', 'confidence': 0.6}
            ],
            'summary': 'Content about Python testing',
            'confidence': 0.8,
            'notes': ['Extracted successfully']
        }
    
    async def generate_structured(self, prompt: str, output_type: type) -> Dict[str, Any]:
        """Mock structured generation."""
        self.generate_structured_calls.append((prompt, output_type))
        
        if self.should_fail and self.fail_operation == "generate_structured":
            raise RuntimeError("Mock LLM failure")
        
        # Determine response based on prompt content
        if "analyze this content" in prompt.lower():
            return self.responses.get('analysis', self.default_analysis)
        elif "extract knowledge" in prompt.lower():
            return self.responses.get('extraction', self.default_extraction)
        else:
            return {}


class TestIntelligentLearningFlow:
    """Test IntelligentLearningFlow class."""
    
    @pytest.fixture
    def flow(self):
        """Create learning flow instance."""
        return IntelligentLearningFlow()
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        return MockLLMProvider()
    
    @pytest.fixture
    def sample_content(self):
        """Create sample content for testing."""
        return "Python is a powerful programming language used for testing applications."
    
    @pytest.mark.asyncio
    async def test_learn_from_content_success(self, flow, mock_llm, sample_content):
        """Test successful learning from content."""
        result = await flow.learn_from_content(
            content=sample_content,
            llm=mock_llm
        )
        
        assert result.success is True
        assert result.knowledge.total_items == 4  # 1 entity + 1 concept + 1 relationship + 1 pattern
        assert result.processing_time_seconds > 0
        assert "Successfully learned 4 items" in result.message
        assert len(result.errors) == 0
        
        # Verify LLM was called for analysis and extraction
        assert len(mock_llm.generate_structured_calls) == 2
    
    @pytest.mark.asyncio
    async def test_learn_from_content_empty_content(self, flow, mock_llm):
        """Test learning from empty content."""
        result = await flow.learn_from_content(
            content="",
            llm=mock_llm
        )
        
        assert result.success is False
        assert result.knowledge.total_items == 0
        assert "No content provided" in result.message
        assert "Empty or missing content" in result.errors
        
        # Should not call LLM with empty content
        assert len(mock_llm.generate_structured_calls) == 0
    
    @pytest.mark.asyncio
    async def test_learn_from_content_whitespace_only(self, flow, mock_llm):
        """Test learning from whitespace-only content."""
        result = await flow.learn_from_content(
            content="   \n\t   ",
            llm=mock_llm
        )
        
        assert result.success is False
        assert "No content provided" in result.message
    
    @pytest.mark.asyncio
    async def test_learn_from_content_with_focus_areas(self, flow, mock_llm, sample_content):
        """Test learning with specific focus areas."""
        result = await flow.learn_from_content(
            content=sample_content,
            focus_areas=['entities', 'concepts'],
            llm=mock_llm
        )
        
        assert result.success is True
        # Verify extraction prompt includes focus areas
        extraction_prompt = mock_llm.generate_structured_calls[1][0]
        assert 'entities' in extraction_prompt
        assert 'concepts' in extraction_prompt
    
    @pytest.mark.asyncio
    async def test_learn_from_content_with_context(self, flow, mock_llm, sample_content):
        """Test learning with additional context."""
        context = "This is about software development best practices"
        result = await flow.learn_from_content(
            content=sample_content,
            context=context,
            llm=mock_llm
        )
        
        assert result.success is True
        # Verify analysis prompt includes context
        analysis_prompt = mock_llm.generate_structured_calls[0][0]
        assert context in analysis_prompt
    
    @pytest.mark.asyncio
    async def test_learn_from_content_llm_failure(self, flow, mock_llm, sample_content):
        """Test learning with LLM failure - flow is resilient and continues with fallbacks."""
        mock_llm.should_fail = True
        mock_llm.fail_operation = "generate_structured"
        
        result = await flow.learn_from_content(
            content=sample_content,
            llm=mock_llm
        )
        
        # Flow continues with fallback behavior
        assert result.success is True  # Still succeeds due to graceful degradation
        assert result.knowledge.total_items == 0  # But no knowledge extracted
        assert "Successfully learned 0 items" in result.message
        assert result.knowledge.summary == "Knowledge extraction failed"
        assert result.knowledge.confidence == 0.0
        assert "Extraction error: Mock LLM failure" in result.knowledge.processing_notes
    
    @pytest.mark.asyncio
    async def test_analyze_content_success(self, flow, mock_llm, sample_content):
        """Test successful content analysis."""
        analysis = await flow._analyze_content(sample_content, None, mock_llm)
        
        assert analysis.content_type == "technical"
        assert "testing" in analysis.key_topics
        assert analysis.complexity_level == "medium"
        assert analysis.language == "en"
        assert analysis.structure_type == "unstructured"
        assert "entities" in analysis.suggested_focus
        assert analysis.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_analyze_content_with_context(self, flow, mock_llm, sample_content):
        """Test content analysis with context."""
        context = "Educational material"
        analysis = await flow._analyze_content(sample_content, context, mock_llm)
        
        # Verify context was included in prompt
        analysis_prompt = mock_llm.generate_structured_calls[0][0]
        assert context in analysis_prompt
    
    @pytest.mark.asyncio
    async def test_analyze_content_llm_failure(self, flow, mock_llm, sample_content):
        """Test content analysis with LLM failure."""
        mock_llm.should_fail = True
        mock_llm.fail_operation = "generate_structured"
        
        analysis = await flow._analyze_content(sample_content, None, mock_llm)
        
        # Should return fallback analysis
        assert analysis.content_type == "general"
        assert analysis.complexity_level == "medium"
        assert analysis.confidence == 0.5
        assert "entities" in analysis.suggested_focus
    
    @pytest.mark.asyncio
    async def test_extract_knowledge_success(self, flow, mock_llm, sample_content):
        """Test successful knowledge extraction."""
        analysis = ContentAnalysis(content_type="technical")
        
        knowledge = await flow._extract_knowledge(sample_content, analysis, None, mock_llm)
        
        assert knowledge.total_items == 4
        assert len(knowledge.entities) == 1
        assert len(knowledge.concepts) == 1
        assert len(knowledge.relationships) == 1
        assert len(knowledge.patterns) == 1
        assert knowledge.summary == "Content about Python testing"
        assert knowledge.confidence == 0.8
        assert "Extracted successfully" in knowledge.processing_notes
    
    @pytest.mark.asyncio
    async def test_extract_knowledge_with_focus_areas(self, flow, mock_llm, sample_content):
        """Test knowledge extraction with focus areas."""
        analysis = ContentAnalysis(content_type="technical")
        focus_areas = ['entities', 'concepts']
        
        knowledge = await flow._extract_knowledge(sample_content, analysis, focus_areas, mock_llm)
        
        # Verify extraction prompt was built with focus areas
        extraction_prompt = mock_llm.generate_structured_calls[0][0]
        assert 'entities' in extraction_prompt
        assert 'concepts' in extraction_prompt
        # Should still extract all types based on LLM response
        assert knowledge.total_items == 4
    
    @pytest.mark.asyncio
    async def test_extract_knowledge_llm_failure(self, flow, mock_llm, sample_content):
        """Test knowledge extraction with LLM failure."""
        analysis = ContentAnalysis(content_type="technical")
        mock_llm.should_fail = True
        mock_llm.fail_operation = "generate_structured"
        
        knowledge = await flow._extract_knowledge(sample_content, analysis, None, mock_llm)
        
        # Should return empty knowledge set
        assert knowledge.total_items == 0
        assert knowledge.summary == "Knowledge extraction failed"
        assert knowledge.confidence == 0.0
        assert len(knowledge.processing_notes) == 1
        assert "Extraction error" in knowledge.processing_notes[0]
    
    def test_build_analysis_prompt(self, flow):
        """Test analysis prompt building."""
        content = "Test content"
        prompt = flow._build_analysis_prompt(content, None)
        
        assert content in prompt
        assert "content_type" in prompt
        assert "complexity_level" in prompt
        assert "suggested_focus" in prompt
        assert "JSON" in prompt
    
    def test_build_analysis_prompt_with_context(self, flow):
        """Test analysis prompt building with context."""
        content = "Test content"
        context = "Educational context"
        prompt = flow._build_analysis_prompt(content, context)
        
        assert content in prompt
        assert context in prompt
    
    def test_build_extraction_prompt_all_types(self, flow):
        """Test extraction prompt building with all knowledge types."""
        content = "Test content"
        analysis = ContentAnalysis(content_type="technical")
        
        prompt = flow._build_extraction_prompt(content, analysis, None)
        
        assert content in prompt
        assert "technical" in prompt
        assert "entities" in prompt
        assert "concepts" in prompt
        assert "relationships" in prompt
        assert "patterns" in prompt
        assert "summary" in prompt
    
    def test_build_extraction_prompt_focused(self, flow):
        """Test extraction prompt building with focus areas."""
        content = "Test content"
        analysis = ContentAnalysis(content_type="technical")
        focus_areas = ['entities', 'concepts']
        
        prompt = flow._build_extraction_prompt(content, analysis, focus_areas)
        
        assert "entities" in prompt
        assert "concepts" in prompt
        # Verify focus is applied in prompt generation logic
        # (relationships and patterns may not be in prompt with focus_areas)
    
    def test_determine_length_category(self, flow):
        """Test content length categorization."""
        short_content = "Short text"
        medium_content = "Medium " * 100  # ~700 chars
        long_content = "Long " * 500  # ~2500 chars
        
        assert flow._determine_length_category(short_content) == "short"
        assert flow._determine_length_category(medium_content) == "medium"
        assert flow._determine_length_category(long_content) == "long"
    
    def test_process_entities_success(self, flow):
        """Test successful entity processing."""
        entities_data = [
            {
                'name': 'Python',
                'type': 'technology',
                'description': 'Programming language',
                'confidence': 0.9,
                'properties': {'category': 'language'},
                'aliases': ['py']
            },
            {
                'name': 'Django',
                'type': 'framework',
                'confidence': 0.8
            }
        ]
        
        entities = flow._process_entities(entities_data)
        
        assert len(entities) == 2
        assert entities[0].name == "Python"
        assert entities[0].type == "technology"
        assert entities[0].confidence == 0.9
        assert entities[0].properties == {'category': 'language'}
        assert entities[0].aliases == ['py']
        
        assert entities[1].name == "Django"
        assert entities[1].type == "framework"
        assert entities[1].confidence == 0.8
    
    def test_process_entities_invalid_data(self, flow):
        """Test entity processing with invalid data."""
        entities_data = [
            {'name': 'Valid Entity', 'type': 'test'},
            {'name': '', 'type': 'test'},  # Invalid: empty name
            {'name': 'Invalid Confidence', 'type': 'test', 'confidence': 2.0},  # Invalid: confidence > 1.0
            {'invalid': 'structure'}  # Invalid: missing required fields
        ]
        
        entities = flow._process_entities(entities_data)
        
        # Should only include valid entity
        assert len(entities) == 1
        assert entities[0].name == "Valid Entity"
    
    def test_process_concepts_success(self, flow):
        """Test successful concept processing."""
        concepts_data = [
            {
                'name': 'Testing',
                'description': 'Quality assurance',
                'category': 'software',
                'examples': ['unit tests', 'integration tests'],
                'confidence': 0.8,
                'abstraction_level': 2
            }
        ]
        
        concepts = flow._process_concepts(concepts_data)
        
        assert len(concepts) == 1
        assert concepts[0].name == "Testing"
        assert concepts[0].description == "Quality assurance"
        assert concepts[0].category == "software"
        assert concepts[0].examples == ['unit tests', 'integration tests']
        assert concepts[0].confidence == 0.8
        assert concepts[0].abstraction_level == 2
    
    def test_process_concepts_invalid_data(self, flow):
        """Test concept processing with invalid data."""
        concepts_data = [
            {'name': 'Valid Concept', 'description': 'Valid description'},
            {'name': '', 'description': 'Invalid'},  # Invalid: empty name
            {'name': 'Invalid', 'description': ''},  # Invalid: empty description
            {'name': 'Invalid Level', 'description': 'Valid', 'abstraction_level': 10}  # Invalid: level > 5
        ]
        
        concepts = flow._process_concepts(concepts_data)
        
        # Should only include valid concept
        assert len(concepts) == 1
        assert concepts[0].name == "Valid Concept"
    
    def test_process_relationships_success(self, flow):
        """Test successful relationship processing."""
        relationships_data = [
            {
                'source': 'Python',
                'target': 'Testing',
                'type': 'used_for',
                'description': 'Python is used for testing',
                'confidence': 0.9,
                'bidirectional': False,
                'strength': 'strong'
            }
        ]
        
        relationships = flow._process_relationships(relationships_data)
        
        assert len(relationships) == 1
        assert relationships[0].source == "Python"
        assert relationships[0].target == "Testing"
        assert relationships[0].type == "used_for"
        assert relationships[0].confidence == 0.9
        assert relationships[0].bidirectional is False
        assert relationships[0].strength == "strong"
    
    def test_process_relationships_invalid_data(self, flow):
        """Test relationship processing with invalid data."""
        relationships_data = [
            {'source': 'Valid', 'target': 'Valid', 'type': 'valid_type'},
            {'source': '', 'target': 'Invalid', 'type': 'type'},  # Invalid: empty source
            {'source': 'Invalid', 'target': '', 'type': 'type'},  # Invalid: empty target
            {'source': 'Invalid', 'target': 'Invalid', 'type': ''},  # Invalid: empty type
            {'source': 'Invalid', 'target': 'Invalid', 'type': 'type', 'strength': 'invalid'}  # Invalid: bad strength
        ]
        
        relationships = flow._process_relationships(relationships_data)
        
        # Should only include valid relationship
        assert len(relationships) == 1
        assert relationships[0].source == "Valid"
    
    def test_process_patterns_success(self, flow):
        """Test successful pattern processing."""
        patterns_data = [
            {
                'name': 'Test Pattern',
                'description': 'Common testing pattern',
                'pattern_type': 'behavioral',
                'examples': ['setup-test-teardown'],
                'variables': ['setup', 'test', 'teardown'],
                'confidence': 0.7,
                'frequency': 'common'
            }
        ]
        
        patterns = flow._process_patterns(patterns_data)
        
        assert len(patterns) == 1
        assert patterns[0].name == "Test Pattern"
        assert patterns[0].description == "Common testing pattern"
        assert patterns[0].pattern_type == "behavioral"
        assert patterns[0].examples == ['setup-test-teardown']
        assert patterns[0].variables == ['setup', 'test', 'teardown']
        assert patterns[0].confidence == 0.7
        assert patterns[0].frequency == "common"
    
    def test_process_patterns_invalid_data(self, flow):
        """Test pattern processing with invalid data."""
        patterns_data = [
            {'name': 'Valid Pattern', 'description': 'Valid description'},
            {'name': '', 'description': 'Invalid'},  # Invalid: empty name
            {'name': 'Invalid', 'description': ''},  # Invalid: empty description
            {'name': 'Invalid', 'description': 'Valid', 'frequency': 'invalid'}  # Invalid: bad frequency
        ]
        
        patterns = flow._process_patterns(patterns_data)
        
        # Should only include valid pattern
        assert len(patterns) == 1
        assert patterns[0].name == "Valid Pattern"


class TestLearnFromTextAPI:
    """Test learn_from_text API function."""
    
    @pytest.mark.asyncio
    async def test_learn_from_text_success(self):
        """Test successful learn_from_text API call."""
        content = "Python is great for testing"
        
        with patch.object(IntelligentLearningFlow, 'learn_from_content') as mock_learn:
            mock_result = LearningResult(
                success=True,
                knowledge=KnowledgeSet(summary="Test result")
            )
            mock_learn.return_value = mock_result
            
            result = await learn_from_text(content)
            
            assert result == mock_result
            mock_learn.assert_called_once_with(content, None, None)
    
    @pytest.mark.asyncio
    async def test_learn_from_text_with_parameters(self):
        """Test learn_from_text with focus areas and context."""
        content = "Python testing content"
        focus_areas = ['entities', 'concepts']
        context = "Educational context"
        
        with patch.object(IntelligentLearningFlow, 'learn_from_content') as mock_learn:
            mock_result = LearningResult(
                success=True,
                knowledge=KnowledgeSet()
            )
            mock_learn.return_value = mock_result
            
            result = await learn_from_text(content, focus_areas, context)
            
            assert result == mock_result
            mock_learn.assert_called_once_with(content, focus_areas, context)


class TestIntelligentLearningFlowIntegration:
    """Integration tests for IntelligentLearningFlow."""
    
    @pytest.fixture
    def flow(self):
        """Create learning flow instance."""
        return IntelligentLearningFlow()
    
    @pytest.mark.asyncio
    async def test_full_learning_workflow(self, flow):
        """Test complete learning workflow with mock LLM."""
        content = "Artificial Intelligence is transforming software development through automated testing and intelligent code generation."
        
        mock_llm = MockLLMProvider()
        mock_llm.responses['analysis'] = {
            'content_type': 'technical',
            'key_topics': ['AI', 'software development', 'testing'],
            'complexity_level': 'medium',
            'language': 'en',
            'structure_type': 'unstructured',
            'suggested_focus': ['entities', 'concepts', 'relationships'],
            'confidence': 0.9
        }
        mock_llm.responses['extraction'] = {
            'entities': [
                {'name': 'Artificial Intelligence', 'type': 'technology', 'description': 'AI technology', 'confidence': 0.9},
                {'name': 'Software Development', 'type': 'process', 'description': 'Development process', 'confidence': 0.8}
            ],
            'concepts': [
                {'name': 'Automation', 'description': 'Automated processes', 'category': 'technical', 'confidence': 0.7},
                {'name': 'Code Generation', 'description': 'Automated code creation', 'confidence': 0.6}
            ],
            'relationships': [
                {'source': 'AI', 'target': 'Software Development', 'type': 'transforms', 'confidence': 0.8},
                {'source': 'AI', 'target': 'Testing', 'type': 'automates', 'confidence': 0.7}
            ],
            'patterns': [
                {'name': 'AI Integration Pattern', 'description': 'Pattern of AI integration', 'pattern_type': 'architectural', 'confidence': 0.6}
            ],
            'summary': 'AI is transforming software development',
            'confidence': 0.8,
            'notes': ['Successfully extracted knowledge']
        }
        
        result = await flow.learn_from_content(content, llm=mock_llm)
        
        assert result.success is True
        assert result.knowledge.total_items == 7  # 2 entities + 2 concepts + 2 relationships + 1 pattern
        assert result.processing_time_seconds > 0
        assert len(mock_llm.generate_structured_calls) == 2
        
        # Verify knowledge content
        assert len(result.knowledge.entities) == 2
        assert len(result.knowledge.concepts) == 2
        assert len(result.knowledge.relationships) == 2
        assert len(result.knowledge.patterns) == 1
        assert result.knowledge.summary == "AI is transforming software development"
        assert result.knowledge.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_partial_extraction_failure(self, flow):
        """Test learning with partial extraction failures."""
        content = "Test content with mixed valid and invalid data"
        
        mock_llm = MockLLMProvider()
        mock_llm.responses['extraction'] = {
            'entities': [
                {'name': 'Valid Entity', 'type': 'test', 'confidence': 0.8},
                {'name': '', 'type': 'invalid'}  # Invalid entity
            ],
            'concepts': [
                {'name': 'Valid Concept', 'description': 'Valid', 'confidence': 0.7},
                {'name': 'Invalid', 'description': ''}  # Invalid concept
            ],
            'relationships': [],
            'patterns': [],
            'summary': 'Mixed extraction results',
            'confidence': 0.6
        }
        
        result = await flow.learn_from_content(content, llm=mock_llm)
        
        assert result.success is True
        assert result.knowledge.total_items == 2  # Only valid items
        assert len(result.knowledge.entities) == 1
        assert len(result.knowledge.concepts) == 1
        assert result.knowledge.entities[0].name == "Valid Entity"
        assert result.knowledge.concepts[0].name == "Valid Concept"
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, flow):
        """Test that processing time is tracked correctly."""
        content = "Test content"
        mock_llm = MockLLMProvider()
        
        start_time = datetime.now()
        result = await flow.learn_from_content(content, llm=mock_llm)
        end_time = datetime.now()
        
        elapsed_time = (end_time - start_time).total_seconds()
        
        assert result.processing_time_seconds > 0
        assert result.processing_time_seconds <= elapsed_time
    
    @pytest.mark.asyncio
    async def test_content_length_handling(self, flow):
        """Test handling of different content lengths."""
        mock_llm = MockLLMProvider()
        
        # Test short content
        short_content = "AI"
        result = await flow.learn_from_content(short_content, llm=mock_llm)
        assert result.success is True
        
        # Test long content
        long_content = "AI " * 1000  # Very long content
        result = await flow.learn_from_content(long_content, llm=mock_llm)
        assert result.success is True
        
        # Verify source content is truncated for long content
        assert len(result.knowledge.source_content) <= 203  # 200 + "..."