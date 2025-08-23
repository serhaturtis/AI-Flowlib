"""Tests for knowledge retrieval flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from flowlib.agent.components.knowledge_flows.knowledge_retrieval import KnowledgeRetrievalFlow
from flowlib.agent.components.knowledge_flows.models import (
    KnowledgeRetrievalInput,
    KnowledgeRetrievalOutput,
    RetrievedKnowledge
)
from flowlib.flows.decorators.decorators import flow, pipeline


class TestKnowledgeRetrievalFlow:
    """Test KnowledgeRetrievalFlow class."""
    
    def test_knowledge_retrieval_flow_decoration(self):
        """Test that KnowledgeRetrievalFlow is properly decorated."""
        # Check flow decorator attributes
        assert hasattr(KnowledgeRetrievalFlow, '__flow_name__')
        assert hasattr(KnowledgeRetrievalFlow, '__flow_description__')
        assert hasattr(KnowledgeRetrievalFlow, '__is_infrastructure__')
        
        assert KnowledgeRetrievalFlow.__flow_name__ == "knowledge-retrieval"
        assert "Retrieve knowledge from plugins" in KnowledgeRetrievalFlow.__flow_description__
        assert KnowledgeRetrievalFlow.__is_infrastructure__ is False
    
    def test_knowledge_retrieval_flow_pipeline_decoration(self):
        """Test that run_pipeline method is properly decorated."""
        # Check that run_pipeline has pipeline decoration
        assert hasattr(KnowledgeRetrievalFlow.run_pipeline, '__pipeline_input_model__')
        assert hasattr(KnowledgeRetrievalFlow.run_pipeline, '__pipeline_output_model__')
        
        assert KnowledgeRetrievalFlow.run_pipeline.__pipeline_input_model__ == KnowledgeRetrievalInput
        assert KnowledgeRetrievalFlow.run_pipeline.__pipeline_output_model__ == KnowledgeRetrievalOutput
    
    def test_knowledge_retrieval_flow_instantiation(self):
        """Test creating KnowledgeRetrievalFlow instance."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        assert isinstance(flow, KnowledgeRetrievalFlow)
        assert hasattr(flow, 'run_pipeline')
        assert hasattr(flow, '_search_agent_memory')
        assert hasattr(flow, '_search_knowledge_plugins')
        assert hasattr(flow, '_synthesize_results')
    
    @pytest.mark.asyncio
    async def test_run_pipeline_successful_retrieval(self):
        """Test successful knowledge retrieval from multiple sources."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(
            query="What is machine learning?",
            domain="ai",
            max_results=5,
            include_plugins=True,
            include_memory=True
        )
        
        # Mock the internal search methods
        memory_results = [
            RetrievedKnowledge(
                content="Machine learning is a subset of AI",
                source="agent_memory",
                domain="ai",
                confidence=0.9
            )
        ]
        
        plugin_results = [
            RetrievedKnowledge(
                content="ML algorithms learn from data",
                source="plugin:ai_knowledge",
                domain="ai",
                confidence=0.85
            )
        ]
        
        with patch.object(flow, '_search_agent_memory', return_value=memory_results) as mock_memory, \
             patch.object(flow, '_search_knowledge_plugins', return_value=plugin_results) as mock_plugins, \
             patch.object(flow, '_synthesize_results', return_value=memory_results + plugin_results) as mock_synthesize:
            
            result = await flow.run_pipeline(input_data)
            
            assert isinstance(result, KnowledgeRetrievalOutput)
            assert len(result.retrieved_knowledge) == 2
            assert result.total_results == 2
            assert "agent_memory" in result.sources_searched
            assert "knowledge_plugins" in result.sources_searched
            assert "Searched 2 sources" in result.search_summary
            
            # Verify method calls
            mock_memory.assert_called_once_with(input_data)
            mock_plugins.assert_called_once_with(input_data)
            mock_synthesize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_pipeline_memory_only(self):
        """Test retrieval with only memory search enabled."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(
            query="Python programming",
            include_plugins=False,
            include_memory=True
        )
        
        memory_results = [
            RetrievedKnowledge(
                content="Python is interpreted",
                source="agent_memory",
                domain="programming",
                confidence=0.8
            )
        ]
        
        with patch.object(flow, '_search_agent_memory', return_value=memory_results) as mock_memory, \
             patch.object(flow, '_search_knowledge_plugins') as mock_plugins, \
             patch.object(flow, '_synthesize_results', return_value=memory_results) as mock_synthesize:
            
            result = await flow.run_pipeline(input_data)
            
            assert len(result.retrieved_knowledge) == 1
            assert "agent_memory" in result.sources_searched
            assert "knowledge_plugins" not in result.sources_searched
            
            mock_memory.assert_called_once()
            mock_plugins.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_run_pipeline_plugins_only(self):
        """Test retrieval with only plugin search enabled."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(
            query="Chemistry facts",
            include_plugins=True,
            include_memory=False
        )
        
        plugin_results = [
            RetrievedKnowledge(
                content="Water is H2O",
                source="plugin:chemistry",
                domain="chemistry",
                confidence=0.95
            )
        ]
        
        with patch.object(flow, '_search_agent_memory') as mock_memory, \
             patch.object(flow, '_search_knowledge_plugins', return_value=plugin_results) as mock_plugins, \
             patch.object(flow, '_synthesize_results', return_value=plugin_results) as mock_synthesize:
            
            result = await flow.run_pipeline(input_data)
            
            assert len(result.retrieved_knowledge) == 1
            assert "knowledge_plugins" in result.sources_searched
            assert "agent_memory" not in result.sources_searched
            
            mock_memory.assert_not_called()
            mock_plugins.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_pipeline_max_results_limiting(self):
        """Test that results are limited to max_results."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(
            query="Test query",
            max_results=2
        )
        
        # Create more results than max_results
        all_results = [
            RetrievedKnowledge(content=f"Result {i}", source="test", domain="test", confidence=0.8)
            for i in range(5)
        ]
        
        with patch.object(flow, '_search_agent_memory', return_value=[]) as mock_memory, \
             patch.object(flow, '_search_knowledge_plugins', return_value=all_results) as mock_plugins, \
             patch.object(flow, '_synthesize_results', return_value=all_results) as mock_synthesize:
            
            result = await flow.run_pipeline(input_data)
            
            # Should only return max_results items
            assert len(result.retrieved_knowledge) == 2
            assert result.total_results == 5  # But total should reflect all found results
    
    @pytest.mark.asyncio
    async def test_run_pipeline_exception_handling(self):
        """Test exception handling in run_pipeline."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        with patch.object(flow, '_search_agent_memory', side_effect=Exception("Memory search failed")):
            result = await flow.run_pipeline(input_data)
            
            # Should return empty result with error message
            assert isinstance(result, KnowledgeRetrievalOutput)
            assert len(result.retrieved_knowledge) == 0
            assert "failed" in result.search_summary.lower()
            assert result.total_results == 0


class TestSearchAgentMemory:
    """Test _search_agent_memory method."""
    
    @pytest.mark.asyncio
    async def test_search_agent_memory_placeholder(self):
        """Test that agent memory search returns placeholder results."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        results = await flow._search_agent_memory(input_data)
        
        # Currently returns empty list as placeholder
        assert isinstance(results, list)
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_agent_memory_exception_handling(self):
        """Test exception handling in agent memory search."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        # Mock logging to trigger an exception path
        with patch('flowlib.agent.components.knowledge_flows.knowledge_retrieval.logger') as mock_logger:
            mock_logger.debug.side_effect = Exception("Logging error")
            
            results = await flow._search_agent_memory(input_data)
            
            # Should handle exception gracefully
            assert isinstance(results, list)
            assert len(results) == 0


class TestSearchKnowledgePlugins:
    """Test _search_knowledge_plugins method."""
    
    @pytest.mark.asyncio
    async def test_search_knowledge_plugins_specific_domain(self):
        """Test plugin search with specific domain."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(
            query="Chemical formulas",
            domain="chemistry",
            max_results=5
        )
        
        # Mock plugin manager
        mock_knowledge_result = MagicMock()
        mock_knowledge_result.content = "H2O is water"
        mock_knowledge_result.source = "chemistry_plugin"
        mock_knowledge_result.domain = "chemistry"
        mock_knowledge_result.confidence = 0.9
        mock_knowledge_result.metadata = {"type": "formula"}
        
        with patch('flowlib.providers.knowledge.plugin_manager.plugin_manager') as mock_plugin_manager:
            mock_plugin_manager.get_available_domains.return_value = ["chemistry"]
            mock_plugin_manager.query_domain = AsyncMock(return_value=[mock_knowledge_result])
            
            results = await flow._search_knowledge_plugins(input_data)
            
            assert len(results) == 1
            assert isinstance(results[0], RetrievedKnowledge)
            assert results[0].content == "H2O is water"
            assert results[0].source == "plugin:chemistry_plugin"
            assert results[0].domain == "chemistry"
            assert results[0].confidence == 0.9
            assert "plugin_source" in results[0].metadata
            
            # Verify plugin manager was called correctly
            mock_plugin_manager.query_domain.assert_called_once_with(
                domain="chemistry",
                query="Chemical formulas",
                limit=5
            )
    
    @pytest.mark.asyncio
    async def test_search_knowledge_plugins_all_domains(self):
        """Test plugin search across all available domains."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(
            query="General knowledge",
            domain=None,  # Search all domains
            max_results=10
        )
        
        # Mock plugin manager
        mock_knowledge_result = MagicMock()
        mock_knowledge_result.content = "General fact"
        mock_knowledge_result.source = "general_plugin"
        mock_knowledge_result.domain = "general"
        mock_knowledge_result.confidence = 0.8
        mock_knowledge_result.metadata = {}
        
        with patch('flowlib.providers.knowledge.plugin_manager.plugin_manager') as mock_plugin_manager:
            mock_plugin_manager.get_available_domains.return_value = ["general", "science"]
            mock_plugin_manager.query_domain = AsyncMock(return_value=[mock_knowledge_result])
            
            results = await flow._search_knowledge_plugins(input_data)
            
            # Should query each available domain
            assert mock_plugin_manager.get_available_domains.called
            assert mock_plugin_manager.query_domain.call_count == 2  # Called for each domain
            assert len(results) == 2  # One result per domain
    
    @pytest.mark.asyncio
    async def test_search_knowledge_plugins_no_domains(self):
        """Test plugin search when no domains are available."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        with patch('flowlib.providers.knowledge.plugin_manager.plugin_manager') as mock_plugin_manager:
            mock_plugin_manager.get_available_domains.return_value = []
            
            results = await flow._search_knowledge_plugins(input_data)
            
            assert len(results) == 0
            mock_plugin_manager.query_domain.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_search_knowledge_plugins_domain_failure(self):
        """Test handling when a domain search fails."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(
            query="Test query",
            domain="failing_domain"
        )
        
        with patch('flowlib.providers.knowledge.plugin_manager.plugin_manager') as mock_plugin_manager:
            mock_plugin_manager.query_domain.side_effect = Exception("Domain search failed")
            
            results = await flow._search_knowledge_plugins(input_data)
            
            # Should handle exception gracefully and return empty results
            assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_knowledge_plugins_import_failure(self):
        """Test handling when plugin manager import fails."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        # Mock import failure by patching builtins __import__
        def mock_import(name, *args, **kwargs):
            if name == 'flowlib.providers.knowledge.plugin_manager':
                raise ImportError("Import failed")
            return __import__(name, *args, **kwargs)
            
        with patch('builtins.__import__', side_effect=mock_import):
            results = await flow._search_knowledge_plugins(input_data)
            
            assert len(results) == 0


class TestSynthesizeResults:
    """Test _synthesize_results method."""
    
    @pytest.mark.asyncio
    async def test_synthesize_results_empty_list(self):
        """Test synthesizing empty results list."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        results = await flow._synthesize_results([], input_data, ["memory"])
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_synthesize_results_confidence_ranking(self):
        """Test that results are ranked by confidence."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        unsorted_results = [
            RetrievedKnowledge(content="Low confidence", source="test", domain="test", confidence=0.3),
            RetrievedKnowledge(content="High confidence", source="test", domain="test", confidence=0.9),
            RetrievedKnowledge(content="Medium confidence", source="test", domain="test", confidence=0.6)
        ]
        
        sorted_results = await flow._synthesize_results(unsorted_results, input_data, ["test"])
        
        # Should be sorted by confidence (high to low)
        assert len(sorted_results) == 3
        assert sorted_results[0].confidence == 0.9
        assert sorted_results[1].confidence == 0.6
        assert sorted_results[2].confidence == 0.3
        assert sorted_results[0].content == "High confidence"
    
    @pytest.mark.asyncio
    async def test_synthesize_results_duplicate_removal(self):
        """Test that duplicate results are removed."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        # Create results with similar content (first 100 chars)
        results_with_duplicates = [
            RetrievedKnowledge(
                content="This is a long piece of content that should be considered duplicate based on first 100 characters exactly matching prefix content",
                source="source1",
                domain="test",
                confidence=0.8
            ),
            RetrievedKnowledge(
                content="This is a long piece of content that should be considered duplicate based on first 100 characters exactly matching prefix but different ending",
                source="source2", 
                domain="test",
                confidence=0.7
            ),
            RetrievedKnowledge(
                content="Completely different content",
                source="source3",
                domain="test", 
                confidence=0.6
            )
        ]
        
        unique_results = await flow._synthesize_results(results_with_duplicates, input_data, ["test"])
        
        # Should remove duplicates based on first 100 characters
        assert len(unique_results) == 2
        assert unique_results[0].confidence == 0.8  # Higher confidence duplicate kept
        assert unique_results[1].content == "Completely different content"
    
    @pytest.mark.asyncio
    async def test_synthesize_results_exception_handling(self):
        """Test exception handling in result synthesis."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        # Create a result that might cause issues
        problematic_results = [
            RetrievedKnowledge(content="Normal content", source="test", domain="test", confidence=0.8)
        ]
        
        # Mock sorting to raise exception
        with patch('builtins.sorted', side_effect=Exception("Sorting failed")):
            results = await flow._synthesize_results(problematic_results, input_data, ["test"])
            
            # Should return original results if synthesis fails
            assert results == problematic_results


class TestFlowIntegration:
    """Test integration aspects of the knowledge retrieval flow."""
    
    @pytest.mark.asyncio
    async def test_full_retrieval_workflow(self):
        """Test complete retrieval workflow with multiple sources and processing."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(
            query="Machine learning algorithms",
            domain="ai",
            max_results=3,
            include_plugins=True,
            include_memory=True
        )
        
        # Mock comprehensive results from multiple sources
        memory_results = [
            RetrievedKnowledge(
                content="Neural networks are ML algorithms",
                source="agent_memory",
                domain="ai",
                confidence=0.9
            ),
            RetrievedKnowledge(
                content="Decision trees classify data",
                source="agent_memory", 
                domain="ai",
                confidence=0.85
            )
        ]
        
        plugin_results = [
            RetrievedKnowledge(
                content="SVM is a supervised learning algorithm",
                source="plugin:ml_knowledge",
                domain="ai",
                confidence=0.88
            ),
            RetrievedKnowledge(
                content="K-means is used for clustering",
                source="plugin:ml_knowledge",
                domain="ai", 
                confidence=0.82
            )
        ]
        
        with patch.object(flow, '_search_agent_memory', return_value=memory_results), \
             patch.object(flow, '_search_knowledge_plugins', return_value=plugin_results):
            
            result = await flow.run_pipeline(input_data)
            
            # Should have max_results items, ranked by confidence
            assert len(result.retrieved_knowledge) == 3
            assert result.total_results == 4  # All found results
            assert len(result.sources_searched) == 2
            
            # Check confidence-based ranking
            confidences = [r.confidence for r in result.retrieved_knowledge]
            assert confidences == sorted(confidences, reverse=True)
            
            # Highest confidence should be first
            assert result.retrieved_knowledge[0].confidence == 0.9
            assert result.retrieved_knowledge[0].content == "Neural networks are ML algorithms"
    
    @pytest.mark.asyncio
    async def test_no_results_found(self):
        """Test handling when no results are found from any source."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Nonexistent topic")
        
        with patch.object(flow, '_search_agent_memory', return_value=[]), \
             patch.object(flow, '_search_knowledge_plugins', return_value=[]):
            
            result = await flow.run_pipeline(input_data)
            
            assert len(result.retrieved_knowledge) == 0
            assert result.total_results == 0
            assert "found 0 total results" in result.search_summary
    
    def test_flow_class_attributes(self):
        """Test that flow class has expected attributes."""
        assert KnowledgeRetrievalFlow.__flow_name__ == "knowledge-retrieval"
        assert "Retrieve knowledge from plugins" in KnowledgeRetrievalFlow.__flow_description__
        assert KnowledgeRetrievalFlow.__is_infrastructure__ is False
    
    def test_pipeline_method_attributes(self):
        """Test that pipeline method has expected attributes."""
        run_pipeline = KnowledgeRetrievalFlow.run_pipeline
        assert hasattr(run_pipeline, '__pipeline_input_model__')
        assert hasattr(run_pipeline, '__pipeline_output_model__')
        assert run_pipeline.__pipeline_input_model__ == KnowledgeRetrievalInput
        assert run_pipeline.__pipeline_output_model__ == KnowledgeRetrievalOutput


class TestLogging:
    """Test logging functionality in the flow."""
    
    @pytest.mark.asyncio
    async def test_logging_calls(self):
        """Test that appropriate logging calls are made."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_retrieval.logger') as mock_logger, \
             patch.object(flow, '_search_agent_memory', return_value=[]), \
             patch.object(flow, '_search_knowledge_plugins', return_value=[]):
            
            await flow.run_pipeline(input_data)
            
            # Check that logging calls were made
            mock_logger.info.assert_called()
            
            # Check specific log messages
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any("Starting knowledge retrieval" in msg for msg in info_calls)
            assert any("Knowledge retrieval completed" in msg for msg in info_calls)
    
    @pytest.mark.asyncio
    async def test_debug_logging_in_search_methods(self):
        """Test debug logging in search methods."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        with patch('flowlib.agent.components.knowledge_flows.knowledge_retrieval.logger') as mock_logger:
            # Test memory search logging
            await flow._search_agent_memory(input_data)
            mock_logger.debug.assert_called_with("Searching agent memory...")
            
            # Test plugin search logging
            with patch('flowlib.providers.knowledge.plugin_manager.plugin_manager') as mock_plugin_manager:
                mock_plugin_manager.get_available_domains.return_value = []
                await flow._search_knowledge_plugins(input_data)
                
                debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
                assert any("Searching knowledge plugins" in msg for msg in debug_calls)


class TestErrorScenarios:
    """Test various error scenarios in the retrieval flow."""
    
    @pytest.mark.asyncio
    async def test_partial_source_failure(self):
        """Test handling when one source fails but others succeed."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(
            query="Test query",
            include_memory=True,
            include_plugins=True
        )
        
        successful_results = [
            RetrievedKnowledge(
                content="Successful result",
                source="working_source",
                domain="test",
                confidence=0.8
            )
        ]
        
        with patch.object(flow, '_search_agent_memory', side_effect=Exception("Memory failed")), \
             patch.object(flow, '_search_knowledge_plugins', return_value=successful_results):
            
            result = await flow.run_pipeline(input_data)
            
            # Current implementation fails entire pipeline if any source fails
            # This could be improved to handle partial failures more gracefully
            assert len(result.retrieved_knowledge) == 0
            assert "failed" in result.search_summary.lower()
    
    @pytest.mark.asyncio
    async def test_synthesis_failure_fallback(self):
        """Test fallback when result synthesis fails."""
        flow = KnowledgeRetrievalFlow("knowledge-retrieval")
        
        input_data = KnowledgeRetrievalInput(query="Test query")
        
        raw_results = [
            RetrievedKnowledge(content="Test result", source="test", domain="test", confidence=0.8)
        ]
        
        with patch.object(flow, '_search_agent_memory', return_value=[]), \
             patch.object(flow, '_search_knowledge_plugins', return_value=raw_results), \
             patch.object(flow, '_synthesize_results', side_effect=Exception("Synthesis failed")):
            
            # Should handle synthesis failure gracefully
            result = await flow.run_pipeline(input_data)
            
            # Exact behavior depends on implementation, but should not crash
            assert isinstance(result, KnowledgeRetrievalOutput)