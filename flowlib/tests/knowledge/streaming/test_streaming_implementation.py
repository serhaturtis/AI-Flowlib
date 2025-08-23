#!/usr/bin/env python3
"""Test streaming knowledge extraction implementation."""

import asyncio
import tempfile
import logging
import pytest
from pathlib import Path

from flowlib.knowledge.models.models import (
    KnowledgeExtractionRequest,
    ExtractionConfig,
    ChunkingStrategy,
    ExtractionState,
    ExtractionProgress
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestExtractionModels:
    """Test streaming model classes."""
    
    def test_extraction_config(self):
        """Test ExtractionConfig model."""
        config = ExtractionConfig(
            batch_size=3,
            checkpoint_interval=5,
            chunking_strategy=ChunkingStrategy.PARAGRAPH_AWARE,
            max_chunk_size=1000,
            overlap_size=100
        )
        
        assert config.batch_size == 3
        assert config.checkpoint_interval == 5
        assert config.chunking_strategy == ChunkingStrategy.PARAGRAPH_AWARE
        assert config.max_chunk_size == 1000
        assert config.overlap_size == 100
    
    def test_streaming_request(self):
        """Test KnowledgeExtractionRequest model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExtractionConfig(batch_size=2, checkpoint_interval=3)
            
            request = KnowledgeExtractionRequest(
                input_directory=temp_dir,
                output_directory=temp_dir,
                extraction_config=config,
                plugin_name_prefix="test_plugin",
                plugin_domains=["technology", "research"]
            )
            
            assert request.input_directory == temp_dir
            assert request.output_directory == temp_dir
            assert request.extraction_config.batch_size == 2
            assert request.plugin_name_prefix == "test_plugin"
            assert "technology" in request.plugin_domains
    
    def test_streaming_state(self):
        """Test ExtractionState model."""
        state = ExtractionState()
        
        # Test initial state
        assert len(state.processed_docs) == 0
        assert len(state.detected_domains) == 0
        assert state.progress.processed_documents == 0
        
        # Test state updates
        state.processed_docs.add("doc1.txt")
        state.processed_docs.add("doc2.txt")
        state.detected_domains.add("technology")
        state.progress.total_documents = 10
        state.progress.processed_documents = 2
        
        assert len(state.processed_docs) == 2
        assert "doc1.txt" in state.processed_docs
        assert "technology" in state.detected_domains
        assert state.progress.processed_documents == 2
        assert state.progress.total_documents == 10


class TestSmartChunking:
    """Test smart chunking functionality."""
    
    @pytest.mark.asyncio
    async def test_smart_chunking_flow(self):
        """Test smart chunking flow."""
        from flowlib.knowledge.chunking.flow import SmartChunkingFlow
        from flowlib.knowledge.chunking.models import ChunkingInput
        from flowlib.knowledge.models.models import (
            DocumentContent, DocumentMetadata, DocumentType, 
            ProcessingStatus, ExtractionConfig, ChunkingStrategy
        )
        
        # Create test document
        metadata = DocumentMetadata(
            file_path="/test/doc.txt",
            file_name="doc.txt",
            file_size=1000,
            file_type=DocumentType.TXT
        )
        
        test_text = """This is the first paragraph about artificial intelligence and machine learning.
It discusses neural networks and deep learning algorithms.

This is the second paragraph that talks about natural language processing.
It covers transformers and attention mechanisms in detail.

This is a third paragraph about computer vision and image recognition.
It explains convolutional neural networks and their applications."""
        
        document = DocumentContent(
            document_id="test_doc",
            metadata=metadata,
            full_text=test_text,
            chunks=[],  # Will be populated by chunking
            status=ProcessingStatus.COMPLETED
        )
        
        # Create chunking config
        config = ExtractionConfig(
            chunking_strategy=ChunkingStrategy.PARAGRAPH_AWARE,
            max_chunk_size=200,
            overlap_size=50
        )
        
        # Create chunking input
        chunking_input = ChunkingInput(
            document=document,
            config=config
        )
        
        # Test chunking flow
        chunking_flow = SmartChunkingFlow()
        result = await chunking_flow.run_pipeline(chunking_input)
        
        assert len(result.document.chunks) >= 3  # Should create at least 3 chunks for 3 paragraphs
        
        # Verify chunk properties
        for i, chunk in enumerate(result.document.chunks):
            assert chunk.chunk_index == i
            assert chunk.document_id == "test_doc"
            assert len(chunk.text) > 0
            assert chunk.word_count > 0
            assert chunk.char_count > 0


class TestCheckpointManager:
    """Test checkpoint management functionality."""
    
    @pytest.mark.asyncio
    async def test_checkpoint_manager(self):
        """Test checkpoint manager."""
        from flowlib.knowledge.streaming.checkpoint_manager import CheckpointManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create checkpoint manager
            checkpoint_manager = CheckpointManager(temp_dir)
            
            # Test directory creation
            assert checkpoint_manager.checkpoints_dir.exists()
            assert checkpoint_manager.incremental_plugins_dir.exists()
            assert checkpoint_manager.streaming_vector_db_dir.exists()
            assert checkpoint_manager.streaming_graph_db_dir.exists()
            
            # Create test streaming state
            state = ExtractionState()
            state.processed_docs.add("doc1.txt")
            state.processed_docs.add("doc2.txt")
            state.progress.total_documents = 10
            state.progress.processed_documents = 2
            state.detected_domains.add("technology")
            
            # Test state saving and loading
            await checkpoint_manager.save_streaming_state(state)
            logger.info("Streaming state saved successfully")
            
            loaded_state = await checkpoint_manager.load_streaming_state()
            assert loaded_state is not None
            assert len(loaded_state.processed_docs) == 2
            assert "doc1.txt" in loaded_state.processed_docs
            assert "technology" in loaded_state.detected_domains
            assert loaded_state.progress.processed_documents == 2
            
            logger.info(f"Streaming state loaded: {len(loaded_state.processed_docs)} docs")
    
    @pytest.mark.asyncio
    async def test_checkpoint_export_logic(self):
        """Test checkpoint export logic."""
        from flowlib.knowledge.streaming.checkpoint_manager import CheckpointManager
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir)
            
            # Create streaming state
            state = ExtractionState()
            state.extraction_config = ExtractionConfig(checkpoint_interval=3)
            state.processed_docs.add("doc1.txt")
            state.processed_docs.add("doc2.txt")
            state.last_checkpoint_at = 0
            
            # Should not export yet (2 docs, interval is 3)
            should_export = await checkpoint_manager.should_export_checkpoint_plugin(state)
            assert not should_export
            
            # Add one more doc
            state.processed_docs.add("doc3.txt")
            
            # Should export now (3 docs, interval is 3)
            should_export = await checkpoint_manager.should_export_checkpoint_plugin(state)
            assert should_export


class TestStreamingIntegration:
    """Test streaming component integration."""
    
    @pytest.mark.asyncio
    async def test_integration(self):
        """Test basic integration of streaming components."""
        
        # Test that all components can be imported together
        from flowlib.knowledge.streaming.flow import KnowledgeExtractionFlow
        from flowlib.knowledge.streaming.checkpoint_manager import CheckpointManager
        from flowlib.knowledge.chunking.flow import SmartChunkingFlow
        
        logger.info("All streaming components imported successfully")
        
        # Test flow instantiation
        streaming_flow = KnowledgeExtractionFlow()
        chunking_flow = SmartChunkingFlow()
        
        logger.info("Streaming flows instantiated successfully")
        
        # Test config creation
        config = ExtractionConfig(
            batch_size=1,
            checkpoint_interval=2
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            request = KnowledgeExtractionRequest(
                input_directory=temp_dir,
                output_directory=temp_dir,
                extraction_config=config
            )
            
            logger.info("Streaming request created successfully")
            
            # Create checkpoint manager
            checkpoint_manager = CheckpointManager(temp_dir)
            
            logger.info("✅ Integration test passed!")


@pytest.mark.asyncio
async def test_streaming_models():
    """Test streaming model functionality."""
    test_instance = TestExtractionModels()
    test_instance.test_extraction_config()
    test_instance.test_streaming_request()
    test_instance.test_streaming_state()
    logger.info("✅ Streaming models tests passed!")


@pytest.mark.asyncio
async def test_smart_chunking():
    """Test smart chunking functionality."""
    test_instance = TestSmartChunking()
    await test_instance.test_smart_chunking_flow()
    logger.info("✅ Smart chunking tests passed!")


@pytest.mark.asyncio
async def test_checkpoint_manager():
    """Test checkpoint manager functionality."""
    test_instance = TestCheckpointManager()
    await test_instance.test_checkpoint_manager()
    await test_instance.test_checkpoint_export_logic()
    logger.info("✅ Checkpoint manager tests passed!")


@pytest.mark.asyncio
async def test_streaming_integration():
    """Test streaming integration."""
    test_instance = TestStreamingIntegration()
    await test_instance.test_integration()
    logger.info("✅ Streaming integration tests passed!")


if __name__ == "__main__":
    async def main():
        """Run all tests."""
        logger.info("🚀 Starting streaming implementation tests...")
        
        try:
            await test_streaming_models()
            await test_smart_chunking()
            await test_checkpoint_manager()
            await test_streaming_integration()
            
            logger.info("🎉 All streaming implementation tests passed!")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise
    
    asyncio.run(main())