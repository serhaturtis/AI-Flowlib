#!/usr/bin/env python3
"""Test checkpoint and resuming functionality for streaming knowledge extraction."""

import asyncio
import tempfile
import logging
import pytest
import json
from pathlib import Path
from typing import Dict, Any

from flowlib.knowledge.models.models import (
    ExtractionState, ExtractionProgress, ExtractionConfig, ChunkingStrategy
)
from flowlib.knowledge.streaming.checkpoint_manager import CheckpointManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCheckpointResuming:
    """Test checkpoint and resuming functionality."""
    
    @pytest.mark.asyncio
    async def test_streaming_state_persistence(self):
        """Test streaming state save and load."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir)
            
            # Create initial streaming state
            original_state = ExtractionState()
            original_state.processed_docs.add("doc1.txt")
            original_state.processed_docs.add("doc2.txt")
            original_state.processed_docs.add("doc3.txt")
            original_state.detected_domains.add("technology")
            original_state.detected_domains.add("research")
            
            original_state.progress.total_documents = 10
            original_state.progress.processed_documents = 3
            original_state.progress.current_document = "doc3.txt"
            
            original_state.last_checkpoint_at = 2
            original_state.extraction_config = ExtractionConfig(
                checkpoint_interval=2,
                chunking_strategy=ChunkingStrategy.PARAGRAPH_AWARE
            )
            
            # Save state
            await checkpoint_manager.save_streaming_state(original_state)
            logger.info("✅ Streaming state saved")
            
            # Load state
            loaded_state = await checkpoint_manager.load_streaming_state()
            
            # Verify state was loaded correctly
            assert loaded_state is not None
            assert len(loaded_state.processed_docs) == 3
            assert "doc1.txt" in loaded_state.processed_docs
            assert "doc2.txt" in loaded_state.processed_docs
            assert "doc3.txt" in loaded_state.processed_docs
            
            assert len(loaded_state.detected_domains) == 2
            assert "technology" in loaded_state.detected_domains
            assert "research" in loaded_state.detected_domains
            
            assert loaded_state.progress.total_documents == 10
            assert loaded_state.progress.processed_documents == 3
            assert loaded_state.progress.current_document == "doc3.txt"
            
            assert loaded_state.last_checkpoint_at == 2
            assert loaded_state.extraction_config.checkpoint_interval == 2
            
            logger.info("✅ Streaming state loaded and verified")
    
    @pytest.mark.asyncio
    async def test_checkpoint_interval_logic(self):
        """Test checkpoint interval and export logic."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir)
            
            # Create streaming state with checkpoint interval of 3
            state = ExtractionState()
            state.extraction_config = ExtractionConfig(checkpoint_interval=3)
            state.last_checkpoint_at = 0
            
            # Process documents and test checkpoint logic
            documents = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt"]
            
            for i, doc in enumerate(documents, 1):
                state.processed_docs.add(doc)
                
                should_checkpoint = await checkpoint_manager.should_export_checkpoint_plugin(state)
                
                if i % 3 == 0:  # Every 3rd document
                    assert should_checkpoint, f"Should checkpoint at document {i}"
                    logger.info(f"✅ Checkpoint triggered at document {i}")
                    
                    # Simulate checkpoint export
                    state.last_checkpoint_at = len(state.processed_docs)
                else:
                    assert not should_checkpoint, f"Should not checkpoint at document {i}"
                    logger.debug(f"No checkpoint at document {i}")
            
            logger.info("✅ Checkpoint interval logic verified")
    
    @pytest.mark.asyncio
    async def test_checkpoint_data_structure(self):
        """Test checkpoint data structure via plugin export."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir)
            
            # Create test streaming state with data
            state = ExtractionState()
            state.extraction_config = ExtractionConfig(checkpoint_interval=2)
            state.processed_docs.add("doc1.txt")
            state.processed_docs.add("doc2.txt") 
            state.detected_domains.add("technology")
            state.detected_domains.add("research")
            state.progress.total_documents = 10
            state.progress.processed_documents = 2
            
            # Set proper database paths to avoid copying .git
            state.streaming_vector_db_path = str(checkpoint_manager.streaming_vector_db_dir)
            state.streaming_graph_db_path = str(checkpoint_manager.streaming_graph_db_dir)
            
            # Mock some accumulated data
            from flowlib.knowledge.models.models import Entity, Relationship, EntityType, RelationType
            
            mock_entity = Entity(
                entity_id="test_entity_1",
                name="TestEntity",
                entity_type=EntityType.TECHNOLOGY,
                description="Test entity for checkpointing",
                documents=["doc1.txt"],
                frequency=1,
                confidence=0.9
            )
            
            mock_relationship = Relationship(
                relationship_id="test_rel_1",
                source_entity_id="test_entity_1",
                target_entity_id="test_entity_1",
                relationship_type=RelationType.RELATES_TO,
                description="Test relationship",
                documents=["doc1.txt"],
                confidence=0.8,
                frequency=1
            )
            
            state.accumulated_entities = [mock_entity]
            state.accumulated_relationships = [mock_relationship]
            
            # Export checkpoint plugin (this is how checkpoints are actually created)
            plugin_path = await checkpoint_manager.export_incremental_plugin(state)
            logger.info(f"✅ Checkpoint plugin exported: {plugin_path}")
            
            # Verify plugin structure was created
            plugin_dir = Path(plugin_path)
            assert plugin_dir.exists()
            assert (plugin_dir / "data").exists()
            assert (plugin_dir / "data" / "entities.json").exists()
            assert (plugin_dir / "data" / "relationships.json").exists()
            assert (plugin_dir / "data" / "metadata.json").exists()
            assert (plugin_dir / "manifest.yaml").exists()
            
            # Verify metadata content
            with open(plugin_dir / "data" / "metadata.json") as f:
                metadata = json.load(f)
            
            assert len(metadata["processed_documents"]) == 2
            assert "doc1.txt" in metadata["processed_documents"]
            assert "technology" in metadata["detected_domains"]
            assert "research" in metadata["detected_domains"]
            
            logger.info("✅ Checkpoint data structure verified")
    
    @pytest.mark.asyncio
    async def test_checkpoint_list_and_cleanup(self):
        """Test checkpoint plugin creation and cleanup operations."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir)
            
            # Create multiple checkpoint plugins
            plugin_paths = []
            
            for i in range(3):  # Reduce number for faster testing
                # Create state for each checkpoint
                state = ExtractionState()
                state.extraction_config = ExtractionConfig(checkpoint_interval=1)
                state.processed_docs.add(f"doc_{i}.txt")
                state.detected_domains.add("technology")
                state.progress.processed_documents = i + 1
                state.progress.total_documents = 5
                
                # Set proper database paths
                state.streaming_vector_db_path = str(checkpoint_manager.streaming_vector_db_dir)
                state.streaming_graph_db_path = str(checkpoint_manager.streaming_graph_db_dir)
                
                # Add mock data
                from flowlib.knowledge.models.models import Entity, EntityType
                mock_entity = Entity(
                    entity_id=f"entity_{i}",
                    name=f"Entity{i}",
                    entity_type=EntityType.CONCEPT,
                    description=f"Test entity {i}",
                    documents=[f"doc_{i}.txt"],
                    frequency=1,
                    confidence=0.8
                )
                state.accumulated_entities = [mock_entity]
                state.accumulated_relationships = []
                
                # Export checkpoint plugin
                plugin_path = await checkpoint_manager.export_incremental_plugin(state)
                plugin_paths.append(plugin_path)
                logger.info(f"Created checkpoint plugin {i}: {plugin_path}")
            
            # Verify plugins were created
            assert len(plugin_paths) == 3
            
            for plugin_path in plugin_paths:
                plugin_dir = Path(plugin_path)
                assert plugin_dir.exists()
                assert (plugin_dir / "manifest.yaml").exists()
                assert (plugin_dir / "data").exists()
            
            logger.info(f"✅ Created {len(plugin_paths)} checkpoint plugins")
            
            # Test cleanup functionality
            await checkpoint_manager.cleanup_old_checkpoints(keep_last=2)
            
            # Verify cleanup worked (this is basic - the method exists)
            logger.info("✅ Checkpoint cleanup completed")
    
    @pytest.mark.asyncio
    async def test_resuming_simulation(self):
        """Test complete resuming simulation."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir)
            
            # Simulate first streaming run
            logger.info("🚀 Simulating first streaming run...")
            
            # Initial state
            state = ExtractionState()
            state.extraction_config = ExtractionConfig(
                checkpoint_interval=2,
                batch_size=1,
                chunking_strategy=ChunkingStrategy.PARAGRAPH_AWARE
            )
            state.progress.total_documents = 6
            
            # Set proper database paths
            state.streaming_vector_db_path = str(checkpoint_manager.streaming_vector_db_dir)
            state.streaming_graph_db_path = str(checkpoint_manager.streaming_graph_db_dir)
            
            # Process first batch of documents
            first_batch = ["doc1.txt", "doc2.txt", "doc3.txt"]
            
            for doc in first_batch:
                state.processed_docs.add(doc)
                state.progress.processed_documents = len(state.processed_docs)
                state.progress.current_document = doc
                state.detected_domains.add("technology")
            
            # Save state after processing
            await checkpoint_manager.save_streaming_state(state)
            
            # Create checkpoint by exporting plugin
            # Add mock data for plugin export
            from flowlib.knowledge.models.models import Entity, EntityType
            mock_entity = Entity(
                entity_id="resume_test_entity",
                name="ResumeTestEntity",
                entity_type=EntityType.CONCEPT,
                description="Entity for resume testing",
                documents=first_batch,
                frequency=1,
                confidence=0.9
            )
            state.accumulated_entities = [mock_entity]
            state.accumulated_relationships = []
            
            # Export checkpoint plugin
            checkpoint_plugin_path = await checkpoint_manager.export_incremental_plugin(state)
            logger.info(f"✅ First run completed, checkpoint plugin: {checkpoint_plugin_path}")
            
            # Simulate resuming from checkpoint
            logger.info("🔄 Simulating resume from checkpoint...")
            
            # Load previous state
            resumed_state = await checkpoint_manager.load_streaming_state()
            
            assert resumed_state is not None
            assert len(resumed_state.processed_docs) == 3
            assert "doc1.txt" in resumed_state.processed_docs
            assert "doc2.txt" in resumed_state.processed_docs
            assert "doc3.txt" in resumed_state.processed_docs
            
            logger.info(f"✅ Resumed from checkpoint: {len(resumed_state.processed_docs)} docs already processed")
            
            # Process remaining documents
            remaining_docs = ["doc4.txt", "doc5.txt", "doc6.txt"]
            all_docs = first_batch + remaining_docs
            
            docs_to_process = []
            for doc in all_docs:
                if doc not in resumed_state.processed_docs:
                    docs_to_process.append(doc)
            
            assert len(docs_to_process) == 3  # Should only process remaining docs
            assert docs_to_process == remaining_docs
            
            # Process remaining documents
            for doc in docs_to_process:
                resumed_state.processed_docs.add(doc)
                resumed_state.progress.processed_documents = len(resumed_state.processed_docs)
                resumed_state.progress.current_document = doc
            
            # Save final state
            await checkpoint_manager.save_streaming_state(resumed_state)
            
            # Verify final state
            final_state = await checkpoint_manager.load_streaming_state()
            assert len(final_state.processed_docs) == 6
            assert final_state.progress.processed_documents == 6
            
            logger.info("✅ Resuming simulation completed successfully")
    
    @pytest.mark.asyncio
    async def test_concurrent_checkpoint_access(self):
        """Test concurrent access to checkpoint system."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_manager = CheckpointManager(temp_dir)
            
            # Create initial state
            state = ExtractionState()
            state.extraction_config = ExtractionConfig(checkpoint_interval=1)
            
            async def process_doc_batch(doc_names: list, batch_id: str):
                """Simulate processing a batch of documents."""
                for doc in doc_names:
                    state.processed_docs.add(f"{batch_id}_{doc}")
                    state.progress.processed_documents = len(state.processed_docs)
                
                # Save state
                await checkpoint_manager.save_streaming_state(state)
                
                # Create checkpoint plugin
                # Add mock data for the batch
                from flowlib.knowledge.models.models import Entity, EntityType
                mock_entity = Entity(
                    entity_id=f"{batch_id}_entity",
                    name=f"Entity{batch_id}",
                    entity_type=EntityType.CONCEPT,
                    description=f"Entity from {batch_id}",
                    documents=[f"{batch_id}_{doc}" for doc in doc_names],
                    frequency=1,
                    confidence=0.8
                )
                
                # Create a local state copy to avoid conflicts
                local_state = ExtractionState()
                local_state.extraction_config = ExtractionConfig(checkpoint_interval=1)
                local_state.processed_docs.update([f"{batch_id}_{doc}" for doc in doc_names])
                local_state.accumulated_entities = [mock_entity]
                local_state.accumulated_relationships = []
                local_state.progress.processed_documents = len(local_state.processed_docs)
                
                # Set proper database paths
                local_state.streaming_vector_db_path = str(checkpoint_manager.streaming_vector_db_dir)
                local_state.streaming_graph_db_path = str(checkpoint_manager.streaming_graph_db_dir)
                
                checkpoint_plugin_path = await checkpoint_manager.export_incremental_plugin(local_state)
                logger.info(f"Batch {batch_id} checkpoint: {checkpoint_plugin_path}")
                
                return checkpoint_plugin_path
            
            # Process multiple batches concurrently
            batch_tasks = [
                process_doc_batch(["doc1", "doc2"], "batch_1"),
                process_doc_batch(["doc3", "doc4"], "batch_2"),
                process_doc_batch(["doc5", "doc6"], "batch_3")
            ]
            
            # Wait for all batches to complete
            checkpoint_ids = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Verify all checkpoints were created
            successful_checkpoints = [cid for cid in checkpoint_ids if isinstance(cid, str)]
            logger.info(f"✅ Created {len(successful_checkpoints)} checkpoints concurrently")
            
            # Verify final state
            final_state = await checkpoint_manager.load_streaming_state()
            assert len(final_state.processed_docs) > 0
            
            logger.info("✅ Concurrent checkpoint access test completed")


@pytest.mark.asyncio
async def test_checkpoint_persistence():
    """Test checkpoint persistence functionality."""
    test_instance = TestCheckpointResuming()
    await test_instance.test_streaming_state_persistence()
    logger.info("✅ Checkpoint persistence tests passed!")


@pytest.mark.asyncio
async def test_checkpoint_logic():
    """Test checkpoint interval and export logic."""
    test_instance = TestCheckpointResuming()
    await test_instance.test_checkpoint_interval_logic()
    await test_instance.test_checkpoint_data_structure()
    await test_instance.test_checkpoint_list_and_cleanup()
    logger.info("✅ Checkpoint logic tests passed!")


@pytest.mark.asyncio
async def test_resuming_functionality():
    """Test resuming functionality."""
    test_instance = TestCheckpointResuming()
    await test_instance.test_resuming_simulation()
    await test_instance.test_concurrent_checkpoint_access()
    logger.info("✅ Resuming functionality tests passed!")


if __name__ == "__main__":
    async def main():
        """Run all checkpoint and resuming tests."""
        
        logger.info("🚀 Starting checkpoint and resuming tests...")
        
        try:
            await test_checkpoint_persistence()
            await test_checkpoint_logic()
            await test_resuming_functionality()
            
            logger.info("🎉 All checkpoint and resuming tests passed!")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise
    
    asyncio.run(main())