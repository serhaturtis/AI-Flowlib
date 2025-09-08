"""Streaming knowledge extraction flow."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Set, AsyncGenerator

from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.knowledge.models import (
    KnowledgeExtractionRequest, KnowledgeExtractionResult,
    ExtractionState, ExtractionConfig, ExtractionProgress,
    DocumentContent, KnowledgeBaseStats, DocumentType
)
from flowlib.knowledge.chunking.flow import SmartChunkingFlow
from flowlib.knowledge.chunking.models import ChunkingInput
from flowlib.knowledge.extraction.flow import DocumentExtractionFlow
from flowlib.knowledge.analysis.flow import EntityAnalysisFlow
from flowlib.knowledge.vector.flow import VectorStorageFlow
from flowlib.knowledge.graph.flow import GraphStorageFlow
from flowlib.knowledge.streaming.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


@flow(name="knowledge-extraction", description="Knowledge extraction with checkpointing and streaming support")
class KnowledgeExtractionFlow:
    """Main orchestrator for knowledge extraction with automatic checkpointing and streaming support."""

    @pipeline(input_model=KnowledgeExtractionRequest, output_model=KnowledgeExtractionResult)
    async def run_pipeline(self, request: KnowledgeExtractionRequest) -> KnowledgeExtractionResult:
        """Execute streaming knowledge base creation with checkpointing."""
        
        logger.info("🚀 Starting streaming knowledge extraction")
        logger.info(f"Input: {request.input_directory}")
        logger.info(f"Output: {request.output_directory}")
        
        start_time = datetime.now()
        
        # Initialize checkpoint management
        checkpoint_manager = CheckpointManager(request.output_directory)
        
        # Load or initialize streaming state
        streaming_state = await self._initialize_streaming_state(request, checkpoint_manager)
        
        # Count total documents for progress tracking
        total_docs = await self._count_total_documents(request.input_directory)
        streaming_state.progress.total_documents = total_docs
        
        logger.info(f"📊 Total documents to process: {total_docs}")
        logger.info(f"📋 Resuming from: {len(streaming_state.processed_docs)} already processed")
        
        # Stream through documents
        async for doc_path in self._stream_document_paths(request.input_directory):
            doc_id = str(doc_path.relative_to(request.input_directory))
            
            # Skip if already processed
            if doc_id in streaming_state.processed_docs:
                logger.debug(f"⏭️  Skipping already processed: {doc_id}")
                continue
            
            # Update current document
            streaming_state.progress.current_document = doc_id
            logger.info(f"📄 Processing: {doc_id} ({len(streaming_state.processed_docs) + 1}/{total_docs})")
            
            try:
                # Process single document through pipeline
                await self._process_single_document(doc_path, request, streaming_state, checkpoint_manager)
                
                # Mark as processed
                streaming_state.processed_docs.add(doc_id)
                streaming_state.progress.processed_documents = len(streaming_state.processed_docs)
                
                # Export checkpoint plugin if needed
                if await checkpoint_manager.should_export_checkpoint_plugin(streaming_state):
                    plugin_path = await checkpoint_manager.export_incremental_plugin(streaming_state)
                    logger.info(f"✅ Checkpoint plugin exported: {Path(plugin_path).name}")
                
                # Save streaming state
                await checkpoint_manager.save_streaming_state(streaming_state)
                
                logger.info(f"✅ Completed {doc_id} - Progress: {streaming_state.progress.progress_percentage:.1f}%")
                
                # Small delay to prevent system overload
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Failed to process {doc_id}: {e}")
                streaming_state.failed_docs[doc_id] = str(e)
                continue
        
        # Export final plugin
        final_plugin_path = await self._export_final_plugin(request, streaming_state, checkpoint_manager)
        
        # Calculate final statistics
        final_stats = await self._calculate_final_stats(streaming_state, start_time)
        
        # Cleanup old checkpoints
        await checkpoint_manager.cleanup_old_checkpoints()
        
        logger.info("🎉 Streaming knowledge extraction completed!")
        logger.info(f"📊 Final stats: {len(streaming_state.accumulated_entities)} entities, {len(streaming_state.accumulated_relationships)} relationships")
        logger.info(f"🔗 Final plugin: {Path(final_plugin_path).name}")
        
        return KnowledgeExtractionResult(
            status="completed",
            message=f"Successfully processed {len(streaming_state.processed_docs)} documents",
            final_stats=final_stats,
            final_plugin_path=final_plugin_path,
            checkpoint_plugins=streaming_state.checkpoint_plugins,
            total_processing_time_seconds=(datetime.now() - start_time).total_seconds(),
            average_documents_per_minute=streaming_state.progress.documents_per_minute,
            final_extraction_state=streaming_state
        )

    async def _initialize_streaming_state(
        self, 
        request: KnowledgeExtractionRequest, 
        checkpoint_manager: CheckpointManager
    ) -> ExtractionState:
        """Initialize or load streaming state."""
        
        # Try to load existing state if resumption is enabled
        if request.resume_from_checkpoint:
            existing_state = await checkpoint_manager.load_streaming_state()
            if existing_state:
                logger.info(f"📂 Resuming from checkpoint: {len(existing_state.processed_docs)} docs processed")
                return existing_state
        
        # Create new streaming state
        logger.info("🆕 Starting fresh streaming extraction")
        
        streaming_state = ExtractionState(
            extraction_config=request.extraction_config,
            streaming_vector_db_path=str(checkpoint_manager.streaming_vector_db_dir),
            streaming_graph_db_path=str(checkpoint_manager.streaming_graph_db_dir)
        )
        
        return streaming_state

    async def _count_total_documents(self, input_directory: str) -> int:
        """Count total documents to process."""
        
        supported_extensions = {
            f".{doc_type.value}" for doc_type in DocumentType
        }
        
        total = 0
        input_path = Path(input_directory)
        
        for file_path in input_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                total += 1
        
        return total

    async def _stream_document_paths(self, input_directory: str) -> AsyncGenerator[Path, None]:
        """Stream document paths one by one."""
        
        supported_extensions = {
            f".{doc_type.value}" for doc_type in DocumentType
        }
        
        input_path = Path(input_directory)
        
        for file_path in input_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                yield file_path

    async def _process_single_document(
        self, 
        doc_path: Path, 
        request: KnowledgeExtractionRequest,
        streaming_state: ExtractionState,
        checkpoint_manager: CheckpointManager
    ) -> None:
        """Process a single document through the full pipeline."""
        
        # Extract document content
        doc_content = await self._extract_document(doc_path, request)
        
        # Apply smart chunking
        enhanced_doc = await self._apply_smart_chunking(doc_content, streaming_state.extraction_config)
        
        # Extract entities and relationships
        extraction_result = await self._extract_entities_relationships(enhanced_doc, request)
        
        # Accumulate to streaming state
        streaming_state.accumulated_entities.extend(extraction_result.entities)
        streaming_state.accumulated_relationships.extend(extraction_result.relationships)
        
        # Update progress statistics
        streaming_state.progress.entities_extracted = len(streaming_state.accumulated_entities)
        streaming_state.progress.relationships_extracted = len(streaming_state.accumulated_relationships)
        streaming_state.progress.chunks_created += len(enhanced_doc.chunks)
        
        # Detect domains from entities
        for entity in extraction_result.entities:
            if hasattr(entity, 'entity_type') and entity.entity_type:
                streaming_state.detected_domains.add(entity.entity_type.value)
        
        # Stream to persistent databases
        if request.use_vector_db:
            await self._stream_to_vector_db(enhanced_doc, streaming_state)
        
        if request.use_graph_db:
            await self._stream_to_graph_db(extraction_result, streaming_state)

    async def _extract_document(self, doc_path: Path, request: KnowledgeExtractionRequest) -> DocumentContent:
        """Extract content from single document."""
        
        from flowlib.knowledge.extraction.flow import DocumentProcessor
        
        # Use DocumentProcessor directly for single document processing
        processor = DocumentProcessor()
        
        try:
            document_content = await processor.process_single_document(str(doc_path))
            return document_content
        except Exception as e:
            raise Exception(f"Failed to extract document {doc_path}: {e}")

    async def _apply_smart_chunking(self, doc_content: DocumentContent, config: ExtractionConfig) -> DocumentContent:
        """Apply smart chunking to document."""
        
        chunking_flow = SmartChunkingFlow()
        
        chunking_input = ChunkingInput(
            document=doc_content,
            config=config
        )
        
        chunking_result = await chunking_flow.run_pipeline(chunking_input)
        return chunking_result.document

    async def _extract_entities_relationships(self, doc_content: DocumentContent, request: KnowledgeExtractionRequest):
        """Extract entities and relationships from document."""
        
        from flowlib.knowledge.models import EntityExtractionInput
        
        analysis_flow = EntityAnalysisFlow()
        
        analysis_input = EntityExtractionInput(
            documents=[doc_content],
            extraction_domain=request.extraction_domain
        )
        
        return await analysis_flow.run_pipeline(analysis_input)

    async def _stream_to_vector_db(self, doc_content: DocumentContent, streaming_state: ExtractionState) -> None:
        """Stream document chunks to persistent vector database."""
        
        from flowlib.knowledge.vector.models import VectorStoreInput
        
        vector_flow = VectorStorageFlow()
        
        # Check if flow has streaming method, otherwise use regular method
        if hasattr(vector_flow, 'stream_upsert'):
            await vector_flow.stream_upsert(doc_content, streaming_state.streaming_vector_db_path)
        else:
            # Use regular pipeline for now
            vector_input = VectorStoreInput(
                documents=[doc_content],
                collection_name="streaming_knowledge"
            )
            await vector_flow.run_pipeline(vector_input)

    async def _stream_to_graph_db(self, extraction_result, streaming_state: ExtractionState) -> None:
        """Stream entities and relationships to persistent graph database."""
        
        from flowlib.knowledge.graph.models import GraphStoreInput
        
        graph_flow = GraphStorageFlow()
        
        # Check if flow has streaming method, otherwise use regular method
        if hasattr(graph_flow, 'stream_upsert'):
            await graph_flow.stream_upsert(extraction_result, streaming_state.streaming_graph_db_path)
        else:
            # Use regular pipeline for now
            graph_input = GraphStoreInput(
                documents=[],  # Empty for streaming
                entities=extraction_result.entities,
                relationships=extraction_result.relationships,
                graph_name="streaming_graph"
            )
            await graph_flow.run_pipeline(graph_input)

    async def _export_final_plugin(
        self, 
        request: KnowledgeExtractionRequest,
        streaming_state: ExtractionState,
        checkpoint_manager: CheckpointManager
    ) -> str:
        """Export final complete plugin without re-running extraction."""
        
        logger.info("📦 Exporting final complete plugin")
        
        # Generate final plugin directly without using PluginGenerationFlow 
        # to avoid infinite recursion
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_plugin_name = f"{request.plugin_name_prefix}_final_{timestamp}"
        final_plugin_dir = checkpoint_manager.output_dir / final_plugin_name
        final_plugin_dir.mkdir(exist_ok=True)
        
        # Copy streaming databases to plugin location
        import shutil
        
        databases_dir = final_plugin_dir / "databases"
        databases_dir.mkdir(exist_ok=True)
        
        if Path(streaming_state.streaming_vector_db_path).exists() and any(Path(streaming_state.streaming_vector_db_path).iterdir()):
            vector_db_dest = databases_dir / "chromadb"
            shutil.copytree(streaming_state.streaming_vector_db_path, vector_db_dest, dirs_exist_ok=True)
        
        if Path(streaming_state.streaming_graph_db_path).exists() and any(Path(streaming_state.streaming_graph_db_path).iterdir()):
            graph_db_dest = databases_dir / "neo4j"
            shutil.copytree(streaming_state.streaming_graph_db_path, graph_db_dest, dirs_exist_ok=True)
        
        # Create data directory and export streaming data as JSON
        data_dir = final_plugin_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Export accumulated data
        import json
        
        # Export documents (create basic document metadata from processed_docs)
        documents_data = []
        for doc_id in streaming_state.processed_docs:
            documents_data.append({
                "document_id": doc_id,
                "status": "completed",
                "processed_at": datetime.now().isoformat()
            })
        
        with open(data_dir / "documents.json", 'w') as f:
            json.dump(documents_data, f, indent=2, default=str)
        
        # Export entities
        entities_data = [entity.model_dump() for entity in streaming_state.accumulated_entities]
        with open(data_dir / "entities.json", 'w') as f:
            json.dump(entities_data, f, indent=2, default=str)
        
        # Export relationships
        relationships_data = [rel.model_dump() for rel in streaming_state.accumulated_relationships]
        with open(data_dir / "relationships.json", 'w') as f:
            json.dump(relationships_data, f, indent=2, default=str)
        
        # Export empty chunks (streaming flow doesn't use chunks)
        chunks_data = []
        with open(data_dir / "chunks.json", 'w') as f:
            json.dump(chunks_data, f, indent=2, default=str)
        
        # Export metadata
        metadata = {
            "processed_documents": list(streaming_state.processed_docs),
            "failed_documents": streaming_state.failed_docs,
            "detected_domains": list(streaming_state.detected_domains),
            "progress": streaming_state.progress.model_dump(),
            "export_timestamp": datetime.now().isoformat(),
            "total_entities": len(streaming_state.accumulated_entities),
            "total_relationships": len(streaming_state.accumulated_relationships)
        }
        
        with open(data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Create simple manifest
        import yaml
        
        manifest_data = {
            "name": final_plugin_name,
            "description": f"Complete knowledge base from {len(streaming_state.processed_docs)} documents",
            "version": "1.0.0",
            "domains": list(streaming_state.detected_domains),
            "stats": {
                "entities": len(streaming_state.accumulated_entities),
                "relationships": len(streaming_state.accumulated_relationships),
                "documents": len(streaming_state.processed_docs)
            },
            "databases": {}
        }
        
        if request.use_vector_db and (databases_dir / "chromadb").exists():
            manifest_data["databases"]["chromadb"] = {
                "enabled": True,
                "config_file": "chromadb_config.yaml"
            }
        
        if request.use_graph_db and (databases_dir / "neo4j").exists():
            manifest_data["databases"]["neo4j"] = {
                "enabled": True,
                "config_file": "neo4j_config.yaml"
            }
        
        with open(final_plugin_dir / "manifest.yaml", 'w') as f:
            yaml.dump(manifest_data, f, default_flow_style=False)
        
        # Create simple README
        readme_content = f"""# {final_plugin_name}

Complete knowledge base extracted from {len(streaming_state.processed_docs)} documents.

## Statistics
- **Entities**: {len(streaming_state.accumulated_entities)}
- **Relationships**: {len(streaming_state.accumulated_relationships)}
- **Domains**: {', '.join(streaming_state.detected_domains)}

## Files
- `manifest.yaml`: Plugin metadata and configuration
- `data/entities.json`: Extracted entities
- `data/relationships.json`: Extracted relationships  
- `data/metadata.json`: Processing metadata
- `databases/`: Embedded database files (if enabled)

Generated by AI-Flowlib streaming knowledge extraction.
"""
        
        with open(final_plugin_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        logger.info(f"✅ Final plugin exported to: {final_plugin_dir}")
        return str(final_plugin_dir)

    async def _calculate_final_stats(self, streaming_state: ExtractionState, start_time: datetime) -> KnowledgeBaseStats:
        """Calculate final processing statistics."""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Count entity types
        entity_types = {}
        for entity in streaming_state.accumulated_entities:
            entity_type = entity.entity_type
            entity_types[entity_type] = entity_types[entity_type] + 1 if entity_type in entity_types else 1
        
        # Count relationship types  
        relationship_types = {}
        for rel in streaming_state.accumulated_relationships:
            rel_type = rel.relationship_type
            relationship_types[rel_type] = relationship_types[rel_type] + 1 if rel_type in relationship_types else 1
        
        # Calculate other stats
        total_words = streaming_state.progress.chunks_created * 100  # Approximate
        total_chars = total_words * 5  # Approximate
        
        return KnowledgeBaseStats(
            total_documents=len(streaming_state.processed_docs),
            total_chunks=streaming_state.progress.chunks_created,
            total_entities=len(streaming_state.accumulated_entities),
            total_relationships=len(streaming_state.accumulated_relationships),
            total_vectors=streaming_state.progress.chunks_created,  # One vector per chunk
            successful_documents=len(streaming_state.processed_docs),
            failed_documents=len(streaming_state.failed_docs),
            processing_time_seconds=processing_time,
            total_words=total_words,
            total_characters=total_chars,
            average_document_length=total_chars / len(streaming_state.processed_docs) if streaming_state.processed_docs else 0,
            entity_types=entity_types,
            relationship_types=relationship_types,
            topics_discovered=len(streaming_state.detected_domains),
            top_topics=[
                {"name": domain, "count": 1} 
                for domain in list(streaming_state.detected_domains)[:5]
            ]
        )