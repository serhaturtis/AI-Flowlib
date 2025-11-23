"""Knowledge orchestration flow."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from flowlib.config.required_resources import RequiredAlias
from flowlib.flows.decorators.decorators import flow, pipeline
from flowlib.flows.registry.registry import flow_registry
from flowlib.providers.core.registry import provider_registry
from flowlib.providers.embedding.base import EmbeddingProvider
from flowlib.knowledge.models import (
    DocumentContent,
    DocumentExtractionInput,
    Entity,
    EntityExtractionInput,
    GraphStoreInput,
    Relationship,
    VectorStoreInput,
)
from flowlib.knowledge.orchestration.models import (
    OrchestrationProgress,
    OrchestrationRequest,
    OrchestrationResult,
)

logger = logging.getLogger(__name__)


@flow(name="knowledge-orchestration", description="Orchestrate knowledge extraction pipeline")  # type: ignore[arg-type]
class KnowledgeOrchestrationFlow:
    """Orchestrates the complete knowledge extraction pipeline.

    This flow coordinates document extraction, entity analysis, and storage
    across vector and graph databases in a structured pipeline.
    """

    @pipeline(input_model=OrchestrationRequest, output_model=OrchestrationResult)
    async def run_pipeline(self, request: OrchestrationRequest) -> OrchestrationResult:
        """Execute the complete knowledge orchestration pipeline."""

        logger.info("🚀 Starting knowledge orchestration pipeline")
        logger.info(f"📂 Input: {request.input_directory}")
        logger.info(f"📁 Output: {request.output_directory}")

        start_time = datetime.now()
        progress = OrchestrationProgress()

        try:
            # Initialize progress tracking
            total_files = await self._count_documents(
                request.input_directory, request.supported_formats
            )
            progress.total_documents = (
                min(total_files, request.max_files) if request.max_files else total_files
            )

            logger.info(f"📊 Total documents to process: {progress.total_documents}")

            # Stage 1: Document Extraction
            progress.current_stage = "document_extraction"
            logger.info("📝 Stage 1: Document Extraction")

            documents = await self._extract_documents(request, progress)
            progress.extraction_complete = True
            progress.processed_documents = len(documents)

            # Stage 2: Entity Analysis
            progress.current_stage = "entity_analysis"
            logger.info("🔍 Stage 2: Entity Analysis")

            entities, relationships = await self._analyze_entities(documents, progress)
            progress.analysis_complete = True
            progress.total_entities = len(entities)
            progress.total_relationships = len(relationships)

            # Stage 3: Vector Storage (parallel)
            progress.current_stage = "vector_storage"
            logger.info("📊 Stage 3: Vector Storage")

            vector_task = asyncio.create_task(self._store_vectors(documents, request, progress))

            # Stage 4: Graph Storage (parallel)
            progress.current_stage = "graph_storage"
            logger.info("🔗 Stage 4: Graph Storage")

            graph_task = asyncio.create_task(
                self._store_graph(documents, entities, relationships, request, progress)
            )

            # Wait for both storage operations
            vector_result, graph_result = await asyncio.gather(vector_task, graph_task)

            progress.vector_storage_complete = True
            progress.graph_storage_complete = True

            # Stage 5: Export and Finalization
            progress.current_stage = "finalization"
            logger.info("📦 Stage 5: Export and Finalization")

            output_files = await self._export_results(request, documents, entities, relationships)

            processing_time = (datetime.now() - start_time).total_seconds()

            logger.info("✅ Knowledge orchestration pipeline completed successfully!")
            logger.info(
                f"📊 Final stats: {len(entities)} entities, {len(relationships)} relationships"
            )
            logger.info(f"⏱️ Processing time: {processing_time:.2f} seconds")

            return OrchestrationResult(
                status="completed",
                message=f"Successfully processed {len(documents)} documents",
                progress=progress,
                documents=documents,
                entities=entities,
                relationships=relationships,
                vector_collection=request.collection_name,
                graph_database=request.graph_name,
                output_files=output_files,
                export_directory=request.output_directory,
                processing_time_seconds=processing_time,
                total_size_bytes=sum(len(doc.full_text.encode()) for doc in documents),
            )

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"❌ Knowledge orchestration pipeline failed: {e}")

            return OrchestrationResult(
                status="failed",
                message=f"Pipeline failed: {str(e)}",
                progress=progress,
                processing_time_seconds=processing_time,
            )

    async def _count_documents(self, input_dir: str, supported_formats: list) -> int:
        """Count total documents to process."""

        extensions = {f".{fmt.value}" for fmt in supported_formats}
        count = 0

        input_path = Path(input_dir)
        for file_path in input_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                count += 1

        return count

    async def _extract_documents(
        self, request: OrchestrationRequest, progress: OrchestrationProgress
    ) -> list[DocumentContent]:
        """Extract documents from input directory."""

        # Collect file paths
        input_path = Path(request.input_directory)
        extensions = {f".{fmt.value}" for fmt in request.supported_formats}
        file_paths = []

        for file_path in input_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                file_paths.append(str(file_path))
                if request.max_files and len(file_paths) >= request.max_files:
                    break

        # Extract documents
        from flowlib.knowledge.models import KnowledgeExtractionRequest

        # Get embedding provider to retrieve model configuration
        embedding_provider_temp = await provider_registry.get_by_config(RequiredAlias.DEFAULT_EMBEDDING.value)
        await embedding_provider_temp.initialize()
        embedding_provider_cast = cast(EmbeddingProvider, embedding_provider_temp)

        # Get model config from provider settings
        embedding_model = embedding_provider_cast.settings.model_name
        vector_dimensions = embedding_provider_cast.settings.embedding_dim

        # Shutdown temporary provider
        await embedding_provider_temp.shutdown()

        # Convert OrchestrationRequest to KnowledgeExtractionRequest
        knowledge_request = KnowledgeExtractionRequest(
            input_directory=request.input_directory,
            output_directory=request.output_directory,
            collection_name=request.collection_name,
            graph_name=request.graph_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            max_files=request.max_files,
            supported_formats=request.supported_formats,
            extract_entities=request.extract_entities,
            extract_relationships=request.extract_relationships,
            create_summaries=request.create_summaries,
            detect_topics=request.detect_topics,
            # Use defaults for fields not in OrchestrationRequest
            extraction_domain="general",
            llm_model_name=RequiredAlias.DEFAULT_LLM.value,
            embedding_model=embedding_model,
            vector_dimensions=vector_dimensions,
            vector_provider_config=request.vector_provider_config or RequiredAlias.DEFAULT_VECTOR_DB.value,
            graph_provider_config=request.graph_provider_config or RequiredAlias.DEFAULT_GRAPH_DB.value,
            embedding_provider_config=RequiredAlias.DEFAULT_EMBEDDING.value,
            enable_graph_analysis=True,
            min_entity_frequency=2,
            min_relationship_confidence=0.7,
            use_vector_db=True,
            use_graph_db=True,
            resume_from_checkpoint=False,  # New orchestration doesn't resume
            plugin_name_prefix="knowledge_extraction",
        )

        extraction_flow = flow_registry.get_flow("document-extraction-flow")
        extraction_input = DocumentExtractionInput(request=knowledge_request, file_paths=file_paths)

        extraction_result = await extraction_flow.run_pipeline(extraction_input)
        documents: list[DocumentContent] = extraction_result.documents
        return documents

    async def _analyze_entities(
        self, documents: list[DocumentContent], progress: OrchestrationProgress
    ) -> tuple[list[Entity], list[Relationship]]:
        """Analyze entities and relationships from documents."""

        analysis_flow = flow_registry.get_flow("entity-analysis-flow")
        analysis_input = EntityExtractionInput(
            documents=documents,
            extraction_domain="general",
            llm_model_name="music-album-model",
            min_entity_frequency=2,
            min_relationship_confidence=0.7,
        )

        analysis_result = await analysis_flow.run_pipeline(analysis_input)
        return analysis_result.entities, analysis_result.relationships

    async def _store_vectors(
        self,
        documents: list[DocumentContent],
        request: OrchestrationRequest,
        progress: OrchestrationProgress,
    ) -> Any:
        """Store documents in vector database."""

        if not request.vector_provider_config:
            logger.info("Skipping vector storage (not configured)")
            return None

        # Get embedding provider to retrieve model configuration
        embedding_provider_temp = await provider_registry.get_by_config(RequiredAlias.DEFAULT_EMBEDDING.value)
        await embedding_provider_temp.initialize()
        embedding_provider_cast = cast(EmbeddingProvider, embedding_provider_temp)

        # Get model config from provider settings
        embedding_model = embedding_provider_cast.settings.model_name
        vector_dimensions = embedding_provider_cast.settings.embedding_dim

        # Shutdown temporary provider
        await embedding_provider_temp.shutdown()

        vector_flow = flow_registry.get_flow("vector-storage-flow")
        vector_input = VectorStoreInput(
            documents=documents,
            collection_name=request.collection_name,
            embedding_model=embedding_model,
            vector_dimensions=vector_dimensions,
            vector_provider_config=request.vector_provider_config or RequiredAlias.DEFAULT_VECTOR_DB.value,
            embedding_provider_config=RequiredAlias.DEFAULT_EMBEDDING.value,
        )

        return await vector_flow.run_pipeline(vector_input)

    async def _store_graph(
        self,
        documents: list[DocumentContent],
        entities: list[Entity],
        relationships: list[Relationship],
        request: OrchestrationRequest,
        progress: OrchestrationProgress,
    ) -> Any:
        """Store entities and relationships in graph database."""

        if not request.graph_provider_config:
            logger.info("Skipping graph storage (not configured)")
            return None

        graph_flow = flow_registry.get_flow("graph-storage-flow")
        graph_input = GraphStoreInput(
            documents=documents,
            entities=entities,
            relationships=relationships,
            graph_name=request.graph_name,
            graph_provider_config=request.graph_provider_config or RequiredAlias.DEFAULT_GRAPH_DB.value,
            query_entity_id="",
            query_entity_type="",
            query_source_id="",
            query_target_id="",
            query_limit=100,
        )

        return await graph_flow.run_pipeline(graph_input)

    async def _export_results(
        self,
        request: OrchestrationRequest,
        documents: list[DocumentContent],
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> list[str]:
        """Export results to files."""

        output_path = Path(request.output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        output_files = []

        # Export documents
        documents_file = output_path / "documents.json"
        with open(documents_file, "w") as f:
            json.dump([doc.model_dump() for doc in documents], f, indent=2, default=str)
        output_files.append(str(documents_file))

        # Export entities
        entities_file = output_path / "entities.json"
        with open(entities_file, "w") as f:
            json.dump([entity.model_dump() for entity in entities], f, indent=2, default=str)
        output_files.append(str(entities_file))

        # Export relationships
        relationships_file = output_path / "relationships.json"
        with open(relationships_file, "w") as f:
            json.dump([rel.model_dump() for rel in relationships], f, indent=2, default=str)
        output_files.append(str(relationships_file))

        # Export summary
        summary_file = output_path / "summary.json"
        summary = {
            "total_documents": len(documents),
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "export_time": datetime.now().isoformat(),
            "collection_name": request.collection_name,
            "graph_name": request.graph_name,
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        output_files.append(str(summary_file))

        logger.info(f"Exported results to {len(output_files)} files")
        return output_files
