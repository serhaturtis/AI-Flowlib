"""Knowledge plugin service - Single source of truth for plugin operations.

This service manages the complete lifecycle of knowledge plugins following
flowlib-server patterns: strict validation, fail-fast, no fallbacks.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from flowlib.flows.registry.registry import flow_registry
from flowlib.knowledge.plugin_generation.models import (
    PluginGenerationRequest as FlowlibPluginRequest,
    PluginGenerationResult as FlowlibPluginResult,
)
from server.models.knowledge import (
    DocumentListResponse,
    DocumentSummary,
    DocumentType,
    DocumentUploadResponse,
    DocumentUploadResult,
    EntityListResponse,
    EntitySummary,
    ExtractionStats,
    PluginCapabilities,
    PluginDeleteResponse,
    PluginDetails,
    PluginGenerationRequest,
    PluginGenerationResponse,
    PluginListResponse,
    PluginQueryRequest,
    PluginQueryResponse,
    PluginSummary,
    QueryResultItem,
    RelationshipListResponse,
    RelationshipSummary,
)

logger = logging.getLogger(__name__)


class KnowledgePluginService:
    """Manage knowledge plugin lifecycle with strict validation.

    This service provides operations for:
    - Plugin generation from documents
    - Plugin listing and details
    - Plugin querying
    - Plugin deletion
    - Entity/relationship browsing
    - Document upload management

    All operations fail-fast with clear error messages.
    No fallbacks or silent failures.
    """

    def __init__(self, projects_root: str = "./projects") -> None:
        """Initialize the knowledge plugin service.

        Args:
            projects_root: Root directory for all projects
        """
        self._root = Path(projects_root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._uploads_dir = self._root / "_uploads"
        self._uploads_dir.mkdir(exist_ok=True)

    # ========== PLUGIN LISTING AND DETAILS ==========

    def list_plugins(self, project_id: str) -> PluginListResponse:
        """List all knowledge plugins for a project.

        Args:
            project_id: Project identifier

        Returns:
            PluginListResponse with all available plugins

        Raises:
            FileNotFoundError: If project doesn't exist
            ValueError: If project_id is invalid
        """
        project_path = self._resolve_project_path(project_id)
        plugins_dir = project_path / "knowledge_plugins"

        if not plugins_dir.exists():
            # No plugins directory means no plugins - return empty list
            return PluginListResponse(project_id=project_id, plugins=[], total=0)

        plugins: list[PluginSummary] = []

        for plugin_dir in plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue

            manifest_path = plugin_dir / "manifest.yaml"
            if not manifest_path.exists():
                logger.warning(f"Plugin directory missing manifest: {plugin_dir}")
                continue

            try:
                plugin_summary = self._load_plugin_summary(project_id, plugin_dir.name)
                plugins.append(plugin_summary)
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_dir.name}: {e}")
                # Fail-fast: corrupted plugins should be reported, not silently skipped
                raise RuntimeError(
                    f"Plugin '{plugin_dir.name}' is corrupted or has invalid manifest. "
                    f"Error: {e}"
                ) from e

        # Sort by creation date (newest first)
        plugins.sort(key=lambda p: p.created_at, reverse=True)

        return PluginListResponse(project_id=project_id, plugins=plugins, total=len(plugins))

    def get_plugin_details(self, project_id: str, plugin_id: str) -> PluginDetails:
        """Get detailed information about a specific plugin.

        Args:
            project_id: Project identifier
            plugin_id: Plugin identifier

        Returns:
            PluginDetails with comprehensive plugin information

        Raises:
            FileNotFoundError: If project or plugin doesn't exist
            ValueError: If identifiers are invalid
        """
        plugin_dir = self._resolve_plugin_path(project_id, plugin_id)
        manifest = self._load_manifest(plugin_dir)

        # Load extraction stats
        extraction_stats = ExtractionStats(
            total_documents=manifest.get("extraction_stats", {}).get("total_documents", 0),
            successful_documents=manifest.get("extraction_stats", {}).get("successful_documents", 0),
            failed_documents=manifest.get("extraction_stats", {}).get("failed_documents", 0),
            total_entities=manifest.get("extraction_stats", {}).get("total_entities", 0),
            total_relationships=manifest.get("extraction_stats", {}).get("total_relationships", 0),
            total_chunks=manifest.get("extraction_stats", {}).get("total_chunks", 0),
            processing_time_seconds=manifest.get("extraction_stats", {}).get("processing_time", 0.0),
        )

        # Determine capabilities
        capabilities = PluginCapabilities(
            has_vector_db=manifest.get("requires_vector_db", False),
            has_graph_db=manifest.get("requires_graph_db", False),
            supports_semantic_search=manifest.get("requires_vector_db", False),
            supports_relationship_queries=manifest.get("requires_graph_db", False),
        )

        return PluginDetails(
            plugin_id=plugin_id,
            name=manifest["name"],
            version=manifest["version"],
            description=manifest.get("description", ""),
            author=manifest.get("author", "Unknown"),
            domains=manifest.get("domains", []),
            created_at=manifest.get("created_at", ""),
            extraction_stats=extraction_stats,
            capabilities=capabilities,
            plugin_path=str(plugin_dir),
            chunk_size=manifest.get("chunk_settings", {}).get("chunk_size", 1000),
            chunk_overlap=manifest.get("chunk_settings", {}).get("chunk_overlap", 200),
            domain_strategy=manifest.get("domain_strategy", "generic"),
            files_created=self._list_plugin_files(plugin_dir),
        )

    # ========== PLUGIN GENERATION ==========

    async def generate_plugin(
        self, request: PluginGenerationRequest
    ) -> PluginGenerationResponse:
        """Generate a new knowledge plugin from documents.

        Args:
            request: Plugin generation request with all parameters

        Returns:
            PluginGenerationResponse with generation results

        Raises:
            FileNotFoundError: If input directory doesn't exist
            ValueError: If request parameters are invalid
            RuntimeError: If generation fails
        """
        logger.info(f"Starting plugin generation: {request.plugin_name}")

        # Validate project exists
        project_path = self._resolve_project_path(request.project_id)
        plugins_dir = project_path / "knowledge_plugins"
        plugins_dir.mkdir(exist_ok=True)

        # Check if plugin already exists
        plugin_dir = plugins_dir / request.plugin_name
        if plugin_dir.exists():
            raise ValueError(
                f"Plugin '{request.plugin_name}' already exists in project '{request.project_id}'. "
                f"Delete existing plugin first or choose a different name."
            )

        # Validate input directory exists
        input_path = Path(request.input_directory).resolve()
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input directory not found: {request.input_directory}. "
                f"Ensure documents are uploaded first."
            )

        # Create flowlib plugin generation request
        flowlib_request = FlowlibPluginRequest(
            input_directory=str(input_path),
            output_directory=str(plugin_dir),
            plugin_name=request.plugin_name,
            domains=request.domains,
            description=request.description,
            author=request.author,
            version=request.version,
            use_vector_db=request.use_vector_db,
            use_graph_db=request.use_graph_db,
            max_files=request.max_files,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            domain_strategy=request.domain_strategy,
        )

        try:
            # Get plugin generation flow
            flow = flow_registry.get_flow("plugin-generation")

            # Run generation pipeline
            result: FlowlibPluginResult = await flow.run_pipeline(flowlib_request)

            if not result.success:
                raise RuntimeError(
                    f"Plugin generation failed: {result.error_message or 'Unknown error'}"
                )

            # Map flowlib result to API response
            extraction_stats = ExtractionStats(
                total_documents=result.summary.extraction_stats.total_documents,
                successful_documents=result.summary.extraction_stats.successful_documents,
                failed_documents=result.summary.extraction_stats.failed_documents,
                total_entities=result.summary.extraction_stats.total_entities,
                total_relationships=result.summary.extraction_stats.total_relationships,
                total_chunks=result.summary.extraction_stats.total_chunks,
                processing_time_seconds=result.summary.extraction_stats.processing_time,
            )

            logger.info(f"Plugin generation completed: {request.plugin_name}")

            return PluginGenerationResponse(
                success=True,
                plugin_id=request.plugin_name,
                plugin_path=result.plugin_path,
                extraction_stats=extraction_stats,
                files_created=result.summary.files_created,
                message=f"Successfully generated plugin '{request.plugin_name}'",
                error_message=None,
            )

        except Exception as e:
            logger.error(f"Plugin generation failed: {e}")

            # Clean up failed plugin directory
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir, ignore_errors=True)

            # Re-raise with clear error message
            raise RuntimeError(
                f"Plugin generation failed for '{request.plugin_name}': {str(e)}"
            ) from e

    # ========== PLUGIN DELETION ==========

    def delete_plugin(self, project_id: str, plugin_id: str) -> PluginDeleteResponse:
        """Delete a knowledge plugin.

        Args:
            project_id: Project identifier
            plugin_id: Plugin identifier

        Returns:
            PluginDeleteResponse with deletion confirmation

        Raises:
            FileNotFoundError: If project or plugin doesn't exist
            ValueError: If identifiers are invalid
        """
        plugin_dir = self._resolve_plugin_path(project_id, plugin_id)

        logger.info(f"Deleting plugin: {plugin_id}")

        try:
            shutil.rmtree(plugin_dir)
            logger.info(f"Plugin deleted successfully: {plugin_id}")

            return PluginDeleteResponse(
                success=True,
                plugin_id=plugin_id,
                message=f"Successfully deleted plugin '{plugin_id}'",
            )

        except Exception as e:
            logger.error(f"Plugin deletion failed: {e}")
            raise RuntimeError(f"Failed to delete plugin '{plugin_id}': {str(e)}") from e

    # ========== PLUGIN QUERYING ==========

    async def query_plugin(
        self, project_id: str, plugin_id: str, request: PluginQueryRequest
    ) -> PluginQueryResponse:
        """Query a knowledge plugin for relevant information.

        Args:
            project_id: Project identifier
            plugin_id: Plugin identifier
            request: Query request with search parameters

        Returns:
            PluginQueryResponse with search results

        Raises:
            FileNotFoundError: If project or plugin doesn't exist
            ValueError: If request parameters are invalid
        """
        plugin_dir = self._resolve_plugin_path(project_id, plugin_id)
        data_dir = plugin_dir / "data"

        if not data_dir.exists():
            raise RuntimeError(f"Plugin '{plugin_id}' has no data directory")

        start_time = datetime.now()
        results: list[QueryResultItem] = []

        # Load plugin data
        entities = self._load_json_data(data_dir / "entities.json")
        relationships = self._load_json_data(data_dir / "relationships.json")

        query_lower = request.query.lower()

        # Search entities
        for entity in entities:
            if request.entity_types and entity.get("entity_type") not in request.entity_types:
                continue

            name = entity.get("name", "").lower()
            description = entity.get("description", "").lower()
            confidence = entity.get("confidence", 0.0)

            if confidence < request.min_confidence:
                continue

            if query_lower in name or query_lower in description:
                relevance = 1.0 if query_lower in name else 0.7
                results.append(
                    QueryResultItem(
                        item_type="entity",
                        item_id=entity.get("entity_id", ""),
                        name=entity.get("name", ""),
                        description=entity.get("description"),
                        relevance_score=relevance * confidence,
                        metadata={
                            "entity_type": entity.get("entity_type"),
                            "frequency": entity.get("frequency", 0),
                        },
                    )
                )

        # Search relationships
        for relationship in relationships:
            if (
                request.relationship_types
                and relationship.get("relationship_type") not in request.relationship_types
            ):
                continue

            description = relationship.get("description", "").lower()
            confidence = relationship.get("confidence", 0.0)

            if confidence < request.min_confidence:
                continue

            if query_lower in description:
                results.append(
                    QueryResultItem(
                        item_type="relationship",
                        item_id=relationship.get("relationship_id", ""),
                        name=f"{relationship.get('source_entity_id')} -> {relationship.get('target_entity_id')}",
                        description=relationship.get("description"),
                        relevance_score=0.6 * confidence,
                        metadata={"relationship_type": relationship.get("relationship_type")},
                    )
                )

        # Sort by relevance and apply limit
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        results = results[: request.limit]

        processing_time = (datetime.now() - start_time).total_seconds()

        return PluginQueryResponse(
            query=request.query,
            results=results,
            total_found=len(results),
            processing_time_seconds=processing_time,
        )

    # ========== ENTITY/RELATIONSHIP/DOCUMENT BROWSING ==========

    def list_entities(
        self, project_id: str, plugin_id: str, limit: int = 100, offset: int = 0
    ) -> EntityListResponse:
        """List entities from a knowledge plugin.

        Args:
            project_id: Project identifier
            plugin_id: Plugin identifier
            limit: Maximum entities to return
            offset: Number of entities to skip

        Returns:
            EntityListResponse with entity list

        Raises:
            FileNotFoundError: If project or plugin doesn't exist
        """
        plugin_dir = self._resolve_plugin_path(project_id, plugin_id)
        entities_file = plugin_dir / "data" / "entities.json"

        if not entities_file.exists():
            return EntityListResponse(plugin_id=plugin_id, entities=[], total=0)

        entities_data = self._load_json_data(entities_file)
        total = len(entities_data)

        # Apply pagination
        paginated = entities_data[offset : offset + limit]

        entities = [
            EntitySummary(
                entity_id=e.get("entity_id", ""),
                name=e.get("name", ""),
                entity_type=e.get("entity_type", ""),
                description=e.get("description"),
                confidence=e.get("confidence", 0.0),
                frequency=e.get("frequency", 0),
                documents=e.get("documents", []),
            )
            for e in paginated
        ]

        return EntityListResponse(plugin_id=plugin_id, entities=entities, total=total)

    def list_relationships(
        self, project_id: str, plugin_id: str, limit: int = 100, offset: int = 0
    ) -> RelationshipListResponse:
        """List relationships from a knowledge plugin.

        Args:
            project_id: Project identifier
            plugin_id: Plugin identifier
            limit: Maximum relationships to return
            offset: Number of relationships to skip

        Returns:
            RelationshipListResponse with relationship list

        Raises:
            FileNotFoundError: If project or plugin doesn't exist
        """
        plugin_dir = self._resolve_plugin_path(project_id, plugin_id)
        relationships_file = plugin_dir / "data" / "relationships.json"

        if not relationships_file.exists():
            return RelationshipListResponse(plugin_id=plugin_id, relationships=[], total=0)

        relationships_data = self._load_json_data(relationships_file)
        total = len(relationships_data)

        # Apply pagination
        paginated = relationships_data[offset : offset + limit]

        relationships = [
            RelationshipSummary(
                relationship_id=r.get("relationship_id", ""),
                source_entity_id=r.get("source_entity_id", ""),
                target_entity_id=r.get("target_entity_id", ""),
                relationship_type=r.get("relationship_type", ""),
                description=r.get("description"),
                confidence=r.get("confidence", 0.0),
                frequency=r.get("frequency", 0),
                documents=r.get("documents", []),
            )
            for r in paginated
        ]

        return RelationshipListResponse(
            plugin_id=plugin_id, relationships=relationships, total=total
        )

    def list_documents(
        self, project_id: str, plugin_id: str, limit: int = 100, offset: int = 0
    ) -> DocumentListResponse:
        """List documents from a knowledge plugin.

        Args:
            project_id: Project identifier
            plugin_id: Plugin identifier
            limit: Maximum documents to return
            offset: Number of documents to skip

        Returns:
            DocumentListResponse with document list

        Raises:
            FileNotFoundError: If project or plugin doesn't exist
        """
        plugin_dir = self._resolve_plugin_path(project_id, plugin_id)
        documents_file = plugin_dir / "data" / "documents.json"

        if not documents_file.exists():
            return DocumentListResponse(plugin_id=plugin_id, documents=[], total=0)

        documents_data = self._load_json_data(documents_file)
        total = len(documents_data)

        # Apply pagination
        paginated = documents_data[offset : offset + limit]

        documents = [
            DocumentSummary(
                document_id=d.get("document_id", ""),
                file_name=d.get("metadata", {}).get("file_name", ""),
                file_type=DocumentType(d.get("metadata", {}).get("file_type", "txt")),
                word_count=d.get("word_count", 0),
                chunk_count=len(d.get("chunks", [])),
                summary=d.get("summary"),
            )
            for d in paginated
        ]

        return DocumentListResponse(plugin_id=plugin_id, documents=documents, total=total)

    # ========== DOCUMENT UPLOAD ==========

    async def upload_documents(
        self, project_id: str, files: list[tuple[str, bytes, str]]
    ) -> DocumentUploadResponse:
        """Upload documents for plugin generation.

        Args:
            project_id: Project identifier
            files: List of (filename, content, content_type) tuples

        Returns:
            DocumentUploadResponse with upload results

        Raises:
            ValueError: If files are invalid
        """
        # Validate project exists
        self._resolve_project_path(project_id)

        # Create unique upload session
        upload_id = str(uuid.uuid4())
        upload_dir = self._uploads_dir / project_id / upload_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        uploaded_files: list[DocumentUploadResult] = []
        total_size = 0

        for filename, content, content_type in files:
            # Validate file type
            file_ext = Path(filename).suffix.lower().lstrip(".")
            try:
                doc_type = DocumentType(file_ext)
            except ValueError:
                raise ValueError(
                    f"Unsupported file type '{file_ext}' for file '{filename}'. "
                    f"Supported types: {[t.value for t in DocumentType]}"
                )

            # Save file
            file_path = upload_dir / filename
            file_path.write_bytes(content)
            file_size = len(content)
            total_size += file_size

            uploaded_files.append(
                DocumentUploadResult(
                    filename=filename,
                    file_path=str(file_path),
                    file_size=file_size,
                    file_type=doc_type,
                )
            )

            logger.info(f"Uploaded file: {filename} ({file_size} bytes)")

        return DocumentUploadResponse(
            upload_id=upload_id,
            upload_directory=str(upload_dir),
            uploaded_files=uploaded_files,
            total_uploaded=len(uploaded_files),
            total_size_bytes=total_size,
        )

    def cleanup_upload(self, project_id: str, upload_id: str) -> None:
        """Clean up uploaded files after plugin generation.

        Args:
            project_id: Project identifier
            upload_id: Upload session identifier
        """
        upload_dir = self._uploads_dir / project_id / upload_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            logger.info(f"Cleaned up upload: {upload_id}")

    # ========== PRIVATE HELPER METHODS ==========

    def _resolve_project_path(self, project_id: str) -> Path:
        """Resolve and validate project path.

        Args:
            project_id: Project identifier

        Returns:
            Resolved project path

        Raises:
            FileNotFoundError: If project doesn't exist
            ValueError: If project_id escapes root
        """
        project_path = (self._root / project_id).resolve()

        if not project_path.exists() or not project_path.is_dir():
            raise FileNotFoundError(
                f"Project '{project_id}' not found under {self._root}. "
                f"Create project first before managing knowledge plugins."
            )

        if project_path == self._root or self._root not in project_path.parents:
            raise ValueError(
                f"Project '{project_id}' resolved outside managed root {self._root}. "
                f"This is a security violation."
            )

        return project_path

    def _resolve_plugin_path(self, project_id: str, plugin_id: str) -> Path:
        """Resolve and validate plugin path.

        Args:
            project_id: Project identifier
            plugin_id: Plugin identifier

        Returns:
            Resolved plugin path

        Raises:
            FileNotFoundError: If project or plugin doesn't exist
            ValueError: If plugin_id escapes project
        """
        project_path = self._resolve_project_path(project_id)
        plugin_path = (project_path / "knowledge_plugins" / plugin_id).resolve()

        if not plugin_path.exists():
            raise FileNotFoundError(
                f"Plugin '{plugin_id}' not found in project '{project_id}'. "
                f"Verify plugin name or generate it first."
            )

        plugins_dir = project_path / "knowledge_plugins"
        if plugin_path.parent != plugins_dir:
            raise ValueError(
                f"Plugin '{plugin_id}' resolved outside plugins directory. "
                f"This is a security violation."
            )

        return plugin_path

    def _load_plugin_summary(self, project_id: str, plugin_id: str) -> PluginSummary:
        """Load plugin summary from manifest.

        Args:
            project_id: Project identifier
            plugin_id: Plugin identifier

        Returns:
            PluginSummary with basic plugin information

        Raises:
            RuntimeError: If manifest is corrupted
        """
        project_path = self._resolve_project_path(project_id)
        plugin_dir = project_path / "knowledge_plugins" / plugin_id
        manifest = self._load_manifest(plugin_dir)

        extraction_stats = ExtractionStats(
            total_documents=manifest.get("extraction_stats", {}).get("total_documents", 0),
            successful_documents=manifest.get("extraction_stats", {}).get("successful_documents", 0),
            failed_documents=manifest.get("extraction_stats", {}).get("failed_documents", 0),
            total_entities=manifest.get("extraction_stats", {}).get("total_entities", 0),
            total_relationships=manifest.get("extraction_stats", {}).get("total_relationships", 0),
            total_chunks=manifest.get("extraction_stats", {}).get("total_chunks", 0),
            processing_time_seconds=manifest.get("extraction_stats", {}).get("processing_time", 0.0),
        )

        capabilities = PluginCapabilities(
            has_vector_db=manifest.get("requires_vector_db", False),
            has_graph_db=manifest.get("requires_graph_db", False),
            supports_semantic_search=manifest.get("requires_vector_db", False),
            supports_relationship_queries=manifest.get("requires_graph_db", False),
        )

        return PluginSummary(
            plugin_id=plugin_id,
            name=manifest["name"],
            version=manifest["version"],
            description=manifest.get("description", ""),
            author=manifest.get("author", "Unknown"),
            domains=manifest.get("domains", []),
            created_at=manifest.get("created_at", ""),
            extraction_stats=extraction_stats,
            capabilities=capabilities,
        )

    def _load_manifest(self, plugin_dir: Path) -> dict[str, Any]:
        """Load and validate plugin manifest.

        Args:
            plugin_dir: Plugin directory path

        Returns:
            Manifest data dictionary

        Raises:
            RuntimeError: If manifest is missing or invalid
        """
        manifest_path = plugin_dir / "manifest.yaml"

        if not manifest_path.exists():
            raise RuntimeError(
                f"Plugin manifest not found: {manifest_path}. "
                f"Plugin may be corrupted."
            )

        try:
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)

            # Validate required fields
            required_fields = ["name", "version"]
            for field in required_fields:
                if field not in manifest:
                    raise ValueError(f"Manifest missing required field: {field}")

            return manifest

        except Exception as e:
            raise RuntimeError(f"Failed to load plugin manifest: {e}") from e

    def _load_json_data(self, file_path: Path) -> list[dict[str, Any]]:
        """Load JSON data file.

        Args:
            file_path: Path to JSON file

        Returns:
            Loaded data as list of dictionaries

        Raises:
            RuntimeError: If file is missing or invalid
        """
        if not file_path.exists():
            return []

        try:
            with open(file_path) as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON data from {file_path}: {e}") from e

    def _list_plugin_files(self, plugin_dir: Path) -> list[str]:
        """List all files in plugin directory.

        Args:
            plugin_dir: Plugin directory path

        Returns:
            List of relative file paths
        """
        files = []
        for path in plugin_dir.rglob("*"):
            if path.is_file():
                relative = path.relative_to(plugin_dir)
                files.append(str(relative))
        return sorted(files)
