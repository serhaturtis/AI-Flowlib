"""Knowledge plugin management endpoints."""

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from server.core.config import settings
from server.core.workspace import WorkspaceNotConfiguredError, get_workspace_path
from server.models.knowledge import (
    DocumentListResponse,
    EntityListResponse,
    PluginDeleteResponse,
    PluginDetails,
    PluginGenerationRequest,
    PluginGenerationResponse,
    PluginListResponse,
    PluginQueryRequest,
    PluginQueryResponse,
    RelationshipListResponse,
)
from server.services.knowledge_plugin_service import KnowledgePluginService

router = APIRouter()


def _get_knowledge_service() -> KnowledgePluginService:
    """Get knowledge service with current workspace path."""
    try:
        workspace_path = get_workspace_path()
        return KnowledgePluginService(workspace_path)
    except WorkspaceNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


# ========== PLUGIN LISTING AND DETAILS ==========


@router.get("/plugins", response_model=PluginListResponse)
async def list_plugins(
    project_id: str = Query(..., min_length=1, description="Project identifier")
) -> PluginListResponse:
    """List all knowledge plugins for a project.

    Returns all available knowledge plugins with their metadata and statistics.
    Plugins are sorted by creation date (newest first).

    Args:
        project_id: Target project identifier

    Returns:
        PluginListResponse with all available plugins

    Raises:
        404: Project not found
        400: Invalid project_id
    """
    service = _get_knowledge_service()
    try:
        return await run_in_threadpool(service.list_plugins, project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/plugins/{plugin_id}", response_model=PluginDetails)
async def get_plugin_details(
    plugin_id: str,
    project_id: str = Query(..., min_length=1, description="Project identifier"),
) -> PluginDetails:
    """Get detailed information about a specific plugin.

    Returns comprehensive plugin details including extraction statistics,
    capabilities, configuration, and generated files.

    Args:
        plugin_id: Plugin identifier
        project_id: Project identifier

    Returns:
        PluginDetails with complete plugin information

    Raises:
        404: Project or plugin not found
        400: Invalid identifiers
    """
    service = _get_knowledge_service()
    try:
        return await run_in_threadpool(service.get_plugin_details, project_id, plugin_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


# ========== PLUGIN GENERATION ==========


@router.post("/plugins/generate", response_model=PluginGenerationResponse, status_code=status.HTTP_201_CREATED)
async def generate_plugin(request: PluginGenerationRequest) -> PluginGenerationResponse:
    """Generate a new knowledge plugin from documents.

    Creates a new knowledge plugin by:
    1. Extracting text from uploaded documents
    2. Analyzing content with LLM to extract entities and relationships
    3. Creating vector embeddings for semantic search
    4. Building knowledge graph for relationship queries
    5. Packaging everything into a self-contained plugin

    This is a long-running operation that may take several minutes depending
    on the number and size of documents.

    Args:
        request: Plugin generation configuration

    Returns:
        PluginGenerationResponse with generation results and statistics

    Raises:
        404: Input directory not found
        409: Plugin with same name already exists
        400: Invalid request parameters
        500: Generation failed
    """
    service = _get_knowledge_service()
    try:
        return await service.generate_plugin(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        # Plugin exists or invalid parameters
        if "already exists" in str(exc):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc


# ========== PLUGIN DELETION ==========


@router.delete("/plugins/{plugin_id}", response_model=PluginDeleteResponse)
async def delete_plugin(
    plugin_id: str,
    project_id: str = Query(..., min_length=1, description="Project identifier"),
) -> PluginDeleteResponse:
    """Delete a knowledge plugin.

    Permanently deletes a plugin and all its associated data including:
    - Extracted entities and relationships
    - Vector embeddings
    - Knowledge graph
    - All generated files

    This operation cannot be undone.

    Args:
        plugin_id: Plugin identifier to delete
        project_id: Project identifier

    Returns:
        PluginDeleteResponse with deletion confirmation

    Raises:
        404: Project or plugin not found
        400: Invalid identifiers
        500: Deletion failed
    """
    service = _get_knowledge_service()
    try:
        return await run_in_threadpool(service.delete_plugin, project_id, plugin_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc


# ========== PLUGIN QUERYING ==========


@router.post("/plugins/{plugin_id}/query", response_model=PluginQueryResponse)
async def query_plugin(
    plugin_id: str,
    request: PluginQueryRequest,
    project_id: str = Query(..., min_length=1, description="Project identifier"),
) -> PluginQueryResponse:
    """Query a knowledge plugin for relevant information.

    Searches the plugin's knowledge base using text-based search across:
    - Entity names and descriptions
    - Relationship descriptions
    - Document content

    Results are ranked by relevance and confidence scores.

    Args:
        plugin_id: Plugin identifier to query
        request: Query configuration with search parameters
        project_id: Project identifier

    Returns:
        PluginQueryResponse with ranked search results

    Raises:
        404: Project or plugin not found
        400: Invalid request parameters
        500: Query execution failed
    """
    service = _get_knowledge_service()
    try:
        return await service.query_plugin(project_id, plugin_id, request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc


# ========== ENTITY/RELATIONSHIP/DOCUMENT BROWSING ==========


@router.get("/plugins/{plugin_id}/entities", response_model=EntityListResponse)
async def list_entities(
    plugin_id: str,
    project_id: str = Query(..., min_length=1, description="Project identifier"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum entities to return"),
    offset: int = Query(default=0, ge=0, description="Number of entities to skip"),
) -> EntityListResponse:
    """List entities extracted by a knowledge plugin.

    Returns paginated list of all entities found in the processed documents.
    Entities include people, organizations, locations, concepts, and other
    domain-specific types based on the extraction strategy used.

    Args:
        plugin_id: Plugin identifier
        project_id: Project identifier
        limit: Maximum entities to return (1-1000)
        offset: Pagination offset

    Returns:
        EntityListResponse with entity list and total count

    Raises:
        404: Project or plugin not found
        400: Invalid parameters
    """
    service = _get_knowledge_service()
    try:
        return await run_in_threadpool(service.list_entities, project_id, plugin_id, limit, offset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/plugins/{plugin_id}/relationships", response_model=RelationshipListResponse)
async def list_relationships(
    plugin_id: str,
    project_id: str = Query(..., min_length=1, description="Project identifier"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum relationships to return"),
    offset: int = Query(default=0, ge=0, description="Number of relationships to skip"),
) -> RelationshipListResponse:
    """List relationships extracted by a knowledge plugin.

    Returns paginated list of all relationships found between entities.
    Relationships describe how entities are connected, such as
    "mentions", "defines", "influences", etc.

    Args:
        plugin_id: Plugin identifier
        project_id: Project identifier
        limit: Maximum relationships to return (1-1000)
        offset: Pagination offset

    Returns:
        RelationshipListResponse with relationship list and total count

    Raises:
        404: Project or plugin not found
        400: Invalid parameters
    """
    service = _get_knowledge_service()
    try:
        return await run_in_threadpool(
            service.list_relationships, project_id, plugin_id, limit, offset
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/plugins/{plugin_id}/documents", response_model=DocumentListResponse)
async def list_documents(
    plugin_id: str,
    project_id: str = Query(..., min_length=1, description="Project identifier"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum documents to return"),
    offset: int = Query(default=0, ge=0, description="Number of documents to skip"),
) -> DocumentListResponse:
    """List documents processed by a knowledge plugin.

    Returns paginated list of all documents that were successfully processed
    during plugin generation, including metadata and statistics.

    Args:
        plugin_id: Plugin identifier
        project_id: Project identifier
        limit: Maximum documents to return (1-1000)
        offset: Pagination offset

    Returns:
        DocumentListResponse with document list and total count

    Raises:
        404: Project or plugin not found
        400: Invalid parameters
    """
    service = _get_knowledge_service()
    try:
        return await run_in_threadpool(service.list_documents, project_id, plugin_id, limit, offset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


# ========== DOCUMENT UPLOAD ==========


@router.post("/documents/upload")
async def upload_documents(
    project_id: str = Query(..., min_length=1, description="Project identifier"),
    files: list[UploadFile] = File(..., description="Documents to upload"),
) -> dict[str, str]:
    """Upload documents for plugin generation.

    Uploads one or more documents to a temporary storage location.
    The returned upload_directory should be used as input_directory
    when generating a plugin.

    Supported formats: PDF, TXT, EPUB, MOBI, DOCX, HTML, MD

    Args:
        project_id: Target project identifier
        files: List of files to upload

    Returns:
        Dictionary with upload_id and upload_directory

    Raises:
        404: Project not found
        400: Invalid file types or parameters
    """
    service = _get_knowledge_service()
    try:
        # Convert UploadFile to (filename, content, content_type) tuples
        file_data = []
        for file in files:
            content = await file.read()
            file_data.append((file.filename or "unknown", content, file.content_type or ""))

        result = await service.upload_documents(project_id, file_data)

        return {
            "upload_id": result.upload_id,
            "upload_directory": result.upload_directory,
            "total_uploaded": result.total_uploaded,
            "total_size_bytes": result.total_size_bytes,
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
