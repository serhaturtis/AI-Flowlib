"""Configuration management endpoints."""

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.concurrency import run_in_threadpool

from flowlib.resources.models.constants import ResourceType
from server.core.config import settings
from server.core.workspace import WorkspaceNotConfiguredError, get_workspace_path
from server.models.configs import (
    AgentConfigListResponse,
    AgentConfigResponse,
    AliasListResponse,
    AliasApplyRequest,
    AgentCreateRequest,
    ProviderConfigCreateRequest,
    ProviderConfigUpdateRequest,
    ProviderConfigListResponse,
    ResourceConfigCreateRequest,
    ResourceConfigUpdateRequest,
    ResourceConfigListResponse,
    SchemaResponse,
    ProviderConfigRenderRequest,
    ResourceConfigRenderRequest,
    RenderResponse,
)
from server.models.diff import ConfigApplyResponse
from server.models.diff import ConfigRenameRequest, ConfigDeleteRequest
from server.services.config_service import ConfigService
from server.services.config_scaffold import ConfigScaffoldService
from server.services.schema_service import SchemaService
from server.services.alias_service import AliasService
from server.services.diff_service import DiffService, ConfigValidationError
from server.services.agent_scaffold import AgentScaffoldService
from flowlib.providers.core.registry import provider_registry

router = APIRouter()


def _get_config_services() -> tuple[
    ConfigService, ConfigScaffoldService, AliasService, SchemaService, DiffService, AgentScaffoldService
]:
    """Get all config services with current workspace path."""
    try:
        workspace_path = get_workspace_path()
        config_service = ConfigService(workspace_path)
        config_scaffold_service = ConfigScaffoldService(workspace_path, config_service)
        alias_service = AliasService(workspace_path, config_service)
        schema_service = SchemaService(workspace_path)
        diff_service = DiffService(workspace_path, config_service)
        agent_scaffold_service = AgentScaffoldService(workspace_path, config_service)
        return (
            config_service,
            config_scaffold_service,
            alias_service,
            schema_service,
            diff_service,
            agent_scaffold_service,
        )
    except WorkspaceNotConfiguredError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

@router.get("/providers/schema", response_model=SchemaResponse)
async def get_provider_schema(
    resource_type: ResourceType,
    project_id: str = Query(..., min_length=1),
    provider_type: str | None = Query(default=None),
) -> SchemaResponse:
    """Return provider config schema metadata derived from live project configs."""
    _, _, _, schema_service, _, _ = _get_config_services()
    try:
        return await run_in_threadpool(
            schema_service.provider_schema, project_id, resource_type, provider_type
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/resources/schema", response_model=SchemaResponse)
async def get_resource_schema(
    resource_type: ResourceType,
    project_id: str = Query(..., min_length=1),
    provider_type: str | None = Query(default=None),
) -> SchemaResponse:
    """Return resource config schema metadata derived from live project configs."""
    _, _, _, schema_service, _, _ = _get_config_services()
    try:
        return await run_in_threadpool(
            schema_service.resource_schema, project_id, resource_type, provider_type
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/agents/schema", response_model=SchemaResponse)
async def get_agent_schema(project_id: str = Query(..., min_length=1)) -> SchemaResponse:
    """Return agent config schema metadata."""
    _, _, _, schema_service, _, _ = _get_config_services()
    try:
        return await run_in_threadpool(schema_service.agent_schema, project_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

@router.get("/providers/types", response_model=list[str])
async def list_provider_types(resource_type: ResourceType) -> list[str]:
    """List known provider implementation types for a given provider resource type."""
    # Map provider resource type to category key used in provider_registry factories
    mapping: dict[ResourceType, str] = {
        ResourceType.LLM_CONFIG: "llm",
        ResourceType.MULTIMODAL_LLM_CONFIG: "multimodal_llm",
        ResourceType.VECTOR_DB_CONFIG: "vector_db",
        ResourceType.DATABASE_CONFIG: "db",
        ResourceType.CACHE_CONFIG: "cache",
        ResourceType.STORAGE_CONFIG: "storage",
        ResourceType.EMBEDDING_CONFIG: "embedding",
        ResourceType.GRAPH_DB_CONFIG: "graph_db",
        ResourceType.MESSAGE_QUEUE_CONFIG: "mq",
    }
    if resource_type not in mapping:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported provider resource type '{resource_type}'")
    category = mapping[resource_type]
    # Access registry directly - let any errors propagate (fail-fast)
    factories = getattr(provider_registry, "_factories", {})
    types = sorted({ptype for (cat, ptype) in factories.keys() if cat == category})
    return types


@router.get("/agents", response_model=AgentConfigListResponse)
async def list_agent_configs(project_id: str = Query(..., min_length=1)) -> AgentConfigListResponse:
    """List all agent configurations for a project."""
    config_service, _, _, _, _, _ = _get_config_services()
    try:
        configs = await run_in_threadpool(config_service.list_agent_configs, project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return AgentConfigListResponse(project_id=project_id, configs=configs, total=len(configs))


@router.get("/agents/{config_id}", response_model=AgentConfigResponse)
async def get_agent_config(config_id: str, project_id: str = Query(..., min_length=1)) -> AgentConfigResponse:
    """Get agent configuration."""
    config_service, _, _, _, _, _ = _get_config_services()
    try:
        return await run_in_threadpool(config_service.get_agent_config, project_id, config_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/agents/create", response_model=ConfigApplyResponse, status_code=status.HTTP_201_CREATED)
async def create_agent_config(payload: AgentCreateRequest) -> ConfigApplyResponse:
    """Create agent configuration file (scaffold)."""
    _, _, _, _, _, agent_scaffold_service = _get_config_services()
    try:
        return await run_in_threadpool(agent_scaffold_service.create_agent, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

@router.post("/agents/apply", response_model=ConfigApplyResponse, status_code=status.HTTP_200_OK)
async def apply_agent_structured(
    project_id: str,
    name: str,
    persona: str,
    allowed_tool_categories: list[str],
    knowledge_plugins: list[str] | None = None,
    description: str | None = None,
    model_name: str = "default-model",
    llm_name: str = "default-llm",
    temperature: float = 0.7,
    max_iterations: int = 10,
    enable_learning: bool = False,
    verbose: bool = False,
) -> ConfigApplyResponse:
    """Apply an agent configuration using structured fields (overwrite file content)."""
    _, _, _, _, _, agent_scaffold_service = _get_config_services()
    try:
        return await run_in_threadpool(
            agent_scaffold_service.apply_agent_structured,
            project_id,
            name,
            persona,
            allowed_tool_categories,
            knowledge_plugins if knowledge_plugins is not None else [],
            description,
            model_name,
            llm_name,
            temperature,
            max_iterations,
            enable_learning,
            verbose,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

@router.post("/agents/render", response_model=RenderResponse)
async def render_agent_content(
    name: str,
    persona: str,
    allowed_tool_categories: list[str],
    knowledge_plugins: list[str] | None = None,
    description: str | None = None,
) -> RenderResponse:
    """Render agent config content without writing file (for diff preview)."""
    _, _, _, _, _, agent_scaffold_service = _get_config_services()
    try:
        content = agent_scaffold_service.render_agent_content(
            name=name,
            persona=persona,
            allowed_tool_categories=allowed_tool_categories,
            knowledge_plugins=knowledge_plugins if knowledge_plugins is not None else [],
            description=description,
        )
        return RenderResponse(content=content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/providers", response_model=ProviderConfigListResponse)
async def list_provider_configs(project_id: str = Query(..., min_length=1)) -> ProviderConfigListResponse:
    """List all provider configurations."""
    config_service, _, _, _, _, _ = _get_config_services()
    try:
        configs = await run_in_threadpool(config_service.list_provider_configs, project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return ProviderConfigListResponse(project_id=project_id, configs=configs, total=len(configs))


@router.get("/resources", response_model=ResourceConfigListResponse)
async def list_resource_configs(project_id: str = Query(..., min_length=1)) -> ResourceConfigListResponse:
    """List all resource configurations."""
    config_service, _, _, _, _, _ = _get_config_services()
    try:
        configs = await run_in_threadpool(config_service.list_resource_configs, project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return ResourceConfigListResponse(project_id=project_id, configs=configs, total=len(configs))


@router.get("/aliases", response_model=AliasListResponse)
async def get_aliases(project_id: str = Query(..., min_length=1)) -> AliasListResponse:
    """Get alias mappings."""
    config_service, _, _, _, _, _ = _get_config_services()
    try:
        aliases = await run_in_threadpool(config_service.list_aliases, project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return AliasListResponse(project_id=project_id, aliases=aliases, total=len(aliases))


@router.post("/aliases/apply", response_model=ConfigApplyResponse, status_code=status.HTTP_200_OK)
async def apply_aliases(payload: AliasApplyRequest) -> ConfigApplyResponse:
    """Replace the entire project's alias bindings with given list."""
    _, _, alias_service, _, _, _ = _get_config_services()
    try:
        return await run_in_threadpool(alias_service.apply_aliases, payload.project_id, payload.aliases)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

@router.post(
    "/providers/create",
    response_model=ConfigApplyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_provider_config(payload: ProviderConfigCreateRequest) -> ConfigApplyResponse:
    """Create a provider configuration file."""
    _, config_scaffold_service, _, _, _, _ = _get_config_services()
    try:
        return await run_in_threadpool(config_scaffold_service.create_provider_config, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConfigValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "issues": [i.model_dump() for i in exc.issues]},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/resources/create",
    response_model=ConfigApplyResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_resource_config(payload: ResourceConfigCreateRequest) -> ConfigApplyResponse:
    """Create a resource configuration file."""
    _, config_scaffold_service, _, _, _, _ = _get_config_services()
    try:
        return await run_in_threadpool(config_scaffold_service.create_resource_config, payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConfigValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "issues": [i.model_dump() for i in exc.issues]},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/providers/apply", response_model=ConfigApplyResponse, status_code=status.HTTP_200_OK)
async def apply_provider_structured(payload: ProviderConfigUpdateRequest) -> ConfigApplyResponse:
    """Apply a provider configuration using structured fields (overwrites file content)."""
    _, config_scaffold_service, _, _, _, _ = _get_config_services()
    try:
        return await run_in_threadpool(
            config_scaffold_service.apply_provider_structured,
            payload.project_id,
            payload.name,
            payload.resource_type,
            payload.provider_type,
            payload.description,
            payload.settings,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/resources/apply", response_model=ConfigApplyResponse, status_code=status.HTTP_200_OK)
async def apply_resource_structured(payload: ResourceConfigUpdateRequest) -> ConfigApplyResponse:
    """Apply a resource configuration using structured fields (overwrites file content)."""
    _, config_scaffold_service, _, _, _, _ = _get_config_services()
    try:
        return await run_in_threadpool(
            config_scaffold_service.apply_resource_structured,
            payload.project_id,
            payload.name,
            payload.resource_type,
            payload.provider_type,
            payload.description,
            payload.config,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

@router.post("/configs/rename", status_code=status.HTTP_204_NO_CONTENT)
async def rename_config(payload: ConfigRenameRequest) -> None:
    """Rename/move a configuration file; validates project and rolls back on failure."""
    _, _, _, _, diff_service, _ = _get_config_services()
    try:
        return await run_in_threadpool(
            diff_service.rename_config, payload.project_id, payload.old_relative_path, payload.new_relative_path
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConfigValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "issues": [i.model_dump() for i in exc.issues]},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

@router.post("/configs/delete", status_code=status.HTTP_204_NO_CONTENT)
async def delete_config(payload: ConfigDeleteRequest) -> None:
    """Delete a configuration file; validates project and rolls back on failure."""
    _, _, _, _, diff_service, _ = _get_config_services()
    try:
        return await run_in_threadpool(diff_service.delete_config, payload.project_id, payload.relative_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ConfigValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "issues": [i.model_dump() for i in exc.issues]},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

@router.post("/providers/render", response_model=RenderResponse)
async def render_provider_content(payload: ProviderConfigRenderRequest) -> RenderResponse:
    """Render provider config content without writing file (for diff preview)."""
    _, config_scaffold_service, _, _, _, _ = _get_config_services()
    try:
        content = config_scaffold_service.render_provider_content(
            decorator=config_scaffold_service.provider_decorator(payload.resource_type),
            name=payload.name,
            provider_type=payload.provider_type,
            description=payload.description,
            settings=payload.settings,
            class_name=config_scaffold_service.camel_case(payload.name),
        )
        return RenderResponse(content=content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/resources/render", response_model=RenderResponse)
async def render_resource_content(payload: ResourceConfigRenderRequest) -> RenderResponse:
    """Render resource config content without writing file (for diff preview)."""
    _, config_scaffold_service, _, _, _, _ = _get_config_services()
    try:
        content = config_scaffold_service.render_resource_content(
            name=payload.name,
            provider_type=payload.provider_type,
            description=payload.description,
            config=payload.config,
            class_name=config_scaffold_service.camel_case(payload.name),
        )
        return RenderResponse(content=content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
