"""Configuration service leveraging Flowlib registries."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from flowlib.core.project.project import Project
from flowlib.resources.models.agent_config_resource import AgentConfigResource
from flowlib.resources.models.constants import ResourceType
from flowlib.resources.registry.registry import resource_registry

from flowlib.config.alias_manager import alias_manager

from server.core.registry_lock import registry_lock
from server.models.configs import (
    AgentConfigResponse,
    AgentConfigSummary,
    AliasEntry,
    ProviderConfigSummary,
    ResourceConfigSummary,
)

T = TypeVar("T")

# Cache size limit to prevent unbounded memory growth
MAX_CACHED_PROJECTS = 100


PROVIDER_RESOURCE_TYPES: tuple[str, ...] = (
    ResourceType.LLM_CONFIG,
    ResourceType.MULTIMODAL_LLM_CONFIG,
    ResourceType.VECTOR_DB_CONFIG,
    ResourceType.DATABASE_CONFIG,
    ResourceType.CACHE_CONFIG,
    ResourceType.STORAGE_CONFIG,
    ResourceType.EMBEDDING_CONFIG,
    ResourceType.GRAPH_DB_CONFIG,
    ResourceType.MESSAGE_QUEUE_CONFIG,
)

RESOURCE_ONLY_TYPES: tuple[str, ...] = (
    ResourceType.MODEL_CONFIG,
    ResourceType.PROMPT_CONFIG,
    ResourceType.TEMPLATE_CONFIG,
)


@dataclass
class ProjectCache:
    """Cache entry for loaded project data."""

    agents: list[AgentConfigSummary]
    providers: list[ProviderConfigSummary]
    resources: list[ResourceConfigSummary]
    aliases: list[AliasEntry]
    timestamp: float
    ttl: float = 60.0  # Cache TTL in seconds

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return (time.time() - self.timestamp) > self.ttl


class ConfigService:
    """Expose project configuration data via safe registry access."""

    def __init__(self, projects_root: str = "./projects", cache_ttl: float = 60.0) -> None:
        self._root = Path(projects_root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        # Use OrderedDict for LRU eviction support
        self._cache: OrderedDict[str, ProjectCache] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_ttl = cache_ttl

    def invalidate_cache(self, project_id: str) -> None:
        """Invalidate cache for a specific project."""
        with self._cache_lock:
            self._cache.pop(project_id, None)

    def _get_or_load_cache(self, project_id: str) -> ProjectCache:
        """Get cached project data or load from disk if expired/missing.

        Uses proper double-checked locking to prevent race conditions.
        Implements LRU eviction when cache size exceeds MAX_CACHED_PROJECTS.
        """
        # First check (fast path - no lock)
        cache = self._cache.get(project_id)
        if cache and not cache.is_expired():
            # Move to end for LRU (most recently used)
            with self._cache_lock:
                self._cache.move_to_end(project_id)
            return cache

        # Acquire lock for cache update
        with self._cache_lock:
            # Second check (slow path - under lock)
            cache = self._cache.get(project_id)
            if cache and not cache.is_expired():
                # Move to end for LRU
                self._cache.move_to_end(project_id)
                return cache

            # Cache miss or expired - load project data under lock
            # This prevents multiple threads from loading simultaneously
            cache = self._load_project_cache(project_id)

            # LRU eviction if cache is full
            while len(self._cache) >= MAX_CACHED_PROJECTS:
                # Remove oldest (least recently used)
                evicted_id, _ = self._cache.popitem(last=False)
                # Note: Using logger here would require importing logging
                # logger.debug("Evicted project '%s' from config cache", evicted_id)

            self._cache[project_id] = cache
            return cache

    def _load_project_cache(self, project_id: str) -> ProjectCache:
        """Load all project configuration data into cache."""
        project_path = self._resolve_project_path(project_id)

        with registry_lock:
            resource_registry.clear()
            project = Project(str(project_path))
            project.initialize()
            project.load_configurations()

            try:
                # Load agents
                agents = resource_registry.get_by_type(ResourceType.AGENT_CONFIG)
                agent_summaries: list[AgentConfigSummary] = []
                for name in sorted(agents.keys()):
                    resource = agents[name]
                    if isinstance(resource, AgentConfigResource):
                        agent_summaries.append(
                            AgentConfigSummary(
                                name=name,
                                persona=resource.persona,
                                allowed_tool_categories=resource.allowed_tool_categories,
                                knowledge_plugins=resource.knowledge_plugins,
                                model_name=resource.model_name,
                                llm_name=resource.llm_name,
                                temperature=resource.temperature,
                                max_iterations=resource.max_iterations,
                                enable_learning=resource.enable_learning,
                                verbose=resource.verbose,
                            )
                        )

                # Load providers
                provider_configs: list[ProviderConfigSummary] = []
                for resource_type in PROVIDER_RESOURCE_TYPES:
                    entries = resource_registry.get_by_type(resource_type)
                    for name in sorted(entries.keys()):
                        resource = entries[name]
                        provider_type = getattr(resource, "provider_type", None)
                        settings = getattr(resource, "settings", None)
                        provider_configs.append(
                            ProviderConfigSummary(
                                name=name,
                                resource_type=resource_type,
                                provider_type=str(provider_type or ""),
                                settings=dict(settings or {}),
                            )
                        )

                # Load resources
                resource_configs: list[ResourceConfigSummary] = []
                for resource_type in RESOURCE_ONLY_TYPES:
                    entries = resource_registry.get_by_type(resource_type)
                    for name in sorted(entries.keys()):
                        resource = entries[name]
                        metadata = resource.model_dump(exclude={"name", "type"})
                        resource_configs.append(
                            ResourceConfigSummary(
                                name=name,
                                resource_type=resource_type,
                                metadata=metadata,
                            )
                        )

                # Load aliases
                aliases = alias_manager.list_all_aliases()
                alias_entries = [
                    AliasEntry(alias=alias, canonical=canonical)
                    for alias, canonical in sorted(aliases.items())
                ]

                return ProjectCache(
                    agents=agent_summaries,
                    providers=provider_configs,
                    resources=resource_configs,
                    aliases=alias_entries,
                    timestamp=time.time(),
                    ttl=self._cache_ttl,
                )
            finally:
                resource_registry.clear()

    def list_agent_configs(self, project_id: str) -> list[AgentConfigSummary]:
        """Return agent configurations for the project (cached)."""
        cache = self._get_or_load_cache(project_id)
        return cache.agents

    def get_agent_config(self, project_id: str, config_id: str) -> AgentConfigResponse:
        """Return a specific agent config (cached)."""
        cache = self._get_or_load_cache(project_id)
        for agent in cache.agents:
            if agent.name == config_id:
                return AgentConfigResponse(
                    name=agent.name,
                    persona=agent.persona,
                    allowed_tool_categories=agent.allowed_tool_categories,
                    knowledge_plugins=agent.knowledge_plugins,
                    model_name=agent.model_name,
                    llm_name=agent.llm_name,
                    temperature=agent.temperature,
                    max_iterations=agent.max_iterations,
                    enable_learning=agent.enable_learning,
                    verbose=agent.verbose,
                )
        raise KeyError(f"Agent config '{config_id}' not found")

    def list_provider_configs(self, project_id: str) -> list[ProviderConfigSummary]:
        """Return provider configurations (cached)."""
        cache = self._get_or_load_cache(project_id)
        return cache.providers

    def list_resource_configs(self, project_id: str) -> list[ResourceConfigSummary]:
        """Return non-provider registry entries (cached)."""
        cache = self._get_or_load_cache(project_id)
        return cache.resources

    def list_aliases(self, project_id: str) -> list[AliasEntry]:
        """Return alias mappings (cached)."""
        cache = self._get_or_load_cache(project_id)
        return cache.aliases

    def _with_loaded_project(self, project_id: str, handler: Callable[[], T]) -> T:
        project_path = self._resolve_project_path(project_id)

        with registry_lock:
            resource_registry.clear()
            project = Project(str(project_path))
            project.initialize()
            project.load_configurations()
            try:
                return handler()
            finally:
                resource_registry.clear()

    def _resolve_project_path(self, project_id: str) -> Path:
        project_path = (self._root / project_id).resolve()
        if not project_path.exists() or not project_path.is_dir():
            raise FileNotFoundError(f"Project '{project_id}' not found under {self._root}")
        if project_path == self._root or self._root not in project_path.parents:
            # Prevent escape outside managed directory
            raise ValueError(f"Project '{project_id}' resolved outside managed root {self._root}")
        return project_path

