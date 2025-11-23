"""Generate provider/resource configs for project setup workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flowlib.resources.models.constants import ResourceType

if TYPE_CHECKING:
    from server.models.projects import FastSetupConfig, GuidedSetupConfig
    from server.services.config_scaffold import ConfigScaffoldService


class SetupConfigGenerator:
    """Generate configs for FAST and GUIDED setup workflows."""

    def __init__(self, config_scaffold: ConfigScaffoldService) -> None:
        self._scaffold = config_scaffold

    def generate_fast_setup_configs(
        self, project_id: str, config: FastSetupConfig
    ) -> dict[str, str]:
        """Generate configs for fast setup and return alias mappings.

        Args:
            project_id: Project identifier
            config: Fast setup configuration

        Returns:
            Dictionary mapping alias names to canonical config names

        Raises:
            ValueError: If config generation fails
        """
        from server.models.configs import (
            ProviderConfigCreateRequest,
            ResourceConfigCreateRequest,
        )

        aliases = {}

        # 1. LLM Provider (llamacpp)
        llm_provider_name = "llamacpp-llm-provider"
        self._scaffold.create_provider_config(
            ProviderConfigCreateRequest(
                project_id=project_id,
                name=llm_provider_name,
                resource_type=ResourceType.LLM_CONFIG.value,
                provider_type="llamacpp",
                description="LLaMA.cpp LLM provider",
                settings={},
            )
        )
        aliases["default-llm"] = llm_provider_name

        # 2. LLM Model Config
        llm_model_name = "default-llm-model"
        self._scaffold.create_resource_config(
            ResourceConfigCreateRequest(
                project_id=project_id,
                name=llm_model_name,
                resource_type=ResourceType.MODEL_CONFIG.value,
                provider_type="llamacpp",
                description="Default LLM model configuration",
                config={
                    "path": config.llm_model_path,
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "n_ctx": 8192,
                    "n_gpu_layers": -1,
                },
            )
        )
        aliases["default-model"] = llm_model_name

        # 3. Embedding Provider (llamacpp)
        embedding_provider_name = "llamacpp-embedding-provider"
        self._scaffold.create_provider_config(
            ProviderConfigCreateRequest(
                project_id=project_id,
                name=embedding_provider_name,
                resource_type=ResourceType.EMBEDDING_CONFIG.value,
                provider_type="llamacpp_embedding",
                description="LLaMA.cpp embedding provider",
                settings={},
            )
        )
        aliases["default-embedding"] = embedding_provider_name

        # 4. Embedding Model Config
        embedding_model_name = "default-embedding-model"
        self._scaffold.create_resource_config(
            ResourceConfigCreateRequest(
                project_id=project_id,
                name=embedding_model_name,
                resource_type=ResourceType.MODEL_CONFIG.value,
                provider_type="llamacpp_embedding",
                description="Default embedding model configuration",
                config={
                    "path": config.embedding_model_path,
                    "dimensions": 1024,
                    "max_length": 8192,
                    "n_ctx": 8192,
                    "n_gpu_layers": -1,
                    "normalize": True,
                    "batch_size": 32,
                },
            )
        )
        aliases["default-embedding-model"] = embedding_model_name

        # 5. Vector DB (Qdrant)
        vector_db_name = "qdrant-vector-db"
        vector_db_path = config.vector_db_path or ":memory:"
        self._scaffold.create_provider_config(
            ProviderConfigCreateRequest(
                project_id=project_id,
                name=vector_db_name,
                resource_type=ResourceType.VECTOR_DB_CONFIG.value,
                provider_type="qdrant",
                description="Qdrant vector database",
                settings={
                    "location": vector_db_path,
                    "prefer_grpc": False,
                },
            )
        )
        aliases["default-vector-db"] = vector_db_name

        # Note: Graph DB is conditionally required (only if knowledge memory enabled)
        # Not included in FAST setup by default - users can add later if needed

        return aliases

    def generate_guided_setup_configs(
        self, project_id: str, config: GuidedSetupConfig
    ) -> dict[str, str]:
        """Generate configs for guided setup and return alias mappings.

        Args:
            project_id: Project identifier
            config: Guided setup configuration

        Returns:
            Dictionary mapping alias names to canonical config names

        Raises:
            ValueError: If config generation fails
        """
        from server.models.configs import (
            ProviderConfigCreateRequest,
            ResourceConfigCreateRequest,
        )

        # Create provider configs
        for provider in config.providers:
            self._scaffold.create_provider_config(
                ProviderConfigCreateRequest(
                    project_id=project_id,
                    name=provider.name,
                    resource_type=provider.resource_type,
                    provider_type=provider.provider_type,
                    description=provider.description,
                    settings=provider.settings,
                )
            )

        # Create resource configs
        for resource in config.resources:
            self._scaffold.create_resource_config(
                ResourceConfigCreateRequest(
                    project_id=project_id,
                    name=resource.name,
                    resource_type=resource.resource_type,
                    provider_type=resource.provider_type,
                    description=resource.description,
                    config=resource.config,
                )
            )

        # Return provided alias mappings
        return config.aliases
