"""Project service wrapping Flowlib core."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from flowlib.core.project.scaffold import ProjectScaffold, _slugify
from flowlib.core.project.validator import ProjectValidator, ValidationResult
from server.models.projects import ProjectCreateRequest, ProjectMetadata, SetupType
from server.services.alias_file_generator import AliasFileGenerator
from server.services.config_scaffold import ConfigScaffoldService
from server.services.setup_config_generator import SetupConfigGenerator

logger = logging.getLogger(__name__)


class ProjectService:
    """Service for managing Flowlib projects."""

    def __init__(self, projects_root: str = "./projects"):
        """Initialize project service.

        Args:
            projects_root: Root directory for projects
        """
        self.projects_root = Path(projects_root).expanduser().resolve()
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self._scaffold = ProjectScaffold(self.projects_root)
        self._validator = ProjectValidator()

        # Setup-specific services
        self._config_scaffold = ConfigScaffoldService(str(self.projects_root))
        self._setup_generator = SetupConfigGenerator(self._config_scaffold)
        self._alias_generator = AliasFileGenerator()

    def list_projects(self) -> list[ProjectMetadata]:
        """List all projects."""
        projects: list[ProjectMetadata] = []
        for project_path in sorted(self.projects_root.iterdir()):
            if project_path.is_dir():
                # Skip utility/internal directories (starting with underscore)
                if project_path.name.startswith("_"):
                    logger.debug("Skipping utility directory: %s", project_path.name)
                    continue

                # Skip directories without README.md (not valid projects)
                if not (project_path / "README.md").exists():
                    logger.debug("Skipping directory without README.md: %s", project_path.name)
                    continue

                try:
                    projects.append(self._build_metadata(project_path))
                except (FileNotFoundError, OSError, UnicodeDecodeError) as e:
                    # Project directory malformed or unreadable - skip it
                    logger.warning(
                        "Failed to load project metadata for %s: %s", project_path.name, e
                    )
                    continue
        return projects

    def create_project(self, payload: ProjectCreateRequest) -> ProjectMetadata:
        """Create a new project with optional config generation.

        Raises:
            ValueError: If project already exists or setup validation fails
        """
        slug = _slugify(payload.name)
        project_path = self.projects_root / slug

        if project_path.exists():
            raise ValueError(
                f"Project '{slug}' already exists at path '{project_path}'. "
                "Choose a different name."
            )

        # Deduplicate agent names / categories while preserving order
        unique_agents = list(dict.fromkeys(payload.agent_names))
        unique_categories = list(dict.fromkeys(payload.tool_categories))

        logger.info(
            "Creating project '%s' at %s (setup_type=%s)", slug, project_path, payload.setup_type
        )

        # Step 1: Create basic project structure (without aliases.py)
        created_path = self._scaffold.create_project(
            name=payload.name,
            description=payload.description,
            setup_type=payload.setup_type.value,
            agent_names=unique_agents,
            tool_categories=unique_categories,
        )

        # Step 2: Generate configs and aliases based on setup type
        try:
            aliases: dict[str, str] = {}

            if payload.setup_type == SetupType.FAST:
                if payload.fast_config is None:
                    raise ValueError("fast_config is required for FAST setup")
                logger.info("Generating FAST setup configs for project '%s'", slug)
                aliases = self._setup_generator.generate_fast_setup_configs(
                    slug, payload.fast_config
                )

            elif payload.setup_type == SetupType.GUIDED:
                if payload.guided_config is None:
                    raise ValueError("guided_config is required for GUIDED setup")
                logger.info("Generating GUIDED setup configs for project '%s'", slug)
                aliases = self._setup_generator.generate_guided_setup_configs(
                    slug, payload.guided_config
                )

            # Step 3: Generate aliases.py file
            aliases_path = created_path / "configs" / "aliases.py"
            if payload.setup_type == SetupType.EMPTY:
                aliases_content = self._alias_generator.generate_empty_aliases(payload.name)
            else:
                aliases_content = self._alias_generator.generate_configured_aliases(
                    payload.name, aliases
                )

            aliases_path.write_text(aliases_content)
            logger.info("Generated aliases.py for project '%s'", slug)

        except (ValueError, RuntimeError, OSError, PermissionError) as exc:
            # Cleanup: Remove partially created project on failure (fail-fast)
            logger.error("Project creation failed, cleaning up: %s", exc)
            if created_path.exists():
                shutil.rmtree(created_path)
            raise ValueError(f"Failed to create project: {exc}") from exc

        return self._build_metadata(created_path)

    def get_project(self, project_id: str) -> ProjectMetadata:
        """Get project details by slug/directory name."""
        project_path = self._resolve_project_path(project_id)
        return self._build_metadata(project_path)

    def delete_project(self, project_id: str) -> None:
        """Delete a project permanently.

        Args:
            project_id: The project slug/directory name to delete

        Raises:
            FileNotFoundError: If project doesn't exist
            ValueError: If project path is invalid or escapes root
        """
        project_path = self._resolve_project_path(project_id)
        logger.info("Deleting project '%s' at %s", project_id, project_path)
        shutil.rmtree(project_path)
        logger.info("Successfully deleted project '%s'", project_id)

    def validate_project(self, project_id: str) -> ValidationResult:
        """Validate project structure."""
        project_path = self._resolve_project_path(project_id)
        return self._validator.validate(project_path)

    def _resolve_project_path(self, project_id: str) -> Path:
        path = (self.projects_root / project_id).resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Project '{project_id}' not found under {self.projects_root}")
        if self.projects_root not in path.parents and path != self.projects_root:
            raise ValueError(f"Project path {path} escapes managed root {self.projects_root}")
        return path

    def _build_metadata(self, project_path: Path) -> ProjectMetadata:
        readme_path = project_path / "README.md"
        if not readme_path.exists():
            raise FileNotFoundError(
                f"Project at '{project_path}' is missing README.md. "
                "Ensure scaffolding completed successfully."
            )

        description = self._extract_description(readme_path)
        stats = project_path.stat()
        created = datetime.fromtimestamp(stats.st_ctime, tz=timezone.utc)
        updated = datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc)

        return ProjectMetadata(
            id=project_path.name,
            name=self._derive_display_name(project_path.name),
            description=description,
            path=str(project_path),
            created_at=created,
            updated_at=updated,
        )

    def _extract_description(self, readme_path: Path) -> str:
        for line in readme_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return stripped
        return "Flowlib project"

    def _derive_display_name(self, slug: str) -> str:
        return slug.replace("-", " ").title()
