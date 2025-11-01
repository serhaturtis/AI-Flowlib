"""Central context management system for agents.

This module provides the AgentContextManager - THE context authority
that replaces all manual context assembly with unified context management.
"""

import json
import logging
import os
import tempfile
from datetime import datetime
from typing import Any, Literal

from flowlib.agent.core.base import AgentComponent
from flowlib.agent.core.errors import ExecutionError

from .models import (
    ComponentContext,
    ContextManagerConfig,
    ConversationMessage,
    ExecutionContext,
    LearningContext,
    SessionContext,
    SuccessfulPattern,
    TaskContext,
    UserProfile,
    WorkspaceKnowledge,
)

logger = logging.getLogger(__name__)


class AgentContextManager(AgentComponent):
    """Central context management system - THE context authority.

    This component is the single source of truth for all agent context.
    All components receive unified ExecutionContext from this manager.
    No manual context assembly allowed anywhere else.
    """

    def __init__(self, config: ContextManagerConfig, name: str = "context_manager"):
        super().__init__(name)
        self._config = config

        # Core context state
        self._session_context: SessionContext | None = None
        self._task_context: TaskContext | None = None
        self._learning_context: LearningContext = LearningContext()

        # Auto-compaction tracking
        self._token_usage = 0
        self._needs_compaction = False

    async def _initialize_impl(self) -> None:
        """Initialize context manager."""
        logger.info("AgentContextManager initialized - THE context authority")

    async def _shutdown_impl(self) -> None:
        """Shutdown context manager."""
        # Persist session context if configured
        if self._config.session_persistence and self._session_context:
            await self._persist_session_context()
        logger.info("AgentContextManager shutdown")

    async def initialize_session(
        self,
        session_id: str,
        agent_name: str,
        agent_persona: str,
        agent_role: str,
        working_directory: str,
        user_id: str | None = None,
    ) -> None:
        """Initialize a new session context."""

        # Scan workspace if enabled
        workspace_knowledge = WorkspaceKnowledge()
        if self._config.workspace_scanning:
            workspace_knowledge = await self._scan_workspace(working_directory)

        self._session_context = SessionContext(
            session_id=session_id,
            user_id=user_id,
            agent_name=agent_name,
            agent_persona=agent_persona,
            agent_role=agent_role,
            working_directory=working_directory,
            current_message="",
            conversation_history=[],
            user_profile=UserProfile(),
            workspace_knowledge=workspace_knowledge,
        )

        logger.info(f"Session context initialized: {session_id}")

    async def start_task(self, task_description: str) -> None:
        """Start a new task context."""
        if not self._session_context:
            raise ExecutionError("Session context must be initialized before starting task")

        self._session_context.current_message = task_description
        self._task_context = TaskContext(
            description=task_description, cycle=1, todos=[], execution_results=[]
        )

        # Add to conversation history
        self._session_context.conversation_history.append(
            ConversationMessage(role="user", content=task_description)
        )

        # Trim conversation history if needed
        await self._trim_conversation_history()

        logger.info(f"Task context started: {task_description[:50]}...")

    async def create_execution_context(
        self,
        component_type: Literal[
            "task_generation",
            "task_decomposition",
            "task_execution",
            "task_debriefing",
            "task_planning",
            "task_evaluation",
        ],
        **component_config: Any,
    ) -> ExecutionContext:
        """Create execution context for component - THE context creation method.

        This is the ONLY way components get context. No manual assembly allowed.
        """

        if not self._session_context:
            raise ExecutionError("Session context not initialized - fix configuration")
        if not self._task_context:
            raise ExecutionError("Task context not initialized - call start_task first")

        # Check if auto-compaction needed
        await self._auto_compact_if_needed()

        # Update task context with component-specific data (e.g., todos for task_execution)
        if component_type == "task_execution" and "todos" in component_config:
            self._task_context.todos = component_config["todos"]

        # Create component context
        component_context = ComponentContext(
            component_type=component_type,
            component_config=component_config,
        )

        # Create unified execution context
        execution_context = ExecutionContext(
            session=self._session_context,
            task=self._task_context,
            component=component_context,
            learning=self._learning_context,
        )

        # Update token usage for auto-compaction
        self._token_usage = execution_context.token_count

        logger.debug(
            f"Created execution context for {component_type} (tokens: {self._token_usage})"
        )
        return execution_context

    async def update_from_execution(
        self, component_type: str, execution_result: Any, success: bool
    ) -> None:
        """Learn from execution results to improve future context."""

        # Update learning metrics
        self._learning_context.total_executions += 1
        if success:
            self._learning_context.successful_executions += 1

        # Update task context with results
        if self._task_context:
            result_summary = {
                "component": component_type,
                "result": str(execution_result)[:500],  # Truncate for storage
                "success": success,
                "timestamp": datetime.now().isoformat(),
            }
            self._task_context.execution_results.append(result_summary)

        # Learn successful patterns if learning enabled
        if success and self._config.learning_enabled:
            await self._learn_successful_pattern(component_type, execution_result)

        logger.debug(f"Updated context from {component_type} execution (success: {success})")

    async def increment_cycle(self) -> None:
        """Increment task execution cycle."""
        if self._task_context:
            self._task_context.cycle += 1
            logger.debug(f"Task cycle incremented to {self._task_context.cycle}")

    async def add_assistant_response(self, response: str) -> None:
        """Add assistant response to conversation history."""
        if self._session_context:
            self._session_context.conversation_history.append(
                ConversationMessage(role="assistant", content=response)
            )

            # Trim conversation history if needed
            await self._trim_conversation_history()

            logger.debug("Added assistant response to conversation history")

    async def _auto_compact_if_needed(self) -> None:
        """Auto-compact context when approaching token limits."""
        if self._token_usage > self._config.auto_compaction_threshold:
            logger.info(
                f"Auto-compacting context (tokens: {self._token_usage} > {self._config.auto_compaction_threshold})"
            )

            if self._session_context and len(self._session_context.conversation_history) > 10:
                # Keep recent messages, summarize older ones
                recent_messages = self._session_context.conversation_history[-5:]
                old_messages = self._session_context.conversation_history[:-5]

                # Create summary of old messages
                summary_parts = []
                for msg in old_messages:
                    summary_parts.append(f"{msg.role}: {msg.content[:100]}...")

                summary = (
                    f"Previous conversation summary ({len(old_messages)} messages):\n"
                    + "\n".join(summary_parts)
                )

                # Update session context
                self._session_context.conversation_summary = summary
                self._session_context.conversation_history = recent_messages

                logger.info(f"Compacted {len(old_messages)} messages into summary")

    async def _trim_conversation_history(self) -> None:
        """Trim conversation history to configured maximum."""
        if not self._session_context:
            return

        max_history = self._config.max_conversation_history
        if len(self._session_context.conversation_history) > max_history:
            # Keep most recent messages
            trimmed_count = len(self._session_context.conversation_history) - max_history
            self._session_context.conversation_history = self._session_context.conversation_history[
                -max_history:
            ]

            logger.debug(f"Trimmed {trimmed_count} messages from conversation history")

    async def _learn_successful_pattern(self, component_type: str, execution_result: Any) -> None:
        """Learn from successful execution patterns."""
        pattern_description = f"Successful {component_type} execution"

        # Find existing pattern or create new one
        existing_pattern = None
        for pattern in self._learning_context.successful_patterns:
            if (
                pattern.pattern_type == component_type
                and pattern.description == pattern_description
            ):
                existing_pattern = pattern
                break

        if existing_pattern:
            existing_pattern.success_count += 1
            existing_pattern.last_used = datetime.now()
        else:
            new_pattern = SuccessfulPattern(
                pattern_type=component_type,
                description=pattern_description,
                success_count=1,
                last_used=datetime.now(),
            )
            self._learning_context.successful_patterns.append(new_pattern)

        # Update user preferences based on patterns
        if (
            existing_pattern
            and existing_pattern.success_count >= self._config.pattern_learning_threshold
        ):
            preference_key = f"preferred_{component_type}_pattern"
            self._learning_context.user_preferences[preference_key] = pattern_description

    async def _scan_workspace(self, working_directory: str) -> WorkspaceKnowledge:
        """Scan workspace for project knowledge."""
        try:
            if not os.path.exists(working_directory):
                logger.warning(f"Working directory does not exist: {working_directory}")
                return WorkspaceKnowledge()

            # Collect files for analysis
            files = []
            directories = []

            for _root, dirs, filenames in os.walk(working_directory):
                # Skip hidden directories and common build/cache directories
                dirs[:] = [
                    d
                    for d in dirs
                    if not d.startswith(".")
                    and d not in ["node_modules", "__pycache__", "target", "build", "dist", ".git"]
                ]

                directories.extend(dirs)
                files.extend([f for f in filenames if not f.startswith(".")])

            # Detect languages by file extensions
            extensions = {os.path.splitext(f)[1].lower() for f in files if "." in f}

            language_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".java": "java",
                ".cpp": "cpp",
                ".c": "c",
                ".rs": "rust",
                ".go": "go",
                ".rb": "ruby",
                ".php": "php",
                ".swift": "swift",
                ".kt": "kotlin",
                ".scala": "scala",
                ".sh": "shell",
            }

            detected_languages = [language_map[ext] for ext in extensions if ext in language_map]

            # Detect project type based on key files
            project_indicators = {
                "package.json": "nodejs",
                "requirements.txt": "python",
                "pyproject.toml": "python",
                "setup.py": "python",
                "pom.xml": "java",
                "build.gradle": "java",
                "Cargo.toml": "rust",
                "go.mod": "go",
                "Gemfile": "ruby",
                "composer.json": "php",
            }

            detected_project_type = None
            for indicator_file, project_type in project_indicators.items():
                if indicator_file in files:
                    detected_project_type = project_type
                    break

            # Extract dependencies (basic implementation)
            detected_dependencies = await self._extract_dependencies(working_directory, files)

            # Identify common patterns
            detected_patterns = self._identify_code_patterns(files, detected_languages)

            # Create file structure overview
            file_structure = {
                "total_files": len(files),
                "directories": len(set(directories)),
                "languages_detected": len(detected_languages),
                "project_type": detected_project_type,
            }

            # Create WorkspaceKnowledge with all data at once (required for frozen models)
            knowledge = WorkspaceKnowledge(
                project_type=detected_project_type,
                languages=detected_languages,
                dependencies=detected_dependencies,
                common_patterns=detected_patterns,
                file_structure=file_structure,
            )

            logger.debug(
                f"Workspace scan complete: {detected_project_type}, {len(detected_languages)} languages"
            )

            return knowledge

        except Exception as e:
            logger.warning(f"Workspace scan failed: {e}")
            return WorkspaceKnowledge()

    async def _extract_dependencies(self, working_directory: str, files: list[str]) -> list[str]:
        """Extract project dependencies from common files."""
        dependencies = []

        try:
            # Python dependencies
            if "requirements.txt" in files:
                req_path = os.path.join(working_directory, "requirements.txt")
                with open(req_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Extract package name (before version specifiers)
                            pkg_name = (
                                line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0]
                            )
                            dependencies.append(pkg_name.strip())

            # Node.js dependencies
            if "package.json" in files:
                pkg_path = os.path.join(working_directory, "package.json")
                with open(pkg_path) as f:
                    pkg_data = json.load(f)
                    if "dependencies" in pkg_data:
                        dependencies.extend(pkg_data["dependencies"].keys())
                    if "devDependencies" in pkg_data:
                        dependencies.extend(pkg_data["devDependencies"].keys())

        except Exception as e:
            logger.debug(f"Could not extract dependencies: {e}")

        return dependencies[:20]  # Limit to first 20 for context size

    def _identify_code_patterns(self, files: list[str], languages: list[str]) -> list[str]:
        """Identify common code patterns based on file names and languages."""
        patterns = []

        # Framework patterns
        if "python" in languages:
            if any(f.startswith("test_") or f.endswith("_test.py") for f in files):
                patterns.append("pytest_testing")
            if "manage.py" in files:
                patterns.append("django_project")
            if "app.py" in files or "main.py" in files:
                patterns.append("python_application")

        if "javascript" in languages or "typescript" in languages:
            if "package.json" in files:
                patterns.append("npm_project")
            if any(f.endswith(".test.js") or f.endswith(".spec.js") for f in files):
                patterns.append("javascript_testing")

        # Common file organization patterns
        if any("src" in f for f in files):
            patterns.append("src_directory_structure")
        if any("test" in f.lower() for f in files):
            patterns.append("testing_included")
        if "README.md" in files or "readme.md" in files:
            patterns.append("documented_project")

        return patterns

    async def _persist_session_context(self) -> None:
        """Persist session context to storage."""
        if not self._session_context:
            return

        try:
            # Simple file-based persistence (can be enhanced with database later)
            session_file = os.path.join(
                tempfile.gettempdir(), f"flowlib_session_{self._session_context.session_id}.json"
            )

            # Serialize session context
            session_data = self._session_context.model_dump(mode="json")

            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2, default=str)

            logger.debug(f"Session context persisted to {session_file}")

        except Exception as e:
            logger.warning(f"Failed to persist session context: {e}")

    async def load_session_context(self, session_id: str) -> bool:
        """Load persisted session context."""
        if not self._config.session_persistence:
            return False

        try:
            session_file = os.path.join(tempfile.gettempdir(), f"flowlib_session_{session_id}.json")

            if not os.path.exists(session_file):
                return False

            with open(session_file) as f:
                session_data = json.load(f)

            # Reconstruct session context
            self._session_context = SessionContext(**session_data)

            logger.info(f"Session context loaded: {session_id}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load session context: {e}")
            return False

    @property
    def current_session(self) -> SessionContext | None:
        """Get current session context."""
        return self._session_context

    @property
    def current_task(self) -> TaskContext | None:
        """Get current task context."""
        return self._task_context

    @property
    def learning_stats(self) -> dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_executions": self._learning_context.total_executions,
            "successful_executions": self._learning_context.successful_executions,
            "success_rate": self._learning_context.successful_executions
            / max(self._learning_context.total_executions, 1),
            "patterns_learned": len(self._learning_context.successful_patterns),
            "preferences_learned": len(self._learning_context.user_preferences),
        }
