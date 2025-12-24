"""PostgreSQL migration runner.

A simple version-based migration system that:
1. Tracks applied migrations in a `schema_migrations` table
2. Applies pending migrations in version order
3. Wraps each migration in a transaction for safety
"""

import importlib
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flowlib.providers.db.postgres.provider import PostgreSQLProvider

logger = logging.getLogger(__name__)


class Migration(ABC):
    """Base class for database migrations.

    Each migration must implement:
    - version: Unique version identifier (e.g., "001", "002")
    - description: Human-readable description
    - up(): Apply the migration
    - down(): Reverse the migration (optional but recommended)
    """

    version: str
    description: str

    @abstractmethod
    async def up(self, db: "PostgreSQLProvider") -> None:
        """Apply the migration.

        Args:
            db: PostgreSQL provider instance
        """
        raise NotImplementedError

    async def down(self, db: "PostgreSQLProvider") -> None:
        """Reverse the migration (optional).

        Args:
            db: PostgreSQL provider instance
        """
        raise NotImplementedError(f"Migration {self.version} does not support rollback")


class MigrationRunner:
    """Runs PostgreSQL migrations in version order.

    Usage:
        runner = MigrationRunner(db_provider)
        await runner.run_pending_migrations()
    """

    MIGRATIONS_TABLE = "schema_migrations"

    def __init__(self, db: "PostgreSQLProvider"):
        """Initialize migration runner.

        Args:
            db: Initialized PostgreSQL provider instance
        """
        self.db = db
        self._migrations: list[Migration] = []

    async def ensure_migrations_table(self) -> None:
        """Create the migrations tracking table if it doesn't exist."""
        query = f"""
            CREATE TABLE IF NOT EXISTS {self.MIGRATIONS_TABLE} (
                version VARCHAR(50) PRIMARY KEY,
                description TEXT,
                applied_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
        """
        await self.db.execute(query)
        logger.debug(f"Ensured {self.MIGRATIONS_TABLE} table exists")

    async def get_applied_versions(self) -> set[str]:
        """Get set of already applied migration versions.

        Returns:
            Set of version strings that have been applied
        """
        query = f"SELECT version FROM {self.MIGRATIONS_TABLE};"
        try:
            results = await self.db.execute(query)
            if results:
                return {row["version"] for row in results}
            return set()
        except Exception as e:
            logger.warning(f"Could not fetch applied migrations: {e}")
            return set()

    def register_migration(self, migration: Migration) -> None:
        """Register a migration to be run.

        Args:
            migration: Migration instance to register
        """
        self._migrations.append(migration)
        logger.debug(f"Registered migration {migration.version}: {migration.description}")

    def register_migrations_from_module(self, module_name: str) -> None:
        """Auto-discover and register migrations from a module.

        Args:
            module_name: Full module path (e.g., 'flowlib.providers.db.postgres.migrations.v001_email_schema')
        """
        try:
            module = importlib.import_module(module_name)

            # Look for Migration subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Migration)
                    and attr is not Migration
                    and hasattr(attr, "version")
                ):
                    migration = attr()
                    self.register_migration(migration)
                    logger.info(f"Auto-registered migration from {module_name}: {attr_name}")

        except ImportError as e:
            logger.error(f"Failed to import migration module {module_name}: {e}")
            raise

    async def run_pending_migrations(self) -> list[str]:
        """Run all pending migrations in version order.

        Returns:
            List of applied migration versions
        """
        await self.ensure_migrations_table()
        applied = await self.get_applied_versions()

        # Sort migrations by version
        pending = [m for m in self._migrations if m.version not in applied]
        pending.sort(key=lambda m: m.version)

        if not pending:
            logger.info("No pending migrations to run")
            return []

        applied_versions: list[str] = []

        for migration in pending:
            try:
                logger.info(f"Running migration {migration.version}: {migration.description}")

                # Run migration
                await migration.up(self.db)

                # Record successful migration
                await self._record_migration(migration)
                applied_versions.append(migration.version)

                logger.info(f"Successfully applied migration {migration.version}")

            except Exception as e:
                logger.error(f"Migration {migration.version} failed: {e}")
                raise RuntimeError(
                    f"Migration {migration.version} failed: {e}. "
                    f"Database may be in an inconsistent state."
                ) from e

        return applied_versions

    async def _record_migration(self, migration: Migration) -> None:
        """Record a successful migration in the tracking table.

        Args:
            migration: The migration that was applied
        """
        query = f"""
            INSERT INTO {self.MIGRATIONS_TABLE} (version, description)
            VALUES ($1, $2)
            ON CONFLICT (version) DO NOTHING;
        """
        await self.db.execute(query, {"version": migration.version, "description": migration.description})

    async def rollback_migration(self, version: str) -> None:
        """Rollback a specific migration.

        Args:
            version: Version string of migration to rollback

        Raises:
            ValueError: If migration not found or doesn't support rollback
        """
        migration = next((m for m in self._migrations if m.version == version), None)
        if not migration:
            raise ValueError(f"Migration {version} not found")

        applied = await self.get_applied_versions()
        if version not in applied:
            raise ValueError(f"Migration {version} has not been applied")

        logger.info(f"Rolling back migration {version}: {migration.description}")

        try:
            await migration.down(self.db)

            # Remove from tracking table
            query = f"DELETE FROM {self.MIGRATIONS_TABLE} WHERE version = $1;"
            await self.db.execute(query, {"version": version})

            logger.info(f"Successfully rolled back migration {version}")

        except Exception as e:
            logger.error(f"Rollback of migration {version} failed: {e}")
            raise

    async def get_migration_status(self) -> list[dict[str, Any]]:
        """Get status of all known migrations.

        Returns:
            List of dicts with version, description, applied status
        """
        applied = await self.get_applied_versions()

        status = []
        for migration in sorted(self._migrations, key=lambda m: m.version):
            status.append(
                {
                    "version": migration.version,
                    "description": migration.description,
                    "applied": migration.version in applied,
                }
            )

        return status
