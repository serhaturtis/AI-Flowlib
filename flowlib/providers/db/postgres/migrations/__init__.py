"""PostgreSQL migrations system.

This module provides a simple version-based migration system for managing
database schema changes in PostgreSQL.
"""

from flowlib.providers.db.postgres.migrations.migration_runner import MigrationRunner

__all__ = ["MigrationRunner"]
