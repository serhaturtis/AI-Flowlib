from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session

from server.core.config import settings


class Base(DeclarativeBase):
    pass


# Connection pooling configuration for multi-worker deployments
# QueuePool provides a fixed pool of connections shared by all threads
engine = create_engine(
    settings.DATABASE_URL,
    future=True,
    poolclass=pool.QueuePool,
    pool_size=10,  # Minimum number of connections to keep in pool
    max_overflow=20,  # Additional connections beyond pool_size
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Note: expire_on_commit=True (default) means objects are expired after commit
# and require the session to still be open to access attributes.
# We use False here for convenience, but this means objects may contain stale data
# if the database is modified outside the session. In a multi-worker environment,
# always re-fetch critical data from DB rather than relying on in-memory objects.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
    expire_on_commit=False,  # For convenience; see note above
)


def init_db() -> None:
    from .models import RunHistory  # noqa: F401

    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session() -> Iterator[Session]:
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:  # noqa: BLE001 - Re-raise all exceptions after rollback
        session.rollback()
        raise
    finally:
        session.close()


