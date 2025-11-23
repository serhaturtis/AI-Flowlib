from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column

from server.persistence.db import Base
from server.models.agents import AgentRunStatus


class RunHistory(Base):
    __tablename__ = "run_history"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    agent_name: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[AgentRunStatus] = mapped_column(SAEnum(AgentRunStatus), nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    message: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)


