"""Database service for email sales agent.

This module provides all PostgreSQL operations for:
- Contacts: CRUD operations and profile management
- Threads: Conversation chain management
- Emails: Storage, search, and retrieval
- Topics: Topic taxonomy management
- Interactions: Customer interaction logging
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from flowlib.agent.components.task.execution.tool_implementations.email_db.models import (
    ContactProfile,
    EmailAnalysis,
    EmailSummary,
    InteractionRecord,
    ThreadSummary,
)

if TYPE_CHECKING:
    from flowlib.providers.db.postgres.provider import PostgreSQLProvider

logger = logging.getLogger(__name__)


class EmailDBService:
    """Database service for email sales agent operations.

    This service provides all database operations needed by the email tools.
    It follows the principle of keeping SQL in one place for maintainability.

    Usage:
        service = EmailDBService(db_provider)
        contact = await service.get_or_create_contact("user@example.com")
    """

    def __init__(self, db: "PostgreSQLProvider"):
        """Initialize service with database provider.

        Args:
            db: Initialized PostgreSQL provider instance
        """
        self.db = db

    # =========================================================================
    # Contact Operations
    # =========================================================================

    async def get_contact_by_email(self, email: str) -> ContactProfile | None:
        """Get contact by email address.

        Args:
            email: Email address to look up

        Returns:
            ContactProfile if found, None otherwise
        """
        query = """
            SELECT id, email, display_name, company, first_seen_at, last_contact_at,
                   total_emails_sent, total_emails_received, sentiment_score, tags,
                   profile_data, created_at, updated_at
            FROM contacts
            WHERE email = $1;
        """
        results = await self.db.execute(query, {"email": email.lower()})

        if not results:
            return None

        row = results[0]
        return ContactProfile(
            id=str(row["id"]),
            email=row["email"],
            display_name=row["display_name"],
            company=row["company"],
            first_seen_at=row["first_seen_at"],
            last_contact_at=row["last_contact_at"],
            total_emails_sent=row["total_emails_sent"],
            total_emails_received=row["total_emails_received"],
            sentiment_score=row["sentiment_score"],
            tags=row["tags"] or [],
            profile_data=row["profile_data"] or {},
        )

    async def get_contact_by_id(self, contact_id: str) -> ContactProfile | None:
        """Get contact by UUID.

        Args:
            contact_id: Contact UUID

        Returns:
            ContactProfile if found, None otherwise
        """
        query = """
            SELECT id, email, display_name, company, first_seen_at, last_contact_at,
                   total_emails_sent, total_emails_received, sentiment_score, tags,
                   profile_data, created_at, updated_at
            FROM contacts
            WHERE id = $1;
        """
        results = await self.db.execute(query, {"id": contact_id})

        if not results:
            return None

        row = results[0]
        return ContactProfile(
            id=str(row["id"]),
            email=row["email"],
            display_name=row["display_name"],
            company=row["company"],
            first_seen_at=row["first_seen_at"],
            last_contact_at=row["last_contact_at"],
            total_emails_sent=row["total_emails_sent"],
            total_emails_received=row["total_emails_received"],
            sentiment_score=row["sentiment_score"],
            tags=row["tags"] or [],
            profile_data=row["profile_data"] or {},
        )

    async def get_or_create_contact(
        self, email: str, display_name: str | None = None
    ) -> tuple[ContactProfile, bool]:
        """Get existing contact or create new one.

        Args:
            email: Email address
            display_name: Optional display name for new contacts

        Returns:
            Tuple of (ContactProfile, is_new)
        """
        existing = await self.get_contact_by_email(email)
        if existing:
            return existing, False

        # Create new contact
        query = """
            INSERT INTO contacts (email, display_name)
            VALUES ($1, $2)
            RETURNING id, email, display_name, company, first_seen_at, last_contact_at,
                      total_emails_sent, total_emails_received, sentiment_score, tags,
                      profile_data;
        """
        results = await self.db.execute(
            query, {"email": email.lower(), "display_name": display_name}
        )

        row = results[0]
        contact = ContactProfile(
            id=str(row["id"]),
            email=row["email"],
            display_name=row["display_name"],
            company=row["company"],
            first_seen_at=row["first_seen_at"],
            last_contact_at=row["last_contact_at"],
            total_emails_sent=row["total_emails_sent"] or 0,
            total_emails_received=row["total_emails_received"] or 0,
            sentiment_score=row["sentiment_score"],
            tags=row["tags"] or [],
            profile_data=row["profile_data"] or {},
        )

        logger.info(f"Created new contact: {email}")
        return contact, True

    async def update_contact_stats(
        self,
        contact_id: str,
        increment_sent: int = 0,
        increment_received: int = 0,
        sentiment_score: float | None = None,
    ) -> None:
        """Update contact statistics.

        Args:
            contact_id: Contact UUID
            increment_sent: Number to add to emails sent
            increment_received: Number to add to emails received
            sentiment_score: New sentiment score (will be averaged with existing)
        """
        updates = ["last_contact_at = NOW()"]
        params: dict[str, Any] = {"id": contact_id}
        param_idx = 2  # $1 is always contact_id

        if increment_sent > 0:
            updates.append(f"total_emails_sent = total_emails_sent + ${param_idx}")
            params["increment_sent"] = increment_sent
            param_idx += 1

        if increment_received > 0:
            updates.append(f"total_emails_received = total_emails_received + ${param_idx}")
            params["increment_received"] = increment_received
            param_idx += 1

        if sentiment_score is not None:
            # Running average: new_avg = (old_avg * count + new_value) / (count + 1)
            updates.append(
                f"""sentiment_score = CASE
                    WHEN sentiment_score IS NULL THEN ${param_idx}
                    ELSE (sentiment_score * (total_emails_received + total_emails_sent) + ${param_idx})
                         / (total_emails_received + total_emails_sent + 1)
                END"""
            )
            params["sentiment"] = sentiment_score
            param_idx += 1

        query = f"""
            UPDATE contacts
            SET {', '.join(updates)}
            WHERE id = $1;
        """
        await self.db.execute(query, params)

    # =========================================================================
    # Thread Operations
    # =========================================================================

    async def get_thread_by_id(self, thread_id: str) -> ThreadSummary | None:
        """Get thread by UUID.

        Args:
            thread_id: Thread UUID

        Returns:
            ThreadSummary if found, None otherwise
        """
        query = """
            SELECT t.id, t.subject, t.contact_id, c.email as contact_email,
                   t.status, t.priority, t.category, t.started_at,
                   t.last_activity_at, t.message_count
            FROM threads t
            LEFT JOIN contacts c ON t.contact_id = c.id
            WHERE t.id = $1;
        """
        results = await self.db.execute(query, {"id": thread_id})

        if not results:
            return None

        row = results[0]
        return ThreadSummary(
            id=str(row["id"]),
            subject=row["subject"],
            contact_id=str(row["contact_id"]) if row["contact_id"] else None,
            contact_email=row["contact_email"],
            status=row["status"],
            priority=row["priority"],
            category=row["category"],
            started_at=row["started_at"],
            last_activity_at=row["last_activity_at"],
            message_count=row["message_count"],
        )

    async def get_thread_by_message_id(self, message_id: str) -> ThreadSummary | None:
        """Get thread containing a specific message.

        Args:
            message_id: Email Message-ID header

        Returns:
            ThreadSummary if found, None otherwise
        """
        query = """
            SELECT t.id, t.subject, t.contact_id, c.email as contact_email,
                   t.status, t.priority, t.category, t.started_at,
                   t.last_activity_at, t.message_count
            FROM threads t
            LEFT JOIN contacts c ON t.contact_id = c.id
            JOIN emails e ON e.thread_id = t.id
            WHERE e.message_id = $1;
        """
        results = await self.db.execute(query, {"message_id": message_id})

        if not results:
            return None

        row = results[0]
        return ThreadSummary(
            id=str(row["id"]),
            subject=row["subject"],
            contact_id=str(row["contact_id"]) if row["contact_id"] else None,
            contact_email=row["contact_email"],
            status=row["status"],
            priority=row["priority"],
            category=row["category"],
            started_at=row["started_at"],
            last_activity_at=row["last_activity_at"],
            message_count=row["message_count"],
        )

    async def find_thread_for_reply(self, in_reply_to: str) -> str | None:
        """Find thread for a reply email.

        Args:
            in_reply_to: Message-ID being replied to

        Returns:
            Thread UUID if found, None otherwise
        """
        query = """
            SELECT thread_id FROM emails WHERE message_id = $1;
        """
        results = await self.db.execute(query, {"message_id": in_reply_to})

        if results and results[0]["thread_id"]:
            return str(results[0]["thread_id"])
        return None

    async def create_thread(
        self,
        subject: str | None,
        contact_id: str | None,
        original_message_id: str | None = None,
    ) -> str:
        """Create a new thread.

        Args:
            subject: Thread subject
            contact_id: Primary contact UUID
            original_message_id: Message-ID of first email

        Returns:
            UUID of created thread
        """
        query = """
            INSERT INTO threads (subject, contact_id, original_message_id)
            VALUES ($1, $2, $3)
            RETURNING id;
        """
        results = await self.db.execute(
            query,
            {
                "subject": subject,
                "contact_id": contact_id,
                "original_message_id": original_message_id,
            },
        )

        thread_id = str(results[0]["id"])
        logger.info(f"Created new thread: {thread_id}")
        return thread_id

    async def update_thread_activity(
        self,
        thread_id: str,
        status: str | None = None,
        priority: str | None = None,
        category: str | None = None,
        increment_message_count: bool = True,
    ) -> None:
        """Update thread metadata and activity timestamp.

        Args:
            thread_id: Thread UUID
            status: Optional new status
            priority: Optional new priority
            category: Optional new category
            increment_message_count: Whether to increment message count
        """
        updates = ["last_activity_at = NOW()"]
        params: dict[str, Any] = {"id": thread_id}
        param_idx = 2  # $1 is always thread_id

        if increment_message_count:
            updates.append("message_count = message_count + 1")

        if status:
            updates.append(f"status = ${param_idx}")
            params["status"] = status
            param_idx += 1

        if priority:
            updates.append(f"priority = ${param_idx}")
            params["priority"] = priority
            param_idx += 1

        if category:
            updates.append(f"category = ${param_idx}")
            params["category"] = category
            param_idx += 1

        query = f"""
            UPDATE threads SET {', '.join(updates)} WHERE id = $1;
        """
        await self.db.execute(query, params)

    async def get_contact_threads(
        self, contact_id: str, limit: int = 5
    ) -> list[ThreadSummary]:
        """Get recent threads for a contact.

        Args:
            contact_id: Contact UUID
            limit: Maximum threads to return

        Returns:
            List of ThreadSummary objects
        """
        query = """
            SELECT t.id, t.subject, t.contact_id, c.email as contact_email,
                   t.status, t.priority, t.category, t.started_at,
                   t.last_activity_at, t.message_count
            FROM threads t
            LEFT JOIN contacts c ON t.contact_id = c.id
            WHERE t.contact_id = $1
            ORDER BY t.last_activity_at DESC
            LIMIT $2;
        """
        results = await self.db.execute(query, {"contact_id": contact_id, "limit": limit})

        return [
            ThreadSummary(
                id=str(row["id"]),
                subject=row["subject"],
                contact_id=str(row["contact_id"]) if row["contact_id"] else None,
                contact_email=row["contact_email"],
                status=row["status"],
                priority=row["priority"],
                category=row["category"],
                started_at=row["started_at"],
                last_activity_at=row["last_activity_at"],
                message_count=row["message_count"],
            )
            for row in results
        ]

    # =========================================================================
    # Email Operations
    # =========================================================================

    async def store_email(
        self,
        message_id: str,
        thread_id: str,
        contact_id: str | None,
        direction: str,
        from_address: str,
        to_addresses: list[str],
        cc_addresses: list[str],
        subject: str | None,
        body: str,
        html_body: str | None,
        received_at: datetime,
        in_reply_to: str | None,
        raw_headers: dict | None,
        analysis: EmailAnalysis | None = None,
    ) -> str:
        """Store an email in the database.

        Args:
            message_id: Email Message-ID header
            thread_id: Thread UUID
            contact_id: Contact UUID
            direction: 'inbound' or 'outbound'
            from_address: Sender email
            to_addresses: Recipients
            cc_addresses: CC recipients
            subject: Email subject
            body: Email body text
            html_body: HTML body if available
            received_at: Timestamp
            in_reply_to: Message-ID being replied to
            raw_headers: Raw email headers
            analysis: Optional EmailAnalysis from LLM

        Returns:
            UUID of stored email
        """
        sentiment = None
        sentiment_score = None
        topics: list[str] = []
        intent = None
        urgency = None

        if analysis:
            sentiment = analysis.sentiment
            sentiment_score = analysis.sentiment_score
            topics = analysis.topics
            intent = analysis.intent
            urgency = analysis.urgency

        query = """
            INSERT INTO emails (
                message_id, thread_id, contact_id, direction, from_address,
                to_addresses, cc_addresses, subject, body, html_body,
                received_at, in_reply_to, raw_headers, sentiment, sentiment_score,
                topics, intent, urgency
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14, $15, $16, $17, $18)
            RETURNING id;
        """
        results = await self.db.execute(
            query,
            {
                "message_id": message_id,
                "thread_id": thread_id,
                "contact_id": contact_id,
                "direction": direction,
                "from_address": from_address,
                "to_addresses": to_addresses,
                "cc_addresses": cc_addresses,
                "subject": subject,
                "body": body,
                "html_body": html_body,
                "received_at": received_at,
                "in_reply_to": in_reply_to,
                "raw_headers": raw_headers,
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "topics": topics,
                "intent": intent,
                "urgency": urgency,
            },
        )

        email_id = str(results[0]["id"])
        logger.info(f"Stored email: {email_id} ({direction})")
        return email_id

    async def get_email_by_message_id(self, message_id: str) -> EmailSummary | None:
        """Get email by Message-ID header.

        Args:
            message_id: Email Message-ID header

        Returns:
            EmailSummary if found, None otherwise
        """
        query = """
            SELECT id, message_id, thread_id, contact_id, direction, from_address,
                   to_addresses, subject, body, LEFT(body, 500) as body_preview,
                   received_at, sentiment, sentiment_score, topics, intent, urgency, is_replied
            FROM emails
            WHERE message_id = $1;
        """
        results = await self.db.execute(query, {"message_id": message_id})

        if not results:
            return None

        row = results[0]
        return EmailSummary(
            id=str(row["id"]),
            message_id=row["message_id"],
            thread_id=str(row["thread_id"]) if row["thread_id"] else None,
            contact_id=str(row["contact_id"]) if row["contact_id"] else None,
            direction=row["direction"],
            from_address=row["from_address"],
            to_addresses=row["to_addresses"],
            subject=row["subject"],
            body=row["body"],
            body_preview=row["body_preview"] or "",
            received_at=row["received_at"],
            sentiment=row["sentiment"],
            sentiment_score=row["sentiment_score"],
            topics=row["topics"] or [],
            intent=row["intent"],
            urgency=row["urgency"],
            is_replied=row["is_replied"],
        )

    async def get_thread_emails(
        self, thread_id: str, limit: int = 50
    ) -> list[EmailSummary]:
        """Get all emails in a thread.

        Args:
            thread_id: Thread UUID
            limit: Maximum emails to return

        Returns:
            List of EmailSummary objects (chronological order)
        """
        query = """
            SELECT id, message_id, thread_id, contact_id, direction, from_address,
                   to_addresses, subject, body, LEFT(body, 500) as body_preview,
                   received_at, sentiment, sentiment_score, topics, intent, urgency, is_replied
            FROM emails
            WHERE thread_id = $1
            ORDER BY received_at ASC
            LIMIT $2;
        """
        results = await self.db.execute(query, {"thread_id": thread_id, "limit": limit})

        return [
            EmailSummary(
                id=str(row["id"]),
                message_id=row["message_id"],
                thread_id=str(row["thread_id"]) if row["thread_id"] else None,
                contact_id=str(row["contact_id"]) if row["contact_id"] else None,
                direction=row["direction"],
                from_address=row["from_address"],
                to_addresses=row["to_addresses"],
                subject=row["subject"],
                body=row["body"],
                body_preview=row["body_preview"] or "",
                received_at=row["received_at"],
                sentiment=row["sentiment"],
                sentiment_score=row["sentiment_score"],
                topics=row["topics"] or [],
                intent=row["intent"],
                urgency=row["urgency"],
                is_replied=row["is_replied"],
            )
            for row in results
        ]

    async def search_emails(
        self,
        query_text: str | None = None,
        contact_id: str | None = None,
        contact_email: str | None = None,
        thread_id: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        sentiment: str | None = None,
        min_sentiment: float | None = None,
        max_sentiment: float | None = None,
        intent: str | None = None,
        topic: str | None = None,
        direction: str | None = None,
        is_replied: bool | None = None,
        sort_by: str = "date",
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[EmailSummary], int]:
        """Search emails with multiple criteria.

        Args:
            query_text: Full-text search in subject and body
            contact_id: Filter by contact UUID
            contact_email: Filter by contact email
            thread_id: Filter by thread
            date_from: Filter after this date
            date_to: Filter before this date
            sentiment: Filter by sentiment category
            min_sentiment: Minimum sentiment score
            max_sentiment: Maximum sentiment score
            intent: Filter by intent
            topic: Filter by topic
            direction: Filter by direction
            is_replied: Filter by replied status
            sort_by: Sort order ('date' or 'sentiment')
            limit: Maximum results
            offset: Pagination offset

        Returns:
            Tuple of (list of EmailSummary, total count)
        """
        conditions = ["1=1"]
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        param_idx = 3  # Start after limit and offset

        if query_text:
            conditions.append(
                f"(subject ILIKE ${param_idx} OR body ILIKE ${param_idx})"
            )
            params["query"] = f"%{query_text}%"
            param_idx += 1

        if contact_id:
            conditions.append(f"contact_id = ${param_idx}")
            params["contact_id"] = contact_id
            param_idx += 1

        if contact_email:
            conditions.append(f"contact_id = (SELECT id FROM contacts WHERE email = ${param_idx})")
            params["contact_email"] = contact_email.lower()
            param_idx += 1

        if thread_id:
            conditions.append(f"thread_id = ${param_idx}")
            params["thread_id"] = thread_id
            param_idx += 1

        if date_from:
            conditions.append(f"received_at >= ${param_idx}")
            params["date_from"] = date_from
            param_idx += 1

        if date_to:
            conditions.append(f"received_at <= ${param_idx}")
            params["date_to"] = date_to
            param_idx += 1

        if sentiment:
            conditions.append(f"sentiment = ${param_idx}")
            params["sentiment"] = sentiment
            param_idx += 1

        if min_sentiment is not None:
            conditions.append(f"sentiment_score >= ${param_idx}")
            params["min_sentiment"] = min_sentiment
            param_idx += 1

        if max_sentiment is not None:
            conditions.append(f"sentiment_score <= ${param_idx}")
            params["max_sentiment"] = max_sentiment
            param_idx += 1

        if intent:
            conditions.append(f"intent = ${param_idx}")
            params["intent"] = intent
            param_idx += 1

        if topic:
            conditions.append(f"${param_idx} = ANY(topics)")
            params["topic"] = topic
            param_idx += 1

        if direction:
            conditions.append(f"direction = ${param_idx}")
            params["direction"] = direction
            param_idx += 1

        if is_replied is not None:
            conditions.append(f"is_replied = ${param_idx}")
            params["is_replied"] = is_replied
            param_idx += 1

        where_clause = " AND ".join(conditions)

        # Determine sort order
        order_clause = "received_at DESC"
        if sort_by == "sentiment":
            order_clause = "sentiment_score DESC NULLS LAST, received_at DESC"

        # Count query
        count_query = f"SELECT COUNT(*) as total FROM emails WHERE {where_clause};"
        count_result = await self.db.execute(count_query, params)
        total_count = count_result[0]["total"] if count_result else 0

        # Search query
        search_query = f"""
            SELECT id, message_id, thread_id, contact_id, direction, from_address,
                   to_addresses, subject, body, LEFT(body, 500) as body_preview,
                   received_at, sentiment, sentiment_score, topics, intent, urgency, is_replied
            FROM emails
            WHERE {where_clause}
            ORDER BY {order_clause}
            LIMIT $1 OFFSET $2;
        """
        results = await self.db.execute(search_query, params)

        emails = [
            EmailSummary(
                id=str(row["id"]),
                message_id=row["message_id"],
                thread_id=str(row["thread_id"]) if row["thread_id"] else None,
                contact_id=str(row["contact_id"]) if row["contact_id"] else None,
                direction=row["direction"],
                from_address=row["from_address"],
                to_addresses=row["to_addresses"],
                subject=row["subject"],
                body=row["body"],
                body_preview=row["body_preview"] or "",
                received_at=row["received_at"],
                sentiment=row["sentiment"],
                sentiment_score=row["sentiment_score"],
                topics=row["topics"] or [],
                intent=row["intent"],
                urgency=row["urgency"],
                is_replied=row["is_replied"],
            )
            for row in results
        ]

        return emails, total_count

    async def mark_email_replied(self, email_id: str) -> None:
        """Mark an email as replied.

        Args:
            email_id: Email UUID
        """
        query = "UPDATE emails SET is_replied = TRUE WHERE id = $1;"
        await self.db.execute(query, {"id": email_id})

    # =========================================================================
    # Interaction Operations
    # =========================================================================

    async def log_interaction(
        self,
        contact_id: str,
        interaction_type: str,
        summary: str | None = None,
        outcome: str | None = None,
        thread_id: str | None = None,
        email_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Log a customer interaction.

        Args:
            contact_id: Contact UUID
            interaction_type: Type of interaction (email_received, email_sent, etc.)
            summary: Brief summary
            outcome: Outcome if applicable
            thread_id: Related thread UUID
            email_id: Related email UUID
            metadata: Additional metadata

        Returns:
            UUID of created interaction
        """
        query = """
            INSERT INTO interactions (contact_id, interaction_type, summary, outcome,
                                      thread_id, email_id, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id;
        """
        results = await self.db.execute(
            query,
            {
                "contact_id": contact_id,
                "interaction_type": interaction_type,
                "summary": summary,
                "outcome": outcome,
                "thread_id": thread_id,
                "email_id": email_id,
                "metadata": metadata or {},
            },
        )

        return str(results[0]["id"])

    async def get_contact_interactions(
        self, contact_id: str, limit: int = 10
    ) -> list[InteractionRecord]:
        """Get recent interactions for a contact.

        Args:
            contact_id: Contact UUID
            limit: Maximum interactions to return

        Returns:
            List of InteractionRecord objects
        """
        query = """
            SELECT id, interaction_type, summary, outcome, created_at, email_id, thread_id
            FROM interactions
            WHERE contact_id = $1
            ORDER BY created_at DESC
            LIMIT $2;
        """
        results = await self.db.execute(query, {"contact_id": contact_id, "limit": limit})

        return [
            InteractionRecord(
                id=str(row["id"]),
                interaction_type=row["interaction_type"],
                summary=row["summary"],
                outcome=row["outcome"],
                created_at=row["created_at"],
                email_id=str(row["email_id"]) if row["email_id"] else None,
                thread_id=str(row["thread_id"]) if row["thread_id"] else None,
            )
            for row in results
        ]
