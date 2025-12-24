"""Migration v001: Email Sales Agent Schema.

Creates the core tables for the smart email sales agent:
- contacts: Customer profiles
- threads: Email conversation chains
- emails: Individual email messages
- topics: Extracted topic taxonomy
- email_topics: Many-to-many relationship
- interactions: Customer interaction timeline
"""

from typing import TYPE_CHECKING

from flowlib.providers.db.postgres.migrations.migration_runner import Migration

if TYPE_CHECKING:
    from flowlib.providers.db.postgres.provider import PostgreSQLProvider


class EmailSalesSchema(Migration):
    """Create email sales agent database schema."""

    version = "001"
    description = "Email sales agent schema - contacts, threads, emails, topics, interactions"

    async def up(self, db: "PostgreSQLProvider") -> None:
        """Create all email sales tables and indexes."""

        # Enable UUID extension
        await db.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

        # 1. Contacts table - Customer profiles
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS contacts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                display_name VARCHAR(255),
                company VARCHAR(255),
                first_seen_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_contact_at TIMESTAMP NOT NULL DEFAULT NOW(),
                total_emails_sent INT DEFAULT 0,
                total_emails_received INT DEFAULT 0,
                profile_data JSONB DEFAULT '{}',
                sentiment_score FLOAT,
                tags TEXT[],
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            """
        )

        # 2. Threads table - Email conversation chains
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                subject VARCHAR(1000),
                original_message_id VARCHAR(500),
                contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
                status VARCHAR(50) DEFAULT 'open',
                priority VARCHAR(20) DEFAULT 'normal',
                category VARCHAR(100),
                started_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_activity_at TIMESTAMP NOT NULL DEFAULT NOW(),
                message_count INT DEFAULT 0,
                metadata JSONB DEFAULT '{}'
            );
            """
        )

        # 3. Emails table - Individual email messages
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS emails (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                message_id VARCHAR(500) UNIQUE,
                thread_id UUID REFERENCES threads(id) ON DELETE CASCADE,
                contact_id UUID REFERENCES contacts(id) ON DELETE SET NULL,
                direction VARCHAR(10) NOT NULL,
                from_address VARCHAR(255) NOT NULL,
                to_addresses TEXT[] NOT NULL,
                cc_addresses TEXT[],
                subject VARCHAR(1000),
                body TEXT NOT NULL,
                html_body TEXT,
                in_reply_to VARCHAR(500),
                received_at TIMESTAMP NOT NULL,
                processed_at TIMESTAMP DEFAULT NOW(),
                sentiment VARCHAR(20),
                sentiment_score FLOAT,
                topics TEXT[],
                intent VARCHAR(50),
                urgency VARCHAR(20),
                is_read BOOLEAN DEFAULT FALSE,
                is_replied BOOLEAN DEFAULT FALSE,
                raw_headers JSONB,
                metadata JSONB DEFAULT '{}'
            );
            """
        )

        # 4. Topics table - Extracted topic taxonomy
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL,
                category VARCHAR(100),
                description TEXT,
                email_count INT DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            """
        )

        # 5. Email-Topics junction table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS email_topics (
                email_id UUID REFERENCES emails(id) ON DELETE CASCADE,
                topic_id INT REFERENCES topics(id) ON DELETE CASCADE,
                confidence FLOAT,
                PRIMARY KEY (email_id, topic_id)
            );
            """
        )

        # 6. Interactions table - Customer interaction timeline
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                contact_id UUID REFERENCES contacts(id) ON DELETE CASCADE NOT NULL,
                thread_id UUID REFERENCES threads(id) ON DELETE SET NULL,
                email_id UUID REFERENCES emails(id) ON DELETE SET NULL,
                interaction_type VARCHAR(50) NOT NULL,
                summary TEXT,
                outcome VARCHAR(100),
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'
            );
            """
        )

        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_emails_thread_id ON emails(thread_id);",
            "CREATE INDEX IF NOT EXISTS idx_emails_contact_id ON emails(contact_id);",
            "CREATE INDEX IF NOT EXISTS idx_emails_received_at ON emails(received_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_emails_message_id ON emails(message_id);",
            "CREATE INDEX IF NOT EXISTS idx_emails_direction ON emails(direction);",
            "CREATE INDEX IF NOT EXISTS idx_emails_sentiment ON emails(sentiment);",
            "CREATE INDEX IF NOT EXISTS idx_emails_intent ON emails(intent);",
            "CREATE INDEX IF NOT EXISTS idx_threads_contact_id ON threads(contact_id);",
            "CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);",
            "CREATE INDEX IF NOT EXISTS idx_threads_last_activity ON threads(last_activity_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_contacts_email ON contacts(email);",
            "CREATE INDEX IF NOT EXISTS idx_contacts_last_contact ON contacts(last_contact_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_interactions_contact_id ON interactions(contact_id);",
            "CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON interactions(created_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_interactions_type ON interactions(interaction_type);",
        ]

        for index_sql in indexes:
            await db.execute(index_sql)

        # Create trigger function for updating timestamps
        await db.execute(
            """
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        )

        # Create trigger for contacts updated_at
        await db.execute(
            """
            DROP TRIGGER IF EXISTS contacts_updated_at ON contacts;
            CREATE TRIGGER contacts_updated_at
                BEFORE UPDATE ON contacts
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
            """
        )

    async def down(self, db: "PostgreSQLProvider") -> None:
        """Drop all email sales tables (reverse migration)."""

        # Drop in reverse order to respect foreign key constraints
        drop_statements = [
            "DROP TABLE IF EXISTS email_topics CASCADE;",
            "DROP TABLE IF EXISTS interactions CASCADE;",
            "DROP TABLE IF EXISTS emails CASCADE;",
            "DROP TABLE IF EXISTS threads CASCADE;",
            "DROP TABLE IF EXISTS topics CASCADE;",
            "DROP TABLE IF EXISTS contacts CASCADE;",
            "DROP FUNCTION IF EXISTS update_updated_at_column CASCADE;",
        ]

        for statement in drop_statements:
            await db.execute(statement)
