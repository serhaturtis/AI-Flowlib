"""Migration v002: Business Discovery Agent Schema.

Creates the core tables for the business development agent:
- events: Business exhibitions, trade shows, conferences
- exhibitors: Companies exhibiting at events
- opportunities: Identified business opportunities
- discovery_runs: Track each discovery execution
- discovery_reports: Generated reports
"""

from typing import TYPE_CHECKING

from flowlib.providers.db.postgres.migrations.migration_runner import Migration

if TYPE_CHECKING:
    from flowlib.providers.db.postgres.provider import PostgreSQLProvider


class BusinessDiscoverySchema(Migration):
    """Create business discovery agent database schema."""

    version = "002"
    description = "Business discovery schema - events, exhibitors, opportunities, reports"

    async def up(self, db: "PostgreSQLProvider") -> None:
        """Create all business discovery tables and indexes."""

        # Ensure UUID extension is available (may already exist from v001)
        await db.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

        # 1. Events table - Business exhibitions, trade shows, conferences
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                external_id VARCHAR(500),
                name VARCHAR(500) NOT NULL,
                event_type VARCHAR(100),
                description TEXT,
                industry_sectors TEXT[],
                country VARCHAR(100) NOT NULL,
                city VARCHAR(200),
                venue VARCHAR(500),
                start_date DATE,
                end_date DATE,
                registration_deadline DATE,
                website_url VARCHAR(1000),
                source_url VARCHAR(1000),
                organizer VARCHAR(500),
                expected_attendees INT,
                relevance_score FLOAT,
                status VARCHAR(50) DEFAULT 'discovered',
                discovered_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'
            );
            """
        )

        # 2. Exhibitors table - Companies exhibiting at events
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS exhibitors (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                event_id UUID REFERENCES events(id) ON DELETE CASCADE,
                company_name VARCHAR(500) NOT NULL,
                booth_number VARCHAR(50),
                company_website VARCHAR(1000),
                industry VARCHAR(200),
                company_size VARCHAR(50),
                description TEXT,
                contact_email VARCHAR(255),
                contact_phone VARCHAR(50),
                relevance_score FLOAT,
                is_prospect BOOLEAN DEFAULT FALSE,
                notes TEXT,
                discovered_at TIMESTAMP NOT NULL DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'
            );
            """
        )

        # 3. Opportunities table - Identified business opportunities
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS opportunities (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                event_id UUID REFERENCES events(id) ON DELETE SET NULL,
                exhibitor_id UUID REFERENCES exhibitors(id) ON DELETE SET NULL,
                opportunity_type VARCHAR(100),
                title VARCHAR(500) NOT NULL,
                description TEXT,
                potential_value VARCHAR(100),
                priority VARCHAR(50) DEFAULT 'normal',
                recommended_action TEXT,
                services_matched TEXT[],
                status VARCHAR(50) DEFAULT 'new',
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'
            );
            """
        )

        # 4. Discovery runs table - Track each discovery execution
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS discovery_runs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                started_at TIMESTAMP NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMP,
                status VARCHAR(50) DEFAULT 'running',
                country_filter VARCHAR(100),
                events_found INT DEFAULT 0,
                exhibitors_found INT DEFAULT 0,
                opportunities_identified INT DEFAULT 0,
                report_id UUID,
                error_message TEXT,
                metadata JSONB DEFAULT '{}'
            );
            """
        )

        # 5. Discovery reports table - Generated reports
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS discovery_reports (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                run_id UUID REFERENCES discovery_runs(id) ON DELETE SET NULL,
                report_type VARCHAR(50),
                title VARCHAR(500) NOT NULL,
                summary TEXT,
                content_html TEXT,
                content_markdown TEXT,
                event_count INT,
                opportunity_count INT,
                top_events JSONB,
                top_opportunities JSONB,
                email_sent BOOLEAN DEFAULT FALSE,
                email_sent_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
            """
        )

        # Create indexes for performance
        indexes = [
            # Events indexes
            "CREATE INDEX IF NOT EXISTS idx_events_country ON events(country);",
            "CREATE INDEX IF NOT EXISTS idx_events_start_date ON events(start_date);",
            "CREATE INDEX IF NOT EXISTS idx_events_end_date ON events(end_date);",
            "CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);",
            "CREATE INDEX IF NOT EXISTS idx_events_relevance ON events(relevance_score DESC);",
            "CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);",
            "CREATE INDEX IF NOT EXISTS idx_events_discovered_at ON events(discovered_at DESC);",
            # Exhibitors indexes
            "CREATE INDEX IF NOT EXISTS idx_exhibitors_event_id ON exhibitors(event_id);",
            "CREATE INDEX IF NOT EXISTS idx_exhibitors_is_prospect ON exhibitors(is_prospect);",
            "CREATE INDEX IF NOT EXISTS idx_exhibitors_relevance ON exhibitors(relevance_score DESC);",
            "CREATE INDEX IF NOT EXISTS idx_exhibitors_industry ON exhibitors(industry);",
            # Opportunities indexes
            "CREATE INDEX IF NOT EXISTS idx_opportunities_event_id ON opportunities(event_id);",
            "CREATE INDEX IF NOT EXISTS idx_opportunities_exhibitor_id ON opportunities(exhibitor_id);",
            "CREATE INDEX IF NOT EXISTS idx_opportunities_status ON opportunities(status);",
            "CREATE INDEX IF NOT EXISTS idx_opportunities_priority ON opportunities(priority);",
            "CREATE INDEX IF NOT EXISTS idx_opportunities_potential_value ON opportunities(potential_value);",
            "CREATE INDEX IF NOT EXISTS idx_opportunities_created_at ON opportunities(created_at DESC);",
            # Discovery runs indexes
            "CREATE INDEX IF NOT EXISTS idx_discovery_runs_started_at ON discovery_runs(started_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_discovery_runs_status ON discovery_runs(status);",
            # Discovery reports indexes
            "CREATE INDEX IF NOT EXISTS idx_discovery_reports_run_id ON discovery_reports(run_id);",
            "CREATE INDEX IF NOT EXISTS idx_discovery_reports_created_at ON discovery_reports(created_at DESC);",
        ]

        for index_sql in indexes:
            await db.execute(index_sql)

        # Create triggers for updating timestamps (reuse function from v001 if exists)
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

        # Create trigger for events updated_at
        await db.execute(
            """
            DROP TRIGGER IF EXISTS events_updated_at ON events;
            CREATE TRIGGER events_updated_at
                BEFORE UPDATE ON events
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
            """
        )

        # Create trigger for opportunities updated_at
        await db.execute(
            """
            DROP TRIGGER IF EXISTS opportunities_updated_at ON opportunities;
            CREATE TRIGGER opportunities_updated_at
                BEFORE UPDATE ON opportunities
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
            """
        )

    async def down(self, db: "PostgreSQLProvider") -> None:
        """Drop all business discovery tables (reverse migration)."""

        # Drop in reverse order to respect foreign key constraints
        drop_statements = [
            "DROP TABLE IF EXISTS discovery_reports CASCADE;",
            "DROP TABLE IF EXISTS discovery_runs CASCADE;",
            "DROP TABLE IF EXISTS opportunities CASCADE;",
            "DROP TABLE IF EXISTS exhibitors CASCADE;",
            "DROP TABLE IF EXISTS events CASCADE;",
            # Don't drop update_updated_at_column as it may be used by v001
        ]

        for statement in drop_statements:
            await db.execute(statement)
