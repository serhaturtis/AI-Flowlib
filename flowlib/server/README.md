# Flowlib Server

FastAPI backend for Flowlib project and agent management.

## Development

### Setup

```bash
# Install dependencies
cd flowlib/server
pip install -e ".[dev]"

# Run development server
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Architecture

- `server/api/` - FastAPI routers
- `server/services/` - Business logic wrapping Flowlib core
- `server/models/` - Pydantic request/response schemas
- `server/persistence/` - SQLAlchemy models and database setup
- `server/workers/` - Celery/RQ tasks for background jobs
- `server/core/` - Application configuration, logging, security

