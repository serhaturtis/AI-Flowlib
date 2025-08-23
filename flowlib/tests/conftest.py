"""Global pytest configuration and fixtures for integration tests."""

import os
import pytest
from typing import Optional


@pytest.fixture(scope="session", autouse=True)
def setup_test_resources():
    """Automatically load test resources for all tests."""
    from .test_utils import setup_test_environment
    setup_test_environment()


def get_env_setting(env_var: str, env_type: type = str, default=None):
    """Get environment variable with type conversion."""
    value = os.environ.get(env_var)
    if value is None:
        if default is not None:
            return default
        # For integration tests, skip if env var not set
        pytest.skip(f"Integration test skipped: {env_var} not set")
    
    if env_type == int:
        return int(value)
    elif env_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    return value


@pytest.fixture(scope="session")
def redis_settings():
    """Redis settings for integration tests from environment."""
    from flowlib.providers.cache.redis.provider import RedisCacheProviderSettings
    
    return RedisCacheProviderSettings(
        host=get_env_setting("FLOWLIB_REDIS_HOST", default="localhost"),
        port=get_env_setting("FLOWLIB_REDIS_PORT", int, default=6379),
        db=get_env_setting("FLOWLIB_REDIS_DB", int, default=0),
        namespace="integration_test",
        default_ttl=60
    )


@pytest.fixture(scope="session")
def postgres_settings():
    """PostgreSQL settings for integration tests from environment."""
    from flowlib.providers.db.postgres.provider import PostgreSQLProviderSettings
    
    return PostgreSQLProviderSettings(
        host=get_env_setting("FLOWLIB_POSTGRES_HOST", default="localhost"),
        port=get_env_setting("FLOWLIB_POSTGRES_PORT", int, default=5432),
        database=get_env_setting("FLOWLIB_POSTGRES_DATABASE", default="flowlib_test"),
        username=get_env_setting("FLOWLIB_POSTGRES_USERNAME", default="test"),
        password=get_env_setting("FLOWLIB_POSTGRES_PASSWORD", default="test"),
        pool_size=5
    )


@pytest.fixture(scope="session")
def mongodb_settings():
    """MongoDB settings for integration tests from environment."""
    from flowlib.providers.db.mongodb.provider import MongoDBProviderSettings
    
    return MongoDBProviderSettings(
        host=get_env_setting("FLOWLIB_MONGODB_HOST", default="localhost"),
        port=get_env_setting("FLOWLIB_MONGODB_PORT", int, default=27017),
        database="flowlib_test",
        username=get_env_setting("FLOWLIB_MONGODB_USERNAME", default="test"),
        password=get_env_setting("FLOWLIB_MONGODB_PASSWORD", default="test"),
        auth_source="admin"
    )


@pytest.fixture(scope="session")
def chroma_settings():
    """ChromaDB settings for integration tests from environment."""
    from flowlib.providers.vector.chroma.provider import ChromaDBProviderSettings
    import tempfile
    
    # Use local persistent client by default for integration tests
    return ChromaDBProviderSettings(
        persist_directory=tempfile.mkdtemp(prefix="chroma_test_"),
        client_type="persistent",
        collection_name="test_collection"
    )


@pytest.fixture(scope="session") 
def qdrant_settings():
    """Qdrant settings for integration tests from environment."""
    from flowlib.providers.vector.qdrant.provider import QdrantProviderSettings
    
    return QdrantProviderSettings(
        host=get_env_setting("FLOWLIB_QDRANT_HOST", default="localhost"),
        port=get_env_setting("FLOWLIB_QDRANT_PORT", int, default=6333),
        prefer_grpc=False
    )


@pytest.fixture(scope="session")
def sqlite_settings():
    """SQLite settings for integration tests."""
    from flowlib.providers.db.sqlite.provider import SQLiteProviderSettings
    import tempfile
    
    # Use temporary file for integration tests
    return SQLiteProviderSettings(
        database_path=":memory:",  # In-memory for tests
        create_if_missing=True
    )


@pytest.fixture(scope="session")
def pinecone_settings():
    """Pinecone settings for integration tests from environment."""
    from flowlib.providers.vector.pinecone.provider import PineconeProviderSettings
    
    api_key = os.environ.get("FLOWLIB_PINECONE_API_KEY", "test_api_key")
    host = os.environ.get("FLOWLIB_PINECONE_HOST")
    
    return PineconeProviderSettings(
        api_key=api_key,
        environment=get_env_setting("FLOWLIB_PINECONE_ENVIRONMENT", default="us-west1-gcp"),
        index_name="integration_test"
    )


@pytest.fixture(scope="session")
def s3_settings():
    """S3 settings for integration tests from environment."""
    from flowlib.providers.storage.s3.provider import S3ProviderSettings
    
    return S3ProviderSettings(
        bucket_name=get_env_setting("FLOWLIB_S3_BUCKET"),
        region=get_env_setting("FLOWLIB_S3_REGION", default="us-east-1"),
        access_key_id=get_env_setting("FLOWLIB_S3_ACCESS_KEY_ID"),
        secret_access_key=get_env_setting("FLOWLIB_S3_SECRET_ACCESS_KEY"),
        endpoint_url=os.environ.get("FLOWLIB_S3_ENDPOINT_URL")  # Optional
    )


@pytest.fixture(scope="session")
def neo4j_settings():
    """Neo4j settings for integration tests from environment."""
    from flowlib.providers.graph.neo4j.provider import Neo4jProviderSettings
    
    return Neo4jProviderSettings(
        uri=get_env_setting("FLOWLIB_NEO4J_URI"),
        username=get_env_setting("FLOWLIB_NEO4J_USERNAME"),
        password=get_env_setting("FLOWLIB_NEO4J_PASSWORD"),
        database=get_env_setting("FLOWLIB_NEO4J_DATABASE", default="neo4j")
    )


@pytest.fixture(scope="session")
def arango_settings():
    """ArangoDB settings for integration tests from environment."""
    from flowlib.providers.graph.arango.provider import ArangoProviderSettings
    
    return ArangoProviderSettings(
        url=get_env_setting("FLOWLIB_ARANGO_URL", default="http://localhost:8529"),
        username=get_env_setting("FLOWLIB_ARANGO_USERNAME", default="root"),
        password=get_env_setting("FLOWLIB_ARANGO_PASSWORD", default=""),
        database=get_env_setting("FLOWLIB_ARANGO_DATABASE", default="flowlib_test"),
        verify=False  # Disable SSL verification for Docker testing
    )


@pytest.fixture(scope="session")
def janus_settings():
    """JanusGraph settings for integration tests from environment."""
    from flowlib.providers.graph.janus.provider import JanusProviderSettings
    
    return JanusProviderSettings(
        host=get_env_setting("FLOWLIB_JANUS_HOST"),
        port=get_env_setting("FLOWLIB_JANUS_PORT", int),
        graph_name=get_env_setting("FLOWLIB_JANUS_GRAPH", default="g")
    )


@pytest.fixture(scope="session")
def kafka_settings():
    """Kafka settings for integration tests from environment."""
    from flowlib.providers.mq.kafka.provider import KafkaProviderSettings
    
    return KafkaProviderSettings(
        bootstrap_servers=get_env_setting("FLOWLIB_KAFKA_BOOTSTRAP_SERVERS"),
        group_id="integration_test",
        auto_offset_reset="earliest"
    )


@pytest.fixture(scope="session")
def rabbitmq_settings():
    """RabbitMQ settings for integration tests from environment."""
    from flowlib.providers.mq.rabbitmq.provider import RabbitMQProviderSettings
    
    return RabbitMQProviderSettings(
        host=get_env_setting("FLOWLIB_RABBITMQ_HOST"),
        port=get_env_setting("FLOWLIB_RABBITMQ_PORT", int),
        username=get_env_setting("FLOWLIB_RABBITMQ_USERNAME"),
        password=get_env_setting("FLOWLIB_RABBITMQ_PASSWORD"),
        virtual_host=get_env_setting("FLOWLIB_RABBITMQ_VHOST", default="/")
    )


@pytest.fixture(scope="session")
def google_ai_settings():
    """Google AI settings for integration tests from environment."""
    from flowlib.providers.llm.google_ai.provider import GoogleAIProviderSettings
    
    api_key = os.environ.get("FLOWLIB_GOOGLE_AI_API_KEY")
    if not api_key:
        pytest.skip("Integration test skipped: FLOWLIB_GOOGLE_AI_API_KEY not set")
    
    return GoogleAIProviderSettings(
        api_key=api_key,
        model="gemini-1.5-flash",
        temperature=0.7
    )


@pytest.fixture(scope="session")
def llama_cpp_settings():
    """Llama.cpp settings for integration tests from environment."""
    from flowlib.providers.llm.llama_cpp.provider import LlamaCppProviderSettings
    
    model_path = os.environ.get("FLOWLIB_LLAMA_MODEL_PATH")
    if not model_path:
        pytest.skip("Integration test skipped: FLOWLIB_LLAMA_MODEL_PATH not set")
    
    return LlamaCppProviderSettings(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0
    )


@pytest.fixture(scope="session")
def llama_embedding_settings():
    """Llama.cpp embedding settings for integration tests from environment."""
    from flowlib.providers.embedding.llama_cpp.provider import LlamaCppEmbeddingProviderSettings
    
    model_path = os.environ.get("FLOWLIB_LLAMA_EMBEDDING_MODEL_PATH")
    if not model_path:
        pytest.skip("Integration test skipped: FLOWLIB_LLAMA_EMBEDDING_MODEL_PATH not set")
    
    return LlamaCppEmbeddingProviderSettings(
        model_path=model_path,
        n_ctx=512,
        embedding_dimension=384
    )