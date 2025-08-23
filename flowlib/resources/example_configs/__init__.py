"""Example configuration files for Flowlib.

This package contains example configuration files for all standard Flowlib roles.
These files should be copied to ~/.flowlib/active_configs/ and modified as needed.

Standard configuration roles:
- default-llm: Primary language model
- default-embedding: Embedding model for vectors  
- default-vector-db: Vector database for similarity search
- default-graph-db: Graph database for knowledge graphs
- default-database: Database for persistence
- default-cache: Cache for temporary data

Usage:
1. Copy example files to ~/.flowlib/active_configs/
2. Rename files (remove "example_" prefix)  
3. Modify configuration values for your setup
4. Restart flowlib applications to load new configs
"""

# List of all example configuration files
EXAMPLE_CONFIGS = [
    "example_default_llm.py",
    "example_default_embedding.py", 
    "example_default_vector_db.py",
    "example_default_graph_db.py",
    "example_default_database.py",
    "example_default_cache.py",
    "example_model_config.py",
    "example_embedding_model_config.py"
]

# Mapping from example file to target file name
EXAMPLE_TO_TARGET = {
    "example_default_llm.py": "default_llm.py",
    "example_default_embedding.py": "default_embedding.py",
    "example_default_vector_db.py": "default_vector_db.py", 
    "example_default_graph_db.py": "default_graph_db.py",
    "example_default_database.py": "default_database.py",
    "example_default_cache.py": "default_cache.py",
    "example_model_config.py": "model_config_example.py",
    "example_embedding_model_config.py": "embedding_model_config_example.py"
}