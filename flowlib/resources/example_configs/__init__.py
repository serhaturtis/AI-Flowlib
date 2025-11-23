"""Example configuration files for Flowlib.

This directory contains example configuration files that are automatically
copied to ~/.flowlib/configs/ during initialization.

These files are NOT imported as Python modules - they are copied as templates
for users to modify. This prevents accidental registration during package import.

Usage:
1. Example files are automatically copied to ~/.flowlib/configs/ on first run
2. Users define aliases in ~/.flowlib/configs/aliases.py as needed
3. Modify configuration values and alias bindings as needed
4. Restart flowlib applications to load changes
"""

# Mapping for copying example files to project structure
# Provider configs go to configs/providers/, resources to configs/resources/, agents to agents/
EXAMPLE_TO_TARGET = {
    # Provider configs to configs/providers/
    "example_default_llm.py": "providers/example_llamacpp_provider.py",
    "example_default_embedding.py": "providers/example_llamacpp_embedding_provider.py",
    "example_default_vector_db.py": "providers/example_vector_db_provider.py",
    "example_default_graph_db.py": "providers/example_graph_db_provider.py",
    "example_default_database.py": "providers/example_database_provider.py",
    "example_default_cache.py": "providers/example_cache_provider.py",
    "example_default_multimodal_llm.py": "providers/example_default_multimodal_llm.py",
    # Resource configs to configs/resources/
    "example_model_config.py": "resources/example_model_config.py",
    "example_embedding_model_config.py": "resources/example_embedding_model_config.py",
    "example_multimodal_model_config.py": "resources/example_multimodal_model_config.py",
    # Agent configs to agents/
    "example_default_agent_config.py": "../agents/default_agent_config.py",
    "example_creative_agent.py": "../agents/creative_agent_config.py",
    "example_precise_agent.py": "../agents/precise_agent_config.py",
    "example_devops_agent.py": "../agents/devops_agent_config.py",
    "example_custom_agent_template.py": "../agents/custom_agent_template.py",
    # Alias bindings to configs/
    "example_aliases.py": "aliases.py",
}
