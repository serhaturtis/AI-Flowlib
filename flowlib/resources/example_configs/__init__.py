"""Example configuration files for Flowlib.

This directory contains example configuration files that are automatically
copied to ~/.flowlib/configs/ during initialization.

These files are NOT imported as Python modules - they are copied as templates
for users to modify. This prevents accidental registration during package import.

Usage:
1. Example files are automatically copied to ~/.flowlib/configs/ on first run
2. Users create role assignments in ~/.flowlib/roles/assignments.py as needed
3. Modify configuration values and role assignments as needed
4. Restart flowlib applications to load changes
"""

# Mapping for copying example files to project structure
# Configs go to configs/, agents to agents/, profiles to profiles/
EXAMPLE_TO_TARGET = {
    # Provider configs to configs/
    "example_default_llm.py": "example_llamacpp_provider.py",
    "example_default_embedding.py": "example_llamacpp_embedding_provider.py",
    "example_default_vector_db.py": "example_vector_db_provider.py",
    "example_default_graph_db.py": "example_graph_db_provider.py",
    "example_default_database.py": "example_database_provider.py",
    "example_default_cache.py": "example_cache_provider.py",
    "example_model_config.py": "example_model_config.py",
    "example_embedding_model_config.py": "example_embedding_model_config.py",
    # Agent profiles to profiles/
    "example_agent_profiles.py": "../profiles/agent_profiles.py",
    # Agent configs to agents/
    "example_default_agent_config.py": "../agents/default_agent_config.py",
    "example_creative_agent.py": "../agents/creative_agent_config.py",
    "example_precise_agent.py": "../agents/precise_agent_config.py",
    "example_devops_agent.py": "../agents/devops_agent_config.py",
    "example_custom_agent_template.py": "../agents/custom_agent_template.py",
    # Role assignments to roles/
    "example_role_assignments.py": "../roles/assignments.py",
}
