"""Prompts for plugin generation."""

from flowlib.providers.llm import PromptConfigOverride
from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase


@prompt(name="plugin-manifest-generation")
class PluginManifestPrompt(ResourceBase):
    """Generate plugin manifest metadata."""

    template: str = """
Generate a comprehensive manifest for a knowledge plugin based on the extracted data:

Plugin Name: {{plugin_name}}
Domains: {{domains}}
Extracted Statistics:
- Documents: {{total_documents}}
- Entities: {{total_entities}} 
- Relationships: {{total_relationships}}
- Chunks: {{total_chunks}}

Generate appropriate:
- Description that captures the knowledge domain
- Tags that categorize the content
- Version information
- Priority level (1-100 based on data quality and completeness)

Consider the extracted data quality and coverage when determining metadata.
"""

    config: PromptConfigOverride = PromptConfigOverride(
        temperature=0.3,
        max_tokens=800
    )


@prompt(name="plugin-readme-generation")
class PluginReadmePrompt(ResourceBase):
    """Generate plugin README documentation."""

    template: str = """
Generate comprehensive README documentation for a knowledge plugin:

Plugin: {{plugin_name}}
Description: {{description}}
Domains: {{domains}}

Statistics:
- {{total_documents}} documents processed
- {{total_entities}} entities extracted
- {{total_relationships}} relationships found
- Processing time: {{processing_time}} seconds

Generate sections for:
1. Overview and purpose
2. Installation instructions
3. Usage examples with sample queries
4. Plugin structure explanation
5. Configuration options
6. Troubleshooting common issues

Make it informative and user-friendly for developers who will use this plugin.
"""

    config: PromptConfigOverride = PromptConfigOverride(
        temperature=0.4,
        max_tokens=1500
    )


@prompt(name="plugin-provider-optimization")
class PluginProviderOptimizationPrompt(ResourceBase):
    """Optimize plugin provider code for specific use cases."""

    template: str = """
Optimize the knowledge plugin provider implementation for:

Domains: {{domains}}
Data characteristics:
- Entity types: {{entity_types}}
- Relationship patterns: {{relationship_patterns}}
- Data volume: {{data_volume}}

Current provider structure:
{{current_provider_code}}

Suggest optimizations for:
1. Query performance based on data patterns
2. Domain-specific search strategies
3. Caching strategies for frequently accessed data
4. Memory efficiency for large datasets
5. Error handling for edge cases

Provide specific code improvements and explain the reasoning.
"""

    config: PromptConfigOverride = PromptConfigOverride(
        temperature=0.2,
        max_tokens=2000
    )
