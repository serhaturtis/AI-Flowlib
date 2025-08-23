"""Prompts for knowledge orchestration."""

from flowlib.resources.decorators.decorators import prompt
from flowlib.resources.models.base import ResourceBase
from flowlib.providers.llm import PromptConfigOverride


@prompt(name="orchestration-progress-summary")
class OrchestrationProgressPrompt(ResourceBase):
    """Generate progress summary for orchestration pipeline."""
    
    template: str = """
Generate a comprehensive progress summary for the knowledge extraction pipeline:

Current Status:
- Stage: {{current_stage}}
- Documents: {{processed_documents}}/{{total_documents}} ({{progress_percentage}}%)
- Current document: {{current_document}}

Stage Completion:
- Document extraction: {{extraction_complete}}
- Entity analysis: {{analysis_complete}}
- Vector storage: {{vector_storage_complete}}
- Graph storage: {{graph_storage_complete}}

Statistics:
- Entities extracted: {{total_entities}}
- Relationships found: {{total_relationships}}
- Text chunks created: {{total_chunks}}

Provide:
1. Overall pipeline health assessment
2. Estimated completion time
3. Performance insights
4. Any bottlenecks or issues detected
5. Recommendations for optimization
"""
    
    config: PromptConfigOverride = PromptConfigOverride(
        temperature=0.3,
        max_tokens=3000
    )


@prompt(name="orchestration-error-analysis")
class OrchestrationErrorAnalysisPrompt(ResourceBase):
    """Analyze errors in orchestration pipeline."""
    
    template: str = """
Analyze the errors that occurred during knowledge extraction pipeline:

Error Context:
- Stage: {{error_stage}}
- Document: {{error_document}}
- Error type: {{error_type}}
- Error message: {{error_message}}

Pipeline State:
- Total documents: {{total_documents}}
- Processed successfully: {{successful_documents}}
- Failed documents: {{failed_documents}}

Provide:
1. Root cause analysis of the error
2. Impact assessment on overall pipeline
3. Recovery strategies
4. Prevention recommendations
5. Whether to continue or abort processing

Focus on actionable insights for pipeline reliability.
"""
    
    config: PromptConfigOverride = PromptConfigOverride(
        temperature=0.2,
        max_tokens=2500
    )


@prompt(name="orchestration-optimization")
class OrchestrationOptimizationPrompt(ResourceBase):
    """Suggest optimizations for orchestration pipeline."""
    
    template: str = """
Analyze the knowledge extraction pipeline performance and suggest optimizations:

Performance Metrics:
- Total processing time: {{processing_time}} seconds
- Documents per minute: {{documents_per_minute}}
- Average document size: {{avg_document_size}} bytes
- Memory usage: {{memory_usage}}

Bottlenecks Identified:
- Slowest stage: {{slowest_stage}}
- Resource constraints: {{resource_constraints}}
- I/O patterns: {{io_patterns}}

Current Configuration:
- Chunk size: {{chunk_size}}
- Chunk overlap: {{chunk_overlap}}
- Parallel processing: {{parallel_processing}}

Suggest optimizations for:
1. Processing speed improvements
2. Memory efficiency gains
3. I/O optimization strategies
4. Configuration parameter tuning
5. Parallel processing enhancements

Provide specific, actionable recommendations with expected impact.
"""
    
    config: PromptConfigOverride = PromptConfigOverride(
        temperature=0.4,
        max_tokens=3500
    )
