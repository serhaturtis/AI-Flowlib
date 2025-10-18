"""Task completion evaluation component."""

from . import flow, prompts  # Import to register flows and prompts
from .component import CompletionEvaluatorComponent
from .models import EvaluationInput, EvaluationOutput, EvaluationResult

__all__ = [
    "CompletionEvaluatorComponent",
    "EvaluationResult",
    "EvaluationInput",
    "EvaluationOutput",
    "flow",
    "prompts",
]
