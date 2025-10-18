"""Project management system for flowlib.

This package provides project-based configuration and extension loading,
replacing the previous auto-discovery system with a cleaner project concept.
"""

from .project import Project, get_project

__all__ = [
    'Project',
    'get_project',
]
