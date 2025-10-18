"""Workspace scanners package.

Exports scanner interface and registry for domain scanner implementations.
"""

from .base import BaseDomainScanner
from .registry import ScannerRegistry

__all__ = [
    "BaseDomainScanner",
    "ScannerRegistry",
]
