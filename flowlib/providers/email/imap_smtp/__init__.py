"""IMAP/SMTP email provider implementation.

This module provides email functionality using standard IMAP and SMTP protocols.
"""

from .provider import IMAPSMTPProvider, IMAPSMTPSettings

__all__ = [
    "IMAPSMTPProvider",
    "IMAPSMTPSettings",
]
