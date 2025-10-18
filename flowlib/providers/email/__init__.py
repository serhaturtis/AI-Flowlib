"""Email provider implementations for email operations.

This package provides email provider functionality including:
- IMAP/SMTP provider for standard email protocols
- Base email provider interface
- Email message models
"""

from .base import EmailMessage, EmailProvider, EmailProviderSettings

__all__ = [
    "EmailMessage",
    "EmailProvider",
    "EmailProviderSettings",
]
