"""Global lock for resource registry access.

The resource_registry from flowlib is a global singleton that is not thread-safe.
This module provides a shared lock to coordinate access across all services.

Usage:
    from server.core.registry_lock import registry_lock

    with registry_lock:
        resource_registry.clear()
        # ... other registry operations ...
"""

from __future__ import annotations

import threading

# Global lock for coordinating resource_registry access across all services
# This prevents race conditions when multiple services clear/load the registry
registry_lock = threading.Lock()
