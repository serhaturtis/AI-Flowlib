"""JanusGraph database provider module."""

# Import availability and components
try:
    from .provider import JanusProvider
    from .models import JanusProviderSettings
    JANUS_AVAILABLE = True
    
    # Re-export gremlin components for test compatibility
    try:
        from gremlin_python.driver.client import Client
        from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
        from gremlin_python.process.anonymous_traversal import traversal
        from gremlin_python.driver.serializer import GraphBinaryMessageSerializer
    except ImportError:
        Client = None
        DriverRemoteConnection = None
        traversal = None
        GraphBinaryMessageSerializer = None
        
except ImportError:
    JanusProvider = None
    JanusProviderSettings = None
    JANUS_AVAILABLE = False
    Client = None
    DriverRemoteConnection = None
    traversal = None
    GraphBinaryMessageSerializer = None

__all__ = [
    "JanusProvider",
    "JanusProviderSettings", 
    "JANUS_AVAILABLE",
    "Client",
    "DriverRemoteConnection",
    "traversal", 
    "GraphBinaryMessageSerializer",
]