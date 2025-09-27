"""
Decorators for registering embedding models and providers.
"""

import logging

# Removed ProviderType import - using config-driven provider access


logger = logging.getLogger(__name__)

# Mapping from a implementation name (if specified) to provider class
# This might need expansion if more embedding providers are added



# TODO: Add @embedding_provider decorator if needed for direct class registration