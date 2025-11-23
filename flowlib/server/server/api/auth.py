"""Authentication endpoints.

NOTE: All endpoints in this module are placeholders and not yet implemented.
JWT authentication and token management will be added in a future release.
"""

from fastapi import APIRouter

router = APIRouter()


@router.post("/login")
async def login() -> dict[str, str]:
    """Login endpoint (NOT IMPLEMENTED).

    This is a placeholder. JWT authentication will be implemented in a future release.
    """
    return {"message": "Login endpoint - not implemented"}


@router.post("/logout")
async def logout() -> dict[str, str]:
    """Logout endpoint (NOT IMPLEMENTED).

    This is a placeholder. Session invalidation will be implemented in a future release.
    """
    return {"message": "Logout endpoint - not implemented"}


@router.post("/token/refresh")
async def refresh_token() -> dict[str, str]:
    """Refresh access token (NOT IMPLEMENTED).

    This is a placeholder. Token refresh mechanism will be implemented in a future release.
    """
    return {"message": "Token refresh endpoint - not implemented"}

