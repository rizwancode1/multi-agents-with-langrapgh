"""
Optional API-key authentication + per-client rate-limit keying.

When API_KEY is configured (env), mutating endpoints require an
``X-API-Key`` header matching it. When unset (local dev), everything stays
open so the demo works out of the box.
"""

from fastapi import Header, HTTPException
from slowapi.util import get_remote_address

from app.config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _rate_limit_key(request) -> str:
    """Per-client rate limiting: API key when present, otherwise client IP."""
    return (
        f"key:{request.headers.get('x-api-key')}"
        if request.headers.get("x-api-key")
        else get_remote_address(request)
    )
