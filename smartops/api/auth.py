"""Optional API-key authentication with user/admin RBAC."""

from __future__ import annotations

import hmac
from enum import Enum

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from smartops.api.deps import AppState, get_state

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer = HTTPBearer(auto_error=False)


class Role(str, Enum):
    user = "user"
    admin = "admin"


def _extract_api_key(
    api_key_header: str | None,
    bearer: HTTPAuthorizationCredentials | None,
) -> str | None:
    if api_key_header and api_key_header.strip():
        return api_key_header.strip()
    if bearer and bearer.credentials:
        return bearer.credentials.strip()
    return None


def _matches(provided: str, expected: str) -> bool:
    return bool(expected) and len(provided) == len(expected) and hmac.compare_digest(provided, expected)


def _resolve_role(state: AppState, provided: str | None) -> Role | None:
    settings = state.settings
    user_key = (settings.api_key or "").strip()
    admin_key = (settings.admin_api_key or "").strip()

    # Auth disabled for local/CI
    if not user_key and not admin_key:
        return Role.admin

    if not provided:
        return None

    # Admin key wins when configured
    if admin_key and _matches(provided, admin_key):
        return Role.admin
    if user_key and _matches(provided, user_key):
        # If admin key unset, user key is also admin (single-key demos)
        return Role.admin if not admin_key else Role.user
    return None


async def require_user(
    request: Request,
    state: AppState = Depends(get_state),
    api_key_header: str | None = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> Role:
    """Allow user or admin keys when auth is enabled."""
    provided = _extract_api_key(api_key_header, bearer)
    role = _resolve_role(state, provided)
    if role is None:
        # Auth disabled path already returned admin above
        user_key = (state.settings.api_key or "").strip()
        admin_key = (state.settings.admin_api_key or "").strip()
        if not user_key and not admin_key:
            request.state.role = Role.admin
            return Role.admin
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass X-API-Key or Authorization: Bearer <key>.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.role = role
    request.state.authenticated = True
    return role


async def require_admin(
    request: Request,
    state: AppState = Depends(get_state),
    api_key_header: str | None = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> Role:
    """Admin-only endpoints (/rl/state, /metrics)."""
    role = await require_user(request, state, api_key_header, bearer)
    if role != Role.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin API key required for this endpoint.",
        )
    return role
