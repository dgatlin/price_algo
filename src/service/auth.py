"""Authentication utilities for the pricing service.

This module provides light-weight auth helpers that are easy to wire into
FastAPI route dependencies:

- **JWT creation & verification** using HS256 (via PyJWT).
- **IP allowlisting** (both direct helper and FastAPI `Request` variant).
- **`get_current_user` dependency** that validates a Bearer token when
  `SERVICE_ENABLE_AUTH=true` and returns the `user_id` claim; otherwise
  it resolves to `"anonymous"`.

Security notes
--------------
- Store the JWT secret outside of source control (e.g., env var).
- Set short expirations and rotate secrets regularly.
- Always use HTTPS in production so Authorization headers are protected.
- Consider audience/issuer claims for multi-service deployments.
"""

import jwt
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from pricing_rf.config import Config

# Initialize security scheme for FastAPI dependency injection
security = HTTPBearer()


def create_jwt_token(user_id: str, secret_key: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT for the given `user_id`.

    Parameters
    ----------
    user_id : str
        Subject identifier to embed in the token.
    secret_key : str
        HS256 secret used to sign the token.
    expires_delta : Optional[timedelta]
        Optional TTL. Defaults to 24 hours if not provided.

    Returns
    -------
    str
        Encoded JWT string (compact JWS).

    Claims
    ------
    - `user_id`: provided subject identifier.
    - `exp`: UTC expiration (`datetime.utcnow() + expires_delta`).

    Notes
    -----
    - Uses HS256; for asymmetric keys consider RS256/ES256.
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    payload = {
        "user_id": user_id,
        "exp": expire
    }
    
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return token


def verify_jwt_token(token: str, secret_key: str) -> Optional[str]:
    """Validate a JWT and extract the `user_id` claim.

    Parameters
    ----------
    token : str
        Bearer token string (without the 'Bearer ' prefix).
    secret_key : str
        HS256 secret used for verification.

    Returns
    -------
    Optional[str]
        The `user_id` if verification succeeds; otherwise `None`.

    Logs
    ----
    - Warns on expired or invalid tokens; does not raise.
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        user_id = payload.get("user_id")
        return user_id
    except jwt.ExpiredSignatureError:
        logging.warning("JWT token has expired")
        return None
    except jwt.InvalidTokenError:
        logging.warning("Invalid JWT token")
        return None


def verify_ip_address(client_ip: str, allowed_ips: List[str]) -> bool:
    """Check whether `client_ip` is included in `allowed_ips`.

    Parameters
    ----------
    client_ip : str
        Requesting client's IP address.
    allowed_ips : List[str]
        Allowlist of IP addresses. If empty, no restriction is applied.

    Returns
    -------
    bool
        True if allowed (or if list is empty), else False.

    Notes
    -----
    - This is a pure utility and does not raise; use the wrappers below to
      enforce policy via HTTP exceptions.
    """
    if not allowed_ips:
        return True  # No IP restrictions
    
    return client_ip in allowed_ips


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """FastAPI dependency: return the current user or raise 401.

    Behavior
    --------
    - If `SERVICE_ENABLE_AUTH=false`, returns `"anonymous"`.
    - Otherwise, validates the Bearer token using `SERVICE_JWT_SECRET` and
      returns its `user_id` claim. Raises 401 on failure.

    Parameters
    ----------
    credentials : HTTPAuthorizationCredentials
        Injected by FastAPI's `HTTPBearer` security scheme.

    Returns
    -------
    str
        The `user_id` from the verified token (or `"anonymous"` if auth is disabled).

    Raises
    ------
    HTTPException
        500 if the JWT secret is not configured.
        401 if the token is invalid or expired.

    Client usage
    ------------
    Send header: `Authorization: Bearer <token>`
    """
    config = Config()
    
    if not config.service.enable_auth:
        return "anonymous"
    
    if not config.service.jwt_secret:
        raise HTTPException(status_code=500, detail="JWT secret not configured")
    
    token = credentials.credentials
    user_id = verify_jwt_token(token, config.service.jwt_secret)
    
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_id


def verify_ip_address_from_request(request: Request) -> bool:
    """FastAPI helper: enforce IP allowlist using the incoming `Request`.

    Parameters
    ----------
    request : fastapi.Request
        The inbound HTTP request.

    Returns
    -------
    bool
        True if allowed or if auth is disabled.

    Raises
    ------
    HTTPException
        403 if the client IP is not on the allowlist (when enabled).
    """
    config = Config()
    
    if not config.service.enable_auth:
        return True
    
    # Get client IP (consider X-Forwarded-For if behind a proxy/ELB)
    client_ip = request.client.host
    
    # Check if IP is allowed
    if not verify_ip_address(client_ip, config.service.allowed_ips):
        raise HTTPException(status_code=403, detail="IP address not allowed")
    
    return True


def verify_ip_address(client_ip: str) -> bool:
    """Convenience wrapper: enforce IP allowlist using app `Config`.

    Parameters
    ----------
    client_ip : str
        Requesting client's IP address.

    Returns
    -------
    bool
        True if allowed or if auth is disabled.

    Raises
    ------
    HTTPException
        403 if the client IP is not on the allowlist (when enabled).

    Notes
    -----
    - This wrapper reads `SERVICE_ENABLE_AUTH` and `SERVICE_ALLOWED_IPS` from
      the environment via `Config()`.
    """
    config = Config()
    
    if not config.service.enable_auth:
        return True
    
    if not verify_ip_address(client_ip, config.service.allowed_ips):
        raise HTTPException(status_code=403, detail="IP address not allowed")
    
    return True


