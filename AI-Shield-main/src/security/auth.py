"""JWT authentication with role-based access (admin / analyst). Stdlib PBKDF2 password hashing."""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Callable, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from configs.settings import settings

_DEFAULT_SECRET = "change-me-in-.env"
_ITERATIONS = 200_000
_bearer = HTTPBearer(auto_error=False)


def assert_secure_config() -> None:
    """Refuse to start in prod with default secrets."""
    if settings.environment == "prod" and (
        settings.jwt_secret_key in (_DEFAULT_SECRET, "change-me")
        or settings.admin_password.startswith("change-me")
        or settings.analyst_password.startswith("change-me")
    ):
        raise RuntimeError("Insecure default credentials/secret in prod; set them via env vars")


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, digest_hex = stored.split("$", 1)
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), _ITERATIONS
    )
    return hmac.compare_digest(candidate.hex(), digest_hex)


@lru_cache(maxsize=1)
def _users() -> Dict[str, Dict[str, str]]:
    return {
        settings.admin_username: {"hash": hash_password(settings.admin_password), "role": "admin"},
        settings.analyst_username: {
            "hash": hash_password(settings.analyst_password),
            "role": "analyst",
        },
    }


def authenticate(username: str, password: str) -> Optional[str]:
    """Return the user's role on success, else None (always does a hash check)."""
    user = _users().get(username)
    dummy = "00" * 16 + "$" + "00" * 32
    ok = verify_password(password, user["hash"] if user else dummy)
    return user["role"] if user and ok else None


def create_access_token(subject: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Dict[str, str]:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        "Invalid or missing token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if creds is None:
        raise unauthorized
    try:
        payload = jwt.decode(
            creds.credentials, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except JWTError as exc:
        raise unauthorized from exc
    return {"username": payload["sub"], "role": payload["role"]}


def require_role(role: str) -> Callable:
    def dependency(user: Dict[str, str] = Depends(get_current_user)) -> Dict[str, str]:
        if user["role"] != role:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires role '{role}'")
        return user

    return dependency
