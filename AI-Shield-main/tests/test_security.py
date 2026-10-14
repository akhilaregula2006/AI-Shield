import pytest
from fastapi import HTTPException
from jose import jwt

from configs.settings import settings
from src.security import auth
from src.security.rate_limit import SlidingWindowLimiter


def test_password_hash_roundtrip_and_salting():
    h1, h2 = auth.hash_password("s3cret"), auth.hash_password("s3cret")
    assert h1 != h2
    assert auth.verify_password("s3cret", h1) and not auth.verify_password("wrong", h1)


def test_authenticate_roles():
    assert auth.authenticate(settings.admin_username, settings.admin_password) == "admin"
    assert auth.authenticate(settings.analyst_username, settings.analyst_password) == "analyst"
    assert auth.authenticate("admin", "nope") is None
    assert auth.authenticate("ghost", "x") is None


def test_token_contains_claims():
    token = auth.create_access_token("alice", "admin")
    claims = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    assert claims["sub"] == "alice" and claims["role"] == "admin"


def test_require_role_blocks_wrong_role():
    dep = auth.require_role("admin")
    with pytest.raises(HTTPException) as exc:
        dep(user={"username": "a", "role": "analyst"})
    assert exc.value.status_code == 403


def test_prod_refuses_default_secrets(monkeypatch):
    monkeypatch.setattr(settings, "environment", "prod")
    with pytest.raises(RuntimeError):
        auth.assert_secure_config()
    monkeypatch.setattr(settings, "jwt_secret_key", "x" * 40)
    monkeypatch.setattr(settings, "admin_password", "strong-admin-pass")
    monkeypatch.setattr(settings, "analyst_password", "strong-analyst-pass")
    auth.assert_secure_config()


def test_sliding_window_limiter():
    limiter = SlidingWindowLimiter(limit=2, window_seconds=10)
    assert limiter.allow("ip", now=0)[0] and limiter.allow("ip", now=1)[0]
    allowed, retry = limiter.allow("ip", now=2)
    assert not allowed and retry >= 1
    assert limiter.allow("other", now=2)[0]  # per-key isolation
    assert limiter.allow("ip", now=11)[0]  # window slides
