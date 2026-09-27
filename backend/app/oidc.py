"""Single sign-on: validate OIDC access/ID tokens and map them to principals.

Works with any OpenID Connect provider that publishes a discovery document
and a JWKS: Microsoft Entra ID, ProConnect, Keycloak, and so on. Nothing here
is provider-specific; the provider differences are all configuration:

- Microsoft Entra ID: ``OIDC_ISSUER=https://login.microsoftonline.com/<tid>/v2.0``,
  groups in ``groups`` (object ids) or app roles in ``roles``, tenant in ``tid``.
- ProConnect: the issuer of your ProConnect environment; the audience is your
  client id.
- Keycloak: ``OIDC_ISSUER=https://<host>/realms/<realm>``, roles in
  ``realm_access.roles``.

A token is accepted only if its signature verifies against a key from the
issuer's JWKS and ``iss``, ``aud``, ``exp`` (and ``nbf``, when present) all
check out. Only asymmetric algorithms are accepted, so a token cannot be
"signed" with the public key as an HMAC secret.

The discovery document and JWKS are fetched synchronously (the auth
dependency runs in FastAPI's threadpool) and cached for
``OIDC_JWKS_TTL_SECONDS``. A token signed with a key id the cache does not
know triggers an early refresh. Fetch attempts are spaced at least a minute
apart, so neither forged key ids nor a provider outage turn every request into
an outbound call.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections.abc import Mapping
from typing import Any

import jwt
from fastapi import HTTPException

from app.access import Principal, normalize_groups
from app.config import settings
from app.logging_config import get_structured_logger

logger = get_structured_logger(__name__)

_FETCH_TIMEOUT = 5.0
# Minimum spacing between refreshes forced by an unknown key id.
_MIN_REFRESH_SECONDS = 60.0
# Accepted clock skew on exp/nbf/iat.
_LEEWAY_SECONDS = 30


class OIDCError(Exception):
    """The token was rejected. The message is safe to return to the client."""


def looks_like_jwt(token: str) -> bool:
    """A compact JWS has three base64url segments; API keys have none."""
    return token.count(".") == 2 and not token.startswith("sk_")


def _fetch_json(url: str) -> dict[str, Any]:
    """GET a JSON document. Patched in tests."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:  # noqa: S310 - https issuer URL
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return data


class JWKSCache:
    """The issuer's signing keys, fetched via discovery and cached with a TTL."""

    def __init__(self, issuer: str, ttl_seconds: float) -> None:
        self.issuer = issuer
        self.ttl = ttl_seconds
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at = 0.0
        self._last_attempt = float("-inf")
        self._lock = threading.Lock()

    def _refresh(self) -> None:
        discovery_url = self.issuer.rstrip("/") + "/.well-known/openid-configuration"
        discovery = _fetch_json(discovery_url)
        jwks_uri = discovery.get("jwks_uri")
        if not isinstance(jwks_uri, str) or not jwks_uri:
            raise ValueError("The OIDC discovery document has no jwks_uri")
        jwks = _fetch_json(jwks_uri)

        keys: dict[str, jwt.PyJWK] = {}
        for raw in jwks.get("keys", []):
            if raw.get("use", "sig") != "sig" or not raw.get("kid"):
                continue
            try:
                keys[raw["kid"]] = jwt.PyJWK(raw)
            except jwt.PyJWTError as e:
                # One unusable key (say, an unsupported curve) must not take
                # the others down with it.
                logger.warning(f"Skipping JWKS key {raw.get('kid')}: {e}")
        self._keys = keys
        self._fetched_at = time.monotonic()
        logger.info("OIDC signing keys refreshed", extra_fields={"keys": len(keys)})

    def get(self, kid: str) -> jwt.PyJWK:
        """Return the signing key ``kid``.

        Raises:
            OIDCError: If the key is unknown even after a refresh, or the
                issuer's keys cannot be fetched.
        """
        with self._lock:
            now = time.monotonic()
            stale = not self._keys or now - self._fetched_at > self.ttl
            # An unknown kid is most likely a key rotation: look once more
            # before rejecting. Attempts are spaced out either way, so neither
            # forged kids nor an outage turn every request into a fetch.
            if (stale or kid not in self._keys) and (
                now - self._last_attempt >= _MIN_REFRESH_SECONDS
            ):
                self._last_attempt = now
                try:
                    self._refresh()
                except Exception as e:
                    # Keep serving the keys we have; they are usually still valid.
                    logger.error(
                        f"Could not fetch OIDC signing keys: {type(e).__name__}: {e}"
                    )
            if not self._keys:
                raise OIDCError("The identity provider's signing keys are unavailable.")
            key = self._keys.get(kid)
        if key is None:
            raise OIDCError("Token is signed with an unknown key.")
        return key


_cache: JWKSCache | None = None
_cache_lock = threading.Lock()


def _jwks() -> JWKSCache:
    """The process-wide key cache, rebuilt if the issuer setting changed."""
    global _cache
    with _cache_lock:
        if (
            _cache is None
            or _cache.issuer != settings.oidc_issuer
            or _cache.ttl != settings.oidc_jwks_ttl_seconds
        ):
            _cache = JWKSCache(settings.oidc_issuer, settings.oidc_jwks_ttl_seconds)
        return _cache


def reset_cache() -> None:
    """Forget cached keys. For tests and issuer changes."""
    global _cache
    with _cache_lock:
        _cache = None


def _claim(claims: Mapping[str, Any], path: str) -> Any:
    """Read a claim by dotted path, e.g. ``realm_access.roles``."""
    value: Any = claims
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def extract_groups(claims: Mapping[str, Any]) -> tuple[str, ...]:
    """The groups named by ``OIDC_GROUPS_CLAIM``, as a clean tuple.

    Accepts a list, or a space/comma separated string (some providers flatten
    it). Entra ID replaces ``groups`` with ``_claim_names`` when a user is in
    too many groups to fit the token; that user gets no groups and a warning.
    """
    raw = _claim(claims, settings.oidc_groups_claim)
    if raw is None:
        if "groups" in (claims.get("_claim_names") or {}):
            logger.warning(
                "Token has a groups overage claim; the user's groups are not in the "
                "token. Use app roles (OIDC_GROUPS_CLAIM=roles) or group filtering.",
                extra_fields={"sub": claims.get("sub")},
            )
        return ()
    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()
    if not isinstance(raw, list):
        return ()
    try:
        return normalize_groups(raw)
    except ValueError as e:
        raise OIDCError(f"Token groups rejected: {e}") from e


def principal_from_claims(claims: Mapping[str, Any]) -> Principal:
    """Map validated token claims to a principal.

    Raises:
        OIDCError: If ``sub`` is missing, or ``OIDC_TENANT_CLAIM`` is configured
            and the token lacks it (falling back to the default tenant would
            silently put the user in someone else's tenant).
    """
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise OIDCError("Token has no subject.")

    tenant = settings.default_tenant
    if settings.oidc_tenant_claim:
        value = _claim(claims, settings.oidc_tenant_claim)
        if not isinstance(value, str) or not value.strip():
            raise OIDCError(f"Token has no '{settings.oidc_tenant_claim}' claim.")
        tenant = value.strip()

    groups = extract_groups(claims)
    email = claims.get("email") if isinstance(claims.get("email"), str) else None
    name = next(
        (
            claims[k]
            for k in ("name", "preferred_username", "email")
            if isinstance(claims.get(k), str) and claims[k]
        ),
        sub,
    )
    return Principal(
        id=f"oidc:{sub}",
        kind="oidc_user",
        display_name=name,
        email=email,
        tenant=tenant,
        groups=groups,
        is_admin=bool(settings.oidc_admin_group) and settings.oidc_admin_group in groups,
    )


def verify_token(token: str) -> dict[str, Any]:
    """Validate a JWT against the configured issuer and return its claims.

    Raises:
        OIDCError: For any validation failure.
    """
    algorithms = settings.oidc_algorithm_list
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as e:
        raise OIDCError("Malformed token.") from e

    alg = header.get("alg")
    if alg not in algorithms:
        raise OIDCError(f"Token algorithm '{alg}' is not accepted.")
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise OIDCError("Token header has no key id.")

    key = _jwks().get(kid)
    try:
        return jwt.decode(
            token,
            key=key.key,
            algorithms=[alg],
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer,
            leeway=_LEEWAY_SECONDS,
            options={"require": ["exp", "iss", "aud", "sub"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise OIDCError("Token has expired.") from e
    except jwt.ImmatureSignatureError as e:
        raise OIDCError("Token is not valid yet.") from e
    except jwt.InvalidAudienceError as e:
        raise OIDCError("Token audience is not accepted.") from e
    except jwt.InvalidIssuerError as e:
        raise OIDCError("Token issuer is not accepted.") from e
    except jwt.PyJWTError as e:
        raise OIDCError(f"Invalid token ({type(e).__name__}).") from e


def authenticate_oidc(token: str) -> Principal:
    """Validate a bearer JWT and return its principal.

    Raises:
        HTTPException: 401 when OIDC is not configured or the token is rejected.
    """
    if not settings.oidc_issuer:
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        return principal_from_claims(verify_token(token))
    except OIDCError as e:
        logger.info(f"OIDC token rejected: {e}")
        raise HTTPException(
            status_code=401, detail=str(e), headers={"WWW-Authenticate": "Bearer"}
        ) from e


__all__ = [
    "JWKSCache",
    "OIDCError",
    "authenticate_oidc",
    "extract_groups",
    "looks_like_jwt",
    "principal_from_claims",
    "reset_cache",
    "verify_token",
]
