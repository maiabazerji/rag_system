"""SSO via OIDC: token validation against a locally generated RSA key and a mocked JWKS.

Nothing here touches the network: ``app.oidc._fetch_json`` serves the
discovery document and the key set from memory, and counts the fetches.
"""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from app import auth, oidc
from app.access import PUBLIC_GROUP, AccessScope

ISSUER = "https://login.example.test/tenant-1/v2.0"
AUDIENCE = "api://evalrag"
JWKS_URI = "https://login.example.test/tenant-1/discovery/v2.0/keys"


def _rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


KEY_A = _rsa_key()
KEY_B = _rsa_key()


def _jwk(private_key, kid: str) -> dict:
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    return {**jwk, "kid": kid, "use": "sig", "alg": "RS256"}


class FakeIdP:
    """Serves discovery and JWKS documents and counts how often each is fetched."""

    def __init__(self):
        self.keys = [_jwk(KEY_A, "a")]
        self.fetches: list[str] = []
        self.down = False

    def __call__(self, url: str) -> dict:
        self.fetches.append(url)
        if self.down:
            raise OSError("identity provider unreachable")
        if url == ISSUER + "/.well-known/openid-configuration":
            return {"issuer": ISSUER, "jwks_uri": JWKS_URI}
        if url == JWKS_URI:
            return {"keys": list(self.keys)}
        raise AssertionError(f"unexpected fetch: {url}")


@pytest.fixture
def idp(settings, monkeypatch):
    monkeypatch.setattr(settings, "oidc_issuer", ISSUER)
    monkeypatch.setattr(settings, "oidc_audience", AUDIENCE)
    monkeypatch.setattr(settings, "oidc_groups_claim", "groups")
    monkeypatch.setattr(settings, "oidc_admin_group", "evalrag-admins")
    monkeypatch.setattr(settings, "oidc_tenant_claim", "")
    fake = FakeIdP()
    monkeypatch.setattr(oidc, "_fetch_json", fake)
    oidc.reset_cache()
    yield fake
    oidc.reset_cache()


def make_token(
    key=KEY_A, kid="a", alg="RS256", headers=None, **overrides
) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "user-123",
        "iat": now,
        "nbf": now,
        "exp": now + 600,
        "name": "Ada Lovelace",
        "email": "ada@example.test",
        "groups": ["hr", "engineering"],
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm=alg, headers={"kid": kid, **(headers or {})})


class TestValidTokens:
    def test_maps_claims_to_a_principal(self, idp):
        p = oidc.authenticate_oidc(make_token())
        assert p.id == "oidc:user-123"
        assert p.kind == "oidc_user"
        assert p.display_name == "Ada Lovelace"
        assert p.email == "ada@example.test"
        assert p.groups == ("engineering", "hr")
        assert not p.is_admin
        assert p.tenant == "default"
        assert p.scope() == AccessScope("default", frozenset({"hr", "engineering", PUBLIC_GROUP}))

    def test_admin_group_makes_an_admin(self, idp):
        p = oidc.authenticate_oidc(make_token(groups=["evalrag-admins"]))
        assert p.is_admin

    def test_no_admin_group_configured_means_no_admins(self, idp, settings, monkeypatch):
        monkeypatch.setattr(settings, "oidc_admin_group", "")
        assert not oidc.authenticate_oidc(make_token(groups=[""])).is_admin

    def test_nested_groups_claim(self, idp, settings, monkeypatch):
        """Keycloak puts realm roles under realm_access.roles."""
        monkeypatch.setattr(settings, "oidc_groups_claim", "realm_access.roles")
        p = oidc.authenticate_oidc(make_token(realm_access={"roles": ["finance"]}))
        assert p.groups == ("finance",)

    def test_roles_claim(self, idp, settings, monkeypatch):
        """Entra app roles arrive in `roles`."""
        monkeypatch.setattr(settings, "oidc_groups_claim", "roles")
        assert oidc.authenticate_oidc(make_token(roles=["Reader"])).groups == ("Reader",)

    def test_space_separated_groups_string(self, idp):
        assert oidc.authenticate_oidc(make_token(groups="a b,c")).groups == ("a", "b", "c")

    def test_missing_groups_claim_means_public_only(self, idp):
        p = oidc.authenticate_oidc(make_token(groups=None))
        assert p.groups == ()
        assert p.scope().groups == {PUBLIC_GROUP}

    def test_entra_group_overage_yields_no_groups(self, idp, caplog):
        token = make_token(groups=None, _claim_names={"groups": "src1"})
        assert oidc.authenticate_oidc(token).groups == ()
        assert "overage" in caplog.text

    def test_tenant_claim(self, idp, settings, monkeypatch):
        monkeypatch.setattr(settings, "oidc_tenant_claim", "tid")
        assert oidc.authenticate_oidc(make_token(tid="contoso")).tenant == "contoso"

    def test_display_name_falls_back(self, idp):
        p = oidc.authenticate_oidc(make_token(name=None, preferred_username="ada"))
        assert p.display_name == "ada"

    def test_ec_keys_work_too(self, idp):
        from cryptography.hazmat.primitives.asymmetric import ec

        ec_key = ec.generate_private_key(ec.SECP256R1())
        jwk = json.loads(jwt.algorithms.ECAlgorithm.to_jwk(ec_key.public_key()))
        idp.keys.append({**jwk, "kid": "ec", "use": "sig"})
        p = oidc.authenticate_oidc(make_token(key=ec_key, kid="ec", alg="ES256"))
        assert p.id == "oidc:user-123"


def _rejected(token: str) -> str:
    with pytest.raises(HTTPException) as exc:
        oidc.authenticate_oidc(token)
    assert exc.value.status_code == 401
    return exc.value.detail


class TestRejectedTokens:
    def test_expired(self, idp):
        assert "expired" in _rejected(make_token(exp=int(time.time()) - 3600))

    def test_not_yet_valid(self, idp):
        assert "not valid yet" in _rejected(make_token(nbf=int(time.time()) + 3600))

    def test_wrong_audience(self, idp):
        assert "audience" in _rejected(make_token(aud="api://someone-else"))

    def test_wrong_issuer(self, idp):
        assert "issuer" in _rejected(make_token(iss="https://evil.example.test"))

    @pytest.mark.parametrize("claim", ["exp", "aud", "iss", "sub"])
    def test_required_claims(self, idp, claim):
        _rejected(make_token(**{claim: None}))

    def test_signed_by_another_key_with_a_known_kid(self, idp):
        assert "Invalid token" in _rejected(make_token(key=KEY_B, kid="a"))

    def test_unknown_kid(self, idp):
        assert "unknown key" in _rejected(make_token(key=KEY_B, kid="b"))

    def test_missing_kid(self, idp):
        token = jwt.encode(
            {"sub": "x", "iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 60},
            KEY_A,
            algorithm="RS256",
        )
        assert "key id" in _rejected(token)

    def test_hmac_with_the_public_key_is_refused(self, idp):
        """The classic algorithm-confusion attack: HS256 keyed with the public key."""
        public_pem = KEY_A.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        claims = {"sub": "x", "iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 60}
        assert "algorithm" in _rejected(_hs256_token(claims, public_pem))

    def test_alg_none_is_refused(self, idp):
        token = jwt.encode(
            {"sub": "x", "iss": ISSUER, "aud": AUDIENCE}, None, algorithm="none",
            headers={"kid": "a"},
        )
        assert "algorithm" in _rejected(token)

    def test_garbage(self, idp):
        assert "Malformed" in _rejected("not.a.jwt")

    def test_tenant_claim_required_when_configured(self, idp, settings, monkeypatch):
        monkeypatch.setattr(settings, "oidc_tenant_claim", "tid")
        assert "tid" in _rejected(make_token())

    def test_oidc_not_configured(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "oidc_issuer", "")
        _rejected(make_token())


def _hs256_token(claims: dict, secret: bytes) -> str:
    """Hand-roll an HS256 JWS: PyJWT itself refuses a PEM public key as an HMAC secret."""
    import base64
    import hashlib
    import hmac

    def b64(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    header = b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": "a"}).encode())
    payload = b64(json.dumps(claims).encode())
    sig = hmac.new(secret, f"{header}.{payload}".encode(), hashlib.sha256).digest()
    return f"{header}.{payload}.{b64(sig)}"


class TestKeyCache:
    def test_keys_are_fetched_once_and_cached(self, idp):
        oidc.authenticate_oidc(make_token())
        oidc.authenticate_oidc(make_token(sub="someone-else"))
        assert idp.fetches.count(JWKS_URI) == 1

    def test_ttl_expiry_refetches(self, idp, monkeypatch):
        oidc.authenticate_oidc(make_token())
        cache = oidc._jwks()
        monkeypatch.setattr(cache, "_fetched_at", cache._fetched_at - 10_000)
        monkeypatch.setattr(cache, "_last_attempt", cache._last_attempt - 10_000)
        oidc.authenticate_oidc(make_token())
        assert idp.fetches.count(JWKS_URI) == 2

    def test_key_rotation_is_picked_up(self, idp, monkeypatch):
        monkeypatch.setattr(oidc, "_MIN_REFRESH_SECONDS", 0.0)
        oidc.authenticate_oidc(make_token())
        idp.keys = [_jwk(KEY_B, "b")]
        p = oidc.authenticate_oidc(make_token(key=KEY_B, kid="b"))
        assert p.id == "oidc:user-123"

    def test_unknown_kids_do_not_hammer_the_idp(self, idp):
        oidc.authenticate_oidc(make_token())
        for _ in range(5):
            _rejected(make_token(key=KEY_B, kid=f"forged-{_}"))
        assert idp.fetches.count(JWKS_URI) == 1

    def test_outage_before_first_fetch_is_a_401(self, idp):
        idp.down = True
        assert "unavailable" in _rejected(make_token())

    def test_outage_after_first_fetch_keeps_serving_cached_keys(self, idp, monkeypatch):
        oidc.authenticate_oidc(make_token())
        idp.down = True
        cache = oidc._jwks()
        monkeypatch.setattr(cache, "_fetched_at", cache._fetched_at - 10_000)
        monkeypatch.setattr(cache, "_last_attempt", cache._last_attempt - 10_000)
        assert oidc.authenticate_oidc(make_token()).id == "oidc:user-123"

    def test_issuer_change_resets_the_cache(self, idp, settings, monkeypatch):
        first = oidc._jwks()
        monkeypatch.setattr(settings, "oidc_issuer", ISSUER + "/other")
        assert oidc._jwks() is not first

    def test_keys_for_encryption_or_unusable_are_skipped(self, idp):
        idp.keys.append({**_jwk(KEY_B, "enc"), "use": "enc"})
        idp.keys.append({"kty": "RSA", "kid": "broken", "n": "!!", "e": "AQAB"})
        oidc.authenticate_oidc(make_token())
        assert set(oidc._jwks()._keys) == {"a"}


class TestConfig:
    def test_symmetric_algorithms_are_never_accepted(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "oidc_algorithms", "RS256, HS256,none,ES384,EdDSA")
        assert settings.oidc_algorithm_list == ["RS256", "ES384", "EdDSA"]

    def test_issuer_without_audience_fails_startup(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "oidc_issuer", ISSUER)
        monkeypatch.setattr(settings, "oidc_audience", "")
        with pytest.raises(ValueError, match="OIDC_AUDIENCE"):
            settings.validate_startup()

    def test_looks_like_jwt(self):
        assert oidc.looks_like_jwt(make_token())
        assert not oidc.looks_like_jwt("sk_abc.def.ghi")
        assert not oidc.looks_like_jwt("sk_abcdef")


class TestOverHTTP:
    @pytest.fixture
    def enforced(self, idp, auth_db, settings, monkeypatch):
        monkeypatch.setattr(settings, "require_api_key", True)
        return idp

    def test_bearer_jwt_is_accepted(self, client, enforced):
        mock = AsyncMock(return_value=7)
        with patch("app.api.ingest.store_count", new=mock):
            r = client.get(
                "/ingest/stats", headers={"Authorization": f"Bearer {make_token()}"}
            )
        assert r.status_code == 200
        assert mock.await_args.args[0].groups == {"hr", "engineering", PUBLIC_GROUP}

    def test_api_keys_still_work_alongside(self, client, enforced):
        key = auth.create_api_key("ci")
        with patch("app.api.ingest.store_count", new=AsyncMock(return_value=0)):
            r = client.get("/ingest/stats", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200

    def test_bad_jwt_is_401(self, client, enforced):
        r = client.get(
            "/ingest/stats",
            headers={"Authorization": f"Bearer {make_token(aud='api://other')}"},
        )
        assert r.status_code == 401
        assert r.headers["WWW-Authenticate"] == "Bearer"

    def test_oidc_users_are_not_metered(self, client, enforced):
        with patch("app.api.ingest.store_count", new=AsyncMock(return_value=0)):
            client.get("/ingest/stats", headers={"Authorization": f"Bearer {make_token()}"})
        with auth.session_scope() as db:
            assert db.query(auth.APIKeyUsage).count() == 0

    def test_health_advertises_the_issuer(self, client, enforced):
        assert client.get("/health").json()["oidc_issuer"] == ISSUER

    def test_health_hides_the_issuer_when_auth_is_off(self, client, idp):
        assert client.get("/health").json()["oidc_issuer"] is None

    def test_jwt_ignored_when_auth_is_off(self, client, idp):
        """Local mode is unchanged: every caller is the local principal."""
        with patch("app.api.ingest.store_count", new=AsyncMock(return_value=0)) as mock:
            client.get("/ingest/stats", headers={"Authorization": "Bearer garbage"})
        assert mock.await_args.args[0].groups == {PUBLIC_GROUP}
        assert idp.fetches == []
