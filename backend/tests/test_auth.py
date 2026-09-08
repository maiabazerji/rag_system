"""Tests for API key handling.

Two properties matter most and are both regressions:
  - keys are stored hashed, never in plaintext;
  - a deactivated key stops working immediately, not after a restart (the old
    code memoised validation with `lru_cache`).
"""
import hashlib

import pytest
from fastapi import HTTPException

from app import auth
from app.auth import _parse_bearer, hash_key, require_admin_key


class TestHashKey:
    def test_is_sha256_hex(self):
        assert hash_key("sk_abc") == hashlib.sha256(b"sk_abc").hexdigest()

    def test_is_stable(self):
        assert hash_key("sk_abc") == hash_key("sk_abc")

    def test_differs_per_key(self):
        assert hash_key("sk_abc") != hash_key("sk_abd")

    def test_does_not_contain_the_key(self):
        assert "sk_supersecret" not in hash_key("sk_supersecret")


class TestParseBearer:
    def test_extracts_the_token(self):
        assert _parse_bearer("Bearer sk_abc") == "sk_abc"

    def test_scheme_is_case_insensitive(self):
        assert _parse_bearer("bearer sk_abc") == "sk_abc"

    @pytest.mark.parametrize(
        "header", [None, "", "sk_abc", "Basic sk_abc", "Bearer", "Bearer    "]
    )
    def test_rejects_bad_headers(self, header):
        with pytest.raises(HTTPException) as exc:
            _parse_bearer(header)
        assert exc.value.status_code == 401


class TestRequireAdminKey:
    def test_503_when_unset(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "admin_key", "")
        with pytest.raises(HTTPException) as exc:
            require_admin_key("anything")
        assert exc.value.status_code == 503

    def test_403_on_mismatch(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "admin_key", "right")
        with pytest.raises(HTTPException) as exc:
            require_admin_key("wrong")
        assert exc.value.status_code == 403

    def test_403_when_header_missing(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "admin_key", "right")
        with pytest.raises(HTTPException) as exc:
            require_admin_key(None)
        assert exc.value.status_code == 403

    def test_accepts_the_configured_key(self, settings, monkeypatch):
        monkeypatch.setattr(settings, "admin_key", "right")
        assert require_admin_key("right") is True


@pytest.fixture
def sqlite_auth(tmp_path, monkeypatch):
    """Point the auth module at a throwaway SQLite database."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{tmp_path / 'auth.db'}")
    auth.Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(auth, "_engine", engine)
    monkeypatch.setattr(
        auth, "_SessionLocal", sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )
    yield
    auth.Base.metadata.drop_all(bind=engine)


class TestKeyLifecycle:
    def test_created_key_is_stored_only_as_a_hash(self, sqlite_auth):
        """The plaintext key must never reach the database."""
        key = auth.create_api_key("ci")

        with auth.session_scope() as db:
            rows = db.query(auth.APIKey).all()
            assert len(rows) == 1
            assert rows[0].key_hash == hash_key(key)
            assert not hasattr(rows[0], "key")
            assert key not in rows[0].key_hash

    def test_created_key_has_the_expected_shape(self, sqlite_auth):
        key = auth.create_api_key("ci")
        assert key.startswith("sk_")
        assert len(key) > 32

    def test_keys_are_unique(self, sqlite_auth):
        assert auth.create_api_key("a") != auth.create_api_key("b")

    def test_authenticate_accepts_a_valid_key(self, sqlite_auth):
        key = auth.create_api_key("ci", requests_per_minute=5)
        with auth.session_scope() as db:
            row = auth._authenticate(db, key)
            assert row.name == "ci"
            assert row.requests_per_minute == 5

    def test_authenticate_rejects_an_unknown_key(self, sqlite_auth):
        auth.create_api_key("ci")
        with auth.session_scope() as db:
            with pytest.raises(HTTPException) as exc:
                auth._authenticate(db, "sk_not_a_real_key")
            assert exc.value.status_code == 401

    def test_deactivation_takes_effect_immediately(self, sqlite_auth):
        """The lru_cache regression: revocation used to need a restart."""
        key = auth.create_api_key("ci")

        with auth.session_scope() as db:
            auth._authenticate(db, key)  # populate any cache that might exist

        with auth.session_scope() as db:
            key_id = db.query(auth.APIKey).one().id
        assert auth.deactivate_api_key(key_id) is True

        with auth.session_scope() as db:
            with pytest.raises(HTTPException) as exc:
                auth._authenticate(db, key)
            assert exc.value.status_code == 401
            assert "deactivated" in exc.value.detail

    def test_deactivating_a_missing_key_returns_false(self, sqlite_auth):
        assert auth.deactivate_api_key(9999) is False

    def test_list_api_keys_never_leaks_key_material(self, sqlite_auth):
        key = auth.create_api_key("ci")
        listed = auth.list_api_keys()

        assert len(listed) == 1
        assert "key" not in listed[0]
        assert listed[0]["key_hint"] in key
        assert listed[0]["name"] == "ci"
        assert listed[0]["usage_24h"] == {"requests": 0, "total_tokens": 0}


class TestRateLimit:
    def test_allows_requests_under_the_limit(self, sqlite_auth):
        key_id = 1
        with auth.session_scope() as db:
            auth._enforce_rate_limit(db, key_id, rpm_limit=3)  # no rows yet

    def test_rejects_once_the_limit_is_reached(self, sqlite_auth):
        key_id = 1
        with auth.session_scope() as db:
            for _ in range(3):
                db.add(auth.APIKeyUsage(api_key_id=key_id, endpoint="/ask"))

        with auth.session_scope() as db:
            with pytest.raises(HTTPException) as exc:
                auth._enforce_rate_limit(db, key_id, rpm_limit=3)
            assert exc.value.status_code == 429
            assert exc.value.headers["Retry-After"] == "60"

    def test_counts_every_endpoint_not_just_ask(self, sqlite_auth):
        """/compare used to escape the limiter entirely."""
        key_id = 1
        with auth.session_scope() as db:
            db.add(auth.APIKeyUsage(api_key_id=key_id, endpoint="/compare/strategies"))
            db.add(auth.APIKeyUsage(api_key_id=key_id, endpoint="/ingest"))

        with auth.session_scope() as db:
            with pytest.raises(HTTPException):
                auth._enforce_rate_limit(db, key_id, rpm_limit=2)

    def test_other_keys_have_their_own_budget(self, sqlite_auth):
        with auth.session_scope() as db:
            for _ in range(5):
                db.add(auth.APIKeyUsage(api_key_id=1, endpoint="/ask"))

        with auth.session_scope() as db:
            auth._enforce_rate_limit(db, 2, rpm_limit=1)


class TestRecordTokens:
    def test_attaches_counts_to_the_usage_row(self, sqlite_auth):
        with auth.session_scope() as db:
            usage = auth.APIKeyUsage(api_key_id=1, endpoint="/ask")
            db.add(usage)
            db.flush()
            usage_id = usage.id

        auth.record_tokens(
            {"id": 1, "usage_id": usage_id},
            tokens_input=321,
            tokens_output=45,
            model="claude-sonnet-5",
        )

        with auth.session_scope() as db:
            row = db.get(auth.APIKeyUsage, usage_id)
            assert row.tokens_input == 321
            assert row.tokens_output == 45
            assert row.model == "claude-sonnet-5"

    def test_is_a_noop_for_an_anonymous_principal(self, sqlite_auth):
        auth.record_tokens({"id": None, "usage_id": None}, tokens_input=5)

    def test_never_raises_when_the_row_is_gone(self, sqlite_auth):
        auth.record_tokens({"id": 1, "usage_id": 4242}, tokens_input=5)
