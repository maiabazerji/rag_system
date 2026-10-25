"""Tests for validating strategies inside a private test workspace."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app.access import PUBLIC_GROUP, Principal, local_principal
from app.api.advisor import workspace_scope

WORKSPACE = "audit-0123456789ab"


def _member(groups=()) -> Principal:
    return Principal(id="key:1", kind="api_key", display_name="k", tenant="t", groups=groups)


class TestWorkspaceScope:
    def test_no_workspace_keeps_the_callers_scope(self):
        p = _member(("sales",))
        assert workspace_scope(p, None) == p.scope()

    def test_workspace_scope_sees_only_that_group(self):
        scope = workspace_scope(local_principal(), WORKSPACE)
        assert scope.groups == frozenset({WORKSPACE})
        assert PUBLIC_GROUP not in scope.groups
        assert scope.permits({"tenant": scope.tenant, "acl_groups": [WORKSPACE]})
        assert not scope.permits({"tenant": scope.tenant, "acl_groups": [PUBLIC_GROUP]})

    def test_member_may_use_its_workspace(self):
        assert workspace_scope(_member((WORKSPACE,)), WORKSPACE).groups == frozenset({WORKSPACE})

    def test_non_member_is_refused(self):
        with pytest.raises(HTTPException) as e:
            workspace_scope(_member(("sales",)), WORKSPACE)
        assert e.value.status_code == 403


class TestValidateEndpoint:
    def _fake_run(self, seen: list):
        async def run(question, *, strategy, model=None, access=None, **_):
            seen.append(access)
            result = SimpleNamespace(
                answer="Réponse.", refusal=False, input_tokens=10, output_tokens=5,
                latency_ms=12, sources=[], extra={},
            )
            return result, None
        return run

    def test_validate_retrieves_only_inside_the_workspace(self, client):
        seen: list = []
        with patch("app.advisor.validate.run_strategy_raw", side_effect=self._fake_run(seen)), \
                patch("app.api.advisor.record_tokens"), patch("app.api.advisor.charge"):
            r = client.post(
                "/advise/validate",
                json={"questions": [{"question": "Q ?"}], "strategies": ["classic"],
                      "workspace": WORKSPACE, "judge": False},
            )
        assert r.status_code == 200, r.text
        assert seen and all(a.groups == frozenset({WORKSPACE}) for a in seen)

    @pytest.mark.parametrize("bad", ["public", "audit-", "audit-UPPER123", "audit-x"])
    def test_malformed_workspace_is_rejected(self, client, bad):
        r = client.post(
            "/advise/validate",
            json={"questions": [{"question": "Q ?"}], "workspace": bad},
        )
        assert r.status_code == 422
