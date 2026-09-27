"""Principals, tenants and per-document access control.

Every authenticated request resolves to a :class:`Principal`: an API key, an
OIDC user, or -- with ``REQUIRE_API_KEY`` off -- the local principal. What a
principal may read is its :class:`AccessScope`: one tenant plus a set of
groups. Each indexed chunk carries a ``tenant`` and an ``acl_groups`` list in
its payload, and a chunk is readable when its tenant matches and at least one
of its groups is in the scope.

Two conventions keep this backward compatible:

- Every principal is implicitly a member of ``public``, so a ``public``
  document is readable by everyone in its tenant.
- Chunks indexed before ACLs existed have neither field. They belong to
  ``DEFAULT_TENANT`` and, while ``ACL_LEGACY_PUBLIC`` is on, count as public.

Retrieval code filters in Qdrant (see ``app.rag.store``) and re-checks each
payload with :meth:`AccessScope.permits`, so the rule lives in one place.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import HTTPException

from app.config import settings

PUBLIC_GROUP = "public"
LOCAL_PRINCIPAL_ID = "local"

PrincipalKind = Literal["api_key", "oidc_user", "local"]

# Bounds on group names and counts, so a token or request cannot stuff the
# payload index with arbitrary data.
_MAX_GROUP_LEN = 256
_MAX_GROUPS = 200


def normalize_groups(groups: Iterable[Any] | None) -> tuple[str, ...]:
    """Clean a group list: strings only, stripped, non-empty, de-duplicated, sorted.

    Raises:
        ValueError: If there are more than 200 groups or one is over 256 chars.
    """
    cleaned = {str(g).strip() for g in (groups or []) if str(g).strip()}
    if len(cleaned) > _MAX_GROUPS:
        raise ValueError(f"At most {_MAX_GROUPS} groups are allowed, got {len(cleaned)}.")
    too_long = [g for g in cleaned if len(g) > _MAX_GROUP_LEN]
    if too_long:
        raise ValueError(f"Group names are limited to {_MAX_GROUP_LEN} characters.")
    return tuple(sorted(cleaned))


@dataclass(frozen=True)
class AccessScope:
    """What a caller may read: one tenant and the groups it belongs to.

    ``groups`` always includes ``public``. Construct it with
    :meth:`Principal.scope` rather than by hand.
    """

    tenant: str
    groups: frozenset[str]

    def permits(self, payload: Mapping[str, Any] | None) -> bool:
        """Whether a chunk with this payload is readable within the scope."""
        payload = payload or {}
        if chunk_tenant(payload) != self.tenant:
            return False
        return not self.groups.isdisjoint(chunk_groups(payload))


def chunk_tenant(payload: Mapping[str, Any]) -> str:
    """The tenant a chunk belongs to; legacy chunks belong to the default tenant."""
    return payload.get("tenant") or settings.default_tenant


def chunk_groups(payload: Mapping[str, Any]) -> list[str]:
    """The groups that may read a chunk; legacy chunks may count as public."""
    groups = payload.get("acl_groups")
    if groups:
        return list(groups)
    return [PUBLIC_GROUP] if settings.acl_legacy_public else []


@dataclass(frozen=True)
class Principal:
    """An authenticated caller.

    Attributes:
        id: Stable identifier: ``key:<id>``, ``oidc:<sub>`` or ``local``.
            Used as the document owner and in the audit log.
        kind: How the caller authenticated.
        display_name: Key name, or the user's name from the token.
        email: The user's email, when the token carries one.
        groups: Explicit group memberships (``public`` is implicit).
        is_admin: May label documents with any group. Grants no extra reads.
        tenant: The tenant the caller works in.
        key_id: API key row id, for rate limiting and usage accounting.
        usage_id: The usage row written for this request, if any.
    """

    id: str
    kind: PrincipalKind
    display_name: str
    tenant: str
    email: str | None = None
    groups: tuple[str, ...] = ()
    is_admin: bool = False
    key_id: int | None = None
    usage_id: int | None = None

    def scope(self) -> AccessScope:
        """The read scope for this principal."""
        return AccessScope(
            tenant=self.tenant, groups=frozenset(self.groups) | {PUBLIC_GROUP}
        )

    def as_auth(self) -> dict[str, Any]:
        """The legacy ``auth`` dict shape (``id``, ``name``, ``usage_id``), extended."""
        return {
            "id": self.key_id,
            "name": self.display_name,
            "usage_id": self.usage_id,
            "principal_id": self.id,
            "kind": self.kind,
            "tenant": self.tenant,
            "groups": list(self.groups),
            "is_admin": self.is_admin,
        }


def local_principal() -> Principal:
    """The caller when authentication is off: an admin in the default tenant."""
    return Principal(
        id=LOCAL_PRINCIPAL_ID,
        kind="local",
        display_name="anonymous",
        tenant=settings.default_tenant,
        groups=(PUBLIC_GROUP,),
        is_admin=True,
    )


def resolve_document_acl(
    principal: Principal, requested: Iterable[str] | None = None
) -> tuple[str, list[str]]:
    """Decide the tenant and groups a new document is indexed with.

    Args:
        principal: The uploader.
        requested: Groups the upload asked for. ``None`` means "use the default":
            ``public`` in local mode or with ``ACL_DEFAULT_PUBLIC``, otherwise
            the uploader's own groups (``public`` if it has none).

    Returns:
        ``(tenant, acl_groups)``.

    Raises:
        HTTPException: 422 for an empty or malformed group list; 403 when a
            non-admin asks for a group it is not a member of.
    """
    if requested is None:
        if principal.kind == "local" or settings.acl_default_public or not principal.groups:
            return principal.tenant, [PUBLIC_GROUP]
        return principal.tenant, list(normalize_groups(principal.groups))

    try:
        groups = normalize_groups(requested)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    if not groups:
        raise HTTPException(status_code=422, detail="At least one group is required.")
    if not principal.is_admin:
        allowed = principal.scope().groups
        foreign = sorted(set(groups) - allowed)
        if foreign:
            raise HTTPException(
                status_code=403,
                detail=f"You are not a member of: {', '.join(foreign)}.",
            )
    return principal.tenant, list(groups)


__all__ = [
    "LOCAL_PRINCIPAL_ID",
    "PUBLIC_GROUP",
    "AccessScope",
    "Principal",
    "chunk_groups",
    "chunk_tenant",
    "local_principal",
    "normalize_groups",
    "resolve_document_acl",
]
