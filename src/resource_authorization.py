"""Scope-specific, terminal-confirmed authorization for resource requests.

This is an initial local safeguard, not strong authentication against a
compromised machine or another process running as the same OS user.
"""

import hashlib
import json
import time

from core import database_session
from resource_advisor import ResourceType, RequestStatus
from resource_store import (
    get_resource_request,
    initialize_resource_store,
)


ALLOWED_SCOPES = frozenset({
    "evaluate",
    "download",
    "install",
    "external_data_transfer",
    "purchase",
})

APPROVAL_ROLES = frozenset({"user", "mentor"})


def _validate_scope(scope):
    if not isinstance(scope, str) or scope not in ALLOWED_SCOPES:
        raise ValueError(
            f"scope must be one of: {', '.join(sorted(ALLOWED_SCOPES))}"
        )


def _validate_role(role):
    if not isinstance(role, str) or role not in APPROVAL_ROLES:
        raise ValueError("role must be 'user' or 'mentor'")


def _request_payload(request, scope):
    """Build the canonical, scope-bound representation of a request."""
    from resource_advisor import ResourceRequest

    if not isinstance(request, ResourceRequest):
        raise TypeError("request must be a ResourceRequest")

    _validate_scope(scope)

    return {
        "request_id": request.request_id,
        "resource_type": request.resource_type.value,
        "reason": request.reason,
        "evidence": request.evidence,
        "expected_benefit": request.expected_benefit,
        "estimated_cost": request.estimated_cost,
        "privacy_impact": request.privacy_impact,
        "scope": scope,
    }


def request_fingerprint(request, scope):
    """Hash the exact request contents and the requested action scope."""
    payload = _request_payload(request, scope)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def initialize_authorization_store():
    """Create the append-only-by-API authorization decision table."""
    initialize_resource_store()

    with database_session() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS resource_authorization_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                role TEXT NOT NULL,
                request_fingerprint TEXT NOT NULL,
                approved INTEGER NOT NULL CHECK (approved IN (0, 1)),
                reason TEXT NOT NULL,
                decided_at REAL NOT NULL,
                FOREIGN KEY (request_id)
                    REFERENCES resource_requests(request_id)
            )
        """)

        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_resource_authorization_lookup
            ON resource_authorization_decisions(
                request_id, scope, request_fingerprint, id
            )
        """)


def _load_current_request(request_id):
    """Load the persisted request; never trust caller-supplied request fields."""
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be non-empty text")

    row = get_resource_request(request_id.strip())
    if row is None:
        raise ValueError("Resource request not found")

    from resource_advisor import ResourceRequest

    try:
        resource_type = ResourceType(row["resource_type"])
        status = RequestStatus(row["status"])
    except (KeyError, ValueError) as exc:
        raise ValueError("Stored resource request is invalid") from exc

    request = ResourceRequest(
        request_id=row["request_id"],
        resource_type=resource_type,
        reason=row["reason"],
        evidence=row["evidence"],
        expected_benefit=row["expected_benefit"],
        estimated_cost=row["estimated_cost"],
        privacy_impact=row["privacy_impact"],
        status=status,
    )

    if status in {RequestStatus.REJECTED, RequestStatus.COMPLETED}:
        raise ValueError(
            f"Cannot authorize a request with status '{status.value}'"
        )

    return request


def collect_terminal_approval(request_id, scope, role):
    """Ask a human at the local terminal to approve one exact request scope.

    Approval requires typing the full role- and fingerprint-bound phrase.
    Any other response is recorded as a denial.
    """
    _validate_scope(scope)
    _validate_role(role)
    initialize_authorization_store()

    request = _load_current_request(request_id)
    fingerprint = request_fingerprint(request, scope)

    print("\n=== ACHILLUS RESOURCE AUTHORIZATION ===")
    print(f"Request ID:       {request.request_id}")
    print(f"Approver role:    {role}")
    print(f"Scope:            {scope}")
    print(f"Resource type:    {request.resource_type.value}")
    print(f"Reason:           {request.reason}")
    print(f"Evidence:         {request.evidence}")
    print(f"Expected benefit: {request.expected_benefit}")
    print(f"Estimated cost:   {request.estimated_cost}")
    print(f"Privacy impact:   {request.privacy_impact}")
    print(f"Request SHA-256:  {fingerprint}")
    print("\nApproval applies ONLY to this request and scope.")
    print("It does not authorize other actions.")

    expected = f"APPROVE {role.upper()} {fingerprint}"
    response = input(
        f"\nTo approve, type exactly:\n{expected}\n> "
    ).strip()

    approved = response == expected
    reason = (
        f"Terminal confirmation accepted for role={role}, scope={scope}"
        if approved
        else f"Terminal confirmation denied or mismatched for role={role}, scope={scope}"
    )

    with database_session() as db:
        db.execute("""
            INSERT INTO resource_authorization_decisions (
                request_id, scope, role, request_fingerprint,
                approved, reason, decided_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            request.request_id,
            scope,
            role,
            fingerprint,
            int(approved),
            reason,
            time.time(),
        ))

    return approved


def has_required_approvals(request_id, scope):
    """Return True only if the latest decisions from both roles approve.

    Decisions must match the current persisted request fingerprint and
    the exact requested scope. A later denial overrides an earlier approval.
    """
    _validate_scope(scope)
    initialize_authorization_store()

    request = _load_current_request(request_id)
    fingerprint = request_fingerprint(request, scope)

    with database_session() as db:
        rows = db.execute("""
            SELECT role, approved
            FROM resource_authorization_decisions
            WHERE request_id = ?
              AND scope = ?
              AND request_fingerprint = ?
            ORDER BY id DESC
        """, (request.request_id, scope, fingerprint)).fetchall()

    latest = {}
    for row in rows:
        if row["role"] in APPROVAL_ROLES and row["role"] not in latest:
            latest[row["role"]] = bool(row["approved"])

    return all(latest.get(role, False) for role in APPROVAL_ROLES)


def list_authorization_decisions(request_id):
    """Return the authorization history for a request, oldest first."""
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be non-empty text")

    initialize_authorization_store()

    with database_session() as db:
        rows = db.execute("""
            SELECT id, request_id, scope, role, request_fingerprint,
                   approved, reason, decided_at
            FROM resource_authorization_decisions
            WHERE request_id = ?
            ORDER BY id ASC
        """, (request_id.strip(),)).fetchall()

    return [dict(row) for row in rows]
