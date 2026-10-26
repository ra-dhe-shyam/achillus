
"""SQLite persistence and decision auditing for resource requests."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import database_session
from resource_advisor import ResourceRequest, RequestStatus


ALLOWED_TRANSITIONS = {
    RequestStatus.PROPOSED: {
        RequestStatus.APPROVED,
        RequestStatus.REJECTED,
    },
    RequestStatus.APPROVED: {
        RequestStatus.COMPLETED,
    },
    RequestStatus.REJECTED: set(),
    RequestStatus.COMPLETED: set(),
}


def initialize_resource_store():
    """Create resource-request and audit tables if absent."""
    with database_session() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS resource_requests (
                request_id TEXT PRIMARY KEY,
                resource_type TEXT NOT NULL,
                reason TEXT NOT NULL,
                evidence TEXT NOT NULL,
                expected_benefit TEXT NOT NULL,
                estimated_cost TEXT NOT NULL,
                privacy_impact TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'proposed',
                created_at REAL NOT NULL
            )
        """)

        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_resource_requests_status
            ON resource_requests(status)
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS resource_request_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                old_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                reason TEXT NOT NULL,
                changed_at REAL NOT NULL,
                FOREIGN KEY (request_id)
                    REFERENCES resource_requests(request_id)
            )
        """)

        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_resource_audit_request
            ON resource_request_audit(request_id, id)
        """)


def save_resource_request(request):
    """Save a new request without overwriting an existing one."""
    if not isinstance(request, ResourceRequest):
        raise TypeError("request must be a ResourceRequest")

    with database_session() as db:
        db.execute("""
            INSERT INTO resource_requests (
                request_id, resource_type, reason, evidence,
                expected_benefit, estimated_cost, privacy_impact,
                status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            request.request_id,
            request.resource_type.value,
            request.reason,
            request.evidence,
            request.expected_benefit,
            request.estimated_cost,
            request.privacy_impact,
            request.status.value,
            time.time(),
        ))

    return request.request_id


def get_resource_request(request_id):
    """Retrieve one request by its identifier."""
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be non-empty text")

    with database_session() as db:
        row = db.execute("""
            SELECT request_id, resource_type, reason, evidence,
                   expected_benefit, estimated_cost, privacy_impact,
                   status, created_at
            FROM resource_requests
            WHERE request_id = ?
        """, (request_id.strip(),)).fetchone()

    return dict(row) if row is not None else None


def list_resource_requests(status=None, limit=50):
    """List recent requests, optionally filtered by status."""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 100
    ):
        raise ValueError("limit must be an integer between 1 and 100")

    if status is not None and not isinstance(status, RequestStatus):
        raise TypeError("status must be a RequestStatus")

    sql = """
        SELECT request_id, resource_type, reason, evidence,
               expected_benefit, estimated_cost, privacy_impact,
               status, created_at
        FROM resource_requests
    """
    params = []

    if status is not None:
        sql += " WHERE status = ?"
        params.append(status.value)

    sql += " ORDER BY created_at DESC, request_id DESC LIMIT ?"
    params.append(limit)

    with database_session() as db:
        return [
            dict(row)
            for row in db.execute(sql, params).fetchall()
        ]


def transition_resource_request(
    request_id, new_status, reason, *, scope=None
):
    """Change status and append an audit record in one transaction.

    Approval requires persisted user and mentor approvals for the
    exact action scope. Execution must independently recheck
    authorization immediately before performing the action.
    """
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be non-empty text")

    if not isinstance(new_status, RequestStatus):
        raise TypeError("new_status must be a RequestStatus")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be non-empty text")

    request_id = request_id.strip()
    reason = reason.strip()

    # Enforce persisted, scope-specific authorization before approval.
    if new_status is RequestStatus.APPROVED:
        if scope is None:
            raise ValueError(
                "scope is required when approving a resource request"
            )

        # Import here to avoid a circular import between modules.
        from resource_authorization import has_required_approvals

        if not has_required_approvals(request_id, scope):
            raise PermissionError(
                "Both user and mentor must approve this request and scope"
            )
    elif scope is not None:
        raise ValueError(
            "scope is only valid when approving a request"
        )

    with database_session() as db:
        row = db.execute(
            """
            SELECT status
            FROM resource_requests
            WHERE request_id = ?
            """,
            (request_id,),
        ).fetchone()

        if row is None:
            raise ValueError("Resource request not found")

        try:
            old_status = RequestStatus(row["status"])
        except ValueError as exc:
            raise ValueError(
                "Stored request has an invalid status"
            ) from exc

        if new_status not in ALLOWED_TRANSITIONS[old_status]:
            raise ValueError(
                f"Transition from {old_status.value} "
                f"to {new_status.value} is not allowed"
            )

        changed_at = time.time()

        cursor = db.execute(
            """
            UPDATE resource_requests
            SET status = ?
            WHERE request_id = ? AND status = ?
            """,
            (new_status.value, request_id, old_status.value),
        )

        if cursor.rowcount != 1:
            raise RuntimeError(
                "Request status changed concurrently"
            )

        db.execute(
            """
            INSERT INTO resource_request_audit (
                request_id, old_status, new_status,
                reason, changed_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                request_id,
                old_status.value,
                new_status.value,
                reason,
                changed_at,
            ),
        )

    return get_resource_request(request_id)

def get_resource_request_audit(request_id):
    """Return the transition history for one request, oldest first."""
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be non-empty text")

    with database_session() as db:
        return [
            dict(row)
            for row in db.execute("""
                SELECT id, request_id, old_status, new_status,
                       reason, changed_at
                FROM resource_request_audit
                WHERE request_id = ?
                ORDER BY id ASC
            """, (request_id.strip(),)).fetchall()
        ]
