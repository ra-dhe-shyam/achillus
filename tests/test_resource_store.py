import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from resource_advisor import (
    ResourceType,
    RequestStatus,
    create_resource_request,
)
import resource_store
import resource_authorization


class ResourceStoreTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"

        self.db_patcher = patch("core.DATABASE", self.db_path)
        self.db_patcher.start()

        resource_store.initialize_resource_store()
        resource_authorization.initialize_authorization_store()

        self.request = create_resource_request(
            request_id="resource-001",
            resource_type=ResourceType.MODEL,
            reason="Coding benchmark performance is poor",
            evidence="Only 60% of benchmark tasks passed",
            expected_benefit="Improve coding performance",
            estimated_cost="No purchase proposed",
            privacy_impact="No personal data transmitted",
        )
        resource_store.save_resource_request(self.request)

    def tearDown(self):
        self.db_patcher.stop()
        self.temp_dir.cleanup()

    def approve_scope(self, scope="evaluate"):
        """Record both persisted approvals for a request scope."""
        for role in ("user", "mentor"):
            with patch("builtins.input", return_value=""):
                # Empty input is a denial; the real confirmation is
                # generated from the current persisted request.
                request = resource_authorization._load_current_request(
                    "resource-001"
                )
                fingerprint = resource_authorization.request_fingerprint(
                    request, scope
                )

            with patch(
                "builtins.input",
                return_value=f"APPROVE {role.upper()} {fingerprint}",
            ):
                self.assertTrue(
                    resource_authorization.collect_terminal_approval(
                        "resource-001", scope, role
                    )
                )

    def test_saves_and_retrieves_request(self):
        result = resource_store.get_resource_request("resource-001")
        self.assertEqual(result["reason"], self.request.reason)
        self.assertEqual(result["status"], "proposed")
        self.assertEqual(result["resource_type"], "model")

    def test_missing_request_returns_none(self):
        self.assertIsNone(
            resource_store.get_resource_request("missing")
        )

    def test_duplicate_id_is_rejected(self):
        with self.assertRaises(sqlite3.IntegrityError):
            resource_store.save_resource_request(self.request)

    def test_lists_requests(self):
        self.assertEqual(
            len(resource_store.list_resource_requests()), 1
        )

    def test_filters_by_status(self):
        self.assertEqual(
            resource_store.list_resource_requests(
                status=RequestStatus.PROPOSED
            )[0]["request_id"],
            "resource-001",
        )
        self.assertEqual(
            resource_store.list_resource_requests(
                status=RequestStatus.APPROVED
            ),
            [],
        )

    def test_rejects_invalid_limit(self):
        with self.assertRaises(ValueError):
            resource_store.list_resource_requests(limit=0)

    def test_rejects_invalid_status(self):
        with self.assertRaises(TypeError):
            resource_store.list_resource_requests(status="approved")

    def test_rejects_wrong_request_type(self):
        with self.assertRaises(TypeError):
            resource_store.save_resource_request("not a request")

    def test_approval_requires_scope(self):
        with self.assertRaises(ValueError):
            resource_store.transition_resource_request(
                "resource-001",
                RequestStatus.APPROVED,
                "Approve request",
            )

    def test_approval_requires_both_persisted_approvals(self):
        with self.assertRaises(PermissionError):
            resource_store.transition_resource_request(
                "resource-001",
                RequestStatus.APPROVED,
                "Approve evaluation",
                scope="evaluate",
            )

    def test_approving_request_records_audit(self):
        self.approve_scope("evaluate")

        result = resource_store.transition_resource_request(
            "resource-001",
            RequestStatus.APPROVED,
            "Both approvers authorized evaluation",
            scope="evaluate",
        )

        self.assertEqual(result["status"], "approved")
        audit = resource_store.get_resource_request_audit(
            "resource-001"
        )
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["old_status"], "proposed")
        self.assertEqual(audit[0]["new_status"], "approved")

    def test_approval_for_wrong_scope_is_rejected(self):
        self.approve_scope("evaluate")

        with self.assertRaises(PermissionError):
            resource_store.transition_resource_request(
                "resource-001",
                RequestStatus.APPROVED,
                "Attempt installation",
                scope="install",
            )

    def test_rejection_is_recorded(self):
        result = resource_store.transition_resource_request(
            "resource-001",
            RequestStatus.REJECTED,
            "Cost is too high",
        )
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(
            len(resource_store.get_resource_request_audit(
                "resource-001"
            )),
            1,
        )

    def test_approved_request_can_be_completed(self):
        self.approve_scope("evaluate")
        resource_store.transition_resource_request(
            "resource-001",
            RequestStatus.APPROVED,
            "Approved for evaluation",
            scope="evaluate",
        )
        result = resource_store.transition_resource_request(
            "resource-001",
            RequestStatus.COMPLETED,
            "Evaluation completed",
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            len(resource_store.get_resource_request_audit(
                "resource-001"
            )),
            2,
        )

    def test_rejected_request_cannot_be_approved(self):
        resource_store.transition_resource_request(
            "resource-001",
            RequestStatus.REJECTED,
            "Not needed",
        )

        with self.assertRaises(ValueError):
            resource_store.transition_resource_request(
                "resource-001",
                RequestStatus.APPROVED,
                "Trying to reverse rejection",
                scope="evaluate",
            )

    def test_completed_request_cannot_change_again(self):
        self.approve_scope("evaluate")
        resource_store.transition_resource_request(
            "resource-001",
            RequestStatus.APPROVED,
            "Approved",
            scope="evaluate",
        )
        resource_store.transition_resource_request(
            "resource-001",
            RequestStatus.COMPLETED,
            "Finished",
        )

        with self.assertRaises(ValueError):
            resource_store.transition_resource_request(
                "resource-001",
                RequestStatus.REJECTED,
                "Attempted reversal",
            )

    def test_empty_decision_reason_is_rejected(self):
        with self.assertRaises(ValueError):
            resource_store.transition_resource_request(
                "resource-001",
                RequestStatus.REJECTED,
                " ",
            )

    def test_missing_request_cannot_transition(self):
        with self.assertRaises(ValueError):
            resource_store.transition_resource_request(
                "missing",
                RequestStatus.APPROVED,
                "Approval reason",
                scope="evaluate",
            )

    def test_invalid_target_status_is_rejected(self):
        with self.assertRaises(TypeError):
            resource_store.transition_resource_request(
                "resource-001",
                "approved",
                "Approval reason",
            )


if __name__ == "__main__":
    unittest.main()
