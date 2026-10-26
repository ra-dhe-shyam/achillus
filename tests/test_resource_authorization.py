import builtins
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from resource_advisor import ResourceType, create_resource_request
import resource_store
import resource_authorization as auth


class ResourceAuthorizationTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"

        self.db_patcher = patch("core.DATABASE", self.db_path)
        self.db_patcher.start()

        resource_store.initialize_resource_store()
        auth.initialize_authorization_store()

        self.request = create_resource_request(
            request_id="auth-001",
            resource_type=ResourceType.MODEL,
            reason="Improve coding performance",
            evidence="Benchmark success rate is 60%",
            expected_benefit="Better coding results",
            estimated_cost="No purchase proposed",
            privacy_impact="No personal data transmitted",
        )
        resource_store.save_resource_request(self.request)

    def tearDown(self):
        self.db_patcher.stop()
        self.temp_dir.cleanup()

    def _approve(self, role, scope="evaluate"):
        fingerprint = auth.request_fingerprint(self.request, scope)
        phrase = f"APPROVE {role.upper()} {fingerprint}"

        with patch.object(builtins, "input", return_value=phrase):
            return auth.collect_terminal_approval(
                self.request.request_id, scope, role
            )

    def test_fingerprint_is_deterministic(self):
        first = auth.request_fingerprint(self.request, "evaluate")
        second = auth.request_fingerprint(self.request, "evaluate")
        self.assertEqual(first, second)

    def test_scope_changes_fingerprint(self):
        evaluation = auth.request_fingerprint(self.request, "evaluate")
        installation = auth.request_fingerprint(self.request, "install")
        self.assertNotEqual(evaluation, installation)

    def test_invalid_scope_rejected(self):
        with self.assertRaises(ValueError):
            auth.request_fingerprint(self.request, "execute_everything")

    def test_invalid_role_rejected(self):
        with self.assertRaises(ValueError):
            auth.collect_terminal_approval(
                self.request.request_id, "evaluate", "ai"
            )

    def test_user_approval_alone_is_insufficient(self):
        self.assertTrue(self._approve("user"))
        self.assertFalse(
            auth.has_required_approvals("auth-001", "evaluate")
        )

    def test_both_roles_are_required(self):
        self.assertTrue(self._approve("user"))
        self.assertTrue(self._approve("mentor"))
        self.assertTrue(
            auth.has_required_approvals("auth-001", "evaluate")
        )

    def test_approval_does_not_transfer_between_scopes(self):
        self._approve("user", "evaluate")
        self._approve("mentor", "evaluate")

        self.assertFalse(
            auth.has_required_approvals("auth-001", "install")
        )

    def test_mismatched_confirmation_is_denied(self):
        with patch.object(builtins, "input", return_value="yes"):
            result = auth.collect_terminal_approval(
                "auth-001", "evaluate", "user"
            )

        self.assertFalse(result)
        self.assertFalse(
            auth.has_required_approvals("auth-001", "evaluate")
        )

    def test_later_denial_overrides_approval(self):
        self._approve("user")
        self._approve("mentor")
        self.assertTrue(
            auth.has_required_approvals("auth-001", "evaluate")
        )

        with patch.object(builtins, "input", return_value="no"):
            auth.collect_terminal_approval("auth-001", "evaluate", "user")

        self.assertFalse(
            auth.has_required_approvals("auth-001", "evaluate")
        )

    def test_unknown_request_rejected(self):
        with self.assertRaises(ValueError):
            auth.collect_terminal_approval(
                "missing", "evaluate", "user"
            )

    def test_authorization_history_is_persisted(self):
        self._approve("user")
        decisions = auth.list_authorization_decisions("auth-001")

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["role"], "user")
        self.assertEqual(decisions[0]["scope"], "evaluate")
        self.assertEqual(decisions[0]["approved"], 1)


if __name__ == "__main__":
    unittest.main()
