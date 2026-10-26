
import unittest

from change_manager import ChangeProposal
from self_improvement import ChangeState, Evaluation
from source_change_workflow import SourceChangeWorkflow


class SourceChangeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = SourceChangeWorkflow()
        self.proposal = ChangeProposal.create(
            "change-001",
            "Improve memory ranking",
            "diff --git a/example.py b/example.py\n+return improved_result\n",
        )
        self.workflow.submit(self.proposal)

    def qualify_and_request_approval(self):
        eligible = self.workflow.evaluate(
            "change-001",
            Evaluation(attempts=50, successes=47),
        )
        self.assertTrue(eligible)
        self.assertEqual(
            self.workflow.request_approval("change-001"),
            ChangeState.AWAITING_APPROVAL,
        )

    def test_proposal_starts_as_proposed(self):
        self.assertEqual(
            self.workflow.get_state("change-001"),
            ChangeState.PROPOSED,
        )

    def test_ineligible_evaluation_cannot_request_approval(self):
        eligible = self.workflow.evaluate(
            "change-001",
            Evaluation(attempts=50, successes=20),
        )
        self.assertFalse(eligible)
        with self.assertRaises(ValueError):
            self.workflow.request_approval("change-001")

    def test_both_approvals_are_required(self):
        self.qualify_and_request_approval()

        self.assertEqual(
            self.workflow.record_approval(
                "change-001", approver="user", approved=True
            ),
            ChangeState.AWAITING_APPROVAL,
        )
        self.assertFalse(self.workflow.may_modify_source("change-001"))

        self.assertEqual(
            self.workflow.record_approval(
                "change-001", approver="mentor", approved=True
            ),
            ChangeState.APPROVED,
        )
        self.assertTrue(self.workflow.may_modify_source("change-001"))

    def test_rejection_is_terminal(self):
        self.qualify_and_request_approval()

        self.assertEqual(
            self.workflow.record_approval(
                "change-001", approver="user", approved=False
            ),
            ChangeState.REJECTED,
        )
        with self.assertRaises(ValueError):
            self.workflow.record_approval(
                "change-001", approver="mentor", approved=True
            )
        self.assertFalse(self.workflow.may_modify_source("change-001"))

    def test_duplicate_approver_is_rejected(self):
        self.qualify_and_request_approval()
        self.workflow.record_approval(
            "change-001", approver="user", approved=True
        )

        with self.assertRaises(ValueError):
            self.workflow.record_approval(
                "change-001", approver="user", approved=True
            )

    def test_cannot_skip_evaluation(self):
        with self.assertRaises(ValueError):
            self.workflow.request_approval("change-001")

    def test_duplicate_proposal_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.workflow.submit(self.proposal)

    def test_approval_decisions_are_exposed_as_tuple(self):
        self.qualify_and_request_approval()
        self.workflow.record_approval(
            "change-001", approver="user", approved=True
        )

        approvals = self.workflow.get_approvals("change-001")
        self.assertIsInstance(approvals, tuple)
        self.assertEqual(len(approvals), 1)


if __name__ == "__main__":
    unittest.main()

