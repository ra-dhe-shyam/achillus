import unittest

from self_improvement import (
    Evaluation,
    ChangeState,
    qualifies_for_review,
    may_modify_source,
)


class EvaluationTests(unittest.TestCase):

    def test_success_rate(self):
        self.assertEqual(
            Evaluation(50, 46).success_rate, 0.92
        )

    def test_zero_attempts(self):
        self.assertEqual(
            Evaluation(0, 0).success_rate, 0.0
        )

    def test_insufficient_attempts(self):
        self.assertFalse(
            qualifies_for_review(Evaluation(49, 49))
        )

    def test_threshold_met(self):
        self.assertTrue(
            qualifies_for_review(Evaluation(50, 45))
        )

    def test_threshold_not_met(self):
        self.assertFalse(
            qualifies_for_review(Evaluation(50, 44))
        )

    def test_successes_cannot_exceed_attempts(self):
        with self.assertRaises(ValueError):
            Evaluation(10, 11)

    def test_negative_attempts_rejected(self):
        with self.assertRaises(ValueError):
            Evaluation(-1, 0)

    def test_boolean_attempts_rejected(self):
        with self.assertRaises(ValueError):
            Evaluation(True, 1)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            qualifies_for_review(
                Evaluation(50, 50), threshold=1.5
            )

    def test_both_approvals_required(self):
        self.assertTrue(
            may_modify_source(
                ChangeState.APPROVED,
                user_approved=True,
                mentor_approved=True,
            )
        )

    def test_user_approval_alone_is_insufficient(self):
        self.assertFalse(
            may_modify_source(
                ChangeState.APPROVED,
                user_approved=True,
                mentor_approved=False,
            )
        )

    def test_mentor_approval_alone_is_insufficient(self):
        self.assertFalse(
            may_modify_source(
                ChangeState.APPROVED,
                user_approved=False,
                mentor_approved=True,
            )
        )

    def test_pending_change_cannot_modify_source(self):
        self.assertFalse(
            may_modify_source(
                ChangeState.AWAITING_APPROVAL,
                user_approved=True,
                mentor_approved=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
