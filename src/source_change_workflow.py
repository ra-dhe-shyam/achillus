
"""Controlled lifecycle for proposed source-code changes.

This module coordinates evaluation and patch-bound approval.
It deliberately does not apply, execute, or deploy source changes.
"""

from dataclasses import dataclass, field

from change_manager import (
    ApprovalRecord,
    ChangeProposal,
    create_approval,
    has_required_approvals,
)
from self_improvement import (
    ChangeState,
    Evaluation,
    may_modify_source,
    qualifies_for_review,
)


@dataclass
class _WorkflowEntry:
    proposal: ChangeProposal
    state: ChangeState = ChangeState.PROPOSED
    evaluation: Evaluation | None = None
    eligible_for_review: bool = False
    approvals: list[ApprovalRecord] = field(default_factory=list)


class SourceChangeWorkflow:
    """Manage source-change proposals without applying patches."""

    def __init__(self):
        self._entries: dict[str, _WorkflowEntry] = {}

    def submit(self, proposal: ChangeProposal) -> ChangeState:
        """Register an intact proposal."""
        if not isinstance(proposal, ChangeProposal):
            raise TypeError("proposal must be a ChangeProposal")
        if not proposal.verify_integrity():
            raise ValueError("proposal patch integrity check failed")
        if proposal.proposal_id in self._entries:
            raise ValueError("proposal_id already exists")

        self._entries[proposal.proposal_id] = _WorkflowEntry(
            proposal=proposal
        )
        return ChangeState.PROPOSED

    def evaluate(
        self,
        proposal_id: str,
        evaluation: Evaluation,
        *,
        threshold: float = 0.90,
        minimum_attempts: int = 50,
    ) -> bool:
        """Evaluate a proposal and determine whether it can enter review."""
        entry = self._get_entry(proposal_id)
        self._require_state(entry, ChangeState.PROPOSED)

        if not isinstance(evaluation, Evaluation):
            raise TypeError("evaluation must be an Evaluation")

        entry.evaluation = evaluation
        entry.eligible_for_review = qualifies_for_review(
            evaluation,
            threshold=threshold,
            minimum_attempts=minimum_attempts,
        )
        entry.state = ChangeState.EVALUATED
        return entry.eligible_for_review

    def request_approval(self, proposal_id: str) -> ChangeState:
        """Advance an eligible evaluated proposal to approval."""
        entry = self._get_entry(proposal_id)
        self._require_state(entry, ChangeState.EVALUATED)

        if not entry.eligible_for_review:
            raise ValueError("proposal does not qualify for review")
        if not entry.proposal.verify_integrity():
            raise ValueError("proposal patch integrity check failed")

        entry.state = ChangeState.AWAITING_APPROVAL
        return entry.state

    def record_approval(
        self,
        proposal_id: str,
        *,
        approver: str,
        approved: bool,
    ) -> ChangeState:
        """Record an approval decision bound to the exact patch."""
        entry = self._get_entry(proposal_id)
        self._require_state(entry, ChangeState.AWAITING_APPROVAL)

        if not entry.proposal.verify_integrity():
            entry.state = ChangeState.REJECTED
            raise ValueError("proposal patch integrity check failed")

        # Reject duplicate decisions from the same approver. A new decision
        # requires a new proposal so the audit trail cannot be silently reset.
        if any(record.approver == approver for record in entry.approvals):
            raise ValueError("approver has already recorded a decision")

        record = create_approval(entry.proposal, approver, approved)
        entry.approvals.append(record)

        if not approved:
            entry.state = ChangeState.REJECTED
        elif has_required_approvals(entry.proposal, entry.approvals):
            entry.state = ChangeState.APPROVED

        return entry.state

    def get_state(self, proposal_id: str) -> ChangeState:
        """Return the current lifecycle state."""
        return self._get_entry(proposal_id).state

    def get_approvals(self, proposal_id: str) -> tuple[ApprovalRecord, ...]:
        """Return an immutable view of the recorded approval decisions."""
        return tuple(self._get_entry(proposal_id).approvals)

    def may_modify_source(self, proposal_id: str) -> bool:
        """Check policy eligibility; does not modify any source files."""
        entry = self._get_entry(proposal_id)
        approvals = entry.approvals

        user_approved = any(
            item.approver == "user"
            and item.approved is True
            and item.proposal_id == entry.proposal.proposal_id
            and item.patch_hash == entry.proposal.patch_hash
            for item in approvals
        )
        mentor_approved = any(
            item.approver == "mentor"
            and item.approved is True
            and item.proposal_id == entry.proposal.proposal_id
            and item.patch_hash == entry.proposal.patch_hash
            for item in approvals
        )

        return (
            entry.proposal.verify_integrity()
            and has_required_approvals(entry.proposal, approvals)
            and may_modify_source(
                entry.state,
                user_approved=user_approved,
                mentor_approved=mentor_approved,
            )
        )

    def _get_entry(self, proposal_id: str) -> _WorkflowEntry:
        if not isinstance(proposal_id, str) or not proposal_id.strip():
            raise ValueError("proposal_id must be a non-empty string")
        try:
            return self._entries[proposal_id]
        except KeyError:
            raise KeyError(f"unknown proposal_id: {proposal_id}") from None

    @staticmethod
    def _require_state(entry: _WorkflowEntry, expected: ChangeState) -> None:
        if entry.state is not expected:
            raise ValueError(
                f"operation requires state {expected.name}; "
                f"current state is {entry.state.name}"
            )
