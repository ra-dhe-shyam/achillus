"""Create and verify source-change proposals."""

from dataclasses import dataclass
import hashlib


def calculate_patch_hash(patch):
    """Return a SHA-256 digest for a patch string."""
    if not isinstance(patch, str):
        raise TypeError("patch must be a string")

    if not patch:
        raise ValueError("patch cannot be empty")

    return hashlib.sha256(patch.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChangeProposal:
    proposal_id: str
    objective: str
    patch: str
    patch_hash: str

    @classmethod
    def create(cls, proposal_id, objective, patch):
        for name, value in (
            ("proposal_id", proposal_id),
            ("objective", objective),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")

        digest = calculate_patch_hash(patch)

        return cls(
            proposal_id=proposal_id.strip(),
            objective=objective.strip(),
            patch=patch,
            patch_hash=digest,
        )

    def verify_integrity(self):
        """Check whether the patch still matches its recorded hash."""
        return calculate_patch_hash(self.patch) == self.patch_hash

@dataclass(frozen=True)
class ApprovalRecord:
    proposal_id: str
    patch_hash: str
    approver: str
    approved: bool


def create_approval(proposal, approver, approved):
    if not isinstance(proposal, ChangeProposal):
        raise TypeError("proposal must be a ChangeProposal")

    if approver not in {"user", "mentor"}:
        raise ValueError("approver must be 'user' or 'mentor'")

    if type(approved) is not bool:
        raise TypeError("approved must be a bool")

    if not proposal.verify_integrity():
        raise ValueError("proposal integrity check failed")

    return ApprovalRecord(
        proposal_id=proposal.proposal_id,
        patch_hash=proposal.patch_hash,
        approver=approver,
        approved=approved,
    )


def has_required_approvals(proposal, approvals):
    if not isinstance(proposal, ChangeProposal):
        raise TypeError("proposal must be a ChangeProposal")

    if not proposal.verify_integrity():
        return False

    valid_approvers = set()

    for approval in approvals:
        if (
            isinstance(approval, ApprovalRecord)
            and approval.proposal_id == proposal.proposal_id
            and approval.patch_hash == proposal.patch_hash
            and approval.approved is True
        ):
            valid_approvers.add(approval.approver)

    return valid_approvers >= {"user", "mentor"}
