"""Policy and evaluation for controlled self-improvement."""

from dataclasses import dataclass
from enum import Enum
import math


class ChangeState(Enum):
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"


@dataclass(frozen=True)
class Evaluation:
    attempts: int
    successes: int

    def __post_init__(self):
        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts < 0
        ):
            raise ValueError("attempts must be a non-negative integer")

        if (
            isinstance(self.successes, bool)
            or not isinstance(self.successes, int)
            or not 0 <= self.successes <= self.attempts
        ):
            raise ValueError(
                "successes must be between zero and attempts"
            )

    @property
    def success_rate(self):
        if self.attempts == 0:
            return 0.0

        return self.successes / self.attempts


def qualifies_for_review(
    evaluation,
    threshold=0.90,
    minimum_attempts=50,
):
    if not isinstance(evaluation, Evaluation):
        raise TypeError("evaluation must be an Evaluation")

    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("threshold must be between 0 and 1")

    if (
        isinstance(minimum_attempts, bool)
        or not isinstance(minimum_attempts, int)
        or minimum_attempts < 1
    ):
        raise ValueError("minimum_attempts must be positive")

    return (
        evaluation.attempts >= minimum_attempts
        and evaluation.success_rate >= threshold
    )


def may_modify_source(state, *, user_approved, mentor_approved):
    """Check approval requirements; never modify files here."""

    if not isinstance(state, ChangeState):
        raise TypeError("state must be a ChangeState")

    if type(user_approved) is not bool:
        raise TypeError("user_approved must be a bool")

    if type(mentor_approved) is not bool:
        raise TypeError("mentor_approved must be a bool")

    return (
        state is ChangeState.APPROVED
        and user_approved
        and mentor_approved
    )
