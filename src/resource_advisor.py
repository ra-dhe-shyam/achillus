
"""Generate structured resource requests for AI improvement."""

from dataclasses import dataclass
from enum import Enum


class ResourceType(Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    MODEL = "model"
    DATA = "data"
    TOOL = "tool"
    NETWORK = "network"


class RequestStatus(Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"


@dataclass(frozen=True)
class ResourceRequest:
    request_id: str
    resource_type: ResourceType
    reason: str
    evidence: str
    expected_benefit: str
    estimated_cost: str = "unknown"
    privacy_impact: str = "unknown"
    status: RequestStatus = RequestStatus.PROPOSED

    def __post_init__(self):
        text_fields = (
            "request_id",
            "reason",
            "evidence",
            "expected_benefit",
            "estimated_cost",
            "privacy_impact",
        )

        for field_name in text_fields:
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} must be non-empty text"
                )

        if not isinstance(self.resource_type, ResourceType):
            raise TypeError(
                "resource_type must be a ResourceType"
            )

        if not isinstance(self.status, RequestStatus):
            raise TypeError(
                "status must be a RequestStatus"
            )


def create_resource_request(
    request_id,
    resource_type,
    reason,
    evidence,
    expected_benefit,
    estimated_cost="unknown",
    privacy_impact="unknown",
):
    return ResourceRequest(
        request_id=request_id,
        resource_type=resource_type,
        reason=reason,
        evidence=evidence,
        expected_benefit=expected_benefit,
        estimated_cost=estimated_cost,
        privacy_impact=privacy_impact,
    )
