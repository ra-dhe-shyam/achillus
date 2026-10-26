
import sys
import unittest
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from resource_advisor import (
    ResourceType,
    RequestStatus,
    create_resource_request,
)


class ResourceAdvisorTests(unittest.TestCase):

    def make_request(self, **overrides):
        values = {
            "request_id": "resource-001",
            "resource_type": ResourceType.MODEL,
            "reason": "Coding benchmark performance is poor",
            "evidence": "Only 60% of benchmark tasks passed",
            "expected_benefit": "Improve coding performance",
            "estimated_cost": "No purchase proposed",
            "privacy_impact": "No personal data transmitted",
        }
        values.update(overrides)
        return create_resource_request(**values)

    def test_creates_request(self):
        self.assertEqual(
            self.make_request().request_id, "resource-001"
        )

    def test_new_request_is_not_approved(self):
        self.assertEqual(
            self.make_request().status, RequestStatus.PROPOSED
        )

    def test_resource_type_is_preserved(self):
        request = self.make_request(
            resource_type=ResourceType.COMPUTE
        )
        self.assertEqual(
            request.resource_type, ResourceType.COMPUTE
        )

    def test_rejects_empty_reason(self):
        with self.assertRaises(ValueError):
            self.make_request(reason=" ")

    def test_rejects_empty_evidence(self):
        with self.assertRaises(ValueError):
            self.make_request(evidence="")

    def test_rejects_invalid_resource_type(self):
        with self.assertRaises(TypeError):
            self.make_request(resource_type="model")

    def test_request_is_immutable(self):
        request = self.make_request()
        with self.assertRaises(AttributeError):
            request.status = RequestStatus.APPROVED

    def test_rejects_invalid_status(self):
        with self.assertRaises(TypeError):
            self.make_request(status="approved")


if __name__ == "__main__":
    unittest.main()
