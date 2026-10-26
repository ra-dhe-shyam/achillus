
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

import experience_store


class ExperienceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.patcher = patch("core.DATABASE", self.db_path)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp_dir.cleanup()

    def record(self, **overrides):
        data = {
            "task_type": "coding",
            "outcome": "corrected",
            "source": "user_feedback",
            "summary": (
                "The proposed patch changed the wrong function."
            ),
            "correction": "Patch the actual call site.",
            "metadata": {"test_case": "integration-1"},
            "user_confirmed": True,
        }
        data.update(overrides)
        return experience_store.record_experience(**data)

    def test_explicit_confirmation_required(self):
        with self.assertRaises(PermissionError):
            self.record(user_confirmed=False)

        self.assertEqual(
            experience_store.list_experiences(), []
        )

    def test_round_trip_and_filter(self):
        saved = self.record()

        items = experience_store.list_experiences(
            task_type="coding"
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(
            items[0]["experience_id"],
            saved["experience_id"],
        )
        self.assertEqual(
            items[0]["correction"],
            "Patch the actual call site.",
        )
        self.assertEqual(
            items[0]["metadata"],
            {"test_case": "integration-1"},
        )
        self.assertEqual(
            experience_store.list_experiences(
                outcome="success"
            ),
            [],
        )

    def test_rejects_invalid_outcome_and_source(self):
        with self.assertRaises(ValueError):
            self.record(outcome="unknown")

        with self.assertRaises(ValueError):
            self.record(source="automatic_transcript")

    def test_rejects_null_or_oversized_summary(self):
        with self.assertRaises(ValueError):
            self.record(summary="bad\x00summary")

        with self.assertRaises(ValueError):
            self.record(summary="x" * 1001)

    def test_metadata_must_be_json_serializable_and_bounded(self):
        with self.assertRaises(ValueError):
            self.record(metadata={"bad": object()})

        with self.assertRaises(ValueError):
            self.record(metadata={"payload": "x" * 2100})

    def test_limit_validation(self):
        with self.assertRaises(ValueError):
            experience_store.list_experiences(limit=0)

        with self.assertRaises(ValueError):
            experience_store.list_experiences(limit=True)


if __name__ == "__main__":
    unittest.main()
