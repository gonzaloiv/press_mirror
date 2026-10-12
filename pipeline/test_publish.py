import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import publish


class PublishStateTests(unittest.TestCase):
    def test_success_is_persisted_and_skipped_on_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "buffer-state.json"
            with patch.dict(
                "os.environ", {"BUFFER_STATE_PATH": str(state_path)}
            ):
                completed, records = publish.load_publication_state("2026-10-12")
                post = {
                    "index": 2,
                    "text": "An English headline",
                    "image": "https://example.test/map.png",
                    "due_at": "2026-10-12T08:00:00.000Z",
                }
                channels = [("private-channel-id", "instagram")]
                success = {
                    "data": {
                        "createPost": {"post": {"id": "post-123"}}
                    }
                }

                with patch("pipeline.publish.create_post", return_value=success) as create:
                    failures = publish.publish_post(
                        channels,
                        post,
                        "unused-test-key",
                        "2026-10-12",
                        completed,
                        records,
                    )
                    self.assertEqual(failures, [])
                    self.assertEqual(create.call_count, 1)

                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(state["snapshot_date"], "2026-10-12")
                self.assertEqual(len(state["published"]), 1)
                self.assertNotIn("private-channel-id", state_path.read_text())

                with patch("pipeline.publish.create_post") as create:
                    failures = publish.publish_post(
                        channels,
                        post,
                        "unused-test-key",
                        "2026-10-12",
                        completed,
                        records,
                    )
                    self.assertEqual(failures, [])
                    create.assert_not_called()

    def test_state_for_an_old_day_does_not_block_today(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "buffer-state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "snapshot_date": "2026-10-11",
                        "published": [
                            {"cluster_index": 2, "channel_fingerprint": "hash"}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                "os.environ", {"BUFFER_STATE_PATH": str(state_path)}
            ):
                completed, records = publish.load_publication_state("2026-10-12")

        self.assertEqual(completed, set())
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
