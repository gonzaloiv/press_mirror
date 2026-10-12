import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline.clustering import cluster
from shared import snapshots


class SnapshotAlignmentTests(unittest.TestCase):
    def test_cluster_loads_snapshot_date_from_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings_file = root / "embeddings.npz"
            source_headlines = root / "headlines.json"
            source_headlines.write_text("[]", encoding="utf-8")
            np.savez_compressed(
                embeddings_file,
                snapshot_date=np.array("2026-10-12"),
                embeddings=np.zeros((1, 2)),
                titles=np.array(["Example headline"], dtype=object),
                sources=np.array(["Example source"], dtype=object),
                countries=np.array(["GB"], dtype=object),
                links=np.array(["https://example.test"], dtype=object),
            )

            with patch.object(cluster, "INPUT_FILE", embeddings_file), patch.object(
                cluster, "headlines_file", return_value=source_headlines
            ):
                loaded = cluster.load_data()

        self.assertEqual(loaded[0], "2026-10-12")
        self.assertEqual(list(loaded[2]), ["Example headline"])

    def test_cluster_refuses_embeddings_without_their_source_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            embeddings_file = root / "embeddings.npz"
            np.savez_compressed(
                embeddings_file,
                snapshot_date=np.array("2026-10-12"),
                embeddings=np.zeros((1, 2)),
                titles=np.array(["Example headline"], dtype=object),
                sources=np.array(["Example source"], dtype=object),
                countries=np.array(["GB"], dtype=object),
                links=np.array(["https://example.test"], dtype=object),
            )

            with patch.object(cluster, "INPUT_FILE", embeddings_file), patch.object(
                cluster, "headlines_file", return_value=root / "missing.json"
            ):
                with self.assertRaisesRegex(FileNotFoundError, "refer to 2026-10-12"):
                    cluster.load_data()

    def test_available_snapshots_require_both_data_files(self):
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory)
            complete = history / "2026-10-11"
            incomplete = history / "2026-10-12"
            complete.mkdir()
            incomplete.mkdir()
            (complete / "clusters.json").touch()
            (complete / "headlines.json").touch()
            (incomplete / "clusters.json").touch()

            with patch.object(snapshots, "HISTORY_DIR", history):
                dates = snapshots.snapshot_dates()

        self.assertEqual(dates, ["2026-10-11"])


if __name__ == "__main__":
    unittest.main()
