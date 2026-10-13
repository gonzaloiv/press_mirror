import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pipeline.clustering.cluster import (
    ClusterMembers,
    build_clusters,
    deduplicate_cluster,
    load_summaries,
    merge_clusters,
)
from shared import snapshots
from shared.cluster_selection import (
    MAX_CLUSTER_COUNT,
    MIN_CLUSTER_REPRESENTATIVE_SIMILARITY,
    MIN_CLUSTER_SIMILARITY,
    selected_cluster_indices,
)


def _article(index, source, country):
    return {
        "title": f"Headline {index} shared term",
        "source": source,
        "country": country,
        "link": f"https://example.com/{index}",
    }


_COUNTRIES = ["AT", "BE", "DE", "IT", "ES", "CH"]


def _cluster(avg_similarity, rep=0.7, sources=6, countries=6):
    articles = [
        _article(index, f"source{index}", _COUNTRIES[index % countries])
        for index in range(sources)
    ]
    return {
        "avg_similarity": avg_similarity,
        "min_representative_similarity": rep,
        "articles": articles,
    }


class BuildClustersLabelSelectionTests(unittest.TestCase):
    def test_only_displayed_clusters_get_labels(self):
        shown_articles = [
            _article(index, f"source{index}", country)
            for index, country in enumerate(["FR", "DE", "ES", "IT", "FR", "DE"])
        ]
        hidden_articles = [
            _article(6, "source6", "FR"),
            _article(7, "source7", "DE"),
        ]
        articles = shown_articles + hidden_articles

        titles = [article["title"] for article in articles]
        sources = [article["source"] for article in articles]
        countries = [article["country"] for article in articles]
        links = [article["link"] for article in articles]
        angles = np.deg2rad([-20, -12, -5, 4, 12, 20])
        cluster_embeddings = np.column_stack((np.cos(angles), np.sin(angles)))
        embeddings = np.vstack(
            (cluster_embeddings, np.array([[0.0, 1.0], [0.0, 1.0]]))
        )
        labels = [0, 0, 0, 0, 0, 0, 1, 1]

        labeled_titles = []

        def fake_labeler(cluster_titles):
            labeled_titles.append(list(cluster_titles))
            return ["term"]

        with patch(
            "pipeline.clustering.cluster.translate_shared_terms",
            return_value=["term"],
        ), patch(
            "pipeline.clustering.cluster.generate_llm_label_reviewed",
            return_value=("Shared term event", "accepted"),
        ), patch("builtins.print") as output:
            result = build_clusters(
                labels,
                titles,
                sources,
                countries,
                links,
                embeddings,
                fake_labeler,
                deduplicate=False,
                merge=False,
            )

        self.assertEqual(len(labeled_titles), 1)
        self.assertEqual(
            labeled_titles[0],
            [
                "Headline 3 shared term",
                "Headline 2 shared term",
                "Headline 1 shared term",
                "Headline 4 shared term",
                "Headline 0 shared term",
                "Headline 5 shared term",
            ],
        )
        self.assertEqual(result[0]["source_terms"], ["term"])
        self.assertEqual(result[0]["terms"], ["term"])
        self.assertEqual(result[0]["label"], "Shared term event")
        self.assertEqual(result[0]["label_source"], "llm")
        self.assertEqual(
            result[0]["label_review"],
            {"status": "accepted", "reason": "accepted"},
        )
        output.assert_any_call(
            "Label review 1/1: accepted; kept generated label.",
            flush=True,
        )
        self.assertEqual(result[1]["terms"], [])
        self.assertNotIn("label_source", result[1])

    def test_fills_daily_grid_with_terms_when_no_clean_label(self):
        shown_articles = [
            _article(index, f"source{index}", country)
            for index, country in enumerate(["FR", "DE", "ES", "IT", "FR", "DE"])
        ]
        hidden_articles = [
            _article(6, "source6", "FR"),
            _article(7, "source7", "DE"),
        ]
        articles = shown_articles + hidden_articles
        titles = [article["title"] for article in articles]
        sources = [article["source"] for article in articles]
        countries = [article["country"] for article in articles]
        links = [article["link"] for article in articles]
        angles = np.deg2rad([-20, -12, -5, 4, 12, 20])
        cluster_embeddings = np.column_stack((np.cos(angles), np.sin(angles)))
        embeddings = np.vstack(
            (cluster_embeddings, np.array([[0.0, 1.0], [0.0, 1.0]]))
        )
        labels = [0, 0, 0, 0, 0, 0, 1, 1]

        with patch(
            "pipeline.clustering.cluster.translate_shared_terms",
            return_value=["term"],
        ), patch(
            "pipeline.clustering.cluster.generate_llm_label_reviewed",
            return_value=(None, "insufficient_context_support"),
        ), patch("builtins.print"):
            result = build_clusters(
                labels,
                titles,
                sources,
                countries,
                links,
                embeddings,
                lambda cluster_titles: ["term"],
                deduplicate=False,
                merge=False,
            )

        # The only gated cluster fell back to terms, so it fills the daily
        # selection instead of being dropped; the sub-threshold cluster
        # (never displayed) also remains.
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["label_source"], "terms")
        self.assertNotIn("label_source", result[1])

    def test_falls_back_to_english_headline_when_llm_rejected(self):
        english = "Nine people dead after mass shooting in Pennsylvania"
        articles = [
            _article(index, f"source{index}", country)
            for index, country in enumerate(["UK", "DE", "FR", "ES", "IT", "NL"])
        ]
        for index, article in enumerate(articles):
            article["title"] = f"Pennsylvania earthquake report {index}"
        articles[0]["title"] = english
        titles = [article["title"] for article in articles]
        sources = [article["source"] for article in articles]
        countries = [article["country"] for article in articles]
        links = [article["link"] for article in articles]
        angles = np.deg2rad([-20, -12, -5, 4, 12, 20])
        embeddings = np.column_stack((np.cos(angles), np.sin(angles)))
        labels = [0, 0, 0, 0, 0, 0]

        with patch(
            "pipeline.clustering.cluster.translate_shared_terms",
            return_value=["Pennsylvania", "shooting"],
        ), patch(
            "pipeline.clustering.cluster.generate_llm_label_reviewed",
            return_value=(None, "non_english_output"),
        ), patch("builtins.print"):
            result = build_clusters(
                labels,
                titles,
                sources,
                countries,
                links,
                embeddings,
                lambda cluster_titles: ["term"],
                deduplicate=False,
                merge=False,
            )

        self.assertEqual(result[0]["label"], english)
        self.assertEqual(result[0]["label_source"], "headline")

    def test_low_similarity_to_representative_suppresses_label(self):
        articles = [
            _article(index, f"source{index}", country)
            for index, country in enumerate(
                ["AT", "BE", "DE", "IT", "ES", "CH"]
            )
        ]
        titles = [article["title"] for article in articles]
        sources = [article["source"] for article in articles]
        countries = [article["country"] for article in articles]
        links = [article["link"] for article in articles]
        outlier_similarity = MIN_CLUSTER_REPRESENTATIVE_SIMILARITY - 0.03
        outlier_angle = np.rad2deg(np.arccos(outlier_similarity))
        angles = np.deg2rad([0, 0, 0, 0, 0, outlier_angle])
        embeddings = np.column_stack((np.cos(angles), np.sin(angles)))
        labeled_titles = []

        def fake_labeler(cluster_titles):
            labeled_titles.append(list(cluster_titles))
            return ["term"]

        result = build_clusters(
            [0] * 6,
            titles,
            sources,
            countries,
            links,
            embeddings,
            fake_labeler,
            deduplicate=False,
            merge=False,
        )

        self.assertGreaterEqual(result[0]["avg_similarity"], 0.74)
        self.assertAlmostEqual(
            result[0]["min_representative_similarity"],
            outlier_similarity,
            places=3,
        )
        self.assertEqual(labeled_titles, [])
        self.assertEqual(result[0]["label"], "")

    def test_selection_returns_most_similar_first(self):
        articles = [
            _article(index, f"source{index}", country)
            for index, country in enumerate(
                ["AT", "BE", "DE", "IT", "ES", "CH"]
            )
        ]
        scores = [
            (0.748, 0.716),
            (0.758, 0.694),
            (0.765, 0.682),
            (0.740, 0.699),
        ]
        clusters = [
            {
                "avg_similarity": avg_similarity,
                "min_representative_similarity": representative_similarity,
                "articles": articles,
            }
            for avg_similarity, representative_similarity in scores
        ]

        self.assertEqual(selected_cluster_indices(clusters), [2, 1, 0, 3])

    def test_selection_keeps_top_five_and_strong_clusters(self):
        floor = MIN_CLUSTER_SIMILARITY
        clusters = [
            _cluster(floor + 0.10),
            _cluster(floor + 0.08),
            _cluster(floor + 0.06),
            _cluster(floor + 0.04),
            _cluster(floor + 0.02),
            _cluster(floor + 0.005),
            _cluster(floor - 0.10),
            _cluster(floor + 0.30, sources=2),
            _cluster(floor + 0.28, countries=3),
        ]

        self.assertEqual(
            selected_cluster_indices(clusters), [0, 1, 2, 3, 4, 5]
        )

    def test_selection_fills_to_five_even_below_floor(self):
        floor = MIN_CLUSTER_SIMILARITY
        clusters = [
            _cluster(floor + 0.05),
            _cluster(floor + 0.03),
            _cluster(floor - 0.20),
        ]

        self.assertEqual(selected_cluster_indices(clusters), [0, 1, 2])

    def test_selection_is_capped_at_max(self):
        floor = MIN_CLUSTER_SIMILARITY
        clusters = [
            _cluster(floor + 0.20 - 0.01 * index) for index in range(12)
        ]

        self.assertEqual(
            selected_cluster_indices(clusters),
            list(range(MAX_CLUSTER_COUNT)),
        )


class ClusterPostProcessTests(unittest.TestCase):
    def test_deduplicate_keeps_nearest_article_per_source(self):
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

        members = deduplicate_cluster([0, 1, 2], ["a", "a", "b"], embeddings)

        self.assertEqual(members.indices, [1, 2])
        self.assertEqual(members.dropped_sources, {"a": 1})

    def test_deduplicate_handles_an_empty_cluster(self):
        members = deduplicate_cluster([], [], np.zeros((0, 2)))

        self.assertEqual(members.indices, [])
        self.assertEqual(members.dropped_sources, {})

    def test_merge_combines_close_centroids(self):
        embeddings = np.array(
            [[1.0, 0.0], [0.98, 0.02], [0.999, 0.001], [0.97, 0.03]]
        )
        clusters = [ClusterMembers([0, 1]), ClusterMembers([2, 3])]

        merged = merge_clusters(
            clusters, ["a", "b", "c", "d"], embeddings, threshold=0.5
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].indices, [0, 1, 2, 3])
        self.assertTrue(merged[0].merged)

    def test_merge_keeps_distant_centroids_apart(self):
        embeddings = np.array(
            [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]]
        )
        clusters = [ClusterMembers([0, 1]), ClusterMembers([2, 3])]

        merged = merge_clusters(
            clusters, ["a", "b", "c", "d"], embeddings, threshold=0.5
        )

        self.assertEqual(len(merged), 2)
        self.assertFalse(any(cluster.merged for cluster in merged))


class BuildClustersDegradationTests(unittest.TestCase):
    def _fixture(self):
        shown = [
            _article(index, f"source{index}", country)
            for index, country in enumerate(["FR", "DE", "ES", "IT", "FR", "DE"])
        ]
        hidden = [_article(6, "source6", "FR"), _article(7, "source7", "DE")]
        articles = shown + hidden
        angles = np.deg2rad([-20, -12, -5, 4, 12, 20])
        cluster_embeddings = np.column_stack((np.cos(angles), np.sin(angles)))
        embeddings = np.vstack(
            (cluster_embeddings, np.array([[0.0, 1.0], [0.0, 1.0]]))
        )
        return articles, embeddings

    def test_keeps_term_fallback_clusters_when_model_unavailable(self):
        articles, embeddings = self._fixture()

        with patch(
            "pipeline.clustering.cluster.translate_shared_terms",
            return_value=["term"],
        ), patch(
            "pipeline.clustering.cluster.generate_llm_label_reviewed",
            return_value=(None, "generation_failed"),
        ), patch("builtins.print"):
            result = build_clusters(
                [0, 0, 0, 0, 0, 0, 1, 1],
                [article["title"] for article in articles],
                [article["source"] for article in articles],
                [article["country"] for article in articles],
                [article["link"] for article in articles],
                embeddings,
                lambda cluster_titles: ["term"],
                deduplicate=False,
                merge=False,
                llm_available=False,
            )

        self.assertEqual(len(result), 2)
        self.assertTrue(
            any(cluster.get("label_source") == "terms" for cluster in result)
        )


class DailyFiveTests(unittest.TestCase):
    def _gated_fixture(self, cluster_count):
        """`cluster_count` gate-passing clusters, all equally similar."""
        articles = [
            _article(index, f"source{index}", _COUNTRIES[index % 6])
            for index in range(cluster_count * 6)
        ]
        titles = [article["title"] for article in articles]
        sources = [article["source"] for article in articles]
        countries = [article["country"] for article in articles]
        links = [article["link"] for article in articles]
        angles = [
            np.deg2rad(30 * (index // 6)) for index in range(cluster_count * 6)
        ]
        embeddings = np.column_stack((np.cos(angles), np.sin(angles)))
        labels = [
            cluster for cluster in range(cluster_count) for _ in range(6)
        ]
        return titles, sources, countries, links, embeddings, labels

    def _labeling(self, clean_before_rejection):
        calls = {"count": 0}

        def fake_generate(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] <= clean_before_rejection:
                return ("Shared event label", "accepted")
            return (None, "insufficient_context_support")

        return fake_generate, calls

    def _run(self, cluster_count, clean_before_rejection):
        titles, sources, countries, links, embeddings, labels = (
            self._gated_fixture(cluster_count)
        )
        generate, calls = self._labeling(clean_before_rejection)
        with patch(
            "pipeline.clustering.cluster.translate_shared_terms",
            return_value=["term"],
        ), patch(
            "pipeline.clustering.cluster.generate_llm_label_reviewed",
            side_effect=generate,
        ), patch("builtins.print"):
            result = build_clusters(
                labels,
                titles,
                sources,
                countries,
                links,
                embeddings,
                lambda cluster_titles: ["term"],
                deduplicate=False,
                merge=False,
            )
        return result, calls

    def test_fills_daily_five_with_fallback_labels(self):
        # Six gate-passing clusters, only two label cleanly: the daily five
        # is two LLM labels plus the three best term-assembled names.
        result, _ = self._run(cluster_count=6, clean_before_rejection=2)

        self.assertEqual(len(result), 5)
        sources = [cluster["label_source"] for cluster in result]
        self.assertEqual(sources, ["llm", "llm", "terms", "terms", "terms"])

    def test_labels_deeper_candidates_until_five_are_clean(self):
        # Twelve gate-passing clusters but the display pool holds nine: when
        # only two label cleanly, labeling extends past the pool.
        result, calls = self._run(cluster_count=12, clean_before_rejection=2)

        self.assertEqual(calls["count"], 12)
        self.assertEqual(len(result), 5)
        sources = [cluster["label_source"] for cluster in result]
        self.assertEqual(sources, ["llm", "llm", "terms", "terms", "terms"])

    def test_drops_fallbacks_once_five_clean_labels_exist(self):
        # Six gate-passing clusters, five label cleanly: the sixth cluster's
        # term fallback is dropped again.
        result, _ = self._run(cluster_count=6, clean_before_rejection=5)

        self.assertEqual(len(result), 5)
        self.assertEqual(
            [cluster["label_source"] for cluster in result],
            ["llm", "llm", "llm", "llm", "llm"],
        )


class LoadSummariesTests(unittest.TestCase):
    def test_maps_links_to_non_empty_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory)
            day = history / "2026-10-13"
            day.mkdir()
            (day / "headlines.json").write_text(
                json.dumps(
                    [
                        {"link": "https://a", "summary": "First summary"},
                        {"link": "https://b", "summary": ""},
                        {"link": "https://c"},
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(snapshots, "HISTORY_DIR", history):
                summaries = load_summaries("2026-10-13")

        self.assertEqual(summaries, {"https://a": "First summary"})

    def test_missing_headlines_file_returns_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(snapshots, "HISTORY_DIR", Path(directory)):
                summaries = load_summaries("2026-10-13")

        self.assertEqual(summaries, {})


if __name__ == "__main__":
    unittest.main()
