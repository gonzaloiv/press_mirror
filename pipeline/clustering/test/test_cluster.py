import unittest
from unittest.mock import patch

import numpy as np

from pipeline.clustering.cluster import (
    ClusterMembers,
    build_clusters,
    deduplicate_cluster,
    merge_clusters,
)
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

    def test_drops_display_candidates_without_a_good_label(self):
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

        # The displayed candidate fell back to terms and was dropped; only the
        # sub-threshold cluster (never displayed) remains.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["terms"], [])
        self.assertNotIn("label_source", result[0])

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


if __name__ == "__main__":
    unittest.main()
