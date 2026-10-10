import unittest
from unittest.mock import patch

import numpy as np

from pipeline.clustering.cluster import build_clusters
from shared.cluster_selection import MIN_CLUSTER_SIMILARITY, selected_cluster_indices


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
            "pipeline.clustering.cluster.generate_llm_label_reviewed",
            return_value=(None, "incomplete_ending"),
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
        self.assertEqual(result[0]["terms"], ["term"])
        self.assertEqual(result[0]["label_source"], "terms")
        self.assertEqual(
            result[0]["label_review"],
            {"status": "fallback", "reason": "incomplete_ending"},
        )
        output.assert_any_call(
            "Label review 1/1: incomplete_ending; using term fallback.",
            flush=True,
        )
        self.assertEqual(result[1]["terms"], [])
        self.assertNotIn("label_source", result[1])

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
        outlier_angle = np.rad2deg(np.arccos(0.65))
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
            result[0]["min_representative_similarity"], 0.65, places=3
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


if __name__ == "__main__":
    unittest.main()
