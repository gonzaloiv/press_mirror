import unittest
from unittest.mock import patch

import numpy as np

from pipeline.clustering.cluster import build_clusters


def _article(index, source, country):
    return {
        "title": f"Headline {index} shared term",
        "source": source,
        "country": country,
        "link": f"https://example.com/{index}",
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
            "pipeline.clustering.cluster.generate_llm_label",
            return_value=None,
        ):
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
        self.assertEqual(result[1]["terms"], [])
        self.assertNotIn("label_source", result[1])


if __name__ == "__main__":
    unittest.main()
