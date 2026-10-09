import unittest

from shared.cluster_selection import selected_cluster_indices


def _cluster(sources, countries, similarity=0.8):
    return {
        "avg_similarity": similarity,
        "articles": [
            {"source": source, "country": country}
            for source, country in zip(sources, countries)
        ],
    }


class SelectedClusterIndicesTests(unittest.TestCase):
    def test_default_thresholds_require_six_sources_and_four_countries(self):
        clusters = [
            _cluster(
                ["a", "b", "c", "d", "e", "f"],
                ["FR", "DE", "ES", "IT", "FR", "DE"],
            )
        ]
        self.assertEqual(selected_cluster_indices(clusters), [0])

    def test_single_country_cluster_is_rejected_by_default(self):
        clusters = [
            _cluster(
                ["a", "b", "c", "d", "e", "f"],
                ["ES", "ES", "ES", "ES", "ES", "ES"],
            )
        ]
        self.assertEqual(selected_cluster_indices(clusters), [])

    def test_relaxed_thresholds_accept_a_single_country(self):
        clusters = [
            _cluster(["a", "b", "c"], ["ES", "ES", "ES"])
        ]
        self.assertEqual(
            selected_cluster_indices(
                clusters, min_sources=3, min_countries=1
            ),
            [0],
        )

    def test_similarity_floor_still_applies(self):
        clusters = [
            _cluster(["a", "b", "c"], ["ES", "ES", "ES"], similarity=0.5)
        ]
        self.assertEqual(
            selected_cluster_indices(
                clusters, min_sources=3, min_countries=1
            ),
            [],
        )

    def test_too_few_sources_is_rejected(self):
        clusters = [_cluster(["a", "b"], ["ES", "ES"])]
        self.assertEqual(
            selected_cluster_indices(clusters, min_sources=3, min_countries=1),
            [],
        )


if __name__ == "__main__":
    unittest.main()
