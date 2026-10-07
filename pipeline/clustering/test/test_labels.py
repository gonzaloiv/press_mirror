import unittest
from unicodedata import combining, normalize

import numpy as np

from pipeline.clustering.labels import (
    build_cluster_label,
    extract_common_terms,
)


def _normalize(text: str) -> str:
    decomposed = normalize("NFKD", text.replace("’", "'").casefold())
    return "".join(
        character for character in decomposed if not combining(character)
    )


class _ConceptEncoder:
    """Deterministic stand-in for the multilingual sentence model.

    Words listed in the same group are embedded as one concept vector,
    mimicking how the real model places translations of a term close together.
    Every other token gets its own vector so unrelated words never merge.
    """

    def __init__(self, concept_words=(), titles=()):
        concepts = [set(words) for words in concept_words]
        known = set().union(*concepts) if concepts else set()
        for title in titles:
            for token in _normalize(title).split():
                if token not in known:
                    concepts.append({token})
                    known.add(token)
        self._size = len(concepts)
        self._vectors: dict[str, np.ndarray] = {}
        for index, words in enumerate(concepts):
            vector = np.zeros(self._size)
            vector[index] = 1.0
            for word in words:
                self._vectors[word] = vector

    def encode(self, texts):
        rows = []
        for text in texts:
            parts = [self._vectors[token] for token in text.split()]
            rows.append(
                np.mean(parts, axis=0) if parts else np.zeros(self._size)
            )
        return np.asarray(rows)

    __call__ = encode


def _concepts_present(terms, concept_words):
    present = set()
    for term in terms:
        for token in _normalize(term).split():
            for name, words in concept_words.items():
                if token in words:
                    present.add(name)
    return present


class ExtractCommonTermsTests(unittest.TestCase):
    def test_same_language_cluster_returns_shared_terms(self):
        titles = [
            "Moscow drone attack disrupts the power grid",
            "Drone attack targets Moscow",
            "Russia faces drone strike on Moscow",
        ]

        terms = extract_common_terms(
            titles, encode=_ConceptEncoder(titles=titles)
        )
        normalized = [term.casefold() for term in terms]

        self.assertIn("moscow", normalized)
        self.assertIn("drone attack", normalized)
        self.assertNotIn("drone", normalized)
        self.assertLessEqual(len(terms), 7)
        for term in terms:
            self.assertNotIn("?", term)
            self.assertNotIn("the", _normalize(term).split())

    def test_multilingual_cluster_merges_translated_terms(self):
        concepts = {
            "student": {"students", "eleves", "schuler"},
            "france": {"french", "francais", "franzosische"},
            "protest": {"protest", "manifestent", "protestieren"},
            "reform": {"reform", "reforme"},
        }
        titles = [
            "French students protest school reform",
            "Des élèves français manifestent contre la réforme",
            "Französische Schüler protestieren gegen die Reform",
        ]

        merged = _concepts_present(
            extract_common_terms(
                titles, encode=_ConceptEncoder(concepts.values(), titles)
            ),
            concepts,
        )

        self.assertLessEqual(
            {"student", "france", "protest", "reform"}, merged
        )

    def test_translations_merge_only_through_the_encoder(self):
        concepts = {
            "student": {"students", "eleves", "schuler"},
            "france": {"french", "francais", "franzosische"},
            "protest": {"protest", "manifestent", "protestieren"},
            "reform": {"reform", "reforme"},
        }
        titles = [
            "French students protest school reform",
            "Des élèves français manifestent contre la réforme",
            "Französische Schüler protestieren gegen die Reform",
        ]

        encoded_terms = extract_common_terms(
            titles, encode=_ConceptEncoder(concepts.values(), titles)
        )
        identity_terms = extract_common_terms(
            titles, encode=_ConceptEncoder(titles=titles)
        )

        encoded = _concepts_present(encoded_terms, concepts)
        identity = _concepts_present(identity_terms, concepts)

        self.assertLessEqual({"student", "protest"}, encoded)
        self.assertNotIn("student", identity)
        self.assertNotIn("protest", identity)

    def test_nested_terms_collapse_into_the_longer_phrase(self):
        titles = [
            "Ex chef arrested in Berlin",
            "Ex chef spied for Moscow",
            "Chef arrested in Berlin",
        ]

        terms = extract_common_terms(
            titles, encode=_ConceptEncoder(titles=titles)
        )
        normalized = [term.casefold() for term in terms]

        self.assertIn("ex chef", normalized)
        self.assertNotIn("chef", normalized)
        self.assertNotIn("ex", normalized)

    def test_no_titles_returns_no_terms(self):
        self.assertEqual(extract_common_terms([], encode=_ConceptEncoder()), [])
        self.assertEqual(
            extract_common_terms(["   "], encode=_ConceptEncoder()), []
        )


class BuildClusterLabelTests(unittest.TestCase):
    def test_orders_terms_by_representative_headline(self):
        label = build_cluster_label(
            ["attack", "Moscow", "drone"],
            "Moscow drone attack disrupts the power grid",
        )

        self.assertEqual(label, "Moscow drone attack")

    def test_prefers_the_longer_shared_span(self):
        label = build_cluster_label(
            ["drone", "drone attack"], "Drone attack targets Moscow"
        )

        self.assertEqual(label, "drone attack")

    def test_matches_terms_ignoring_case_and_accents(self):
        label = build_cluster_label(
            ["Schüler", "Frankreich"], "Schüler protestieren in Frankreich"
        )

        self.assertEqual(label, "Schüler Frankreich")

    def test_falls_back_to_joined_terms_when_absent(self):
        label = build_cluster_label(["Moscow", "drone"], "Typhoon hits Taiwan")

        self.assertEqual(label, "Moscow · drone")

    def test_falls_back_when_spans_partially_overlap(self):
        label = build_cluster_label(
            ["drone attack", "attack targets"], "drone attack targets Moscow"
        )

        self.assertEqual(label, "drone attack · attack targets")

    def test_uses_representative_when_no_terms(self):
        self.assertEqual(
            build_cluster_label([], "Some headline"), "Some headline"
        )


if __name__ == "__main__":
    unittest.main()
