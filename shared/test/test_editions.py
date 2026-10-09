import unittest

from shared.editions import (
    DEFAULT_EDITION_KEY,
    EDITIONS,
    edition_from_path,
    get_edition,
)


class EditionRegistryTests(unittest.TestCase):
    def test_default_edition_is_english_and_unprefixed(self):
        english = get_edition("en")
        self.assertEqual(english.language, "en")
        self.assertEqual(english.url_prefix, "")
        self.assertEqual(DEFAULT_EDITION_KEY, "en")

    def test_spanish_edition_is_nested_and_prefixed(self):
        spanish = get_edition("es")
        self.assertEqual(spanish.language, "es")
        self.assertEqual(spanish.url_prefix, "/es")
        self.assertEqual(spanish.history_dir.name, "es")
        self.assertTrue(spanish.maps_url.endswith("/es"))

    def test_spanish_thresholds_allow_single_country(self):
        spanish = get_edition("es")
        self.assertEqual(spanish.min_sources, 3)
        self.assertEqual(spanish.min_countries, 1)

    def test_unknown_edition_raises(self):
        with self.assertRaises(KeyError):
            get_edition("de")

    def test_edition_from_path_matches_prefix(self):
        self.assertEqual(edition_from_path("/es").key, "es")
        self.assertEqual(edition_from_path("/es/map/3").key, "es")
        self.assertEqual(edition_from_path("/es/about").key, "es")

    def test_edition_from_path_defaults_to_english(self):
        self.assertEqual(edition_from_path("/").key, "en")
        self.assertEqual(edition_from_path("/map/3").key, "en")
        self.assertEqual(edition_from_path("/about").key, "en")

    def test_every_edition_has_a_sources_file(self):
        for edition in EDITIONS.values():
            self.assertTrue(
                edition.sources_file.exists(),
                f"Missing sources file: {edition.sources_file}",
            )


if __name__ == "__main__":
    unittest.main()
