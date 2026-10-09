import unittest

from web_app.i18n import (
    ABOUT_TITLE,
    about_sections,
    coverage_line,
    ui_strings,
)
from web_app.map_presentation import european_coverage


class CoverageLineTests(unittest.TestCase):
    def test_english_pluralization(self):
        self.assertEqual(
            coverage_line(6, 4, "en"), "6 outlets in 4 countries"
        )
        self.assertEqual(coverage_line(1, 1, "en"), "1 outlet in 1 country")

    def test_spanish_pluralization(self):
        self.assertEqual(coverage_line(6, 4, "es"), "6 medios en 4 países")
        self.assertEqual(coverage_line(1, 1, "es"), "1 medio en 1 país")


class UiStringsTests(unittest.TestCase):
    def test_site_name_is_substituted(self):
        strings = ui_strings("es", "Press\\Mirror")
        self.assertIn("Press\\Mirror", strings["footer_disclaimer"])
        self.assertNotIn("{app}", strings["footer_disclaimer"])

    def test_known_keys_are_localized(self):
        self.assertEqual(ui_strings("es", "X")["back_all_maps"], "← Todos los mapas")
        self.assertEqual(ui_strings("en", "X")["back_all_maps"], "← All maps")


class AboutSectionsTests(unittest.TestCase):
    def test_english_headings(self):
        sections = about_sections("en", "Press\\Mirror", 40, 18)
        headings = [section["heading"] for section in sections]
        self.assertIn("How the grouping works", headings)

    def test_spanish_headings(self):
        sections = about_sections("es", "Press\\Mirror", 40, 18)
        headings = [section["heading"] for section in sections]
        self.assertIn("Cómo funciona la agrupación", headings)

    def test_sources_html_is_embedded(self):
        sections = about_sections(
            "es", "Press\\Mirror", 40, 18, "<ul id='sources'></ul>"
        )
        blocks = [block for section in sections for block in section["blocks"]]
        self.assertTrue(any("id='sources'" in block for block in blocks))

    def test_about_titles_exist_for_each_language(self):
        self.assertIn("en", ABOUT_TITLE)
        self.assertIn("es", ABOUT_TITLE)


class MapCoverageTests(unittest.TestCase):
    def test_spanish_coverage_copy_uses_spanish_country_names(self):
        self.assertEqual(
            european_coverage(["ES"], language="es"),
            "Cobertura de España.",
        )

    def test_english_coverage_copy_unchanged(self):
        self.assertEqual(
            european_coverage(["ES"], language="en"),
            "Coverage from Spain.",
        )

    def test_spanish_no_coverage_message(self):
        self.assertEqual(
            european_coverage([], language="es"),
            "No se encontró cobertura.",
        )


if __name__ == "__main__":
    unittest.main()
