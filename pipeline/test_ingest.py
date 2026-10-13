import unittest

from pipeline.ingest import SUMMARY_MAX_CHARS, clean_summary


class CleanSummaryTests(unittest.TestCase):
    def test_strips_markup_and_collapses_whitespace(self):
        entry = {"summary": "<p>Hello   <b>world</b></p>\n"}

        self.assertEqual(clean_summary(entry, "Title"), "Hello world")

    def test_unescapes_entities(self):
        entry = {"summary": "Fire &amp; smoke in Berlin"}

        self.assertEqual(clean_summary(entry, "Title"), "Fire & smoke in Berlin")

    def test_drops_a_summary_that_repeats_the_title(self):
        entry = {"summary": "Same title"}

        self.assertEqual(clean_summary(entry, "Same title"), "")

    def test_reads_description_and_content_fallbacks(self):
        self.assertEqual(clean_summary({"description": "Desc"}, "T"), "Desc")
        self.assertEqual(clean_summary({"content": [{"value": "Body"}]}, "T"), "Body")

    def test_empty_summary_returns_nothing(self):
        self.assertEqual(clean_summary({}, "T"), "")
        self.assertEqual(clean_summary({"summary": "   "}, "T"), "")

    def test_caps_length(self):
        entry = {"summary": "x " * 1000}

        self.assertLessEqual(len(clean_summary(entry, "T")), SUMMARY_MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
