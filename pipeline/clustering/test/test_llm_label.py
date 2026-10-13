import unittest

from pipeline.clustering.llm_label import (
    clean_headline,
    english_headline,
    generate_llm_label,
    generate_llm_label_reviewed,
    translate_shared_terms,
)


class _FakeTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        return "PROMPT"


class _FakeGenerator:
    def __init__(self, generated="", error=None):
        self.tokenizer = _FakeTokenizer()
        self.generated = generated
        self.error = error

    def __call__(self, prompt, **kwargs):
        if self.error is not None:
            raise self.error
        return [{"generated_text": self.generated}]


class _HintAwareGenerator(_FakeGenerator):
    def __init__(self, first, second):
        super().__init__(first)
        self.second = second

    def __call__(self, prompt, **kwargs):
        hinted = any(
            "IMPORTANT" in message["content"]
            for message in self.tokenizer.messages
        )
        return [{"generated_text": self.second if hinted else self.generated}]


class GenerateLlmLabelTests(unittest.TestCase):
    def test_returns_cleaned_headline(self):
        generator = _FakeGenerator('"Moscow drone attack"\ntrailing noise')

        label = generate_llm_label(
            ["Moscow", "drone", "attack"],
            ["Moscow drone attack disrupts power"],
            generator=generator,
        )

        self.assertEqual(label, "Moscow drone attack")

    def test_caps_the_word_count(self):
        generator = _FakeGenerator(
            "alpha beta gamma delta epsilon zeta eta theta iota"
        )

        label = generate_llm_label(
            ["alpha", "beta", "gamma", "delta"],
            max_words=7,
            generator=generator,
        )

        self.assertEqual(label, "alpha beta gamma delta epsilon zeta eta")

    def test_default_cap_keeps_a_complete_phrase(self):
        generator = _FakeGenerator(
            "Trump to talk to Putin about Russian plague"
        )

        label = generate_llm_label(["Trump", "Putin"], generator=generator)

        self.assertEqual(
            label, "Trump to talk to Putin about Russian plague"
        )

    def test_truncation_drops_a_dangling_tail(self):
        generator = _FakeGenerator(
            "Bulgaria Drone Attacks: Two Ships Hit, One Sunk"
        )

        label = generate_llm_label(
            ["Bulgaria", "drones", "Ships", "Hit"],
            max_words=7,
            generator=generator,
        )

        self.assertEqual(label, "Bulgaria Drone Attacks: Two Ships Hit")

    def test_truncation_drops_a_possessive_tail(self):
        generator = _FakeGenerator(
            "Quebec election adds new test to Canada's unity"
        )

        label = generate_llm_label(
            ["Quebec", "election"], max_words=7, generator=generator
        )

        self.assertEqual(label, "Quebec election adds new test")

    def test_truncation_drops_a_preposition_and_modifier_tail(self):
        generator = _FakeGenerator(
            "Actor Selma Blair arrested in New York after alleged attack"
        )
        headlines = ["Actor Selma Blair arrested in New York after alleged attack"]

        label = generate_llm_label(
            ["Selma", "arrested", "actor", "New York"],
            headlines,
            generator=generator,
        )

        self.assertEqual(label, "Actor Selma Blair arrested in New York")

    def test_canonicalizes_transliterated_names(self):
        generator = _FakeGenerator(
            "Putin wants more deal with Trump than Selenskyi"
        )

        label = generate_llm_label(
            ["Trump", "Putin", "deal"],
            ["Trump und Putin schließen Diesel-Deal"],
            generator=generator,
        )

        self.assertEqual(label, "Putin wants more deal with Trump than Zelenskyy")

    def test_clean_headline_caps_and_trims_a_dangling_tail(self):
        self.assertEqual(
            clean_headline(
                "A very long fallback headline that keeps going and going"
            ),
            "A very long fallback headline that keeps going",
        )

    def test_clean_headline_canonicalizes_names(self):
        self.assertEqual(
            clean_headline("Selenskyi meets NATO leaders"),
            "Zelenskyy meets NATO leaders",
        )

    def test_english_headline_skips_non_english_markers(self):
        self.assertIsNone(english_headline(["Putin and was killed und Krieg"]))
        self.assertEqual(
            english_headline(["Nine people dead after mass shooting"]),
            "Nine people dead after mass shooting",
        )

    def test_canonicalizes_names_in_translated_terms(self):
        generator = _FakeGenerator("1. Selenskyi\n2. deal")

        translated = translate_shared_terms(
            ["Selenskyi", "deal"], generator=generator
        )

        self.assertEqual(translated, ["Zelenskyy", "deal"])

    def test_rejects_a_cropped_possessive_ending(self):
        generator = _FakeGenerator(
            "Russia's plague fears don't scare us - just its"
        )

        self.assertIsNone(
            generate_llm_label(["Russia", "plague"], generator=generator)
        )

    def test_review_reports_an_incomplete_ending(self):
        generator = _FakeGenerator("Russia's plague fears don't scare us - just its")

        label, reason = generate_llm_label_reviewed(
            ["Russia", "plague"], generator=generator
        )

        self.assertIsNone(label)
        self.assertEqual(reason, "incomplete_ending")

    def test_review_accepts_a_complete_label(self):
        generator = _FakeGenerator("Moscow drone attack")

        label, reason = generate_llm_label_reviewed(
            ["Moscow", "drone"], generator=generator
        )

        self.assertEqual(label, "Moscow drone attack")
        self.assertEqual(reason, "accepted")

    def test_review_rejects_labels_without_context_support(self):
        generator = _FakeGenerator("Parliament votes on budget")

        label, reason = generate_llm_label_reviewed(
            ["Moscow", "plague"],
            ["Moscow plague concerns continue"],
            generator=generator,
        )

        self.assertIsNone(label)
        self.assertEqual(reason, "insufficient_context_support")

    def test_rejects_gibberish_from_multilingual_terms(self):
        generator = _FakeGenerator("Gaza stills attack weapons for year")
        headlines = [
            'Ein Jahr brüchige Waffenruhe Gaza: "Der Krieg lässt uns nicht in Ruhe"',
            "Expert: Gaza har krympt under ett år av vapenvila",
            "Minst fem drept i nye israelske angrep på Gazastripen",
        ]

        label, reason = generate_llm_label_reviewed(
            ["Israel", "Jahr", "Gaza har krympt", "Krieg"],
            headlines,
            generator=generator,
        )

        self.assertIsNone(label)
        self.assertEqual(reason, "insufficient_context_support")

    def test_rejects_non_english_headlines(self):
        generator = _FakeGenerator(
            "Mann erschießt in Pennsylvania acht Menschen und sich selbst"
        )

        label, reason = generate_llm_label_reviewed(
            ["Pennsylvania", "people", "shooting"], generator=generator
        )

        self.assertIsNone(label)
        self.assertEqual(reason, "non_english_output")

    def test_review_reports_generation_failure(self):
        generator = _FakeGenerator(error=RuntimeError("no model"))

        label, reason = generate_llm_label_reviewed(
            ["Moscow"], generator=generator
        )

        self.assertIsNone(label)
        self.assertEqual(reason, "generation_failed")

    def test_rejects_a_cropped_possessive_name(self):
        generator = _FakeGenerator("Russia's")

        self.assertIsNone(
            generate_llm_label(["Russia"], generator=generator)
        )

    def test_strips_trailing_punctuation(self):
        generator = _FakeGenerator("Turkey earthquake.")

        self.assertEqual(
            generate_llm_label(["Turkey", "earthquake"], generator=generator),
            "Turkey earthquake",
        )

    def test_no_terms_returns_none_without_a_generator(self):
        self.assertIsNone(generate_llm_label([]))

    def test_generator_failure_returns_none(self):
        generator = _FakeGenerator(error=RuntimeError("no model"))

        self.assertIsNone(
            generate_llm_label(["Moscow"], generator=generator)
        )

    def test_prompt_includes_terms_and_context(self):
        generator = _FakeGenerator("Moscow drone attack")

        generate_llm_label(
            ["Moscow", "drone"],
            ["Moscow drone attack", "Drone strike on Moscow"],
            generator=generator,
        )

        user_message = generator.tokenizer.messages[-1]["content"]
        self.assertIn("Moscow, drone", user_message)
        self.assertIn("Moscow drone attack", user_message)

    def test_retries_in_english_when_first_output_is_foreign(self):
        generator = _HintAwareGenerator(
            "Mann erschießt in Pennsylvania acht Menschen und sich selbst",
            "Pennsylvania shooting leaves nine dead",
        )

        label, reason = generate_llm_label_reviewed(
            ["Pennsylvania", "shooting", "nine", "dead"],
            generator=generator,
        )

        self.assertEqual(label, "Pennsylvania shooting leaves nine dead")
        self.assertEqual(reason, "accepted")

    def test_prefers_english_headlines_in_context(self):
        english = "Nine people dead after mass shooting in Pennsylvania"
        german = "Mann erschießt in Pennsylvania acht Menschen und sich selbst"
        generator = _FakeGenerator("Pennsylvania mass shooting")

        generate_llm_label(
            ["Pennsylvania", "shooting"], [german, english], generator=generator
        )

        user_message = generator.tokenizer.messages[-1]["content"]
        self.assertLess(user_message.index(english), user_message.index(german))
        self.assertEqual(english_headline([german, english]), english)

    def test_translates_shared_terms_in_order(self):
        generator = _FakeGenerator(
            "1. Israel\n2. year\n3. Gaza has shrunk\n4. war"
        )

        translated = translate_shared_terms(
            ["Israel", "Jahr", "Gaza har krympt", "Krieg"],
            ["Ein Jahr Waffenruhe in Gaza", "Expert: Gaza har krympt"],
            generator=generator,
        )

        self.assertEqual(
            translated, ["Israel", "year", "Gaza has shrunk", "war"]
        )

    def test_discards_terms_left_in_source_language(self):
        generator = _FakeGenerator("1. Israel\n2. Jahr\n3. war")

        translated = translate_shared_terms(
            ["Israel", "Jahr", "Krieg"], generator=generator
        )

        self.assertEqual(translated, ["Israel", "war"])

    def test_retries_malformed_translation_output(self):
        generator = _FakeGenerator("year and war")

        translated = translate_shared_terms(
            ["Jahr", "Krieg"], generator=generator
        )

        self.assertEqual(translated, [])

    def test_rejects_subjective_wording(self):
        generator = _FakeGenerator("Trump and Putin's diesel deal is a bad idea")

        label, reason = generate_llm_label_reviewed(
            ["Trump", "Putin", "diesel", "deal"],
            ["Trump und Putin schließen Diesel-Deal"],
            generator=generator,
        )

        self.assertIsNone(label)
        self.assertEqual(reason, "subjective_wording")

    def test_dedupes_cross_language_name_variants(self):
        generator = _FakeGenerator("1. Putin\n2. Poutine\n3. deal")

        translated = translate_shared_terms(
            ["Putin", "Poutine", "deal"], generator=generator
        )

        self.assertEqual(translated, ["Putin", "deal"])

    def test_keeps_distinct_translated_terms(self):
        generator = _FakeGenerator("1. deal\n2. diesel")

        translated = translate_shared_terms(
            ["deal", "diesel"], generator=generator
        )

        self.assertEqual(translated, ["deal", "diesel"])

    def test_prompt_includes_eight_context_headlines(self):
        generator = _FakeGenerator("Moscow drone attack")
        headlines = [f"Headline {index}" for index in range(10)]

        generate_llm_label(["Moscow", "drone"], headlines, generator=generator)

        user_message = generator.tokenizer.messages[-1]["content"]
        self.assertIn("- Headline 7", user_message)
        self.assertNotIn("- Headline 8", user_message)

    def test_prompt_includes_source_descriptions(self):
        generator = _FakeGenerator("Moscow drone attack")

        generate_llm_label(
            ["Moscow", "drone"],
            ["Moscow drone attack"],
            summaries=["A drone struck a Moscow refinery overnight."],
            generator=generator,
        )

        user_message = generator.tokenizer.messages[-1]["content"]
        self.assertIn("Moscow refinery", user_message)

    def test_caps_source_descriptions(self):
        generator = _FakeGenerator("Moscow drone attack")
        summaries = [f"Summary {index}" for index in range(10)]

        generate_llm_label(
            ["Moscow"],
            ["Moscow drone attack"],
            summaries=summaries,
            generator=generator,
        )

        user_message = generator.tokenizer.messages[-1]["content"]
        self.assertIn("- Summary 4", user_message)
        self.assertNotIn("- Summary 5", user_message)

    def test_translation_prompt_includes_source_descriptions(self):
        generator = _FakeGenerator("1. deal")

        translate_shared_terms(
            ["deal"],
            ["Deal unterzeichnet"],
            summaries=["Der Deal wurde gestern unterschrieben."],
            generator=generator,
        )

        user_message = generator.tokenizer.messages[-1]["content"]
        self.assertIn("gestern unterschrieben", user_message)


if __name__ == "__main__":
    unittest.main()
