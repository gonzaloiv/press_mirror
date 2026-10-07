import unittest

from pipeline.clustering.llm_label import generate_llm_label


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
            ["alpha", "beta"], max_words=7, generator=generator
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
            ["Bulgaria", "drones"], max_words=7, generator=generator
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


if __name__ == "__main__":
    unittest.main()
