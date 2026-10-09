"""Optional LLM formulation of a cluster label from its shared terms.

The extractive terms stay the source of truth. This module only rewrites them
into one short, neutral English headline for display, using the cluster's
context headlines to disambiguate. Any failure returns None so callers fall
back to the deterministic term assembly.
"""

import re

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MAX_WORDS = 9
LLM_MAX_NEW_TOKENS = 32
LLM_CONTEXT_HEADLINES = 8

_SYSTEM_PROMPT = (
    "You write one short, factual English news headline. "
    "Always write in English, even when the shared terms or headlines are in "
    "another language: translate the topic into English. "
    "Use only the given shared terms and headline context; do not invent "
    "names, places, or numbers. "
    "Output a single complete headline of at most {max_words} words, with no "
    "quotes, no list, and no colon. "
    "Always finish the thought: never end on a preposition, article, or an "
    "adjective that is missing its noun."
)

_SYSTEM_PROMPT_ES = (
    "Escribes un titular de prensa breve y factual en español. "
    "Escribe siempre en español. "
    "Usa solo los términos compartidos y el contexto de titulares dados; no "
    "inventes nombres, lugares ni cifras. "
    "Escribe un único titular completo de como máximo {max_words} palabras, "
    "sin comillas, sin listas y sin dos puntos. "
    "Termina siempre la idea: nunca acabes en una preposición, un artículo o "
    "un adjetivo al que le falte el sustantivo."
)

_SYSTEM_PROMPTS = {"en": _SYSTEM_PROMPT, "es": _SYSTEM_PROMPT_ES}

_FEW_SHOT = (
    (["gear", "factory", "fire"], "Fire breaks out at gear factory"),
    (
        ["election", "president", "protests"],
        "Protests follow presidential election",
    ),
)

_FEW_SHOT_ES = (
    (
        ["fábrica", "engranajes", "incendio"],
        "Se incendia una fábrica de engranajes",
    ),
    (
        ["elecciones", "presidente", "protestas"],
        "Protestas tras las elecciones presidenciales",
    ),
)

_FEW_SHOTS = {"en": _FEW_SHOT, "es": _FEW_SHOT_ES}

_WRITE_INSTRUCTION = {
    "en": "Write the English headline.",
    "es": "Escribe el titular en español.",
}


def _user_prompt(terms: list[str], headlines, language: str) -> str:
    lines = ["Shared terms: " + ", ".join(terms)]
    context = [title for title in headlines if title and title.strip()]
    if context:
        lines.append("Headline context:")
        for headline in context[:LLM_CONTEXT_HEADLINES]:
            lines.append(f"- {headline}")
    lines.append(_WRITE_INSTRUCTION.get(language, _WRITE_INSTRUCTION["en"]))
    return "\n".join(lines)


def _messages(terms: list[str], headlines, max_words: int, language: str):
    messages = [
        {
            "role": "system",
            "content": _SYSTEM_PROMPTS.get(language, _SYSTEM_PROMPT).format(
                max_words=max_words
            ),
        }
    ]
    for example_terms, example_headline in _FEW_SHOTS.get(
        language, _FEW_SHOT
    ):
        messages.append(
            {
                "role": "user",
                "content": _user_prompt(example_terms, [], language),
            }
        )
        messages.append({"role": "assistant", "content": example_headline})
    messages.append(
        {"role": "user", "content": _user_prompt(terms, headlines, language)}
    )
    return messages


_DANGLING_WORDS = frozenset(
    """
    a an the and or of to in on with for from at by as is are was were be been
    has have had will would that this these those one two three four five six
    seven eight nine ten
    el la los las un una unos unas del al de en a por para con y o que
    """.split()
)

_INCOMPLETE_TAIL_WORDS = _DANGLING_WORDS.difference(
    "one two three four five six seven eight nine ten".split()
) | frozenset("its their my your our".split())


def _has_incomplete_tail(words: list[str]) -> bool:
    token = words[-1].strip(".,;:!?…\"'").casefold()
    return token in _INCOMPLETE_TAIL_WORDS or token.endswith(("'s", "’s"))


def _has_cropped_ending(text: str) -> bool:
    words = text.split()
    return bool(words) and _has_incomplete_tail(words)


_REVIEW_STOPWORDS = _INCOMPLETE_TAIL_WORDS | frozenset(
    """
    but if then than so without over under after before during while about
    against between through being do does did it his her he she they them we
    you who why what when where how not no now new more most all any only
    """.split()
)

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _content_words(text: str) -> set[str]:
    return {
        match.group().replace("’", "'").casefold()
        for match in _WORD_RE.finditer(text)
        if len(match.group()) > 2
        and match.group().casefold() not in _REVIEW_STOPWORDS
    }


def _has_context_support(label: str, terms: list[str], headlines) -> bool:
    label_words = _content_words(label)
    source_words = _content_words(" ".join([*terms, *headlines]))
    return bool(label_words.intersection(source_words))


def _is_dangling(token: str) -> bool:
    bare = token.strip(".,;:!?…\"'").casefold()
    return (
        not bare
        or len(bare) <= 1
        or bare in _DANGLING_WORDS
        or bare.isdigit()
        or token.endswith(("'s", "’s"))
    )


def _drop_dangling_tail(words: list[str]) -> list[str]:
    while len(words) > 1 and _is_dangling(words[-1]):
        words.pop()
    return words


def _clean_with_review(
    text: str,
    max_words: int,
    terms: list[str],
    headlines,
) -> tuple[str | None, str]:
    first_line = next(
        (line for line in text.strip().splitlines() if line.strip()), ""
    )
    cleaned = first_line.strip().strip("\"'").strip()
    cleaned = cleaned.rstrip(".,;:!?…").strip()
    words = cleaned.split()
    if not words:
        return None, "empty_output"
    was_truncated = len(words) > max_words
    if was_truncated:
        words = _drop_dangling_tail(words[:max_words])
    result = " ".join(words).rstrip(".,;:!?…").strip()
    if not result:
        return None, "empty_output"
    if _has_cropped_ending(result):
        return None, "incomplete_ending"
    if not _has_context_support(result, terms, headlines):
        return result, "accepted_weak_context_support"
    return result, "accepted_after_truncation" if was_truncated else "accepted"


_generator = None


def _load_generator():
    global _generator
    if _generator is None:
        from transformers import pipeline

        _generator = pipeline("text-generation", model=LLM_MODEL_NAME, device=-1)
        _generator.tokenizer.clean_up_tokenization_spaces = False
    return _generator


def _generation_config(model):
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=False,
        num_return_sequences=1,
        pad_token_id=model.tokenizer.eos_token_id,
        clean_up_tokenization_spaces=False,
    )


def generate_llm_label_reviewed(
    terms: list[str],
    headlines=(),
    *,
    max_words: int = LLM_MAX_WORDS,
    language: str = "en",
    generator=None,
) -> tuple[str | None, str]:
    if not terms:
        return None, "no_terms"
    try:
        model = generator if generator is not None else _load_generator()
        prompt = model.tokenizer.apply_chat_template(
            _messages(terms, headlines, max_words, language),
            tokenize=False,
            add_generation_prompt=True,
        )
        kwargs = {"return_full_text": False}
        if generator is None:
            kwargs["generation_config"] = _generation_config(model)
        output = model(prompt, **kwargs)
        generated = output[0]["generated_text"]
    except Exception:
        return None, "generation_failed"
    return _clean_with_review(generated, max_words, terms, headlines)


def generate_llm_label(
    terms: list[str],
    headlines=(),
    *,
    max_words: int = LLM_MAX_WORDS,
    language: str = "en",
    generator=None,
) -> str | None:
    label, _ = generate_llm_label_reviewed(
        terms,
        headlines,
        max_words=max_words,
        language=language,
        generator=generator,
    )
    return label
