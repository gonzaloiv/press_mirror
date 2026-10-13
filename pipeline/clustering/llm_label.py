"""Optional LLM formulation of a cluster label from its shared terms.

The extractive terms stay the source of truth. This module only rewrites them
into one short, neutral English headline for display, using the cluster's
context headlines to disambiguate. Any failure returns None so callers fall
back to the deterministic term assembly.
"""

import re
import unicodedata

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MAX_WORDS = 9
LLM_MAX_NEW_TOKENS = 32
LLM_CONTEXT_HEADLINES = 8

_SYSTEM_PROMPT = (
    "You write one short, factual news headline ALWAYS IN ENGLISH. "
    "The shared terms and headlines are usually in other languages (German, "
    "French, Spanish, Italian, Dutch, Swedish, Finnish and others). Translate "
    "the shared event into natural English; never copy non-English words into "
    "the output. Write personal names and organisations in their standard "
    "English spelling (for example Selenskyi or Zelensky -> Zelenskyy, "
    "Poutine -> Putin). The shared terms are noisy hints, not text to stitch "
    "together. Read the headlines as a group, identify the central event, and "
    "describe it in English. Be specific: name the concrete actor and action, "
    "and include a distinctive shared term when one is available; avoid vague "
    "stock phrases such as 'threatens war' or 'raises concerns'. Do not invent "
    "names, places, numbers, or claims. Output one complete English headline "
    "of at most {max_words} words, ending on a complete phrase: no quotes, no "
    "list, no colon."
)

_FEW_SHOT = (
    (["gear", "factory", "fire"], "Fire breaks out at gear factory"),
    (
        ["election", "president", "protests"],
        "Protests follow presidential election",
    ),
    (["Guerre", "Ukraine", "morts"], "Deaths reported in Ukraine war"),
    (
        ["Erdbeben", "Panama", "Schäden"],
        "Earthquake causes damage in Panama",
    ),
    (
        ["Putin", "Trump", "Diesel", "Deal"],
        "Putin and Trump discuss diesel deal",
    ),
)


_ENGLISH_MARKERS = frozenset(
    "a an the and or but of to in on at by for from with after before while "
    "as is are was were be been being has have had says said who what where "
    "when why how will would can could should people killed dead attack "
    "attacks reported officials government president"
    .split()
)

_NON_ENGLISH_MARKERS = frozenset(
    "und sich selbst menschen erschiesst erschießt acht jahre jahr krieg "
    "har krympt ett vapenvila angrep drept etter minst nye israelske "
    "avec une des dans pour morts attaque frappe les sont est aux "
    "una unos unas los las muertos guerra tras entre contra "
    "sono degli delle alla nella attacco morti "
    "niet van een bij zijn wordt met "
    "para mortos sao esta voimakas maanjäristys kuollut sota "
    "puin aardbeving veel doden gewonden aanval"
    .split()
)

_SUBJECTIVE_MARKERS = frozenset(
    "bad good worst best terrible awful brilliant foolish wise stupid "
    "disastrous shameful disgraceful reckless ridiculous absurd pathetic "
    "outrageous humiliating"
    .split()
)

# Curated stopgap for names and organisations the model transliterates from
# source languages: the standard English form maps to the non-standard variants
# that should collapse onto it. It is intentionally small and shared by both
# the label and the displayed-term paths so a title and its subtitle agree.
_CANONICAL_NAMES = {
    "Zelenskyy": (
        "zelensky", "zelenskyy", "zelenskyj",
        "selensky", "selenskyi", "selenskyj", "zelenskiy",
    ),
    "Putin": ("putin", "poutine"),
    "NATO": ("nato",),
    "EU": ("eu",),
    "UK": ("uk",),
}
_NAME_CANONICALS = {
    variant: canonical
    for canonical, variants in _CANONICAL_NAMES.items()
    for variant in variants
}



def _english_marker_count(text: str) -> int:
    return len(_content_words(text) & _ENGLISH_MARKERS)


def english_headline(headlines) -> str | None:
    """Return a context headline with clear English function/content words."""
    for headline in headlines:
        if not (headline and headline.strip()):
            continue
        if _has_non_english_markers(headline):
            continue
        if _english_marker_count(headline) >= 2:
            return headline.strip()
    return None


def _user_prompt(terms: list[str], headlines) -> str:
    lines = [
        "Noisy shared-term hints (do not repeat literally): " + ", ".join(terms)
    ]
    context = [title for title in headlines if title and title.strip()]
    context.sort(key=lambda title: -_english_marker_count(title))
    if context:
        lines.append("Headlines from this cluster, English first:")
        for headline in context[:LLM_CONTEXT_HEADLINES]:
            lines.append(f"- {headline}")
    lines.append("Write one natural English headline for the shared event.")
    return "\n".join(lines)


def _messages(terms: list[str], headlines, max_words: int, english_hint=False):
    messages = [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT.format(max_words=max_words),
        }
    ]
    for example_terms, example_headline in _FEW_SHOT:
        messages.append(
            {"role": "user", "content": _user_prompt(example_terms, [])}
        )
        messages.append({"role": "assistant", "content": example_headline})
    final_prompt = _user_prompt(terms, headlines)
    if english_hint:
        final_prompt += (
            "\nIMPORTANT: reply in neutral English only. Translate source "
            "terms; do not praise, condemn, or express an opinion about the "
            "event, and avoid vague stock wording."
        )
    messages.append({"role": "user", "content": final_prompt})
    return messages


_DANGLING_WORDS = frozenset(
    """
    a an the and or but if then than so as of at by for from in into on onto
    to with without over under after before during while about against between
    through is are was were be been being has have had will would that this
    these those one two three four five six seven eight nine ten
    """.split()
)

# Attributive modifiers that cannot end a headline; truncation should drop
# them rather than show "…after alleged".
_PRENOMINAL_MODIFIERS = frozenset(
    "alleged suspected unnamed unidentified former".split()
)

_INCOMPLETE_TAIL_WORDS = (
    _DANGLING_WORDS.difference(
        "one two three four five six seven eight nine ten".split()
    )
    | frozenset("its their my your our".split())
    | _PRENOMINAL_MODIFIERS
)


def _has_incomplete_tail(words: list[str]) -> bool:
    token = words[-1].strip(".,;:!?…\"'").casefold()
    return token in _INCOMPLETE_TAIL_WORDS or token.endswith(("'s", "’s"))


def _has_cropped_ending(text: str) -> bool:
    words = text.split()
    return bool(words) and _has_incomplete_tail(words)


_REVIEW_STOPWORDS = _INCOMPLETE_TAIL_WORDS | frozenset(
    """
    do does did it his her he she they them we you who why what when where how
    not no now new more most all any only
    """.split()
)

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _canonicalize_names(text: str) -> str:
    """Rewrite transliterated names into their standard English spelling."""
    def replace(match: re.Match) -> str:
        word = match.group()
        canonical = _NAME_CANONICALS.get(word.replace("’", "'").casefold())
        return canonical if canonical else word

    return _WORD_RE.sub(replace, text)


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
    supported_words = label_words.intersection(source_words)
    if len(label_words) <= 2:
        return bool(supported_words)
    return len(supported_words) >= 2 and (
        len(supported_words) / len(label_words) >= 0.4
    )


def _has_non_english_markers(label: str) -> bool:
    return bool(_content_words(label).intersection(_NON_ENGLISH_MARKERS))


def _has_subjective_wording(label: str) -> bool:
    return bool(_content_words(label).intersection(_SUBJECTIVE_MARKERS))


def _is_dangling(token: str) -> bool:
    bare = token.strip(".,;:!?…\"'").casefold()
    return (
        not bare
        or len(bare) <= 1
        or bare in _DANGLING_WORDS
        or bare in _PRENOMINAL_MODIFIERS
        or bare.isdigit()
        or token.endswith(("'s", "’s"))
    )


def _drop_dangling_tail(words: list[str]) -> list[str]:
    while len(words) > 1 and _is_dangling(words[-1]):
        words.pop()
    return words


def clean_headline(text: str, *, max_words: int = LLM_MAX_WORDS) -> str:
    """Clean and cap a raw headline for use as a fallback label.

    The headline fallback bypasses the LLM review, so cap it and drop a
    dangling tail here to avoid shipping an overlong or cropped title.
    """
    cleaned = text.strip().strip("\"'").strip()
    cleaned = cleaned.rstrip(".,;:!?…").strip()
    cleaned = _canonicalize_names(cleaned)
    words = cleaned.split()
    if len(words) > max_words:
        words = _drop_dangling_tail(words[:max_words])
    return " ".join(words).rstrip(".,;:!?…").strip()


def _review_reason(label: str, terms: list[str], headlines) -> str | None:
    if _has_cropped_ending(label):
        return "incomplete_ending"
    if _has_non_english_markers(label):
        return "non_english_output"
    if _has_subjective_wording(label):
        return "subjective_wording"
    if not _has_context_support(label, terms, headlines):
        return "insufficient_context_support"
    return None


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
    cleaned = _canonicalize_names(cleaned)
    words = cleaned.split()
    if not words:
        return None, "empty_output"
    was_truncated = len(words) > max_words
    if was_truncated:
        words = _drop_dangling_tail(words[:max_words])
    result = " ".join(words).rstrip(".,;:!?…").strip()
    if not result:
        return None, "empty_output"
    reason = _review_reason(result, terms, headlines)
    if reason is not None:
        return None, reason
    return result, "accepted_after_truncation" if was_truncated else "accepted"


_generator = None


def _load_generator():
    global _generator
    if _generator is None:
        from transformers import pipeline

        _generator = pipeline("text-generation", model=LLM_MODEL_NAME, device=-1)
        _generator.tokenizer.clean_up_tokenization_spaces = False
    return _generator


def generator_available() -> bool:
    """True when the label model can be loaded in this process.

    Callers use this to degrade gracefully: when the model cannot run at all,
    label generation falls back to terms for every cluster, so callers should
    keep those clusters instead of dropping them all.
    """
    try:
        _load_generator()
    except Exception:
        return False
    return True


def _generation_config(model):
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=False,
        num_return_sequences=1,
        pad_token_id=model.tokenizer.eos_token_id,
        clean_up_tokenization_spaces=False,
    )


_TERM_TRANSLATION_SYSTEM_PROMPT = (
    "Translate each numbered news keyword or short phrase into concise, "
    "natural English. Use the headline context to resolve meaning. Write "
    "personal names and organisations in their standard English spelling "
    "(for example Poutine -> Putin, Zelensky -> Zelenskyy). Return exactly "
    "one numbered English translation per input line, in the same order, with "
    "no explanation or extra text."
)

_TERM_TRANSLATION_EXAMPLES = (
    (
        ["Jahr", "Waffenruhe", "Krieg"],
        "1. year\n2. ceasefire\n3. war",
    ),
    (
        ["voimakas", "maanjäristys", "Panama"],
        "1. powerful\n2. earthquake\n3. Panama",
    ),
)

_RECOVERABLE_REASONS = frozenset(
    {
        "non_english_output",
        "insufficient_context_support",
        "incomplete_ending",
        "subjective_wording",
    }
)


def _generate_once(model, terms, headlines, max_words, english_hint, use_config):
    prompt = model.tokenizer.apply_chat_template(
        _messages(terms, headlines, max_words, english_hint),
        tokenize=False,
        add_generation_prompt=True,
    )
    kwargs = {"return_full_text": False}
    if use_config:
        kwargs["generation_config"] = _generation_config(model)
    output = model(prompt, **kwargs)
    return output[0]["generated_text"]


def generate_llm_label_reviewed(
    terms: list[str],
    headlines=(),
    *,
    max_words: int = LLM_MAX_WORDS,
    generator=None,
) -> tuple[str | None, str]:
    if not terms:
        return None, "no_terms"
    try:
        model = generator if generator is not None else _load_generator()
    except Exception:
        return None, "generation_failed"

    reason = "generation_failed"
    for attempt in range(2):
        try:
            generated = _generate_once(
                model,
                terms,
                headlines,
                max_words,
                english_hint=attempt > 0,
                use_config=generator is None,
            )
        except Exception:
            return None, "generation_failed"
        label, reason = _clean_with_review(
            generated, max_words, terms, headlines
        )
        if label is not None or reason not in _RECOVERABLE_REASONS:
            return label, reason
    return None, reason


def generate_llm_label(
    terms: list[str],
    headlines=(),
    *,
    max_words: int = LLM_MAX_WORDS,
    generator=None,
) -> str | None:
    label, _ = generate_llm_label_reviewed(
        terms, headlines, max_words=max_words, generator=generator
    )
    return label


def _translation_messages(terms, headlines, english_hint=False):
    messages = [{"role": "system", "content": _TERM_TRANSLATION_SYSTEM_PROMPT}]
    for example_terms, example_output in _TERM_TRANSLATION_EXAMPLES:
        prompt = "Translate these terms:\n" + "\n".join(
            f"{index}. {term}"
            for index, term in enumerate(example_terms, start=1)
        )
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": example_output})

    prompt_lines = ["Translate these terms into English using the headlines:"]
    context = [title for title in headlines if title and title.strip()]
    context.sort(key=lambda title: -_english_marker_count(title))
    for headline in context[:LLM_CONTEXT_HEADLINES]:
        prompt_lines.append(f"Headline: {headline}")
    prompt_lines.append("Terms:")
    for index, term in enumerate(terms, start=1):
        prompt_lines.append(f"{index}. {term}")
    if english_hint:
        prompt_lines.append("Translate every term; do not copy source-language words.")
    messages.append({"role": "user", "content": "\n".join(prompt_lines)})
    return messages


def _parse_translated_terms(text: str, expected_count: int) -> list[str] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != expected_count:
        return None
    translations = []
    for index, line in enumerate(lines, start=1):
        match = re.match(r"^(\d+)[.)-]\s*(.+)$", line)
        if not match or int(match.group(1)) != index:
            return None
        translation = match.group(2).strip().strip("\"'").strip()
        translation = translation.rstrip(".,;:!?…").strip()
        if not translation:
            return None
        translations.append(translation)
    return translations


_VOWELS = frozenset("aeiouy")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return stripped.casefold()


def _name_skeleton(term: str) -> str | None:
    """Consonant skeleton of a single-word name, or None if not applicable."""
    folded = _fold(term).strip()
    if not folded.isalpha() or " " in folded or len(folded) < 5:
        return None
    skeleton = "".join(char for char in folded if char not in _VOWELS)
    return skeleton if len(skeleton) >= 3 else None


def _same_name(term: str, existing: str) -> bool:
    """True when two terms are spelling variants of the same name."""
    if _fold(term) == _fold(existing):
        return True
    left, right = _name_skeleton(term), _name_skeleton(existing)
    return left is not None and left == right


def translate_shared_terms(terms: list[str], headlines=(), generator=None) -> list[str]:
    """Translate displayed cluster keywords into English; discard unsafe output."""
    if not terms:
        return []
    try:
        model = generator if generator is not None else _load_generator()
    except Exception:
        return []

    for attempt in range(2):
        try:
            prompt = model.tokenizer.apply_chat_template(
                _translation_messages(terms, headlines, english_hint=attempt > 0),
                tokenize=False,
                add_generation_prompt=True,
            )
            kwargs = {"return_full_text": False}
            if generator is None:
                kwargs["generation_config"] = _generation_config(model)
            output = model(prompt, **kwargs)
        except Exception:
            return []
        translated = _parse_translated_terms(
            output[0]["generated_text"], len(terms)
        )
        if translated is None:
            continue
        safe_terms = []
        for term in translated:
            term = _canonicalize_names(term)
            if _has_non_english_markers(term):
                continue
            if any(_same_name(term, existing) for existing in safe_terms):
                continue
            safe_terms.append(term)
        return safe_terms
    return []
