"""Extract the terms a cluster's headlines have in common.

This module does not write a sentence. For each cluster it reports the words
and short phrases that recur across the headlines, ranked by how many
headlines they touch. Equivalent surface forms are grouped into a single
concept with the shared multilingual embedding model, so translations of the
same term ("drone" / "Drohne" / "dron") compete as one shared term instead of
three rare ones.
"""

import re
from dataclasses import dataclass
from unicodedata import combining, normalize

import numpy as np

MAX_SHARED_TERMS = 7
MIN_SHARED_HEADLINES = 2
MAX_TERM_WORDS = 3
CONCEPT_SIMILARITY_THRESHOLD = 0.72

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model = None

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)

# Generic function words only. Content-bearing vocabulary is never listed here.
_FUNCTION_WORDS = frozenset(
    """
    a an the and or but if then than so as of at by for from in into on onto
    to with without over under after before during while about against between
    through is are was were be been being has have had do does did this that
    these those it its their his her he she they them we you your who why what
    when where how not no now new more most all any only
    le la les un une des du de dans pour avec sur au aux et est sont ce cette
    ces
    il elle ils elles que qui ne pas plus son sa ses contre chez par vers sous
    entre sans tout tous leur leurs notre votre mais ou donc car ni
    der die das den dem des ein eine einer und oder von fur mit nach auf zu im
    am an als ist sind wird werden nicht auch gegen aus bei vom zum zur vor
    seit durch ohne um uber zwischen dass wie nur noch schon sehr mehr beim
    el los las del una unos unas y o pero si como por para con en al es son fue
    fueron ser estar esta estan se su sus lo les mas sin sobre entre hasta
    desde muy ya todo
    il gli le dei delle nel sulle non piu di della da su per tra fra che si
    os as dos das uma para com na no sao nao do da em ao aos um mais
    het een van bij voor met zijn wordt niet ook de den te op om als dat die
    deze
    och ar det den som att for med av pa inte i till utan over under vid
    w na ze i nie do od po za pod nad
    ja on ovat etta kanssa ei se kuin mutta
    one two three first last how why warum wie was zwei drei jeden deux trois
    comment pourquoi dos tres como porque due come perche dois como porque twee
    drie hoe waarom dwa trzy jak dlaczego tva tre hur varfor kaksi kolme miten
    miksi
    us
    """.split()
)


def _normalize_token(token: str) -> str:
    decomposed = normalize("NFKD", token.replace("’", "'").casefold())
    return "".join(
        character for character in decomposed if not combining(character)
    )


def _tokens(text: str) -> list[str]:
    return [_normalize_token(match.group()) for match in _WORD_RE.finditer(text)]


def _raw_tokens(text: str) -> list[str]:
    return [match.group() for match in _WORD_RE.finditer(text)]


def _is_content(token: str) -> bool:
    return len(token) >= 2 and not token.isdigit() and token not in _FUNCTION_WORDS


@dataclass
class _Form:
    headlines: set[int]
    mentions: int
    surfaces: dict[str, int]


def _collect_forms(sentences: list[str]) -> dict[str, _Form]:
    forms: dict[str, _Form] = {}
    for index, title in enumerate(sentences):
        words = _tokens(title)
        raw = _raw_tokens(title)
        if len(raw) != len(words):
            raw = words
        for size in range(1, MAX_TERM_WORDS + 1):
            for start in range(len(words) - size + 1):
                window = words[start:start + size]
                if not all(_is_content(word) for word in window):
                    continue
                form = " ".join(window)
                surface = " ".join(raw[start:start + size])
                entry = forms.setdefault(form, _Form(set(), 0, {}))
                entry.headlines.add(index)
                entry.mentions += 1
                entry.surfaces[surface] = entry.surfaces.get(surface, 0) + 1
    return forms


def _unit_vectors(rows) -> np.ndarray:
    vectors = np.asarray(rows, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return np.zeros((0, 0))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0.0, 1.0, norms)


def _concept_groups(
    names: list[str],
    vectors: np.ndarray,
    forms: dict[str, _Form],
    threshold: float,
) -> list[list[str]]:
    """Greedily attach each form to the closest existing concept.

    Forms are visited strongest first and a match is only accepted against an
    existing concept's centroid. This avoids the transitive chaining that
    single-linkage clustering produces on noisy word embeddings.
    """
    order = sorted(
        range(len(names)),
        key=lambda index: (
            -len(forms[names[index]].headlines),
            -forms[names[index]].mentions,
            len(names[index].split()),
            len(names[index]),
            names[index],
        ),
    )
    groups: list[list[int]] = []
    centroids: list[np.ndarray] = []
    for index in order:
        vector = vectors[index]
        best = -1
        best_similarity = threshold
        for group_index, centroid in enumerate(centroids):
            similarity = float(vector @ centroid)
            if similarity >= best_similarity:
                best_similarity = similarity
                best = group_index
        if best < 0:
            groups.append([index])
            centroids.append(vector.copy())
            continue
        groups[best].append(index)
        member_vectors = [vectors[member] for member in groups[best]]
        centroid = np.mean(member_vectors, axis=0)
        norm = float(np.linalg.norm(centroid))
        centroids[best] = centroid / norm if norm else centroid

    return [
        [names[index] for index in group]
        for group in groups
    ]


def _merge_concepts(
    forms: dict[str, _Form], encode, threshold: float
) -> list[list[str]]:
    names = list(forms)
    if len(names) < 2:
        return [[name] for name in names]
    vectors = _unit_vectors(encode(names))
    if len(vectors) != len(names):
        return [[name] for name in names]
    return _concept_groups(names, vectors, forms, threshold)


@dataclass
class _Concept:
    display: str
    tokens: tuple[str, ...]
    headlines: frozenset[int]
    mentions: int


def _display_surface(form: str, surfaces: dict[str, int]) -> str:
    if not surfaces:
        return form
    return max(surfaces, key=lambda surface: (surfaces[surface], len(surface)))


def _concepts(groups: list[list[str]], forms: dict[str, _Form]) -> list[_Concept]:
    concepts = []
    for group in groups:
        headlines: set[int] = set()
        mentions = 0
        for form in group:
            headlines |= forms[form].headlines
            mentions += forms[form].mentions
        representative = max(
            group,
            key=lambda form: (
                len(forms[form].headlines),
                forms[form].mentions,
                -len(form.split()),
                -len(form),
                form,
            ),
        )
        concepts.append(
            _Concept(
                display=_display_surface(
                    representative, forms[representative].surfaces
                ),
                tokens=tuple(representative.split()),
                headlines=frozenset(headlines),
                mentions=mentions,
            )
        )
    return concepts


def _ranked(concepts: list[_Concept]) -> list[_Concept]:
    return sorted(
        concepts,
        key=lambda concept: (
            -len(concept.headlines),
            -concept.mentions,
            -len(concept.tokens),
            concept.display,
        ),
    )


def _suppress_overlapping(concepts: list[_Concept]) -> list[_Concept]:
    """Collapse nested concepts into their most specific member.

    When every word of one term reappears inside a longer term ("chef" inside
    "Ex chef"), showing both repeats the same words on screen. The longer
    phrase contains all of the shorter one's words, so the longer phrase is
    the member worth keeping, even when the bare word is the more frequent.
    """
    kept: list[_Concept] = []
    for concept in concepts:
        tokens = set(concept.tokens)
        if any(tokens <= set(other.tokens) for other in kept):
            continue
        kept = [other for other in kept if not set(other.tokens) < tokens]
        kept.append(concept)
    return kept


def _default_encode(texts: list[str]) -> np.ndarray:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        try:
            from config import HF_TOKEN
        except ImportError:
            HF_TOKEN = None
        _model = SentenceTransformer(MODEL_NAME, token=HF_TOKEN)
    return _model.encode(
        list(texts),
        batch_size=32,
        normalize_embeddings=True,
    )


def extract_common_terms(
    titles: list[str],
    *,
    encode=None,
    max_terms: int = MAX_SHARED_TERMS,
    min_headlines: int = MIN_SHARED_HEADLINES,
    similarity_threshold: float = CONCEPT_SIMILARITY_THRESHOLD,
) -> list[str]:
    """Return the strongest terms shared by a cluster's headlines."""
    sentences = [title for title in titles if title and title.strip()]
    if not sentences:
        return []
    forms = _collect_forms(sentences)
    if not forms:
        return []

    encoder = encode or _default_encode
    groups = _merge_concepts(forms, encoder, similarity_threshold)
    concepts = _ranked(_concepts(groups, forms))

    recurring = [
        concept for concept in concepts if len(concept.headlines) >= min_headlines
    ]
    selected = recurring if recurring else concepts[:1]
    selected = _suppress_overlapping(selected)
    return [concept.display for concept in selected[:max_terms]]


def generate_topic_label(titles: list[str], **options) -> str:
    """Legacy entry point: the shared terms assembled for display."""
    terms = extract_common_terms(titles, **options)
    representative = next(
        (title for title in titles if title and title.strip()), ""
    )
    return build_cluster_label(terms, representative)


def _find_span(representative: list[str], tokens: list[str]) -> int | None:
    limit = len(representative) - len(tokens)
    for start in range(limit + 1):
        if representative[start:start + len(tokens)] == tokens:
            return start
    return None


def _select_non_overlapping(spans: list[tuple[int, int, int]]):
    """Keep longest spans first; None when two spans partially overlap."""
    accepted: list[tuple[int, int, int]] = []
    for span in sorted(spans, key=lambda item: (-(item[1] - item[0]), item[0], item[2])):
        start, end, _ = span
        for kept_start, kept_end, _ in accepted:
            if start < kept_end and kept_start < end:
                if kept_start <= start and end <= kept_end:
                    break
                return None
        else:
            accepted.append(span)
    return accepted


def build_cluster_label(terms: list[str], representative: str = "") -> str:
    """Assemble a label from shared terms using the headline's word order.

    This is deliberately conservative: it only reorders when every term is
    found in the representative headline and the spans do not conflict.
    Otherwise it returns the plain joined-term string, so a label is either a
    clean subsequence of a real headline or the same term list as before.
    """
    if not terms:
        return representative.strip()
    representative_tokens = _tokens(representative)
    spans: list[tuple[int, int, int]] = []
    for index, term in enumerate(terms):
        tokens = _tokens(term)
        start = _find_span(representative_tokens, tokens) if tokens else None
        if start is None:
            return " · ".join(terms)
        spans.append((start, start + len(tokens), index))

    accepted = _select_non_overlapping(spans)
    if accepted is None:
        return " · ".join(terms)
    ordered = sorted(accepted, key=lambda span: span[0])
    return " ".join(terms[index] for _, _, index in ordered)

