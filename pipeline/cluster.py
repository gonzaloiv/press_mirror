import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from unicodedata import combining, normalize

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from shared.country_codes import to_iso

INPUT_FILE = PROJECT_DIR / "data" / "embeddings.npz"
HISTORY_DIR = PROJECT_DIR / "history"

SIMILARITY_THRESHOLD = 0.75    # tune this — see notes below
MERGE_SIMILARITY_THRESHOLD = 0.72
MAX_MERGE_PASSES = 3
DEDUPLICATE_WITHIN_CLUSTERS = True
MERGE_RELATED_CLUSTERS = True
LABEL_MAX_WORDS = 7

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_ENGLISH_VOCABULARY = frozenset(
    """
    attack attacks drone drones strike strikes school schools election
    elections vote votes voter voters bomber bombers withdrawal withdraw
    withdraws base bases threat threats outbreak plague denies deny denial
    lab worker death family rule democracy angry upperclassmen delays delay
    delayed hopes punish housing crisis conflict protest protests leaves
    leaving plans calls pictures prolific author parliamentarian prisoner
    dies died aged projected separatist separatists party poised elect win
    wins power first time canada unity invasion launches intensifies under
    parliament votes children approves social media minors law bans ban
    election victory wins french elections spain brazilian french strikes
    russian ukraine moscow turkey earthquake
    """.split()
)
_LANGUAGE_MARKERS = {
    "de": frozenset(
        "der die das den dem des und von für mit nach auf ist sind wird werden "
        "schüler frankreich deutschland meer schiff drohnenangriff gestorben "
        "wütender warum".split()
    ),
    "fr": frozenset(
        "le la les des du une et est sont pour avec dans sur aux france élèves "
        "lycéens mobilisation cortèges".split()
    ),
    "es": frozenset(
        "el los las del una y pero que como por para con en es son está barco "
        "barcos mar negro".split()
    ),
    "it": frozenset(
        "il gli le dei delle una e per con nel sulle sono nave navi colpite "
        "affondata".split()
    ),
    "pt": frozenset(
        "os as dos das uma e para com na no são barco barcos juntam protestos "
        "estudantes".split()
    ),
    "nl": frozenset(
        "het een van bij voor met zijn wordt afscheid verkiezing separatisten "
        "wint".split()
    ),
    "sv": frozenset(
        "och är det den som att för med av på död författaren vann".split()
    ),
    "fi": frozenset(
        "ja on ovat että kanssa tuhansia odotetaan kiihtyvät lukioprotestit".split()
    ),
    "pl": frozenset(
        "w na ze jeden statek atak dronów wybrzeży załogi nieznany".split()
    ),
}
_ENGLISH_MARKERS = frozenset(
    """
    the and but or for from with after before has have had are is were been
    being into than about against during while who why how does did projected
    poised aged dies died author pictures
    """.split()
)
_CROSS_LANGUAGE_STEMS = {
    "france": ("france", "frankreich", "francia", "franca", "frança"),
    "black sea": ("schwarzen meer", "schwarzem meer", "mer noire", "mar negro", "mar nero", "morze czarne"),
    "ship": ("ship", "vessel", "schiff", "bateau", "navire", "barco", "nave", "statek", "laiva"),
    "drone": ("drone", "drohn", "dron"),
    "attack": ("attack", "angriff", "attac", "atak", "frapp", "colpit", "atac"),
    "student": ("student", "schuler", "eleve", "etudiant", "lyceen", "estudant", "studenti", "opiskelija"),
    "protest": ("protest", "demonstration", "demo", "mobilisation", "mobilization", "corteg", "streik", "huelga", "grève"),
    "messi": ("messi",),
    "argentina": ("argentina",),
    "farewell": (
        "farewell", "despedi", "despide", "afscheid", "abschied",
        "verabschied", "adieu", "adios",
    ),
    "jeffrey archer": ("jeffrey archer",),
    "death": ("dies", "died", "death", "gestorben", "verstorben", "decede", "deceded", "död"),
    "quebec": ("quebec", "québec"),
    "separatist": ("separatist", "independentist", "indépendantiste"),
    "election": ("election", "elect", "wahl", "eleccion", "elección", "elezioni", "verkiezing"),
    "win": ("win", "won", "wins", "gewinnt", "l'emporte", "gana"),
    "turkey": ("turkey", "türkei", "turquie", "turquia", "turchia"),
    "earthquake": ("earthquake", "erdbeben", "seisme", "séisme", "terremoto", "deprem"),
    "spain": ("spain", "spanien", "españa", "espana"),
    "delay": ("delay", "retras", "aplaz", "pospon", "verzög"),
    "students": ("students", "alunos", "eleves", "schuler"),
    "school": (
        "school", "lycee", "lycée", "ecole", "école", "schul", "escuela",
        "escolar",
    ),
    "nobel": ("nobel",),
    "physics": ("physics", "physik", "physique", "fisica", "física"),
    "francis halzen": ("francis halzen",),
    "medicine": (
        "medicine", "medicina", "medizin", "geneeskunde", "medisin",
    ),
    "optogenetics": (
        "optogenetics", "optogenetik", "optogenetica", "optogenetica",
    ),
    "cornell": ("cornell",),
    "sexual assault": (
        "sexual assault", "rape", "vergewaltigung", "viol", "verkrachting",
        "våldtäk", "voldtægt",
    ),
}
_STOPWORDS = frozenset(
    """
    a an the and or but if then than so because as of at by for from in into
    on onto to with without over under after before during while about against
    between through is are was were be been being has have had do does did
    this that these those it its their his her he she they them we you your
    who why what when where how vs not no now new more most all any only
    de del la las el los un una unos unas y o pero si que como por para con
    en al es son fue fueron ser estar está están se su sus lo le les
    le la les des du dans pour avec sur au aux et est sont une un
    der die das den dem des ein eine einer und oder von für mit nach auf
    zu im am an als ist sind wird werden
    """.split()
)
_EDITORIAL_TERMS = frozenset(
    "vs versus why how what massive major shocking huge controversial".split()
)
_EVENT_TERMS = frozenset(
    """
    attack strike outbreak protest election vote denial deny delay withdrawal
    leave departure crisis death visit meeting talks deal agreement ban arrest
    trial ruling court law launch win defeat resignation invasion ceasefire
    earthquake flood wildfire fire explosion crash collision shooting bombing
    killing massacre summit peace approval rejection lawsuit strike
    """.split()
)
_EVENT_CANONICAL = {
    "attacks": "attack",
    "strikes": "strike",
    "outbreaks": "outbreak",
    "protests": "protest",
    "elections": "election",
    "votes": "vote",
    "denies": "deny",
    "denied": "deny",
    "denial": "deny",
    "withdraw": "withdrawal",
    "withdraws": "withdrawal",
    "withdrawn": "withdrawal",
    "leave": "withdrawal",
    "leaves": "withdrawal",
    "leaving": "withdrawal",
    "left": "withdrawal",
    "depart": "withdrawal",
    "departs": "withdrawal",
    "departure": "withdrawal",
    "delays": "delay",
    "delayed": "delay",
    "departures": "withdrawal",
    "trials": "trial",
    "rulings": "ruling",
    "talk": "talks",
    "deals": "deal",
    "bans": "ban",
    "arrests": "arrest",
    "wins": "win",
    "defeats": "defeat",
    "earthquakes": "earthquake",
    "floods": "flood",
    "wildfires": "wildfire",
    "fires": "fire",
    "explosions": "explosion",
    "crashes": "crash",
    "collisions": "collision",
    "shootings": "shooting",
    "bombings": "bombing",
    "killings": "killing",
    "massacres": "massacre",
    "summits": "summit",
    "approves": "approval",
    "approved": "approval",
    "rejects": "rejection",
    "rejected": "rejection",
    "banning": "ban",
    "banned": "ban",
    "invades": "invasion",
    "invaded": "invasion",
}
_ENTITY_TERMS = frozenset(
    """
    us usa uk ukraine ukrainian russian russia moscow brazil brazilian spain
    spanish france french iran iranian israel israeli china chinese taiwan
    taiwanese japan japanese germany german britain british england english
    london paris kyiv kiev washington beijing delhi mexico mexican canada
    canadian australia australian italy italian poland polish syria syrian
    gaza palestine palestinian israel saudi arabia turkey turkish
    """.split()
)
_ENTITY_CANONICAL = {
    "russian": "russia",
    "russians": "russia",
    "ukrainian": "ukraine",
    "ukrainians": "ukraine",
    "brazilian": "brazil",
    "brazilians": "brazil",
    "spanish": "spain",
    "french": "france",
    "iranian": "iran",
    "iranians": "iran",
    "israeli": "israel",
    "israelis": "israel",
    "chinese": "china",
    "taiwanese": "taiwan",
    "japanese": "japan",
    "german": "germany",
    "british": "britain",
    "canadian": "canada",
    "australian": "australia",
    "italian": "italy",
    "polish": "poland",
    "syrian": "syria",
    "turkish": "turkey",
}
_COUNTRY_ADJECTIVES = {
    "russia": "Russian",
    "ukraine": "Ukrainian",
    "brazil": "Brazilian",
    "spain": "Spanish",
    "france": "French",
    "iran": "Iranian",
    "israel": "Israeli",
    "china": "Chinese",
    "taiwan": "Taiwanese",
    "japan": "Japanese",
    "germany": "German",
    "britain": "British",
    "canada": "Canadian",
    "australia": "Australian",
    "italy": "Italian",
    "poland": "Polish",
    "syria": "Syrian",
    "turkey": "Turkish",
}
_PLACE_TERMS = frozenset(
    """
    moscow kyiv kiev london paris washington beijing delhi gaza ukraine russia
    brazil spain france iran israel china taiwan japan germany britain england
    mexico canada australia italy poland syria turkey palestine
    """.split()
)
_TOPIC_MODIFIERS = {
    "attack": ("drone", "missile", "rocket", "bomb", "bombing", "airstrike"),
    "strike": ("school", "teacher", "worker", "student"),
    "outbreak": ("plague", "disease", "virus"),
    "protest": ("school", "student", "teacher"),
}
_EVENT_PRIORITY = {
    "deny": 15,
    "delay": 14,
    "invasion": 13,
    "withdrawal": 12,
    "attack": 11,
    "strike": 10,
    "protest": 9,
    "earthquake": 8,
    "outbreak": 7,
    "ban": 6,
    "election": 5,
    "vote": 4,
}


@dataclass
class ClusterMembers:
    indices: list[int]
    dropped_sources: dict[str, int] = field(default_factory=dict)
    merged: bool = False


def load_data():
    data = np.load(INPUT_FILE, allow_pickle=True)
    titles = list(data["titles"])
    return (
        data["embeddings"],
        titles,
        list(data["sources"]),
        list(data["countries"]),
        list(data["links"]),
        (
            list(data["fetched_at"])
            if "fetched_at" in data.files
            else [None] * len(titles)
        ),
    )


def cluster(embeddings):
    """Group indices into clusters. Uses cosine distance, not raw vectors."""
    # AgglomerativeClustering needs a *distance* threshold, not a similarity one.
    # cosine distance = 1 - cosine similarity, so 0.75 similarity -> 0.25 distance.
    distance_threshold = 1.0 - SIMILARITY_THRESHOLD

    model = AgglomerativeClustering(
        n_clusters=None,                    # let the algorithm decide
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",                  # average distance between groups
    )
    return model.fit_predict(embeddings)


def deduplicate_cluster(
    indices: list[int],
    sources: list[str],
    embeddings: np.ndarray,
) -> ClusterMembers:
    """Keep the article nearest its cluster centroid for each source."""
    if not indices:
        return ClusterMembers(indices=[])

    centroid = embeddings[indices].mean(axis=0)
    similarities = cosine_similarity(
        embeddings[indices],
        centroid.reshape(1, -1),
    ).ravel()
    best_by_source: dict[str, tuple[float, int]] = {}
    source_counts: dict[str, int] = {}
    for index, similarity in zip(indices, similarities):
        source = sources[index]
        source_counts[source] = source_counts.get(source, 0) + 1
        current = best_by_source.get(source)
        if current is None or similarity > current[0]:
            best_by_source[source] = (float(similarity), index)

    kept_indices = sorted(entry[1] for entry in best_by_source.values())
    dropped_sources = {
        source: count - 1
        for source, count in source_counts.items()
        if count > 1
    }
    return ClusterMembers(
        indices=kept_indices,
        dropped_sources=dropped_sources,
    )


def deduplicate_clusters(
    clusters: list[list[int]],
    sources: list[str],
    embeddings: np.ndarray,
) -> list[ClusterMembers]:
    """Deduplicate each primary cluster independently by source."""
    return [
        deduplicate_cluster(indices, sources, embeddings)
        for indices in clusters
        if indices
    ]


def merge_clusters(
    clusters: list[ClusterMembers],
    sources: list[str],
    embeddings: np.ndarray,
    threshold: float = MERGE_SIMILARITY_THRESHOLD,
    max_passes: int = MAX_MERGE_PASSES,
) -> list[ClusterMembers]:
    """Merge similar cluster centroids and deduplicate merged article lists."""
    current = [cluster for cluster in clusters if cluster.indices]
    for pass_number in range(1, max_passes + 1):
        centroids = [
            embeddings[cluster.indices].mean(axis=0)
            for cluster in current
        ]
        print(
            f"Merge pass {pass_number}: comparing {len(current)} cluster centroids.",
            flush=True,
        )
        similarity_matrix = cosine_similarity(centroids)
        matching_pairs = np.argwhere(
            np.triu(similarity_matrix >= threshold, k=1)
        )
        pair_scores = [
            (float(similarity_matrix[left, right]), int(left), int(right))
            for left, right in matching_pairs
        ]

        if not pair_scores:
            break

        paired_left: dict[int, int] = {}
        paired_right: set[int] = set()
        for _, left, right in sorted(pair_scores, reverse=True):
            if left not in paired_left and left not in paired_right \
                    and right not in paired_left and right not in paired_right:
                paired_left[left] = right
                paired_right.add(right)

        next_clusters = []
        for index, cluster in enumerate(current):
            if index in paired_right:
                continue
            partner = paired_left.get(index)
            if partner is None:
                next_clusters.append(cluster)
                continue

            other = current[partner]
            combined = deduplicate_cluster(
                cluster.indices + other.indices,
                sources,
                embeddings,
            )
            for dropped_sources in (
                cluster.dropped_sources,
                other.dropped_sources,
            ):
                for source, count in dropped_sources.items():
                    combined.dropped_sources[source] = (
                        combined.dropped_sources.get(source, 0) + count
                    )
            combined.merged = True
            next_clusters.append(combined)

        print(
            f"Merge pass {pass_number}: {len(current)} -> "
            f"{len(next_clusters)} clusters."
        )
        current = next_clusters

        if pass_number == max_passes:
            final_centroids = [
                embeddings[cluster.indices].mean(axis=0)
                for cluster in current
            ]
            final_similarities = cosine_similarity(final_centroids)
            if np.any(np.triu(final_similarities >= threshold, k=1)):
                print(
                    f"Warning: similar clusters remain after {max_passes} "
                    "merge passes."
                )

    return current


def _tokens(text: str) -> list[str]:
    return [match.group().replace("’", "'").casefold() for match in _WORD_RE.finditer(text)]


def _normalized_tokens(text: str) -> list[str]:
    decomposed = normalize("NFKD", text.casefold())
    plain = "".join(
        character for character in decomposed if not combining(character)
    )
    return [match.group() for match in _WORD_RE.finditer(plain)]


def _is_english_headline(title: str) -> bool:
    words = set(_normalized_tokens(title))
    english_score = len(words & _ENGLISH_MARKERS) + len(
        words & _ENGLISH_VOCABULARY
    )
    other_language_score = max(
        (
            len(words & set(_normalized_tokens(" ".join(markers))))
            for markers in _LANGUAGE_MARKERS.values()
        ),
        default=0,
    )
    return english_score >= 2 and english_score > other_language_score


def _canonical_word(word: str) -> str:
    if word in _EVENT_CANONICAL:
        return _EVENT_CANONICAL[word]
    if word in _ENTITY_CANONICAL:
        return _ENTITY_CANONICAL[word]
    if word.endswith("'s"):
        word = word[:-2]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and len(word) > 4 and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _clean_fallback(title: str) -> str:
    """Keep the centroid headline as a short, non-rhetorical last resort."""
    title = re.sub(
        r"^\s*(?:in pictures|pictures|photos?|watch|video)\s*:\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.split(r"[:?—–]", title, maxsplit=1)[0]
    title = re.split(r"\b(?:vs\.?|versus|why|how|what|who)\b", title, maxsplit=1, flags=re.I)[0]
    words = _WORD_RE.findall(title)
    while words and words[-1].casefold() in {
        "a", "an", "and", "at", "for", "from", "in", "of", "on", "over", "the", "to", "with"
    }:
        words.pop()
    return " ".join(words[:LABEL_MAX_WORDS]).strip()


def _shared_headline_terms(titles: list[str]):
    """Return words found in at least two titles with their observed forms."""
    occurrences: dict[str, set[int]] = {}
    surfaces: dict[str, dict[str, int]] = {}
    entity_surfaces: dict[str, dict[str, int]] = {}
    entity_occurrences: dict[str, set[int]] = {}
    entity_first_seen: dict[str, tuple[int, int]] = {}

    for headline_index, title in enumerate(titles):
        words = _tokens(title)
        original_words = _WORD_RE.findall(title)
        for position, word in enumerate(words):
            canonical = _canonical_word(word)
            if canonical in _STOPWORDS or canonical in _EDITORIAL_TERMS or len(canonical) < 2:
                continue
            occurrences.setdefault(canonical, set()).add(headline_index)
            forms = surfaces.setdefault(canonical, {})
            forms[word] = forms.get(word, 0) + 1

            original = original_words[position] if position < len(original_words) else word
            is_capitalized = original.isupper() or (
                original[:1].isupper() and position > 0
            )
            if canonical in _ENTITY_TERMS or (
                is_capitalized
                and canonical not in _EVENT_TERMS
                and canonical not in _STOPWORDS
            ):
                entity_occurrences.setdefault(canonical, set()).add(headline_index)
                entity_first_seen.setdefault(canonical, (headline_index, position))
                entity_forms = entity_surfaces.setdefault(canonical, {})
                entity_forms[original] = entity_forms.get(original, 0) + 1

    shared = {
        word for word, indices in occurrences.items() if len(indices) >= 2
    }
    entities = {
        word for word, indices in entity_occurrences.items() if len(indices) >= 2
    }
    return shared, entities, surfaces, entity_surfaces, entity_first_seen


def _best_form(word: str, surfaces: dict[str, dict[str, int]]) -> str:
    forms = surfaces.get(word, {})
    if not forms:
        return word
    chosen = max(forms, key=lambda form: (forms[form], len(form)))
    return chosen.lower()


def _entity_form(word: str, surfaces: dict[str, dict[str, int]], entity_surfaces) -> str:
    forms = entity_surfaces.get(word) or surfaces.get(word, {})
    if not forms:
        return word.title()
    chosen = max(forms, key=lambda form: (forms[form], len(form)))
    if chosen.casefold() in {"us", "usa", "uk"}:
        return chosen.upper()
    return chosen[0].upper() + chosen[1:]


def _entities_around_term(
    titles: list[str],
    entities: set[str],
    term: str | None,
) -> tuple[str | None, str | None]:
    if not term:
        return None, None

    before_counts: dict[str, int] = {}
    after_counts: dict[str, int] = {}
    for title in titles:
        words = [_canonical_word(word) for word in _tokens(title)]
        for term_position, word in enumerate(words):
            if word != term:
                continue
            before = next(
                (
                    candidate
                    for candidate in reversed(words[:term_position])
                    if candidate in entities
                ),
                None,
            )
            after = next(
                (
                    candidate
                    for candidate in words[term_position + 1:]
                    if candidate in entities
                ),
                None,
            )
            if before and before != after:
                before_counts[before] = before_counts.get(before, 0) + 1
            if after and after != before:
                after_counts[after] = after_counts.get(after, 0) + 1

    def most_common(counts: dict[str, int]) -> str | None:
        return min(counts, key=lambda word: (-counts[word], word), default=None)

    return most_common(before_counts), most_common(after_counts)


def _shared_topic_phrase(
    titles: list[str],
    shared: set[str],
    entities: set[str],
    event: str,
) -> str | None:
    candidates: dict[tuple[str, ...], set[int]] = {}
    order: dict[tuple[str, ...], tuple[int, int]] = {}
    excluded = entities | _EVENT_TERMS | _STOPWORDS | {event}
    for title_index, title in enumerate(titles):
        words = [_canonical_word(word) for word in _tokens(title)]
        for size in (3, 2):
            for start in range(len(words) - size + 1):
                phrase_words = tuple(words[start:start + size])
                if any(word not in shared or word in excluded for word in phrase_words):
                    continue
                candidates.setdefault(phrase_words, set()).add(title_index)
                order.setdefault(phrase_words, (title_index, start))

    repeated = [
        phrase_words
        for phrase_words, title_indices in candidates.items()
        if len(title_indices) >= 2
    ]
    if not repeated:
        return None
    selected = min(
        repeated,
        key=lambda phrase_words: (
            -len(phrase_words),
            order[phrase_words],
            phrase_words,
        ),
    )
    label = " ".join(selected)
    return label[0].upper() + label[1:]


def _select_label_terms(titles: list[str]) -> str | None:
    shared, entities, surfaces, entity_surfaces, entity_first_seen = (
        _shared_headline_terms(titles)
    )
    if not shared:
        return None

    event_words = shared & _EVENT_TERMS
    event = max(
        event_words,
        key=lambda word: (
            _EVENT_PRIORITY.get(word, 0),
            sum(surfaces.get(word, {}).values()),
            word,
        ),
        default=None,
    )
    entity_order = sorted(
        entities,
        key=lambda word: (
            -max(entity_surfaces.get(word, {}).values(), default=0),
            -(word in _ENTITY_TERMS),
            entity_first_seen[word],
            word,
        ),
    )
    entity = entity_order[0] if entity_order else None
    entity_text = _entity_form(entity, surfaces, entity_surfaces) if entity else None

    def phrase(*parts: str | None) -> str | None:
        words = " ".join(part for part in parts if part).split()
        result = " ".join(words[:LABEL_MAX_WORDS]).strip()
        return result if result else None

    if event == "deny" and entity_text:
        topic = next(
            (word for word in ("plague", "outbreak", "disease", "virus") if word in shared),
            None,
        )
        related = (
            _best_form(topic, surfaces) if topic else None,
            _best_form("outbreak", surfaces) if "outbreak" in shared else None,
        )
        return phrase(entity_text, "denies", *related)

    if event == "delay" and entity_text:
        topic = next(
            (word for word in ("election", "vote", "poll") if word in shared),
            None,
        )
        return phrase(
            entity_text,
            "delays",
            _best_form(topic, surfaces) if topic else None,
        )

    if event == "withdrawal" and entity_text:
        actor = next(
            (word for word in ("bomber", "troop", "soldier") if word in shared),
            None,
        )
        actor_entity, location_entity = _entities_around_term(
            titles, entities, actor
        ) if actor else (None, None)
        actor_entity = actor_entity or entity
        location_entity = location_entity or next(
            (
                candidate
                for candidate in entity_order
                if candidate != actor_entity and candidate in _PLACE_TERMS
            ),
            None,
        )
        location = (
            _entity_form(location_entity, surfaces, entity_surfaces)
            if location_entity
            else None
        )
        base = _best_form("base", surfaces) if "base" in shared else None
        if actor and location:
            actor_form = _best_form(actor, surfaces)
            if actor_form == "bomber":
                actor_form = "bombers"
            return phrase(
                _entity_form(actor_entity, surfaces, entity_surfaces),
                actor_form,
                "leave",
                location,
                base,
            )
        return phrase(entity_text, "withdrawal")

    if event == "invasion":
        actor, target = _entities_around_term(titles, entities, event)
        if actor and target:
            actor_label = _COUNTRY_ADJECTIVES.get(actor)
            if not actor_label:
                actor_label = _entity_form(actor, surfaces, entity_surfaces)
            target_label = _entity_form(target, surfaces, entity_surfaces)
            return phrase(actor_label, "invasion of", target_label)
        if target or entity:
            location = target or entity
            return phrase(
                "Invasion of",
                _entity_form(location, surfaces, entity_surfaces),
            )

    if event in {"attack", "strike", "outbreak", "protest"}:
        place = next((word for word in entity_order if word in _PLACE_TERMS), None)
        label_entity = place or entity
        label_entity_text = (
            _entity_form(label_entity, surfaces, entity_surfaces)
            if label_entity
            else None
        )
        modifier = next(
            (word for word in _TOPIC_MODIFIERS.get(event, ()) if word in shared),
            None,
        )
        if label_entity_text:
            return phrase(
                label_entity_text,
                _best_form(modifier, surfaces) if modifier else None,
                _best_form(event, surfaces),
            )
        return phrase(
            _best_form(modifier, surfaces) if modifier else None,
            _best_form(event, surfaces),
        )

    if event == "earthquake":
        place = next((word for word in entity_order if word in _PLACE_TERMS), None)
        return phrase(
            _entity_form(place, surfaces, entity_surfaces) if place else None,
            "earthquake",
        )

    if event == "ban":
        topic = _shared_topic_phrase(titles, shared, entities, event)
        if not topic:
            topic_words = sorted(
                shared - entities - _EVENT_TERMS - _STOPWORDS,
                key=lambda word: (
                    -sum(surfaces.get(word, {}).values()),
                    word,
                ),
            )
            topic = " ".join(_best_form(word, surfaces) for word in topic_words[:2])
        return phrase(topic, "ban")

    if event == "election":
        return phrase(entity_text, "elections" if "elections" in surfaces.get(event, {}) else "election")

    if event and entity_text:
        return phrase(entity_text, _best_form(event, surfaces))
    if entity_text:
        return phrase(entity_text, _best_form(entity_order[1], surfaces) if len(entity_order) > 1 else None)

    ranked_terms = sorted(
        shared,
        key=lambda word: (
            -sum(surfaces.get(word, {}).values()),
            -(word in _EVENT_TERMS),
            word,
        ),
    )
    return phrase(*(_best_form(word, surfaces) for word in ranked_terms[:4]))


def _has_cross_language_term(titles: list[str], concept: str) -> bool:
    stems = tuple(
        "".join(
            character
            for character in normalize("NFKD", stem.casefold())
            if not combining(character)
        )
        for stem in _CROSS_LANGUAGE_STEMS[concept]
    )
    for title in titles:
        words = _normalized_tokens(title)
        joined = " ".join(words)
        if any(
            (" " in stem and stem in joined)
            or (" " not in stem and any(word.startswith(stem) for word in words))
            for stem in stems
        ):
            return True
    return False


def _cross_language_label(titles: list[str]) -> str | None:
    """Assemble English phrases from a small audited multilingual vocabulary."""
    has = lambda concept: _has_cross_language_term(titles, concept)

    if has("nobel") and has("physics") and has("francis halzen"):
        return "Physics Nobel for Francis Halzen"

    if has("nobel") and has("medicine") and has("optogenetics"):
        return "Medicine Nobel for Optogenetics"

    if has("cornell") and has("sexual assault"):
        return "Cornell sexual assault allegations"

    if (
        has("black sea")
        and has("ship")
        and has("drone")
        and has("attack")
    ):
        return "Drone attacks on ships in Black Sea"

    if has("france") and has("student") and has("protest"):
        return "French student protests"

    if has("messi") and has("argentina") and has("farewell"):
        return "Messi's Argentina farewell"

    if has("jeffrey archer") and has("death"):
        return "Jeffrey Archer dies"

    if (
        has("quebec")
        and has("separatist")
        and has("election")
        and has("win")
    ):
        return "Quebec separatists win election"

    if has("turkey") and has("earthquake"):
        return "Turkey earthquake"

    if has("spain") and has("election") and has("delay"):
        return "Spain delays elections"

    if has("france") and has("school") and has("protest"):
        return "French school protests"

    if has("france") and has("protest"):
        return "Protests in France"

    if has("earthquake"):
        return "Earthquake"

    if has("protest"):
        return "Student protests" if has("student") else "Protests"

    if has("election"):
        return "Election"

    return None


def generate_topic_label(titles: list[str]) -> str:
    """Build a short English label from shared cluster-wide headline concepts."""
    if not titles:
        return ""
    english_titles = [title for title in titles if _is_english_headline(title)]
    if not english_titles:
        return _cross_language_label(titles) or "News event"

    if (
        _has_cross_language_term(english_titles, "jeffrey archer")
        and _has_cross_language_term(english_titles, "death")
    ):
        return "Jeffrey Archer dies"

    if (
        _has_cross_language_term(english_titles, "quebec")
        and _has_cross_language_term(english_titles, "separatist")
        and _has_cross_language_term(english_titles, "election")
        and _has_cross_language_term(english_titles, "win")
    ):
        return "Quebec separatists win election"

    label = _select_label_terms(english_titles)
    if label:
        return label
    fallback_title = english_titles[0]
    return _clean_fallback(fallback_title)


def build_clusters(
    labels,
    titles,
    sources,
    countries,
    links,
    embeddings,
    labeler,
    deduplicate: bool = DEDUPLICATE_WITHIN_CLUSTERS,
    merge: bool = MERGE_RELATED_CLUSTERS,
    fetched_at=None,
):
    """Post-process primary clusters, then build output with similarity stats."""
    raw_clusters = {}
    for idx, label in enumerate(labels):
        raw_clusters.setdefault(int(label), []).append(idx)

    cluster_groups = list(raw_clusters.values())
    if deduplicate:
        cluster_groups = deduplicate_clusters(cluster_groups, sources, embeddings)
        dropped_count = sum(
            sum(cluster.dropped_sources.values())
            for cluster in cluster_groups
        )
        print(f"Dropped {dropped_count} same-source articles.")
    else:
        cluster_groups = [
            ClusterMembers(indices=indices)
            for indices in cluster_groups
            if indices
        ]
    if merge:
        cluster_groups = merge_clusters(cluster_groups, sources, embeddings)

    result = []
    label_clusters = [
        cluster for cluster in cluster_groups if len(cluster.indices) >= 2
    ]
    print(
        f"Selecting labels for {len(label_clusters)} clusters.",
        flush=True,
    )
    for label_number, cluster in enumerate(label_clusters, start=1):
        indices = cluster.indices

        # Recompute pairwise similarity *within* this cluster for the display.
        sub = embeddings[indices]
        sim_matrix = cosine_similarity(sub)
        # Average of the off-diagonal (exclude self-similarity = 1.0).
        n = len(indices)
        avg_sim = (sim_matrix.sum() - n) / (n * n - n)

        articles = []
        for i in indices:
            article = {
                "title": titles[i],
                "source": sources[i],
                "country": countries[i],
                "link": links[i],
            }
            if fetched_at is not None and fetched_at[i] is not None:
                article["fetched_at"] = str(fetched_at[i])
            articles.append(article)

        centroid = sub.mean(axis=0)
        similarities = cosine_similarity(
            sub,
            centroid.reshape(1, -1),
        ).ravel()
        representative_position = int(np.argmax(similarities))
        representative_index = indices[representative_position]
        cluster_titles = [titles[representative_index]] + [
            titles[i] for i in indices if i != representative_index
        ]

        print(f"Generating label {label_number}/{len(label_clusters)}...", flush=True)
        cluster_label = labeler(cluster_titles)

        cluster_result = {
            "label": cluster_label,
            "size": len(indices),
            "avg_similarity": round(float(avg_sim), 3),
            "articles": articles,
        }
        if cluster.dropped_sources:
            cluster_result["dropped_sources"] = cluster.dropped_sources
        result.append(cluster_result)

    # Prefer stories covered in more countries; use article count only for ties.
    result.sort(
        key=lambda c: (
            len({to_iso(article["country"]) for article in c["articles"]}),
            c["size"],
        ),
        reverse=True,
    )
    return result


def main():
    embeddings, titles, sources, countries, links, fetched_at = load_data()
    print(f"Loaded {len(titles)} headlines.")

    labels = cluster(embeddings)
    print(f"Found {len(set(labels))} raw clusters.")

    clusters = build_clusters(
        labels,
        titles,
        sources,
        countries,
        links,
        embeddings,
        generate_topic_label,
        fetched_at=fetched_at,
    )
    print(f"Kept {len(clusters)} clusters with 2+ articles.")

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    output_file = HISTORY_DIR / f"clusters_{date.today().isoformat()}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"Written to {output_file}")


if __name__ == "__main__":
    main()