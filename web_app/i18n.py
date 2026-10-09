"""Localized UI copy for the editions.

Templates read a flat ``strings`` mapping for chrome, plus structured about-page
sections. Values may contain the ``{app}`` placeholder, which the server fills
with the site name before rendering.
"""

ABOUT_TITLE = {
    "en": "How {app} works",
    "es": "Cómo funciona {app}",
}


def _plural(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def coverage_line(outlets: int, countries: int, language: str = "en") -> str:
    """Localized "N outlets in M countries" summary for the card image."""
    if language == "es":
        return (
            f"{outlets} {_plural(outlets, 'medio', 'medios')} en "
            f"{countries} {_plural(countries, 'país', 'países')}"
        )
    return (
        f"{outlets} {_plural(outlets, 'outlet', 'outlets')} in "
        f"{countries} {_plural(countries, 'country', 'countries')}"
    )


_UI = {
    "en": {
        "tagline": "What happened today, and who covered it.",
        "how_it_works": "How it works",
        "older_day": "← Older day",
        "newer_day": "Newer day →",
        "collected": "Collected {time}",
        "snapshot_description": (
            "Each card is one event. The map shows which countries covered it."
        ),
        "browse_dates_aria": "Browse snapshot dates",
        "snapshot_nav_aria": "Snapshot navigation",
        "top_stories_aria": "Top stories for {date}",
        "back_all_maps": "← All maps",
        "legend_covered": "Covered",
        "legend_silent": "Not covered",
        "legend_silent_detail": "Not covered by tracked sources",
        "legend_untracked": "Not tracked",
        "map_legend_aria": "Map legend",
        "source_metadata_aria": "Source metadata",
        "more": "+{count} more",
        "footer_disclaimer": (
            "Every headline links to the original article at its original "
            "outlet. {app} does not produce news."
        ),
        "language_switch": "Español",
        "language_switch_aria": "Change language to Spanish",
        "home_link": "← Home",
    },
    "es": {
        "tagline": "Lo que pasó hoy y quién lo cubrió.",
        "how_it_works": "Cómo funciona",
        "older_day": "← Día anterior",
        "newer_day": "Día siguiente →",
        "collected": "Recopilado {time}",
        "snapshot_description": (
            "Cada tarjeta es un acontecimiento. El mapa muestra qué países lo "
            "cubrieron."
        ),
        "browse_dates_aria": "Explorar fechas",
        "snapshot_nav_aria": "Navegación por fechas",
        "top_stories_aria": "Historias destacadas de {date}",
        "back_all_maps": "← Todos los mapas",
        "legend_covered": "Cubierto",
        "legend_silent": "No cubierto",
        "legend_silent_detail": "No cubierto por medios seguidos",
        "legend_untracked": "No seguido",
        "map_legend_aria": "Leyenda del mapa",
        "source_metadata_aria": "Metadatos de las fuentes",
        "more": "+{count} más",
        "footer_disclaimer": (
            "Cada titular enlaza al artículo original en su medio. {app} no "
            "produce noticias."
        ),
        "language_switch": "English",
        "language_switch_aria": "Cambiar el idioma a inglés",
        "home_link": "← Inicio",
    },
}


def ui_strings(language: str, app_name: str) -> dict[str, str]:
    """Return the localized chrome strings with the site name substituted."""
    source = _UI.get(language, _UI["en"])
    return {
        key: value.replace("{app}", app_name)
        for key, value in source.items()
    }


def about_sections(
    language: str,
    app_name: str,
    outlet_count: int,
    country_count: int,
    sources_html: str = "",
) -> list[dict]:
    """Structured, localized about-page content. Blocks are trusted HTML."""
    if language == "es":
        return _about_es(app_name, outlet_count, country_count, sources_html)
    return _about_en(app_name, outlet_count, country_count, sources_html)


def _about_en(
    app_name: str, outlet_count: int, country_count: int, sources_html: str
) -> list[dict]:
    return [
        {
            "heading": f"What {app_name} is",
            "blocks": [
                f"<p>{app_name} groups news articles from different outlets "
                "that cover the same event. Each card on the homepage is one "
                "event, as told by each outlet that published a story about "
                "it. The map shows which countries covered that event — and "
                "which ones didn't.</p>",
            ],
        },
        {
            "heading": "How the grouping works",
            "blocks": [
                f"<p>Every day, {app_name} collects headlines from a fixed "
                "set of European news outlets. It compares each headline with "
                "headlines from other outlets published that day, to see "
                "whether they are about the same event. Headlines about the "
                "same event are grouped into a cluster. When one outlet "
                f"published several stories about the same event, {app_name} "
                "keeps a single one — the story closest to what every outlet "
                "was saying — so each outlet appears once per event.</p>",
                "<p>A cluster is the smallest unit of the product: one event, "
                "many voices. Everything else in the app — the map, the "
                "labels, the source list, the daily selection — is built on "
                "top of clusters.</p>",
                "<p>The grouping compares the meaning of headlines across "
                "languages, not just their words. A French headline and an "
                "English headline about the same event can end up in the same "
                "cluster even when they share no vocabulary.</p>",
            ],
        },
        {
            "heading": "Sources and countries",
            "blocks": [
                f"<p>This is the fixed set: {outlet_count} outlets across "
                f"{country_count} countries, each read every day through its "
                "public RSS feed. The list defines the tracked countries on "
                "the map — a country is tracked when we read at least one "
                "outlet there, and not tracked when we read none. What the map "
                "can say about an event ends where this list ends.</p>",
                sources_html,
            ],
        },
        {
            "heading": "What the map shows",
            "blocks": [
                "<p>The map colors every country in one of three states:</p>",
                "<p><strong>Covered</strong> — we have sources from this "
                "country, and at least one published a story about this "
                "event.</p>",
                "<p><strong>Not covered by tracked sources</strong> — we have "
                "sources from this country, and none of them published a story "
                "about this event.</p>",
                "<p><strong>Not tracked</strong> — we don't have sources from "
                "this country. We make no claim about whether it covered the "
                "event.</p>",
                "<p>The distinction between “not covered” and “not tracked” "
                "matters. When a country is gray, we know our sources there "
                "didn't cover the story. When it's outlined, we don't know — "
                "because we don't read its press. We don't claim a global "
                "view. We show a European view and where that view ends.</p>",
            ],
        },
        {
            "heading": "What each card shows",
            "blocks": [
                "<p>Each card on the homepage represents one event. The card "
                "shows:</p>",
                "<ul><li>A short label naming the event.</li>"
                "<li>The shared terms its headlines have in common.</li>"
                "<li>A map of the countries that covered it.</li>"
                "<li>The number of outlets and countries involved.</li>"
                "<li>The names of a few of those outlets.</li></ul>",
                "<p>On the detail page — reached by clicking any card — every "
                "outlet in the cluster is listed once, with one headline "
                "linked to the original article at its original outlet.</p>",
            ],
        },
        {
            "heading": "How the daily view is built",
            "blocks": [
                "<p>Each day, the pipeline produces a fixed set of stories — "
                "one per event. Only some reach the homepage: an event must be "
                "covered by enough outlets, from enough countries, with "
                "headlines similar enough to be one story. Events that make "
                "the bar are ordered by how many countries covered them. The "
                "selection is designed to answer one question: what is Europe "
                "talking about today?</p>",
                "<p>Some events appear on multiple days as they develop. "
                "Others appear once and fall away. The date navigation at the "
                "top of the homepage lets you browse back through previous "
                "days.</p>",
            ],
        },
        {
            "heading": "What we don't do",
            "blocks": [
                "<p>We don't rate bias or label outlets as left or right in "
                "the interface. We don't tell you what to think about the "
                "coverage we show.</p>",
                "<p>We don't claim to be complete. If a country doesn't appear "
                "on a map, it may be because we don't have sources there — not "
                "because the event wasn't covered. Every headline links to the "
                "original article at its original outlet. To know what an "
                "outlet said, read its article.</p>",
                f"<p>We also don't generate the news. The one thing {app_name} "
                "generates is the short label and the shared terms on each "
                "card, assembled from the headlines themselves. It does not "
                "produce or edit journalism. It reflects what the press has "
                "already published.</p>",
                "<p>What we do is show you, side by side, how different "
                "outlets told the same story. The comparison is yours to "
                "make.</p>",
            ],
        },
    ]


def _about_es(
    app_name: str, outlet_count: int, country_count: int, sources_html: str
) -> list[dict]:
    return [
        {
            "heading": f"Qué es {app_name}",
            "blocks": [
                f"<p>{app_name} agrupa artículos de distintos medios que "
                "cubren el mismo acontecimiento. Cada tarjeta de la portada es "
                "un acontecimiento, contado por cada medio que publicó una "
                "noticia sobre él. El mapa muestra qué países cubrieron ese "
                "acontecimiento y cuáles no.</p>",
            ],
        },
        {
            "heading": "Cómo funciona la agrupación",
            "blocks": [
                f"<p>Cada día, {app_name} recoge titulares de un conjunto fijo "
                "de medios. Compara cada titular con los de otros medios "
                "publicados ese día para ver si hablan del mismo "
                "acontecimiento. Los titulares sobre el mismo acontecimiento "
                "se agrupan en un clúster. Cuando un medio publicó varias "
                f"noticias sobre el mismo acontecimiento, {app_name} conserva "
                "solo una —la más cercana a lo que decían todos los medios— "
                "para que cada medio aparezca una sola vez por "
                "acontecimiento.</p>",
                "<p>Un clúster es la unidad mínima del producto: un "
                "acontecimiento, muchas voces. Todo lo demás —el mapa, las "
                "etiquetas, la lista de fuentes, la selección diaria— se "
                "construye sobre los clústeres.</p>",
                "<p>La agrupación compara el significado de los titulares, no "
                "solo sus palabras. Un titular en francés y otro en inglés "
                "sobre el mismo acontecimiento pueden acabar en el mismo "
                "clúster aunque no compartan vocabulario.</p>",
            ],
        },
        {
            "heading": "Fuentes y países",
            "blocks": [
                f"<p>Este es el conjunto fijo: {outlet_count} medios en "
                f"{country_count} países, cada uno leído cada día a través de "
                "su feed RSS público. La lista define los países seguidos en "
                "el mapa: un país aparece seguido cuando leemos al menos un "
                "medio allí, y no seguido cuando no leemos ninguno. Lo que el "
                "mapa puede decir sobre un acontecimiento termina donde "
                "termina esta lista.</p>",
                sources_html,
            ],
        },
        {
            "heading": "Qué muestra el mapa",
            "blocks": [
                "<p>El mapa colorea cada país en uno de tres estados:</p>",
                "<p><strong>Cubierto</strong> — tenemos medios de este país y "
                "al menos uno publicó una noticia sobre el acontecimiento.</p>",
                "<p><strong>No cubierto por medios seguidos</strong> — tenemos "
                "medios de este país y ninguno publicó una noticia sobre el "
                "acontecimiento.</p>",
                "<p><strong>No seguido</strong> — no tenemos medios de este "
                "país. No afirmamos nada sobre si cubrió el acontecimiento.</p>",
                "<p>La distinción entre «no cubierto» y «no seguido» importa. "
                "Cuando un país está en gris, sabemos que nuestros medios allí "
                "no cubrieron la noticia. Cuando está solo perfilado, no lo "
                "sabemos, porque no leemos su prensa. No afirmamos tener una "
                "visión global. Mostramos una visión europea y dónde termina "
                "esa visión.</p>",
            ],
        },
        {
            "heading": "Qué muestra cada tarjeta",
            "blocks": [
                "<p>Cada tarjeta de la portada representa un acontecimiento. "
                "La tarjeta muestra:</p>",
                "<ul><li>Una etiqueta breve que nombra el acontecimiento.</li>"
                "<li>Los términos que comparten sus titulares.</li>"
                "<li>Un mapa de los países que lo cubrieron.</li>"
                "<li>El número de medios y países implicados.</li>"
                "<li>Los nombres de algunos de esos medios.</li></ul>",
                "<p>En la página de detalle —a la que se llega haciendo clic "
                "en cualquier tarjeta— aparece cada medio del clúster una sola "
                "vez, con un titular enlazado al artículo original en su "
                "medio.</p>",
            ],
        },
        {
            "heading": "Cómo se construye la vista diaria",
            "blocks": [
                "<p>Cada día, el proceso genera un conjunto fijo de historias: "
                "una por acontecimiento. Solo algunas llegan a la portada: un "
                "acontecimiento debe ser cubierto por medios suficientes, con "
                "titulares lo bastante similares como para ser una sola "
                "historia. Los acontecimientos que superan el umbral se "
                "ordenan por cuántos países los cubrieron. La selección "
                "responde a una pregunta: ¿de qué habla Europa hoy?</p>",
                "<p>Algunas historias aparecen en varios días a medida que se "
                "desarrollan. Otras aparecen una vez y desaparecen. La "
                "navegación por fechas en la parte superior de la portada te "
                "permite recorrer los días anteriores.</p>",
            ],
        },
        {
            "heading": "Lo que no hacemos",
            "blocks": [
                "<p>No valoramos el sesgo ni etiquetamos a los medios como de "
                "izquierda o derecha en la interfaz. No te decimos qué pensar "
                "sobre la cobertura que mostramos.</p>",
                "<p>No afirmamos ser completos. Si un país no aparece en un "
                "mapa, puede ser porque no tenemos medios allí, no porque el "
                "acontecimiento no se cubriera. Cada titular enlaza al "
                "artículo original en su medio. Para saber qué dijo un medio, "
                "lee su artículo.</p>",
                f"<p>Tampoco generamos noticias. Lo único que {app_name} "
                "genera es la etiqueta breve y los términos compartidos de "
                "cada tarjeta, a partir de los propios titulares. No produce "
                "ni edita periodismo. Refleja lo que la prensa ya ha "
                "publicado.</p>",
                "<p>Lo que hacemos es mostrarte, en paralelo, cómo distintos "
                "medios contaron la misma historia. La comparación es tuya.</p>",
            ],
        },
    ]
