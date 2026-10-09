"""Formatting and viewport helpers for map pages."""

from shared.country_codes import (
    COUNTRY_CENTERS,
    NON_EUROPEAN_COUNTRIES,
    country_name,
    to_iso,
)

WORLD_BOUNDS = [[-80, -180], [80, 180]]

_COVERAGE_PREFIX = {
    "en": {"coverage": "Coverage from", "european": "European coverage from"},
    "es": {"coverage": "Cobertura de", "european": "Cobertura europea de"},
}

_NO_COVERAGE = {
    "en": "No coverage found.",
    "es": "No se encontró cobertura.",
}


def _join_names(names: list[str], language: str) -> str:
    conjunction = "and" if language == "en" else "y"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} {conjunction} {names[1]}"
    return f"{', '.join(names[:-1])} {conjunction} {names[-1]}"


def european_coverage(
    countries: list[str],
    max_countries: int | None = 5,
    language: str = "en",
) -> str:
    """Format country codes as a concise, naturally sorted coverage subtitle."""
    normalized_countries = {to_iso(code) for code in countries}
    names = sorted(
        (country_name(code, language) for code in normalized_countries),
        key=lambda name: name.removeprefix("the ").casefold(),
    )
    if not names:
        return _NO_COVERAGE.get(language, _NO_COVERAGE["en"])

    country_count = len(names)
    display_names = names
    if max_countries is not None and country_count > max_countries:
        remainder = country_count - max_countries + 1
        more = f"{remainder} more" if language == "en" else f"{remainder} más"
        display_names = names[:max_countries - 1] + [more]

    country_list = _join_names(display_names, language)
    is_international = bool(normalized_countries & NON_EUROPEAN_COUNTRIES)
    prefixes = _COVERAGE_PREFIX.get(language, _COVERAGE_PREFIX["en"])
    prefix = prefixes["coverage"] if is_international or country_count == 1 else prefixes["european"]
    return f"{prefix} {country_list}."


def map_viewport(countries: list[str]) -> list[list[float]] | None:
    """Return padded Leaflet bounds, or None to use the Europe fallback."""
    normalized = {to_iso(country) for country in countries}
    if len(normalized) <= 1:
        return None

    centers = [
        COUNTRY_CENTERS[code]
        for code in normalized
        if code in COUNTRY_CENTERS
    ]
    if len(centers) < 2:
        return None

    latitudes = [center[0] for center in centers]
    longitudes = [center[1] for center in centers]
    min_lat, max_lat = min(latitudes), max(latitudes)
    min_lng, max_lng = min(longitudes), max(longitudes)

    if max_lng - min_lng > 180:
        return WORLD_BOUNDS

    lat_padding = max((max_lat - min_lat) * 0.12, 5.0)
    lng_padding = max((max_lng - min_lng) * 0.12, 6.0)
    return [
        [max(-85, min_lat - lat_padding), max(-180, min_lng - lng_padding)],
        [min(85, max_lat + lat_padding), min(180, max_lng + lng_padding)],
    ]
