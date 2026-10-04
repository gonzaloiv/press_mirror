"""Formatting and viewport helpers for map pages."""

from shared.country_codes import (
    COUNTRY_CENTERS,
    COUNTRY_NAMES,
    NON_EUROPEAN_COUNTRIES,
    to_iso,
)

WORLD_BOUNDS = [[-80, -180], [80, 180]]


def european_coverage(
    countries: list[str], max_countries: int | None = 5
) -> str:
    """Format country codes as a concise, naturally sorted coverage subtitle."""
    normalized_countries = {to_iso(code) for code in countries}
    names = sorted(
        (COUNTRY_NAMES.get(code, code) for code in normalized_countries),
        key=lambda name: name.removeprefix("the ").casefold(),
    )
    if not names:
        return "No coverage found."

    country_count = len(names)
    display_names = names
    if max_countries is not None and country_count > max_countries:
        display_names = names[:max_countries - 1] + [
            f"{country_count - max_countries + 1} others"
        ]

    if len(display_names) == 1:
        country_list = display_names[0]
    elif len(display_names) == 2:
        country_list = " and ".join(display_names)
    else:
        country_list = f"{', '.join(display_names[:-1])}, and {display_names[-1]}"

    is_international = bool(normalized_countries & NON_EUROPEAN_COUNTRIES)
    prefix = (
        "Coverage from"
        if is_international or country_count == 1
        else "European coverage from"
    )
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
