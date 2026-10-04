ISO_FIX = {"UK": "GB"}

COUNTRY_NAMES = {
    "AU": "Australia",
    "BR": "Brazil",
    "CA": "Canada",
    "CN": "China",
    "DE": "Germany",
    "ES": "Spain",
    "FR": "France",
    "GB": "the UK",
    "IN": "India",
    "IT": "Italy",
    "JP": "Japan",
    "MX": "Mexico",
    "NL": "the Netherlands",
    "PL": "Poland",
    "PT": "Portugal",
    "QA": "Qatar",
    "RU": "Russia",
    "UA": "Ukraine",
    "US": "the United States",
}

NON_EUROPEAN_COUNTRIES = {"AU", "BR", "CA", "CN", "IN", "JP", "MX", "QA", "US"}

COUNTRY_CENTERS = {
    "AU": (-25.3, 133.8),
    "BR": (-10.8, -52.9),
    "CA": (56.1, -106.3),
    "CN": (35.9, 104.2),
    "DE": (51.2, 10.4),
    "ES": (40.3, -3.7),
    "FR": (46.2, 2.2),
    "GB": (54.0, -2.0),
    "IN": (20.6, 78.9),
    "IT": (41.9, 12.6),
    "JP": (36.2, 138.3),
    "MX": (23.6, -102.5),
    "NL": (52.1, 5.3),
    "PL": (52.1, 19.1),
    "PT": (39.4, -8.2),
    "QA": (25.3, 51.2),
    "RU": (61.5, 105.3),
    "UA": (48.4, 31.2),
    "US": (39.8, -98.6),
}


def to_iso(country_code: str) -> str:
    """Normalize common country-code aliases to ISO 3166-1 alpha-2."""
    return ISO_FIX.get(country_code, country_code)
