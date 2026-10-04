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


def to_iso(country_code: str) -> str:
    """Normalize common country-code aliases to ISO 3166-1 alpha-2."""
    return ISO_FIX.get(country_code, country_code)
