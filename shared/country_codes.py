ISO_FIX = {"UK": "GB"}


def to_iso(country_code: str) -> str:
    """Normalize common country-code aliases to ISO 3166-1 alpha-2."""
    return ISO_FIX.get(country_code, country_code)
