ISO_FIX = {"UK": "GB"}

COUNTRY_NAMES = {
    "AT": "Austria",
    "AU": "Australia",
    "BE": "Belgium",
    "BR": "Brazil",
    "CA": "Canada",
    "CH": "Switzerland",
    "CN": "China",
    "CZ": "Czechia",
    "DE": "Germany",
    "DK": "Denmark",
    "EE": "Estonia",
    "ES": "Spain",
    "FI": "Finland",
    "FR": "France",
    "GB": "the UK",
    "HU": "Hungary",
    "IE": "Ireland",
    "IN": "India",
    "IT": "Italy",
    "JP": "Japan",
    "MX": "Mexico",
    "NL": "the Netherlands",
    "NO": "Norway",
    "PL": "Poland",
    "PT": "Portugal",
    "QA": "Qatar",
    "RU": "Russia",
    "SE": "Sweden",
    "UA": "Ukraine",
    "US": "the United States",
}

COUNTRY_NAMES_ES = {
    "AT": "Austria",
    "AU": "Australia",
    "BE": "Bélgica",
    "BR": "Brasil",
    "CA": "Canadá",
    "CH": "Suiza",
    "CN": "China",
    "CZ": "Chequia",
    "DE": "Alemania",
    "DK": "Dinamarca",
    "EE": "Estonia",
    "ES": "España",
    "FI": "Finlandia",
    "FR": "Francia",
    "GB": "Reino Unido",
    "HU": "Hungría",
    "IE": "Irlanda",
    "IN": "India",
    "IT": "Italia",
    "JP": "Japón",
    "MX": "México",
    "NL": "Países Bajos",
    "NO": "Noruega",
    "PL": "Polonia",
    "PT": "Portugal",
    "QA": "Catar",
    "RU": "Rusia",
    "SE": "Suecia",
    "UA": "Ucrania",
    "US": "Estados Unidos",
}

NON_EUROPEAN_COUNTRIES = {"AU", "BR", "CA", "CN", "IN", "JP", "MX", "QA", "US"}

EUROPEAN_COUNTRIES = {
    "AD", "AL", "AT", "BA", "BE", "BG", "BY", "CH", "CY", "CZ",
    "DE", "DK", "EE", "ES", "FI", "FR", "GB", "GR", "HR", "HU",
    "IE", "IS", "IT", "LI", "LT", "LU", "LV", "MC", "MD", "ME",
    "MK", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "RU", "SE",
    "SI", "SK", "SM", "TR", "UA", "VA", "XK",
}

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


_LOCALIZED_NAMES = {
    "en": COUNTRY_NAMES,
    "es": COUNTRY_NAMES_ES,
}


def country_name(country_code: str, language: str = "en") -> str:
    """Return a country's display name, keeping the English "the " prefix."""
    code = to_iso(country_code)
    names = _LOCALIZED_NAMES.get(language, COUNTRY_NAMES)
    return names.get(code, COUNTRY_NAMES.get(code, code))
