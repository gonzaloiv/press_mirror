import json
import time
from datetime import datetime, timezone

import feedparser

# Curated list of RSS feeds.
# Grouped by country so you can later filter by region in the UI.
FEEDS = [
    # --- France ---
    {"source": "Le Monde",        "country": "FR", "url": "https://www.lemonde.fr/rss/une.xml"},
    {"source": "Le Figaro",       "country": "FR", "url": "https://www.lefigaro.fr/rss/figaro_actualites.xml"},
    {"source": "France 24",       "country": "FR", "url": "https://www.france24.com/fr/rss"},

    # --- Spain ---
    {"source": "El País",         "country": "ES", "url": "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada"},
    {"source": "El Mundo",        "country": "ES", "url": "https://www.elmundo.es/rss/portada.xml"},
    {"source": "La Vanguardia",   "country": "ES", "url": "https://www.lavanguardia.com/rss/home.xml"},

    # --- UK / US (English) ---
    {"source": "BBC World",       "country": "UK", "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"source": "The Guardian",    "country": "UK", "url": "https://www.theguardian.com/world/rss"},
    {"source": "New York Times",  "country": "US", "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"},
    {"source": "Al Jazeera",      "country": "QA", "url": "https://www.aljazeera.com/xml/rss/all.xml"},

    # --- Germany ---
    {"source": "Deutsche Welle",  "country": "DE", "url": "https://rss.dw.com/rdf/rss-en-world"},
    {"source": "Der Spiegel",     "country": "DE", "url": "https://www.spiegel.de/international/index.rss"},
]

OUTPUT_FILE = "headlines.json"


def fetch_feed(feed: dict) -> list[dict]:
    """Fetch a single RSS feed and return a list of headline dicts."""
    print(f"  Fetching {feed['source']} ({feed['country']})...")

    parsed = feedparser.parse(feed["url"])

    # feedparser doesn't raise on network errors; it sets bozo + bozo_exception.
    if parsed.bozo:
        print(f"    ! Warning: {parsed.bozo_exception}")

    articles = []
    for entry in parsed.entries:
        # Every feed names things slightly differently. Be defensive.
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        if not title or not link:
            continue

        # published_parsed is a time.struct_time in UTC, or None if missing.
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        published_iso = (
            datetime(*published[:6], tzinfo=timezone.utc).isoformat()
            if published
            else None
        )

        articles.append({
            "title": title,
            "link": link,
            "source": feed["source"],
            "country": feed["country"],
            "published": published_iso,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })

    print(f"    -> {len(articles)} headlines")
    return articles


def main():
    all_articles = []

    for feed in FEEDS:
        try:
            all_articles.extend(fetch_feed(feed))
        except Exception as e:
            # One broken feed should never kill the whole run.
            print(f"  ! Failed to fetch {feed['source']}: {e}")
        # Be polite: don't hammer servers.
        time.sleep(0.5)

    # Deduplicate on link (some feeds repeat entries).
    seen = set()
    unique = []
    for a in all_articles:
        if a["link"] not in seen:
            seen.add(a["link"])
            unique.append(a)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(unique)} unique headlines written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()