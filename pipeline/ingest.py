import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import feedparser

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCES_FILE = Path(__file__).resolve().parent / "sources.json"
HISTORY_DIR = PROJECT_DIR / "history"


def load_feeds() -> list[dict]:
    """Load configured RSS feeds from sources.json."""
    with SOURCES_FILE.open(encoding="utf-8") as f:
        config = json.load(f)
    return config["feeds"]


def validate_feed(feed: dict, parsed) -> dict:
    """Summarize a parsed feed response without making another request."""
    entry_count = len(parsed.entries)
    return {
        "source": feed["source"],
        "country": feed["country"],
        "status": "ok" if entry_count > 0 else "empty",
        "entries": entry_count,
        "error": str(parsed.bozo_exception) if parsed.bozo else None,
    }


def fetch_feed(feed: dict, parsed) -> list[dict]:
    """Convert a parsed feed response into headline dicts."""
    print(f"  Fetching {feed['source']} ({feed['country']})...")

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
    feeds = load_feeds()
    print(f"Loaded {len(feeds)} feeds from {SOURCES_FILE}.")

    print("\nValidating feeds and fetching headlines...")
    for feed in feeds:
        try:
            parsed = feedparser.parse(feed["url"])
            result = validate_feed(feed, parsed)
            status = f"{result['status']} ({result['entries']} entries)"
            if result["error"]:
                status += f" — {result['error']}"
            print(f"  {result['source']} ({result['country']}): {status}")
            all_articles.extend(fetch_feed(feed, parsed))
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

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    output_file = HISTORY_DIR / f"headlines_{date.today().isoformat()}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(unique)} unique headlines written to {output_file}")


if __name__ == "__main__":
    main()