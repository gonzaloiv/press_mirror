import json
import re
import socket
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import feedparser

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCES_FILE = Path(__file__).resolve().parent / "sources.json"
HISTORY_DIR = PROJECT_DIR / "history"
USER_AGENT = "Mozilla/5.0 (compatible; PressMirrorBot/0.1; +https://pressmirror.eu)"
FETCH_TIMEOUT_SECONDS = 10
RETRY_DELAY_SECONDS = 2
LENIENT_RECOVERY_FEEDS = {"Público", "Politiken", "Jyllands-Posten"}
SUPPRESSED_PARSE_WARNINGS = (
    "document declared as us-ascii, but parsed as utf-8",
    "xml is not an xml media type",
)
UNESCAPED_AMPERSAND = re.compile(
    rb"&(?!#(?:[0-9]+|[xX][0-9A-Fa-f]+);|[A-Za-z_:][A-Za-z0-9_.:-]*;)"
)
ILLEGAL_XML_CONTROLS = re.compile(rb"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def load_feeds() -> list[dict]:
    """Load configured RSS feeds from sources.json."""
    with SOURCES_FILE.open(encoding="utf-8") as f:
        config = json.load(f)
    return config["feeds"]


class FeedFetchError(Exception):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def fetch_raw_feed(feed: dict) -> bytes:
    request = Request(feed["url"], headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
        return response.read()


def fetch_error(error: Exception) -> FeedFetchError:
    if isinstance(error, HTTPError):
        return FeedFetchError("URL problem", str(error))
    if isinstance(error, URLError) and isinstance(
        error.reason, (TimeoutError, socket.timeout)
    ):
        return FeedFetchError("URL problem", f"Timeout: {error}")
    if isinstance(error, (TimeoutError, socket.timeout)):
        return FeedFetchError("URL problem", f"Timeout: {error}")
    if isinstance(error, OSError):
        return FeedFetchError("URL problem", f"Network error: {error}")
    return FeedFetchError("URL problem", str(error))


def is_suppressed_parse_warning(parsed) -> bool:
    message = str(parsed.bozo_exception).lower() if parsed.bozo else ""
    return any(warning in message for warning in SUPPRESSED_PARSE_WARNINGS)


def sanitize_xml(content: bytes) -> bytes:
    if content.startswith(b"\xef\xbb\xbf"):
        content = content[3:]
    content = UNESCAPED_AMPERSAND.sub(b"&amp;", content)
    return ILLEGAL_XML_CONTROLS.sub(b"", content)


def fetch_parsed_feed(feed: dict):
    """Fetch and parse one feed, with one retry for any fetch or parse failure."""
    for attempt in range(2):
        try:
            content = fetch_raw_feed(feed)
            parsed = feedparser.parse(content, agent=USER_AGENT)
            if parsed.bozo and not is_suppressed_parse_warning(parsed):
                message = str(parsed.bozo_exception) or "Unknown XML parse error"
                reason = (
                    "URL problem"
                    if "xml is not an xml media type" in message.lower()
                    else "Malformed XML"
                )
                raise FeedFetchError(
                    reason,
                    message,
                )
            return parsed
        except FeedFetchError as error:
            reason = error.reason
            message = str(error)
        except Exception as error:
            failure = fetch_error(error)
            reason = failure.reason
            message = str(failure)

        if attempt == 0:
            print(f"    Retrying {feed['source']} in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)
        else:
            raise FeedFetchError(reason, message)


def recover_malformed_feed(feed: dict):
    """Re-fetch and sanitize a known malformed feed once before giving up."""
    try:
        content = sanitize_xml(fetch_raw_feed(feed))
        parsed = feedparser.parse(content, agent=USER_AGENT)
    except Exception as error:
        raise fetch_error(error) from error

    if parsed.entries:
        return parsed

    message = (
        str(parsed.bozo_exception)
        if parsed.bozo
        else "Lenient parse completed but returned 0 entries"
    )
    raise FeedFetchError("Malformed XML", message)


def fetch_feed(feed: dict, parsed) -> list[dict]:
    """Convert a parsed feed response into headline dicts."""
    articles = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for entry in parsed.entries:
        # Every feed names things slightly differently. Be defensive.
        title = str(entry.get("title") or "").strip()
        link = str(entry.get("link") or "").strip()
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
            "fetched_at": fetched_at,
        })

    print(f"    -> {len(articles)} headlines")
    return articles


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    all_articles = []
    failures = []
    feeds = load_feeds()
    print(f"Loaded {len(feeds)} feeds from {SOURCES_FILE}.")

    print("\nValidating feeds and fetching headlines...")
    for feed in feeds:
        print(f"  Fetching {feed['source']} ({feed['country']})...")
        try:
            parsed = fetch_parsed_feed(feed)
        except FeedFetchError as error:
            if feed["source"] in LENIENT_RECOVERY_FEEDS and error.reason == "Malformed XML":
                try:
                    parsed = recover_malformed_feed(feed)
                except FeedFetchError as recovery_error:
                    error = recovery_error
                else:
                    all_articles.extend(fetch_feed(feed, parsed))
                    print(f"    -> {len(parsed.entries)} entries after lenient parse")
                    time.sleep(0.5)
                    continue
            failures.append({
                **feed,
                "reason": error.reason,
                "error": str(error),
            })
            time.sleep(0.5)
            continue
        except Exception as error:
            failure = fetch_error(error)
            failures.append({
                **feed,
                "reason": failure.reason,
                "error": str(failure),
            })
            time.sleep(0.5)
            continue

        if feed["source"] in LENIENT_RECOVERY_FEEDS and not parsed.entries:
            try:
                parsed = recover_malformed_feed(feed)
            except FeedFetchError as error:
                failures.append({
                    **feed,
                    "reason": error.reason,
                    "error": str(error),
                })
                time.sleep(0.5)
                continue

        if not parsed.entries:
            failures.append({
                **feed,
                "reason": "Empty feed",
                "error": "Parsed successfully but contained 0 entries",
            })
            print("    -> 0 entries")
        else:
            try:
                all_articles.extend(fetch_feed(feed, parsed))
                print(f"    -> {len(parsed.entries)} entries")
            except Exception as error:
                failures.append({
                    **feed,
                    "reason": "URL problem",
                    "error": f"Processing error: {error}",
                })
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
    for reason in ("URL problem", "Malformed XML", "Empty feed"):
        print(f"\n{reason}")
        failed = [feed for feed in failures if feed["reason"] == reason]
        if failed:
            for feed in failed:
                print(
                    f"  - {feed['source']} ({feed['country']}) | {feed['url']} | "
                    f"{feed['error']}"
                )
        else:
            print("  (none)")


if __name__ == "__main__":
    main()