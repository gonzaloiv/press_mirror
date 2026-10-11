"""Publish the day's top cluster to Buffer's GraphQL API.

Runs after the site has deployed, so the linked map page and the attached
image URL are publicly reachable. Configure with environment variables:

    BUFFER_API_KEY       - Buffer API key (required to publish)
    BUFFER_CHANNEL_IDS   - comma-separated channel IDs to post to
    SITE_URL             - public site base URL (default https://pressmirror.eu)
    BUFFER_DRY_RUN       - set to "1" to print the payload without sending

With no key or channels set, the module is a no-op so local pipeline runs
never publish by accident.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from shared.cluster_selection import selected_cluster_indices
from shared.country_codes import to_iso
from shared.snapshots import clusters_file, snapshot_dates

BUFFER_API_URL = "https://api.buffer.com"
DEFAULT_SITE_URL = "https://pressmirror.eu"
REQUEST_TIMEOUT_SECONDS = 30
IMAGE_WAIT_ATTEMPTS = 30
IMAGE_WAIT_DELAY_SECONDS = 10
DAILY_POST_COUNT = 3
POST_SPACING_HOURS = 1


def build_text(cluster: dict, link: str) -> str:
    """Compose the two-line post body from a displayed cluster."""
    sources = {
        article["source"]
        for article in cluster["articles"]
        if article.get("source")
    }
    countries = {to_iso(article["country"]) for article in cluster["articles"]}
    return (
        f"How Europe covered {cluster['label']}.\n"
        f"{len(sources)} outlets in {len(countries)} countries. "
        f"See the map: {link}"
    )


def view_url(site_url: str, snapshot_date: str, index: int) -> str:
    return f"{site_url}/map/{index}?date={snapshot_date}"


def image_url(site_url: str, snapshot_date: str, index: int) -> str:
    repository = os.environ.get("GITHUB_REPOSITORY")
    commit = os.environ.get("SNAPSHOT_COMMIT")
    if repository and commit:
        return (
            f"https://raw.githubusercontent.com/{repository}/{commit}/"
            f"history/{snapshot_date}/maps/{index}_large.png"
        )
    return f"{site_url}/maps/{snapshot_date}/{index}_large.png"


def scheduled_time(offset_hours: int) -> str:
    due_at = datetime.now(timezone.utc) + timedelta(hours=offset_hours)
    return due_at.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def wait_for_public_url(url: str) -> bool:
    """Poll until the deployed image URL is reachable (HEAD, bounded retries)."""
    for attempt in range(IMAGE_WAIT_ATTEMPTS):
        try:
            with urlopen(Request(url, method="HEAD"), timeout=10) as response:
                if response.status == 200:
                    return True
        except (HTTPError, URLError, OSError):
            pass
        if attempt < IMAGE_WAIT_ATTEMPTS - 1:
            print(f"    waiting for {url} ({attempt + 1})...", flush=True)
            time.sleep(IMAGE_WAIT_DELAY_SECONDS)
    return False


def _gql_string(value: str) -> str:
    """Return a valid GraphQL string literal (JSON escaping matches)."""
    return json.dumps(value)


def create_post(
    channel_id: str,
    text: str,
    image: str,
    due_at: str,
    api_key: str,
) -> dict:
    """Schedule a Buffer post for one channel."""
    query = f"""mutation {{
  createPost(input: {{
    text: {_gql_string(text)}
    channelId: {_gql_string(channel_id)}
    schedulingType: automatic
    mode: customScheduled
    dueAt: {_gql_string(due_at)}
    assets: [{{ image: {{ url: {_gql_string(image)} }} }}]
  }}) {{
    ... on PostActionSuccess {{ post {{ id }} }}
    ... on MutationError {{ message }}
  }}
}}"""
    request = Request(
        BUFFER_API_URL,
        data=json.dumps({"query": query}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def publish_errors(result: dict) -> list[str]:
    """Extract error messages from a Buffer GraphQL response."""
    errors = [error.get("message", str(error)) for error in result.get("errors", [])]
    mutation = (result.get("data") or {}).get("createPost") or {}
    if mutation.get("message"):
        errors.append(mutation["message"])
    return errors


def channel_ids() -> list[str]:
    raw = os.environ.get("BUFFER_CHANNEL_IDS", "")
    return [channel.strip() for channel in raw.split(",") if channel.strip()]


def load_top_clusters(limit: int):
    """Return (snapshot_date, [(index, cluster), ...]) for the top cards."""
    snapshot_date = snapshot_dates()[0]
    with clusters_file(snapshot_date).open(encoding="utf-8") as f:
        clusters = json.load(f)
    selected = selected_cluster_indices(clusters)[:limit]
    return snapshot_date, [(index, clusters[index]) for index in selected]


def build_post(
    site_url: str,
    snapshot_date: str,
    index: int,
    cluster: dict,
    position: int,
) -> dict:
    """Assemble one scheduled post payload for a cluster."""
    link = view_url(site_url, snapshot_date, index)
    return {
        "index": index,
        "label": cluster["label"],
        "text": build_text(cluster, link),
        "link": link,
        "image": image_url(site_url, snapshot_date, index),
        "due_at": scheduled_time(position * POST_SPACING_HOURS),
    }


def publish_post(channels: list[str], post: dict, api_key: str) -> None:
    """Schedule one post on every configured channel."""
    for channel_id in channels:
        result = create_post(
            channel_id, post["text"], post["image"], post["due_at"], api_key
        )
        errors = publish_errors(result)
        if errors:
            raise SystemExit(
                f"Buffer post failed for {channel_id}: {'; '.join(errors)}"
            )
        post_id = (result.get("data") or {}).get("createPost", {}).get("post", {})
        print(
            f"Scheduled cluster {post['index']} post {post_id.get('id')} on "
            f"channel {channel_id} for {post['due_at']}."
        )


def main() -> None:
    api_key = os.environ.get("BUFFER_API_KEY")
    channels = channel_ids()
    dry_run = os.environ.get("BUFFER_DRY_RUN") == "1"
    if not api_key and not dry_run:
        print("BUFFER_API_KEY not set; skipping Buffer publish.")
        return
    if not channels:
        print("BUFFER_CHANNEL_IDS not set; skipping Buffer publish.")
        return

    snapshot_date, top_clusters = load_top_clusters(DAILY_POST_COUNT)
    if not top_clusters:
        print("No displayed cluster for today; nothing to publish.")
        return

    site_url = os.environ.get("SITE_URL", DEFAULT_SITE_URL).rstrip("/")
    posts = [
        build_post(site_url, snapshot_date, index, cluster, position)
        for position, (index, cluster) in enumerate(top_clusters, start=1)
    ]
    print(f"Prepared {len(posts)} posts for {snapshot_date}.")

    if dry_run:
        print(json.dumps(
            {"channels": channels, "posts": posts},
            ensure_ascii=False,
            indent=2,
        ))
        return

    if not wait_for_public_url(posts[0]["image"]):
        raise SystemExit(
            f"Image not reachable at {posts[0]['image']}; not publishing."
        )

    for post in posts:
        publish_post(channels, post, api_key)

    print("Buffer publishing complete.")


if __name__ == "__main__":
    main()
