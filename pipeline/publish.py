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

import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parent.parent

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
        f"How Europe covered {cluster['label']}.\n\n"
        f"{len(sources)} outlets in {len(countries)} countries.\n\n"
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


def graphql(query: str, api_key: str) -> dict:
    """Run one GraphQL request against the Buffer API."""
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


def channel_service(channel_id: str, api_key: str) -> str | None:
    """Return a channel's Buffer service (e.g. instagram, twitter)."""
    query = (
        f"query {{ channel(input: {{ id: {_gql_string(channel_id)} }}) "
        "{ service } }"
    )
    result = graphql(query, api_key)
    channel = (result.get("data") or {}).get("channel") or {}
    return channel.get("service")


def channel_metadata(service: str | None) -> str:
    """Return the GraphQL metadata fragment a service requires, if any."""
    if service == "instagram":
        return "metadata: { instagram: { type: post, shouldShareToFeed: true } }"
    return ""


def create_post_query(
    channel_id: str,
    text: str,
    image: str,
    due_at: str,
    service: str | None,
) -> str:
    """Build the createPost mutation, including any service metadata."""
    metadata = channel_metadata(service)
    return f"""mutation {{
  createPost(input: {{
    text: {_gql_string(text)}
    channelId: {_gql_string(channel_id)}
    schedulingType: automatic
    mode: customScheduled
    dueAt: {_gql_string(due_at)}
    assets: [{{ image: {{ url: {_gql_string(image)} }} }}]
    {metadata}
  }}) {{
    ... on PostActionSuccess {{ post {{ id }} }}
    ... on MutationError {{ message }}
  }}
}}"""


def create_post(
    channel_id: str,
    text: str,
    image: str,
    due_at: str,
    api_key: str,
    service: str | None = None,
) -> dict:
    """Schedule a Buffer post for one channel."""
    query = create_post_query(channel_id, text, image, due_at, service)
    return graphql(query, api_key)


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


def resolve_channels(api_key: str | None) -> list[tuple[str, str | None]]:
    """Pair each configured channel id with its Buffer service."""
    resolved = []
    for channel_id in channel_ids():
        service = None
        if api_key:
            try:
                service = channel_service(channel_id, api_key)
            except (HTTPError, URLError, OSError) as error:
                print(f"    could not resolve channel {channel_id}: {error}")
        resolved.append((channel_id, service))
    return resolved


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


def publication_state_path() -> Path:
    configured = os.environ.get("BUFFER_STATE_PATH")
    return Path(configured) if configured else PROJECT_DIR / ".github" / "buffer-publications.json"


def channel_fingerprint(channel_id: str) -> str:
    return hashlib.sha256(channel_id.encode("utf-8")).hexdigest()


def load_publication_state(snapshot_date: str):
    path = publication_state_path()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set(), []
    if state.get("snapshot_date") != snapshot_date:
        return set(), []
    records = state.get("published", [])
    completed = {
        (record["cluster_index"], record["channel_fingerprint"])
        for record in records
        if "cluster_index" in record and "channel_fingerprint" in record
    }
    return completed, records


def save_publication_state(snapshot_date: str, records: list[dict]) -> None:
    path = publication_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"snapshot_date": snapshot_date, "published": records},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def publish_post(
    channels: list[tuple[str, str | None]],
    post: dict,
    api_key: str,
    snapshot_date: str,
    completed: set[tuple[int, str]],
    records: list[dict],
) -> list[str]:
    """Schedule one post per unprocessed channel and persist each success."""
    failures = []
    for channel_id, service in channels:
        fingerprint = channel_fingerprint(channel_id)
        key = (post["index"], fingerprint)
        if key in completed:
            print(
                f"Already scheduled cluster {post['index']} on channel "
                f"{channel_id}; skipping."
            )
            continue
        try:
            result = create_post(
                channel_id,
                post["text"],
                post["image"],
                post["due_at"],
                api_key,
                service,
            )
        except (HTTPError, URLError, OSError) as error:
            failures.append(f"{channel_id}: {error}")
            continue
        errors = publish_errors(result)
        if errors:
            failures.append(f"{channel_id}: {'; '.join(errors)}")
            continue
        post_id = (result.get("data") or {}).get("createPost", {}).get("post", {})
        completed.add(key)
        records.append(
            {
                "cluster_index": post["index"],
                "channel_fingerprint": fingerprint,
            }
        )
        save_publication_state(snapshot_date, records)
        print(
            f"Scheduled cluster {post['index']} post {post_id.get('id')} on "
            f"channel {channel_id} for {post['due_at']}."
        )
    return failures


def main() -> None:
    api_key = os.environ.get("BUFFER_API_KEY")
    dry_run = os.environ.get("BUFFER_DRY_RUN") == "1"
    if not api_key and not dry_run:
        print("BUFFER_API_KEY not set; skipping Buffer publish.")
        return
    channels = resolve_channels(api_key)
    if not channels:
        print("BUFFER_CHANNEL_IDS not set; skipping Buffer publish.")
        return

    snapshot_date, top_clusters = load_top_clusters(DAILY_POST_COUNT)
    today = datetime.now(timezone.utc).date().isoformat()
    if snapshot_date != today:
        print(
            f"Latest snapshot {snapshot_date} is not today ({today}); "
            "skipping Buffer publish."
        )
        return
    if not top_clusters:
        print("No displayed cluster for today; nothing to publish.")
        return

    site_url = os.environ.get("SITE_URL", DEFAULT_SITE_URL).rstrip("/")
    posts = [
        build_post(site_url, snapshot_date, index, cluster, position)
        for position, (index, cluster) in enumerate(top_clusters, start=1)
    ]
    print(f"Prepared {len(posts)} posts for {snapshot_date}.")
    completed, records = load_publication_state(snapshot_date)
    print(f"Already scheduled today: {len(completed)} story/channel pair(s).")

    if dry_run:
        print(json.dumps(
            {
                "channels": [
                    {"id": channel_id, "service": service}
                    for channel_id, service in channels
                ],
                "posts": posts,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    if not wait_for_public_url(posts[0]["image"]):
        raise SystemExit(
            f"Image not reachable at {posts[0]['image']}; not publishing."
        )

    failures = []
    for post in posts:
        failures.extend(
            publish_post(
                channels,
                post,
                api_key,
                snapshot_date,
                completed,
                records,
            )
        )

    if failures:
        raise SystemExit("Buffer publish had failures:\n" + "\n".join(failures))
    print("Buffer publishing complete.")


if __name__ == "__main__":
    main()
