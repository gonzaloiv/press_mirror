import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from shared.cluster_selection import selected_cluster_indices
from shared.editions import DEFAULT_EDITION_KEY, EDITIONS, get_edition

BASE_URL = "http://localhost:5000"
IMAGE_WIDTHS = {
    "large": 1080,
    "small": 600,
}


def cluster_files(history_dir: Path) -> list[Path]:
    files = sorted(history_dir.glob("clusters_????-??-??.json"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No dated cluster files found in {history_dir}")
    return files


def render_images(edition, clean: bool = False) -> None:
    try:
        playwright_api = importlib.import_module("playwright.sync_api")
    except ImportError as error:
        raise SystemExit(
            "Playwright is required. Install it with `pip install playwright` "
            "and `playwright install chromium`."
        ) from error
    playwright_error = playwright_api.Error

    maps_dir = edition.maps_dir
    maps_dir.mkdir(parents=True, exist_ok=True)
    snapshots = cluster_files(edition.history_dir)

    with playwright_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            viewport_height = page.viewport_size["height"]
            for snapshot_index, clusters_file in enumerate(snapshots):
                snapshot_date = clusters_file.stem.removeprefix("clusters_")
                dated_maps_dir = maps_dir / snapshot_date
                dated_maps_dir.mkdir(parents=True, exist_ok=True)
                if clean and snapshot_index == 0:
                    for path in dated_maps_dir.iterdir():
                        if path.is_file() or path.is_symlink():
                            path.unlink()
                        elif path.is_dir():
                            raise IsADirectoryError(
                                f"Refusing to remove unexpected directory: {path}"
                            )

                with clusters_file.open(encoding="utf-8") as f:
                    clusters = json.load(f)
                selected_indices = selected_cluster_indices(
                    clusters,
                    min_similarity=edition.min_similarity,
                    min_sources=edition.min_sources,
                    min_countries=edition.min_countries,
                )
                if not selected_indices:
                    print(f"No clusters to render for {snapshot_date}.")
                    continue

                for cluster_index in selected_indices:
                    for size_name, width in IMAGE_WIDTHS.items():
                        output_file = dated_maps_dir / f"{cluster_index}_{size_name}.png"
                        if output_file.exists():
                            continue

                        page.set_viewport_size({
                            "width": width,
                            "height": viewport_height,
                        })
                        url = (
                            f"{BASE_URL}{edition.url_prefix}/map/{cluster_index}"
                            f"?date={snapshot_date}&thumbnail=1"
                        )
                        try:
                            page.goto(url, wait_until="domcontentloaded")
                        except playwright_error as error:
                            if any(
                                marker in str(error).lower()
                                for marker in (
                                    "err_connection_refused",
                                    "econnrefused",
                                )
                            ):
                                print(
                                    "Start the Flask server with "
                                    "`python web_app\\server.py` first.",
                                    file=sys.stderr,
                                )
                                raise SystemExit(1) from error
                            raise

                        page.wait_for_function("window.mapReady === true")
                        page.wait_for_timeout(300)
                        page.locator("#thumbnail-content").screenshot(
                            path=str(output_file)
                        )
                        print(f"Saved {output_file}")
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render large and small PNGs for clusters meeting the "
            "similarity and coverage minimums."
        )
    )
    parser.add_argument(
        "--edition",
        default=DEFAULT_EDITION_KEY,
        choices=sorted(EDITIONS),
        help="Edition to render (default: %(default)s).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing files in the edition's maps directory before rendering.",
    )
    args = parser.parse_args()
    render_images(get_edition(args.edition), clean=args.clean)


if __name__ == "__main__":
    main()
