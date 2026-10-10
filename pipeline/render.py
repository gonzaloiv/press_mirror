import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from shared.cluster_selection import selected_cluster_indices
from shared.snapshots import clusters_file, maps_dir, snapshot_dates

BASE_URL = "http://localhost:5000"
IMAGE_WIDTHS = {
    "large": 1080,
    "small": 600,
}


def render_images(clean: bool = False) -> None:
    try:
        playwright_api = importlib.import_module("playwright.sync_api")
    except ImportError as error:
        raise SystemExit(
            "Playwright is required. Install it with `pip install playwright` "
            "and `playwright install chromium`."
        ) from error
    playwright_error = playwright_api.Error

    snapshots = snapshot_dates()

    with playwright_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            viewport_height = page.viewport_size["height"]
            for snapshot_index, snapshot_date in enumerate(snapshots):
                dated_maps_dir = maps_dir(snapshot_date)
                dated_maps_dir.mkdir(parents=True, exist_ok=True)
                if clean and snapshot_index == 0:
                    for path in dated_maps_dir.iterdir():
                        if path.is_file() or path.is_symlink():
                            path.unlink()
                        elif path.is_dir():
                            raise IsADirectoryError(
                                f"Refusing to remove unexpected directory: {path}"
                            )

                with clusters_file(snapshot_date).open(encoding="utf-8") as f:
                    clusters = json.load(f)
                selected_indices = selected_cluster_indices(clusters)
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
                            f"{BASE_URL}/map/{cluster_index}"
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
        "--clean",
        action="store_true",
        help="Delete existing map images for the latest snapshot before rendering.",
    )
    args = parser.parse_args()
    render_images(clean=args.clean)


if __name__ == "__main__":
    main()
