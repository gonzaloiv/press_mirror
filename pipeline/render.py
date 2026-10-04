import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
HISTORY_DIR = PROJECT_DIR / "history"
STATIC_DIR = PROJECT_DIR / "static"
MAPS_DIR = STATIC_DIR / "maps"
BASE_URL = "http://localhost:5000"
CLUSTER_LIMIT = 6
IMAGE_SIZES = {
    "large": (1080, 1080),
    "small": (600, 450),
}


def latest_clusters_file() -> Path:
    files = sorted(HISTORY_DIR.glob("clusters_????-??-??.json"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No dated cluster files found in {HISTORY_DIR}")
    return files[0]


def render_images(clean: bool = False) -> None:
    try:
        playwright_api = importlib.import_module("playwright.sync_api")
    except ImportError as error:
        raise SystemExit(
            "Playwright is required. Install it with `pip install playwright` "
            "and `playwright install chromium`."
        ) from error
    playwright_error = playwright_api.Error

    clusters_file = latest_clusters_file()
    with clusters_file.open(encoding="utf-8") as f:
        clusters = json.load(f)

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_date = clusters_file.stem.removeprefix("clusters_")
    dated_maps_dir = MAPS_DIR / snapshot_date
    if clean:
        if dated_maps_dir.exists():
            for path in dated_maps_dir.iterdir():
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    raise IsADirectoryError(
                        f"Refusing to remove unexpected directory: {path}"
                    )
        dated_maps_dir.mkdir(parents=True, exist_ok=True)
    else:
        dated_maps_dir.mkdir(parents=True, exist_ok=True)

    selected_clusters = clusters[:CLUSTER_LIMIT]
    if not selected_clusters:
        print("No clusters to render.")
        return

    with playwright_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page()
            for cluster_index in range(len(selected_clusters)):
                for size_name, (width, height) in IMAGE_SIZES.items():
                    page.set_viewport_size({"width": width, "height": height})
                    url = f"{BASE_URL}/map/{cluster_index}"
                    try:
                        page.goto(url, wait_until="domcontentloaded")
                    except playwright_error as error:
                        if any(
                            marker in str(error).lower()
                            for marker in ("err_connection_refused", "econnrefused")
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
                    output_file = dated_maps_dir / f"{cluster_index}_{size_name}.png"
                    page.screenshot(path=str(output_file))
                    print(f"Saved {output_file}")
        finally:
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render large and small PNGs for the six top story clusters."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing files in static/maps before rendering.",
    )
    args = parser.parse_args()
    render_images(clean=args.clean)


if __name__ == "__main__":
    main()
