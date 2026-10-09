import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from shared.editions import EDITIONS

PIPELINE_DIR = PROJECT_DIR / "pipeline"
PIPELINE_STAGES = (
    PIPELINE_DIR / "ingest.py",
    PIPELINE_DIR / "embed.py",
    PIPELINE_DIR / "clustering" / "cluster.py",
    PIPELINE_DIR / "render.py",
)


def run_stage(script: Path, edition_key: str) -> None:
    print(f"\n=== Running {script.name} ({edition_key}) ===", flush=True)
    command = [sys.executable, str(script), "--edition", edition_key]
    if script.name == "render.py":
        command.append("--clean")
    result = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Pipeline stopped: {script.name} ({edition_key}) exited with "
            f"status {result.returncode}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the generation pipeline for one or all editions."
    )
    parser.add_argument(
        "--edition",
        default=None,
        choices=sorted(EDITIONS),
        help="Run a single edition (default: all editions).",
    )
    args = parser.parse_args()

    edition_keys = [args.edition] if args.edition else sorted(EDITIONS)
    for edition_key in edition_keys:
        for script in PIPELINE_STAGES:
            run_stage(script, edition_key)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
