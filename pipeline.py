import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = PROJECT_DIR / "pipeline"
PIPELINE_STAGES = (
    # PIPELINE_DIR / "ingest.py",
    # PIPELINE_DIR / "embed.py",
    PIPELINE_DIR / "clustering" / "cluster.py",
    PIPELINE_DIR / "render.py",
)


def main() -> None:
    for script in PIPELINE_STAGES:
        print(f"\n=== Running {script.name} ===", flush=True)
        command = [sys.executable, str(script)]
        if script.name == "render.py":
            command.append("--clean")
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"Pipeline stopped: {script.name} exited with status "
                f"{result.returncode}."
            )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
