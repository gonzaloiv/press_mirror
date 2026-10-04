import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
PIPELINE_DIR = Path(__file__).resolve().parent
PIPELINE_STAGES = ("ingest.py", "embed.py", "cluster.py")


def main():
    for stage in PIPELINE_STAGES:
        script = PIPELINE_DIR / stage
        print(f"\n=== Running {stage} ===", flush=True)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_DIR,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"Pipeline stopped: {stage} exited with status {result.returncode}."
            )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
