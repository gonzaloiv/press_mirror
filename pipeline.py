import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_STAGES = (
    "pipeline.ingest",
    "pipeline.embed",
    "pipeline.clustering.cluster",
    "pipeline.render",
)


def main() -> None:
    for stage in PIPELINE_STAGES:
        print(f"\n=== Running {stage} ===", flush=True)
        command = [sys.executable, "-m", stage]
        if stage == "pipeline.render":
            command.append("--clean")
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"Pipeline stopped: {stage} exited with status "
                f"{result.returncode}."
            )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
