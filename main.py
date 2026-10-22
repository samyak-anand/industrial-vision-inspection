"""VisionInspect AI: run existing scripts from a single, guarded entry point.

Copy this file to PROJECT_ROOT/main.py. Run `python main.py --help`.
No stage runs without explicit selection. Existing grouped data is never rebuilt here.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    "ingest": ("src.ingestion.data_injection", []),
    "annotations": ("src.validation.annotation_validation", []),
    "eda": ("src.analysis.EDA", []),
    "inspect": ("src.training.model_inspection", []),
    "similarity": (
        "src.validation.similarity_analysis",
        ["--dataset", "yolo_screw_hole_grouped"]
    ),
    "train": (
        "src.training.train_yolo",
        ["--mode", "baseline", "--epochs", "30",
         "--batch", "2", "--imgsz", "320"]
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # 1. Pipeline stages
    parser.add_argument(
        "stages",
        nargs="*",
        metavar="STAGE",
        help="Pipeline stages to execute",
    )

    # 2. Optional arguments
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available stages",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    parser.add_argument(
        "--confirm-train",
        action="store_true",
        help="Explicitly permit GPU training",
    )

    # 3. Parse arguments
    args = parser.parse_args()

    # 4. List available stages
    if args.list or not args.stages:
        print(
            "Available stages:",
            ", ".join(STAGES)
        )
        print("\nExamples:")
        print("  python main.py ingest annotations eda")
        print("  python main.py inspect")
        print("  python main.py train --confirm-train")
        return 0

    # 5. Validate stage names
    invalid_stages = [
        stage
        for stage in args.stages
        if stage not in STAGES
    ]

    if invalid_stages:
        parser.error(
            f"Invalid stage(s): {', '.join(invalid_stages)}. "
            f"Available stages: {', '.join(STAGES)}"
        )

    # 6. Require explicit confirmation for training
    if (
        "train" in args.stages
        and not args.confirm_train
        and not args.dry_run
    ):
        parser.error(
            "Training requires --confirm-train"
        )

    # 7. Configure Python import path
    env = os.environ.copy()

    env["PYTHONPATH"] = (
        str(ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    # 8. Execute selected stages
    for stage in args.stages:

        module, options = STAGES[stage]

        cmd = [
            sys.executable,
            "-m",
            module,
            *options
        ]

        print(
            f"\n[{stage}] {' '.join(cmd)}",
            flush=True
        )

        # Dry-run prints commands without execution
        if args.dry_run:
            continue

        subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            check=True
        )

        print(
            f"[SUCCESS] {stage} completed",
            flush=True
        )

    print("\nPipeline execution completed.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        print(
            f"Pipeline stopped: stage exited with code {exc.returncode}",
            file=sys.stderr
        )
        sys.exit(exc.returncode)