#!/usr/bin/env python
"""
scripts/16_run_model3_pipeline.py
-----------------------------------
End-to-End Orchestrator for Model 3:
1. Single-seed training (Seed 42) using best tuned parameters (per reduced scope).
2. Ensemble and Blend Evaluation (scripts/15_ensemble_and_blend.py).
3. Comprehensive Evaluation (scripts/20_evaluate.py --model-id 3_distilroberta).
4. Publication-Quality Report Generation (scripts/21_build_model_report.py --model-id 3_distilroberta).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str], desc: str) -> None:
    print("\n" + "=" * 75)
    print(f"[PIPELINE STEP] {desc}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 75)
    t0 = time.perf_counter()
    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - t0
    if ret.returncode != 0:
        print(f"\n[ERROR] Step '{desc}' failed with exit code {ret.returncode}! Aborting pipeline.")
        sys.exit(ret.returncode)
    print(f"[SUCCESS] Step '{desc}' completed in {elapsed/60:.2f} min.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Model 3 End-to-End Pipeline")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds to train (default from models.yaml)")
    args = parser.parse_args()

    python_exe = sys.executable
    smoke_flag = ["--smoke"] if args.smoke else []

    params_file = PROJECT_ROOT / "runs/distilroberta_best_params.json"
    params_flag = ["--params", str(params_file)] if params_file.exists() else []

    if args.seeds is not None:
        seeds = args.seeds
    else:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from wtc.config import load_models_config
        seeds = load_models_config().get("3_distilroberta", {}).get("final", {}).get("seeds", [42])

    total_start = time.perf_counter()

    # 1. Train seeds
    for s in seeds:
        desc = f"Train DistilRoBERTa Seed {s} (4 epochs)"
        cmd = [
            python_exe,
            "scripts/13_train_distilroberta.py",
            "--seed", str(s),
            "--save-seed-probs",
        ] + smoke_flag + params_flag
        run_cmd(cmd, desc)

    # 2. Ensemble and Blend Check
    run_cmd(
        [python_exe, "scripts/15_ensemble_and_blend.py"] + smoke_flag,
        "Evaluate Multi-Seed Ensemble and TF-IDF Blend",
    )

    # 3. Model Evaluation (20_evaluate.py)
    run_cmd(
        [python_exe, "scripts/20_evaluate.py", "--model-id", "3_distilroberta"] + smoke_flag,
        "Comprehensive Evaluation & Benchmarking (20_evaluate.py)",
    )

    # 4. Report Generation (21_build_model_report.py)
    run_cmd(
        [python_exe, "scripts/21_build_model_report.py", "--model-id", "3_distilroberta"] + smoke_flag,
        "Generate Final Model Evaluation Report (21_build_model_report.py)",
    )

    total_time = time.perf_counter() - total_start
    print("\n" + "#" * 75)
    print(f"# ALL MODEL 3 PIPELINE STEPS COMPLETED IN {total_time/60:.2f} MINUTES!")
    print("#" * 75)


if __name__ == "__main__":
    main()
