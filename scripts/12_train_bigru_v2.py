#!/usr/bin/env python
"""
scripts/12_train_bigru_v2.py
----------------------------
Train Model 2b: BiGRU + Additive Attention with Fine-Tuned fastText Embeddings.
Setback fixes vs Model 2:
1. No lowercasing (preserves uppercase / ALL CAPS aggression signal).
2. Trainable (unfrozen) fastText embeddings with differential learning rate (0.1x scale).
3. Increased spatial dropout (0.35) for regularizing fine-tuned embeddings.
4. Cosine Annealing learning rate schedule with warmup.
5. Calibrated positive class weighting (power: 0.45, max_weight: 14.0) to eliminate threshold distortion.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Forward arguments to 11_train_bigru.py with default --model-name bigru_attn_v2
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "11_train_bigru.py"

    args = sys.argv[1:]
    has_model_name = any(arg == "--model-name" or arg.startswith("--model-name=") for arg in args)
    if not has_model_name:
        args = ["--model-name", "bigru_attn_v2"] + args

    cmd = [sys.executable, str(train_script)] + args
    print(f"[Launcher] Executing Model 2b training: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
