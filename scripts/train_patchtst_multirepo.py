#!/usr/bin/env python
"""Multi-repo PatchTST trainer — train the sequence model through the pins.

The torch/HF trainer code runs out of the pinned ``renquant-model`` engine
(``renquant_model_patchtst.hf_trainer``, lifted verbatim from the umbrella's
``scripts/patchtst_hf.py``). Its data-side ``kernel.*`` deps (walk_forward_splits,
hmm_regime_labels, config_consistency) resolve from the umbrella via
``RENQUANT_STRATEGY_DIR`` — same split as the GBDT driver (model in the subrepo,
data deps from the baseline).

Unlike GBDT, PatchTST weights are NOT byte-reproducible: torch on Apple MPS is
not bit-deterministic even with a fixed seed (no use_deterministic_algorithms /
cudnn flags). Parity here is structural/procedural (same lifted code + config
contract + a valid checkpoint), not byte-identical weights.

Usage (passes all args through to the lifted trainer's CLI):
    python scripts/train_patchtst_multirepo.py --cut cut1_covid --epochs 1 --device mps
    python scripts/train_patchtst_multirepo.py --cut all --epochs 8 --device mps --save-model
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SIBLINGS = REPO.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"

# Pinned subrepo source roots the engine import needs (renquant_model_patchtst +
# its renquant_common / renquant_artifacts / renquant_model_common deps).
_PIN_SRCS = ["renquant-common", "renquant-base-data", "renquant-artifacts", "renquant-model"]


def _bootstrap() -> None:
    for name in _PIN_SRCS:
        src = SIBLINGS / name / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    # Umbrella strategy dir hosts the data-side kernel.* deps + strategy_config;
    # the lifted trainer reads RENQUANT_STRATEGY_DIR to find them.
    os.environ.setdefault("RENQUANT_STRATEGY_DIR", str(STRATEGY_DIR))
    for p in (str(STRATEGY_DIR), str(REPO)):
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> int:
    _bootstrap()
    trainer = importlib.import_module("renquant_model_patchtst.hf_trainer")
    sys.stderr.write(
        f"[patchtst-multirepo] trainer={trainer.__file__} (pin); "
        f"kernel.* data deps from {STRATEGY_DIR} (umbrella)\n"
    )
    # Hand off to the lifted trainer's CLI with the original args.
    sys.argv = [trainer.__file__] + sys.argv[1:]
    trainer.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
