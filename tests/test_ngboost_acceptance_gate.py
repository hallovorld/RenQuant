"""Regression guards for NGBoost artifact acceptance.

The NGBoost head is optional/dormant in production scoring, but training still
fits it. A negative validation IC head must not overwrite the prior artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


class _StubNGBoostHead:
    def save(self, path: Path, metadata: dict | None = None) -> None:
        Path(path).write_text(json.dumps({
            "kind": "ngboost_head",
            "val_mu_ic": (metadata or {}).get("val_mu_ic"),
        }))


def test_negative_val_mu_ic_rejects_by_default(tmp_path):
    from training_panel.pp_panel_training import NGBoostSaveTask

    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    prior = strategy_dir / "artifacts" / "ngboost-head.json"
    prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps({"kind": "ngboost_head", "sentinel": "prior"}))

    ctx = SimpleNamespace(
        ngboost_head=_StubNGBoostHead(),
        ngboost_fit={
            "val_mu_ic": -0.01,
            "train_mu_mean": 0.0,
            "train_sigma_mean": 1.0,
            "train_mu_ic": 0.1,
            "n_rows": 100,
        },
        config={
            "panel_ltr": {"ngboost": {"artifact_path": "ngboost-head.json"}},
            "ranking": {"panel_scoring": {"ngboost": {}}},
        },
        strategy_dir=strategy_dir,
        ngboost_artifact_path=None,
    )

    NGBoostSaveTask().run(ctx)

    assert json.loads(prior.read_text())["sentinel"] == "prior"
    assert ctx.ngboost_artifact_path is None
    staging = strategy_dir / "artifacts" / "ngboost-head.staging.json"
    assert staging.exists()
    assert json.loads(staging.read_text())["val_mu_ic"] == -0.01
