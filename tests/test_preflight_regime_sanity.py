import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel.preflight import _check_regime_layered_ic  # noqa: E402


def _config(*, require_sanity: bool = True) -> dict:
    return {
        "ranking": {
            "panel_scoring": {
                "enabled": True,
                "kind": "xgb",
                "artifact_path": "artifacts/prod/panel-ltr.json",
                "regime_admission": {
                    "require_sanity_regime_ic": require_sanity,
                },
            }
        }
    }


def _artifact(strategy_dir: Path, *, sanity: dict | None) -> Path:
    p = strategy_dir / "artifacts" / "prod" / "panel-ltr.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    wf = {
        "trade_monotonicity": {
            "passed": True,
            "regimes": [{
                "regime": "BULL_CALM",
                "eligible": True,
                "passed": True,
                "spearman": 0.10,
            }],
        }
    }
    if sanity is not None:
        wf["sanity_regime_ic"] = sanity
    p.write_text(json.dumps({
        "kind": "panel_ltr_xgboost",
        "feature_cols": ["f1"],
        "metadata": {"wf_gate_metadata": wf},
    }))
    return p


def test_preflight_fails_when_required_sanity_regime_ic_missing(tmp_path) -> None:
    _artifact(tmp_path, sanity=None)

    res = _check_regime_layered_ic(_config(require_sanity=True), tmp_path)

    assert res.name == "P-REGIME-IC"
    assert res.severity == "hard"
    assert res.ok is False
    assert "regime sanity IC evidence absent" in res.message


def test_preflight_fails_when_sanity_regime_ic_failed(tmp_path) -> None:
    _artifact(tmp_path, sanity={"passed": False, "reason": "BULL_CALM weak"})

    res = _check_regime_layered_ic(_config(require_sanity=True), tmp_path)

    assert res.severity == "hard"
    assert res.ok is False
    assert "BULL_CALM weak" in res.message


def test_preflight_allows_sell_only_when_required_sanity_missing(tmp_path) -> None:
    _artifact(tmp_path, sanity=None)

    res = _check_regime_layered_ic(
        _config(require_sanity=True),
        tmp_path,
        run_mode="sell-only",
    )

    assert res.severity == "soft"
    assert res.ok is True
    assert "sell-only risk exits are allowed" in res.message


def test_preflight_passes_when_trade_and_sanity_regime_evidence_pass(tmp_path) -> None:
    _artifact(tmp_path, sanity={
        "passed": True,
        "regimes": {
            "BULL_CALM": {
                "eligible": True,
                "passed": True,
                "mean_ic": 0.04,
            }
        },
    })

    res = _check_regime_layered_ic(_config(require_sanity=True), tmp_path)

    assert res.severity == "hard"
    assert res.ok is True
