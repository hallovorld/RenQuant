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


# ── 2026-08-30: a relaxed P-REGIME-IC is never printed as a bare ✓ ─────────

def _relaxed_config() -> dict:
    cfg = _config(require_sanity=True)
    cfg["wf_gate"] = {"sanity_regime_ic_required": False}
    return cfg


def _served_shape_artifact(strategy_dir: Path) -> None:
    """The served 2026-08-02 artifact's regime evidence, shape-for-shape."""
    p = strategy_dir / "artifacts" / "prod" / "panel-ltr.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "kind": "panel_ltr_xgboost",
        "feature_cols": ["f1"],
        "metadata": {"wf_gate_metadata": {
            "sanity_regime_ic": {
                "passed": False,
                "reason": "regime sanity IC failed: BULL_CALM,BULL_VOLATILE,CHOPPY",
            },
            "trade_monotonicity": {
                "passed": False,
                "pooled": {"n": 117, "spearman": 0.03911372202282383},
                "regimes": [
                    {"regime": "BULL_CALM", "eligible": True, "passed": False,
                     "spearman": 0.0023365233812373976},
                    {"regime": "BULL_VOLATILE", "eligible": False, "passed": False,
                     "spearman": 0.2727},
                ],
            },
        }},
    }))


def test_relaxed_regime_ic_pass_leads_with_the_relaxed_state(tmp_path) -> None:
    _served_shape_artifact(tmp_path)

    res = _check_regime_layered_ic(_relaxed_config(), tmp_path)

    assert res.ok is True and res.severity == "hard"
    assert res.message.startswith("RELAXED: ")
    assert "sanity IC failed (regime sanity IC failed: BULL_CALM,BULL_VOLATILE,CHOPPY)" in res.message
    assert "stamp failed BULL_CALM ρ=0.002" in res.message
    assert "sanity_regime_ic_required=false" in res.message
    assert "NOT proven for eligible regimes ['BULL_CALM']" in res.message
    assert "monotonicity passed" not in res.message
    assert res.details["sanity_regime_ic_relaxed"] is True
    assert res.details["trade_monotonicity_relaxed"] is True


def test_strict_config_still_blocks_the_same_artifact(tmp_path) -> None:
    _served_shape_artifact(tmp_path)

    res = _check_regime_layered_ic(_config(require_sanity=True), tmp_path)

    assert res.ok is False and res.severity == "hard"
    assert "RELAXED" not in res.message


def test_genuine_pass_keeps_the_plain_pass_text(tmp_path) -> None:
    _artifact(tmp_path, sanity={"passed": True})

    res = _check_regime_layered_ic(_relaxed_config(), tmp_path)

    assert res.ok is True
    assert res.message.startswith("regime-layered IC/monotonicity passed for eligible regimes ['BULL_CALM']")
    assert "RELAXED" not in res.message
