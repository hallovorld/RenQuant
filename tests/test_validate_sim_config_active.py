import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_sim_config_active.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_sim_config_active", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_meta_label_enabled_is_active_path():
    mod = _load_validator()
    baseline = {"ranking": {"meta_label": {"enabled": False}}}
    candidate = {"ranking": {"meta_label": {"enabled": True}}}

    active, report = mod.static_validate(baseline, candidate)

    joined = "\n".join(report)
    assert active
    assert "ranking.meta_label.enabled" in joined
    assert "ACTIVE" in joined
    assert "✗ ranking.meta_label.enabled" not in joined


def test_nested_underscore_notes_are_inert_metadata():
    mod = _load_validator()
    baseline = {"ranking": {"meta_label": {"enabled": False}}}
    candidate = {
        "ranking": {
            "meta_label": {
                "enabled": True,
                "_codex_ab_reason": "side config explanation",
            }
        }
    }

    active, report = mod.static_validate(baseline, candidate)

    joined = "\n".join(report)
    assert active
    assert "ranking.meta_label._codex_ab_reason" in joined
    assert "INERT_METADATA" in joined
    assert "✗ ranking.meta_label._codex_ab_reason" not in joined


def test_regime_blend_weight_is_active_path():
    mod = _load_validator()
    baseline = {"ranking": {"regime_blend_weights": {"BULL_CALM": [1.0, 0.0]}}}
    candidate = {"ranking": {"regime_blend_weights": {"BULL_CALM": [0.35, 0.65]}}}

    active, report = mod.static_validate(baseline, candidate)

    joined = "\n".join(report)
    assert active
    assert "ranking.regime_blend_weights.BULL_CALM.0" in joined
    assert "ranking.regime_blend_weights.BULL_CALM.1" in joined
    assert "ACTIVE" in joined


def test_qp_mu_source_and_alpha_to_mu_are_active_paths():
    mod = _load_validator()
    baseline = {"ranking": {"qp_mu_source": "mu", "alpha_to_mu": {"enabled": False}}}
    candidate = {
        "ranking": {
            "qp_mu_source": "ranking_composite",
            "alpha_to_mu": {"enabled": True, "ic": 0.08},
        }
    }

    active, report = mod.static_validate(baseline, candidate)

    joined = "\n".join(report)
    assert active
    assert "ranking.qp_mu_source" in joined
    assert "ranking.alpha_to_mu.enabled" in joined
    assert "ranking.alpha_to_mu.ic" in joined
    assert "ACTIVE" in joined


def test_benchmark_sleeve_paths_are_active():
    mod = _load_validator()
    baseline = {"portfolio": {"benchmark_sleeve": {"enabled": False}}}
    candidate = {
        "portfolio": {
            "benchmark_sleeve": {
                "enabled": True,
                "ticker": "SPY",
                "target_exposure_by_regime": {"BULL_CALM": 1.0},
                "fund_alpha_from_sleeve": True,
                "alpha_funding_budget_pct": 0.15,
                "sleeve_counts_as_cash_reserve": True,
                "_research_note": "diagnostic note",
            },
        },
        "execution": {"buying_power_mode": "non_marginable_buying_power"},
    }

    active, report = mod.static_validate(baseline, candidate)

    joined = "\n".join(report)
    assert active
    assert "portfolio.benchmark_sleeve.enabled" in joined
    assert "portfolio.benchmark_sleeve.target_exposure_by_regime.BULL_CALM" in joined
    assert "portfolio.benchmark_sleeve.fund_alpha_from_sleeve" in joined
    assert "portfolio.benchmark_sleeve.alpha_funding_budget_pct" in joined
    assert "portfolio.benchmark_sleeve.sleeve_counts_as_cash_reserve" in joined
    assert "execution.buying_power_mode" in joined
    assert "portfolio.benchmark_sleeve._research_note" in joined
    assert "INERT_METADATA" in joined
    assert "✗ " not in joined


def test_stop_loss_anchor_policy_paths_are_active():
    mod = _load_validator()
    baseline = {"risk": {"stop_loss_anchor_policy": {"mode": "current_regime"}}}
    candidate = {
        "risk": {
            "stop_loss_anchor_policy": {
                "mode": "max_entry_current",
                "entry_regimes": ["BULL_CALM"],
                "current_regimes": ["CHOPPY", "BEAR"],
            }
        }
    }

    active, report = mod.static_validate(baseline, candidate)

    joined = "\n".join(report)
    assert active
    assert "risk.stop_loss_anchor_policy.mode" in joined
    assert "risk.stop_loss_anchor_policy.entry_regimes.0" in joined
    assert "risk.stop_loss_anchor_policy.current_regimes.0" in joined
    assert "risk.stop_loss_anchor_policy.current_regimes.1" in joined
    assert "✗ " not in joined
