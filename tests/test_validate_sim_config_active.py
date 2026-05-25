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
