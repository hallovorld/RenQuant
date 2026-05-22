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
