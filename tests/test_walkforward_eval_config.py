"""Weapon-safety tests for strategy_config.walkforward_eval.json (§5.13.13).

The walkforward_eval config is a sim-only side config used to produce the
honest 27-mo OOS APY/Sharpe baseline (Track P3.2, 2026-05-10). It opts
into the WalkForwardModelLoader manifest contract so each sim bar uses
the latest retrain with `cutoff_date < today` — eliminating the look-ahead
class CLAUDE.md §5.13 flags as the primary corruption mode for in-sample
"baseline" numbers.

Per CLAUDE.md §5.13.13 ("side configs are loaded weapons") this file:
  1. must declare walkforward.enabled=True AND a manifest path that
     resolves to a real, non-empty manifest.
  2. must declare performance.n_trials so the DSR deflator scales with
     the number of retrains the model went through (38 in this case).
  3. must NOT use any production-default artifact_path — even though
     sim only reads, never writes, a future operator could
     accidentally `--strategy-config-name walkforward_eval.json` in a
     training script and clobber prod artifacts. Belt + suspenders.

The tests below enforce these invariants pre-merge.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
CONFIG_PATH = STRATEGY_DIR / "strategy_config.sim_baseline.json"

# 2026-05-11 sim/prod isolation refactor: every prod artifact now lives under
# artifacts/prod/. Sim configs MUST point to artifacts/sim/<file>; using either
# the old flat ``artifacts/<file>`` paths OR the new ``artifacts/prod/<file>``
# would silently load prod-trained models in sim (the leak we just fixed).
PRODUCTION_DEFAULT_ARTIFACTS = {
    "artifacts/panel-ltr.json",
    "artifacts/ngboost-head.json",
    "artifacts/panel-rank-calibration.json",
    "artifacts/prod/panel-ltr.json",
    "artifacts/prod/ngboost-head.json",
    "artifacts/prod/panel-rank-calibration.json",
    "artifacts/prod/panel-ltr.alpha158_fund.json",
    "artifacts/prod/ngboost-head.alpha158_fund.json",
}


def _load_cfg() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _walk_artifact_paths(obj, path=""):
    """Yield (dotted_path, value) for every key named 'artifact_path'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new = f"{path}.{k}" if path else k
            if k == "artifact_path":
                yield (new, v)
            yield from _walk_artifact_paths(v, new)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_artifact_paths(v, f"{path}[{i}]")


def test_walkforward_eval_config_exists_and_parses():
    """Sanity: the side config file is present and valid JSON."""
    assert CONFIG_PATH.exists(), f"sim_baseline side config missing: {CONFIG_PATH}"
    cfg = _load_cfg()
    assert isinstance(cfg, dict)
    assert cfg.get("_side_config_label") == "sim_baseline", (
        "side config must carry _side_config_label='sim_baseline' "
        "so future audits can grep-find it."
    )


def test_walkforward_block_present_and_enabled():
    """Invariant: walkforward.enabled MUST be True with a valid manifest."""
    cfg = _load_cfg()
    wf = cfg.get("walkforward")
    assert wf is not None, "walkforward block missing"
    assert wf.get("enabled") is True, "walkforward.enabled must be True"
    mp = wf.get("manifest_path")
    assert mp, "walkforward.manifest_path must be set"
    # Manifest path can be relative — resolve against strategy_dir.
    manifest_path = STRATEGY_DIR / mp if not Path(mp).is_absolute() else Path(mp)
    assert manifest_path.exists(), (
        f"walkforward.manifest_path={mp} does not resolve to a real file "
        f"(checked {manifest_path}). Sim would FileNotFoundError on init."
    )
    # fail_on_no_model must be True so an empty manifest aborts loudly
    # rather than silently falling back to look-ahead-leaky static load.
    assert wf.get("fail_on_no_model", True) is True, (
        "walkforward.fail_on_no_model must be True for honest OOS sim. "
        "Setting it False reintroduces the leakage class §5.13 forbids."
    )


def test_walkforward_manifest_has_expected_retrain_count():
    """Invariant: manifest contains 38 retrains (Track P3.2 build)."""
    cfg = _load_cfg()
    manifest_path = STRATEGY_DIR / cfg["walkforward"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("retrains", [])
    assert len(rows) == 39, (
        f"Expected 39 retrain entries (13+13+13 from A/B/C v2_clean (all "
        f"2024-12-23 undertrain skip), got {len(rows)}."
    )
    # Each entry must have cutoff_date and artifact_uri pointing at a
    # real file (defense-in-depth — the loader checks too, but failing
    # at config-test time is cheaper than failing mid-sim).
    for row in rows:
        assert "cutoff_date" in row
        assert "artifact_uri" in row
        uri = row["artifact_uri"]
        p = Path(uri) if Path(uri).is_absolute() else STRATEGY_DIR / uri
        assert p.exists(), f"manifest entry artifact missing: {p}"


def test_performance_block_specifies_n_trials_and_rf():
    """§5.13.4: DSR needs n_trials, sim metrics need risk_free_rate_annual."""
    cfg = _load_cfg()
    perf = cfg.get("performance")
    assert perf is not None, "performance block missing"
    n_trials = perf.get("n_trials")
    assert isinstance(n_trials, int) and n_trials >= 1, (
        f"performance.n_trials must be int >= 1, got {n_trials!r}"
    )
    # n_trials=38 reflects the 38 retrain configurations the walkforward
    # baseline searched over. If you change cadence, change n_trials too.
    assert n_trials == 39, (
        f"Expected n_trials=39 (=number of walkforward retrains in manifest, v3_clean), "
        f"got {n_trials}. Sync with manifest entry count."
    )
    rf = perf.get("risk_free_rate_annual")
    assert isinstance(rf, (int, float)) and rf >= 0.0, (
        f"performance.risk_free_rate_annual must be a non-negative float, "
        f"got {rf!r}"
    )


def test_no_production_default_artifact_paths():
    """§5.13.13: side config must NOT reference production artifact paths.

    Even though sim is read-only on artifacts, a misuse case is someone
    invoking a training script with --strategy-config-name walkforward_eval.json
    that would clobber prod. Force every artifact_path to a side path.
    """
    cfg = _load_cfg()
    violations = []
    for dotted, value in _walk_artifact_paths(cfg):
        if value in PRODUCTION_DEFAULT_ARTIFACTS:
            violations.append(f"{dotted} = {value!r}")
    assert not violations, (
        "walkforward_eval config references production artifact paths "
        "(would clobber prod if used by a training script):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_walkforward_loader_resolves_against_manifest():
    """End-to-end: WalkForwardModelLoader can construct from the manifest.

    Catches the class of bug where a manifest exists but is malformed
    (missing required keys, bad timestamps, etc). The loader's
    constructor raises ValueError on any leakage-invariant violation.
    """
    import sys
    sys.path.insert(0, str(STRATEGY_DIR))
    try:
        from kernel.walk_forward.loader import WalkForwardModelLoader
    finally:
        sys.path.pop(0)
    cfg = _load_cfg()
    manifest_path = STRATEGY_DIR / cfg["walkforward"]["manifest_path"]
    loader = WalkForwardModelLoader(manifest_path)
    assert loader.has_walkforward_model()
    assert len(loader.entries) == 39
    # First entry must precede sim start (2024-01-02). 2024-01-01 satisfies.
    import pandas as pd
    assert loader.entries[0].cutoff_date <= pd.Timestamp("2024-01-02")
    # v3_clean (alpha158 walkforward): 2024-12-23 cutoff trained successfully
    # (alpha158 panel is wider than v1's 21-feat panel, no BUG-CV-2 undertrain
    # trigger). At today=2024-12-30, the loader returns 2024-12-23 directly.
    eligible = [e for e in loader.entries if e.cutoff_date < pd.Timestamp("2024-12-30")]
    assert eligible[-1].cutoff_date == pd.Timestamp("2024-12-23"), (
        f"v3_clean: expected latest eligible at 2024-12-23, got "
        f"{eligible[-1].cutoff_date}"
    )
