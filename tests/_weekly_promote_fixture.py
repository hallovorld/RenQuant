"""Shared fixture builder for the weekly_wf_promote.sh / manual_promote.sh
snapshot-backstop integration harnesses (Codex PR #432 round 5 review).

Builds a self-contained fixture repo where:
  - scripts/render_strategy_104_snapshot.py is a genuine copy (so
    check_snapshot_freshness's real diff/regenerate logic runs unmocked).
  - scripts/promote_pin.py is a genuine copy (source of check_snapshot_
    freshness itself).
  - Every OTHER dependency weekly_wf_promote.sh/manual_promote.sh calls
    (smoke test, retrain, WF-manifest stamping, WF gate, kernel.model_
    acceptance.promote, build_dashboard.py) is a trivial stub that succeeds
    instantly and writes whatever the real script expects to find next —
    these steps are not what round 5 is testing; Step 7's snapshot backstop
    (the actual PR content) is.

Called from Python (pytest) which then invokes the real production shell
scripts as subprocesses against this fixture, via the RQ_WEEKLY_PROMOTE_*/
RQ_MANUAL_PROMOTE_* environment overrides added in this same round.
"""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RENDERER_SRC = REPO_ROOT / "scripts" / "render_strategy_104_snapshot.py"
PROMOTE_PIN_SRC = REPO_ROOT / "scripts" / "promote_pin.py"
PAIR_PROMOTE_SRC = REPO_ROOT / "scripts" / "fallback_pair_promote.py"
REJECT_DISPOSITION_SRC = REPO_ROOT / "scripts" / "reject_notify_disposition.py"

# The exact active-artifact/calibrator filenames weekly_wf_promote.sh and
# manual_promote.sh hardcode (ART_DIR/ACTIVE_ART/ACTIVE_CAL) — the fixture's
# config must point artifact_path at these exact names so the real
# production script's writes are what the renderer/backstop actually sees.
ACTIVE_ARTIFACT_NAME = "panel-ltr.alpha158_fund.json"
ACTIVE_CALIBRATOR_NAME = "panel-rank-calibration.json"
STAGING_ARTIFACT_NAME = "panel-ltr.staging.json"  # manual_promote.sh default


def _load_renderer_module():
    spec = importlib.util.spec_from_file_location(
        "render_strategy_104_snapshot_for_weekly_fixture", RENDERER_SRC)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def build_fixture_repo(root: Path) -> object:
    """Build the fixture repo at `root`; returns the loaded renderer module
    (STRATEGY_DIR_REL / config-name constants live on it)."""
    mod = _load_renderer_module()

    configs = root / mod.PINNED_CONFIGS_REL
    configs.mkdir(parents=True)
    git_dir = root / mod.PINNED_GIT_DIR_REL
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("c" * 40 + "\n", encoding="utf-8")
    strategy_dir = root / mod.STRATEGY_DIR_REL

    # Field-for-field, byte-for-byte identical to what the daily_retrain_
    # alpha158_fund.sh stub below writes — the "fresh" test case depends on
    # the initial committed snapshot (rendered from THIS file, below)
    # matching the retrain stub's actual output exactly, since Step 3
    # unconditionally overwrites this file during the real script's run.
    _write_json(strategy_dir / "artifacts" / "prod" / ACTIVE_ARTIFACT_NAME, {
        "trained_date": "2026-06-30",
        "effective_train_cutoff_date": "2026-05-01",
        "lookahead_days": 60,
        "config_fingerprint": "sha256:abc123",
        "label_col": "fwd_60d_excess",
        "feature_cols": ["a", "b"],
        "metadata": {"wf_gate_metadata": {
            "passed": True, "run_at": "2026-06-30T00:00:00",
            "wf_3cut_sharpe_mean": 1.23, "wf_3cut_apy_mean": 12.3,
            "sanity_shuffled_ic": 0.01, "sanity_placebo_ic": 0.02}},
    })
    _write_json(strategy_dir / "artifacts" / "prod" / ACTIVE_CALIBRATOR_NAME, {
        "kind": "global_panel_calibration", "trained_date": "2026-07-01",
        "metadata": {"method": "platt", "pool_ic": 0.0993,
                     "scorer_model_content_fingerprint": "sha256:feedface"},
    })
    ckpt = strategy_dir / "artifacts" / "shadow" / "model.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"not a real checkpoint")
    _write_json(Path(str(ckpt) + ".metadata.json"), {
        "trained_date": "2026-05-22",
        "effective_train_cutoff_date": "2024-11-13",
        "effective_selection_cutoff_date": "2026-02-10",
        "lookahead_days": 60,
        "config_fingerprint": "sha256:f8fb2259b2bf1537",
    })

    _write_json(configs / mod.ACTIVE_CONFIG_NAME, {
        "watchlist": ["AAPL", "MSFT", "NVDA"],
        "max_concurrent_positions": 8,
        "ranking": {
            "panel_scoring": {
                "kind": "xgb",
                "artifact_path": f"artifacts/prod/{ACTIVE_ARTIFACT_NAME}",
                "conviction_gate": {"enabled": True, "mu_floor": 0.03},
                "global_calibration": {
                    "artifact_path": f"artifacts/prod/{ACTIVE_CALIBRATOR_NAME}"},
                "shadow_models": [
                    {"name": "old_primary", "kind": "hf_patchtst",
                     "artifact_path": "artifacts/shadow/model.pt"},
                ],
            },
        },
        "rotation": {"panel_buy_top_n": 3},
    })
    _write_json(configs / mod.SHADOW_CONFIG_NAME, {
        "ranking": {"panel_scoring": {
            "kind": "hf_patchtst",
            "artifact_path": "artifacts/shadow/model.pt",
            "global_calibration": {
                "artifact_path": f"artifacts/prod/{ACTIVE_CALIBRATOR_NAME}"},
        }},
    })
    _write_json(root / mod.LOCK_FILE_REL, {
        "subrepos": [
            {"name": "renquant-strategy-104", "branch": "main",
             "commit": "c" * 40, "status": "bootstrapped"},
            {"name": "renquant-common", "branch": "main",
             "commit": "a" * 40, "status": "bootstrapped"},
        ],
    })

    # Real copies of the two files under test — these are NOT stubs.
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "render_strategy_104_snapshot.py").write_bytes(RENDERER_SRC.read_bytes())
    (root / "scripts" / "promote_pin.py").write_bytes(PROMOTE_PIN_SRC.read_bytes())
    (root / "scripts" / "fallback_pair_promote.py").write_bytes(PAIR_PROMOTE_SRC.read_bytes())
    (root / "scripts" / "reject_notify_disposition.py").write_bytes(
        REJECT_DISPOSITION_SRC.read_bytes())

    # Render the INITIAL fresh snapshot so a run against this fixture with
    # no further mutation is genuinely "fresh" (mirrors production: a
    # correctly-promoted, correctly-snapshotted repo).
    committed = root / "doc" / "arch" / "strategy-104-snapshot.md"
    rc = mod.main(["--repo-root", str(root), "--output", str(committed)])
    assert rc == 0, "fixture setup: initial snapshot render failed"

    # ---- Stubs for every OTHER dependency the production scripts call ----
    py = sys.executable
    _write_executable(root / "scripts" / "smoke_test_model.py",
                       f"#!{py}\nimport sys\nsys.exit(0)\n")
    _write_executable(root / "scripts" / "build_dashboard.py",
                       f"#!{py}\nimport sys\nsys.exit(0)\n")
    _write_executable(root / "scripts" / "stamp_walkforward_fingerprints.py",
                       f"#!{py}\nimport sys\nsys.exit(0)\n")
    _write_executable(root / "scripts" / "run_wf_gate.py",
                       f"#!{py}\nimport sys\nsys.exit(0)\n")

    # daily_retrain_alpha158_fund.sh: write the staging artifact/calibrator
    # WITH wf_gate_metadata.passed=True already baked in (the real pipeline
    # splits writing the artifact and stamping the gate verdict across
    # Steps 3/3.5/4; for this fixture, faithfully replicating which exact
    # step stamps the metadata is not what round 5 is testing — Step 7's
    # snapshot backstop, which runs after promotion, is).
    _write_executable(root / "scripts" / "daily_retrain_alpha158_fund.sh", f"""#!/bin/bash
set -euo pipefail
OUT=""
CAL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --xgb-artifact-out) OUT="$2"; shift 2 ;;
    --calibrator-out) CAL="$2"; shift 2 ;;
    --no-drop-sentiment) shift ;;
    *) shift ;;
  esac
done
{py} -c "
import json, sys
json.dump({{
    'trained_date': '2026-06-30',
    'effective_train_cutoff_date': '2026-05-01',
    'lookahead_days': 60,
    'config_fingerprint': 'sha256:abc123',
    'label_col': 'fwd_60d_excess',
    'feature_cols': ['a', 'b'],
    'metadata': {{'wf_gate_metadata': {{'passed': True, 'run_at': '2026-06-30T00:00:00',
                          'wf_3cut_sharpe_mean': 1.23, 'wf_3cut_apy_mean': 12.3,
                          'sanity_shuffled_ic': 0.01, 'sanity_placebo_ic': 0.02}}}},
}}, open('$OUT', 'w'))
json.dump({{'kind': 'global_panel_calibration', 'trained_date': '2026-07-01',
           'metadata': {{'method': 'platt', 'pool_ic': 0.0993,
                        'scorer_model_content_fingerprint': 'sha256:feedface'}}}},
          open('$CAL', 'w'))
"
""")

    # backtesting/renquant_104/kernel/model_acceptance.py: the umbrella-mode
    # promote() target (RQ_WF_GATE_RUNNER=umbrella, used by both scripts
    # under test) — a real, minimal implementation (copy src->dst), not a
    # mock of behavior under test.
    _write_executable(
        root / "backtesting" / "renquant_104" / "kernel" / "model_acceptance.py",
        f"""#!{py}
import shutil
from pathlib import Path


def promote(src, dst):
    shutil.copy2(Path(src), Path(dst))
""")
    (root / "backtesting" / "renquant_104" / "kernel" / "__init__.py").write_text(
        "", encoding="utf-8")

    # scripts/subrepo_env.sh: a real copy — its functions are lenient
    # no-ops when the subrepo assembly/env files it looks for don't exist,
    # which is exactly this fixture's (intentionally minimal) state.
    subrepo_env_src = REPO_ROOT / "scripts" / "subrepo_env.sh"
    if subrepo_env_src.exists():
        (root / "scripts" / "subrepo_env.sh").write_bytes(subrepo_env_src.read_bytes())

    # PATH shims for notify()'s outbound channels: WITHOUT these, running
    # the real weekly_wf_promote.sh under test would POST real test-noise
    # notifications (including a fake "SNAPSHOT STALE" alert) to the
    # OPERATOR'S production ntfy topic via the real curl, and pop real
    # desktop notifications via terminal-notifier on a macOS dev machine.
    # Tests must prepend shim_bin_dir(root) to the subprocess PATH; the
    # RQ_WEEKLY_PROMOTE_NOTIFY_LOG hook (not these shims) is what the
    # notification assertions observe.
    testbin = shim_bin_dir(root)
    _write_executable(testbin / "curl", "#!/bin/bash\nexit 0\n")
    _write_executable(testbin / "terminal-notifier", "#!/bin/bash\nexit 0\n")

    (root / ".env").write_text("", encoding="utf-8")
    (root / "doc").mkdir(parents=True, exist_ok=True)
    return mod


def shim_bin_dir(root: Path) -> Path:
    """Directory of no-op curl/terminal-notifier shims (created by
    build_fixture_repo); prepend to PATH so notify() cannot reach the real
    ntfy topic or the desktop notifier from a test run."""
    return root / ".testbin"


def make_snapshot_stale(root: Path, mod) -> None:
    """Directly corrupt the COMMITTED snapshot doc so it no longer matches
    what the renderer would produce from current source — deterministic
    regardless of what the retrain/promote steps write during the run
    (Step 3's stub re-derives the artifact from fixed values each time, so
    mutating the artifact pre-run would just get overwritten identically;
    corrupting the committed doc directly is what genuinely guarantees a
    post-promotion regenerate-and-compare finds drift)."""
    committed = root / "doc" / "arch" / "strategy-104-snapshot.md"
    text = committed.read_text(encoding="utf-8")
    committed.write_text(text + "\n<!-- stale marker injected by test -->\n",
                          encoding="utf-8")
