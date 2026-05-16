#!/usr/bin/env python3
"""Pre-flight validator: refuse to launch a sim panel whose config knobs
are no-ops vs baseline.

Two layers of validation:

  1. STATIC — every diff key is checked against a hand-curated map of
     kernel-reachable config paths (built by greping the kernel for each
     `cfg.get("<key>")`, `regime_p.get("<key>")`, etc.). Diff keys that
     land outside any known reader path are flagged DEAD_PATH.

  2. SMOKE — if --smoke, runs a 1-month sim window of both baseline and
     candidate, compares final_value + APY to 1e-6. Identical → DEAD.
     This catches kernel-reader bugs the static map can't see (renamed
     keys, gated branches, etc.).

Exit code:
   0  — config is ACTIVE (at least one knob change reaches the kernel)
   1  — config is a NO-OP (every diff key is dead)
   2  — usage error / file missing

Reference (CLAUDE.md):
  §5.13.10  "if optional_field is not None defaults to dead code unless
            verified"
  feedback_config_knob_path_audit memory  "Grep-verify every new knob's
            JSON path against the kernel reader AND verify value differs
            from baseline; pre-flight ~30s saves ~80min per botched panel"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CFG_DIR = REPO / "backtesting" / "renquant_104"

# ── KERNEL-READ PATH MAP (curated from grep on backtesting/renquant_104/kernel) ──
#
# Each entry: glob-style config path → kernel evidence (file:line).
# A diff-key whose JSON path matches one of these globs is ACTIVE.
#
# To extend: grep kernel/ for `cfg.get("<key>")` / `regime_p.get(...)`
# and add the resolved config path here. Keep evidence comments so a
# future reviewer can verify the entry without re-deriving it.
ACTIVE_PATHS: list[tuple[str, str]] = [
    # ── per-regime params (pp_inference.py:_build_exit_params) ───────────
    ("regime_params.*.trailing_stop_trigger_pct", "pp_inference.py:41"),
    ("regime_params.*.trailing_stop_trail_pct",   "pp_inference.py:42"),
    ("regime_params.*.stop_loss_pct",             "pp_inference.py:43"),
    ("regime_params.*.stop_n_sigma",              "pp_inference.py:46"),
    ("regime_params.*.sdl_n_sigma",               "pp_inference.py:55"),
    ("regime_params.*.atr_n_multiplier",          "pp_inference.py:50"),
    ("regime_params.*.max_single_day_loss_pct",   "pp_inference.py:51"),
    ("regime_params.*.max_hold_days",             "pp_inference.py:56"),
    ("regime_params.*.min_model_score",           "task_candidates.py:174"),
    ("regime_params.*.take_profit_pct",           "pp_inference.py"),
    ("regime_params.*.drawdown_resume_pct",       "pp_inference.py"),
    ("regime_params.*.entry_mode",                "pp_inference.py"),
    ("regime_params.*.min_price_move_pct",        "pp_inference.py"),
    ("regime_params.*.long_short_enabled",        "task_short_candidates.py"),
    # ── QP knobs (rotation.joint_actions via _qp_cfg) ────────────────────
    ("rotation.joint_actions.qp_cvar_lambda",          "tasks.py:981"),
    ("rotation.joint_actions.qp_cvar_alpha",           "tasks.py"),
    ("rotation.joint_actions.qp_robust_mu_kappa",      "tasks.py"),
    ("rotation.joint_actions.qp_use_full_sigma",       "tasks.py"),
    ("rotation.joint_actions.qp_ledoit_wolf_lambda",   "tasks.py"),
    ("rotation.joint_actions.qp_tax_aware",            "tasks.py"),
    ("rotation.joint_actions.qp_tax_rate_st",          "tasks.py"),
    ("rotation.joint_actions.qp_signal_decay",         "tasks.py"),
    ("rotation.joint_actions.qp_drawdown_limit",       "tasks.py"),
    ("rotation.joint_actions.qp_no_trade_band_factor", "tasks.py"),
    ("rotation.joint_actions.qp_no_trade_band_cap",    "tasks.py"),
    ("rotation.joint_actions.qp_min_invested_pct",     "tasks.py"),
    ("rotation.joint_actions.qp_sector_cap_enabled",   "tasks.py"),
    ("rotation.joint_actions.qp_correlation_cap_enabled", "tasks.py"),
    # ── QP regime overrides (_resolve_regime_override) ───────────────────
    ("rotation.joint_actions.regime_overrides.*.*",  "tasks.py:397"),
    ("ranking.alpha_to_mu.regime_overrides.*.*",     "tasks.py:535"),
    # ── kelly_sizing knobs (kelly_cfg = ctx.config['ranking']['kelly_sizing']) ──
    ("ranking.kelly_sizing.enabled",                  "task_selection.py:149"),
    ("ranking.kelly_sizing.disable_extra_multipliers","task_selection.py:155"),
    ("ranking.kelly_sizing.top_up_threshold",         "task_topup.py:46"),
    ("ranking.kelly_sizing.topup_conviction_floor",   "task_topup.py:118"),
    ("ranking.kelly_sizing.per_session_buy_cap",      "task_topup.py:159"),
    ("ranking.kelly_sizing.rotation_advantage",       "task_rotation.py:117"),
    ("ranking.kelly_sizing.rotation_target_floor",    "task_rotation.py:529"),
    ("ranking.kelly_sizing.trim_enabled",             "task_trim.py:60"),
    ("ranking.kelly_sizing.trim_threshold",           "task_trim.py:62"),
    # ── tiered_thresholds (read in selection/joint_actions) ──────────────
    ("tiered_thresholds.*.min_model_score",           "selection.py:400, task_joint_actions.py:284"),
    # ── exposure_scaling (top-level) ─────────────────────────────────────
    ("exposure_scaling.*",                            "tasks.py"),
    # ── top-level scalars consulted via ctx.config.get(...) ──────────────
    ("consecutive_sell_signals",  "pp_inference.py"),
    ("min_hold_days",             "pp_inference.py"),
    ("max_concurrent_positions",  "selection.py"),
    ("max_positions_per_sector",  "selection.py"),
    ("sharpe_floor",              "selection.py"),
    ("wash_sale_days",            "task_qp_*"),
    ("bear_defensive_slots",      "task_bear_branch.py"),
    ("bear_defensive_pct",        "task_bear_branch.py"),
    ("defensive_tickers",         "task_bear_branch.py"),
    # ── buy_quality_gates (recently added 2026-05-15) ─────────────────────
    ("buy_quality_gates.*.enabled",            "task_buy_gates.py"),
    ("buy_quality_gates.*.disabled_in_regimes","task_buy_gates.py"),
    # ── Phase 2 long-short ──────────────────────────────────────────────
    ("long_short.enabled",                "task_short_candidates.py"),
    ("long_short.max_shorts",             "task_short_candidates.py"),
    ("long_short.max_short_pct",          "task_short_candidates.py"),
    ("long_short.max_gross_exposure",     "tasks.py"),
    # ── known METADATA / comment fields (always inert) ───────────────────
    # listed so the validator marks them INERT_METADATA, not DEAD_PATH
]
INERT_KEYS = {
    "_2026-05-15_re_eval_hypothesis", "_side_config_label",
    "_activation_log", "_backtest_start_note",
}


def _flatten(prefix: str, obj, out: dict) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}.{i}", v, out)
    else:
        out[prefix] = obj


def _match_glob(path: str, pattern: str) -> bool:
    p_parts = path.split(".")
    g_parts = pattern.split(".")
    if len(p_parts) != len(g_parts):
        return False
    return all(g == "*" or g == p for g, p in zip(g_parts, p_parts))


def _is_active(path: str) -> tuple[bool, str]:
    # Skip top-level inert metadata
    head = path.split(".", 1)[0]
    if head in INERT_KEYS:
        return False, "INERT_METADATA"
    for pat, evidence in ACTIVE_PATHS:
        if _match_glob(path, pat):
            return True, f"ACTIVE ({evidence})"
    return False, "DEAD_PATH"


def static_validate(baseline: dict, candidate: dict) -> tuple[bool, list[str]]:
    b_flat: dict[str, object] = {}
    c_flat: dict[str, object] = {}
    _flatten("", baseline, b_flat)
    _flatten("", candidate, c_flat)
    diff_paths = sorted(set(b_flat) | set(c_flat))
    report: list[str] = []
    n_active = 0
    n_dead = 0
    n_inert = 0
    for p in diff_paths:
        bv = b_flat.get(p, "<absent>")
        cv = c_flat.get(p, "<absent>")
        if bv == cv:
            continue
        ok, why = _is_active(p)
        marker = "✓" if ok else ("·" if "INERT" in why else "✗")
        report.append(f"  {marker} {p}: {bv!r} → {cv!r}   [{why}]")
        if ok:
            n_active += 1
        elif "INERT" in why:
            n_inert += 1
        else:
            n_dead += 1
    summary = (
        f"\nSTATIC SUMMARY: {n_active} ACTIVE / {n_dead} DEAD_PATH / "
        f"{n_inert} INERT (metadata)"
    )
    report.append(summary)
    return (n_active > 0), report


def smoke_validate(
    baseline_cfg_name: str, candidate_cfg_name: str,
    smoke_start: str = "2024-04-01", smoke_end: str = "2024-05-01",
) -> tuple[bool, list[str]]:
    """Run 1-month sim of both configs, compare APY."""
    report: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        outs: dict[str, dict] = {}
        for label, cfg in [("baseline", baseline_cfg_name),
                            ("candidate", candidate_cfg_name)]:
            eq = td / f"{label}.json"
            cmd = [
                sys.executable, "scripts/run_sim_104.py",
                "--strategy-config-name", cfg,
                "--start", smoke_start, "--end", smoke_end,
                "--no-compare", "--no-persist",
                "--equity-json", str(eq),
            ]
            report.append(f"  running smoke {label}: {' '.join(cmd)}")
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
            if r.returncode != 0 or not eq.exists():
                report.append(f"  smoke {label} FAILED rc={r.returncode}")
                report.append(f"  stderr: {r.stderr[-500:]}")
                return False, report
            outs[label] = json.loads(eq.read_text())
        b_apy = outs["baseline"]["apy"]
        c_apy = outs["candidate"]["apy"]
        b_fv  = outs["baseline"]["final_value"]
        c_fv  = outs["candidate"]["final_value"]
        report.append(
            f"  smoke baseline:  APY={b_apy:+.6f}  final_value={b_fv:.2f}")
        report.append(
            f"  smoke candidate: APY={c_apy:+.6f}  final_value={c_fv:.2f}")
        identical = (abs(b_apy - c_apy) < 1e-6 and abs(b_fv - c_fv) < 1e-3)
        report.append(f"  SMOKE VERDICT: {'NO-OP (identical)' if identical else 'ACTIVE (differs)'}")
        return (not identical), report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="baseline config name in backtesting/renquant_104/")
    p.add_argument("--candidate", required=True, help="candidate config name in backtesting/renquant_104/")
    p.add_argument("--smoke", action="store_true", help="also run 1-month smoke sim")
    p.add_argument("--smoke-start", default="2024-04-01")
    p.add_argument("--smoke-end",   default="2024-05-01")
    args = p.parse_args()

    bp = CFG_DIR / args.baseline
    cp = CFG_DIR / args.candidate
    if not bp.exists() or not cp.exists():
        print(f"ERROR: missing {bp if not bp.exists() else cp}", file=sys.stderr)
        return 2
    baseline = json.loads(bp.read_text())
    candidate = json.loads(cp.read_text())

    print(f"\n── static path validation ({args.candidate} vs {args.baseline}) ──")
    static_ok, lines = static_validate(baseline, candidate)
    print("\n".join(lines))

    smoke_ok = True
    if args.smoke:
        print(f"\n── smoke validation ({args.smoke_start}..{args.smoke_end}) ──")
        smoke_ok, lines = smoke_validate(
            args.baseline, args.candidate,
            args.smoke_start, args.smoke_end,
        )
        print("\n".join(lines))

    verdict_static = "ACTIVE" if static_ok else "NO-OP"
    verdict_smoke  = ("ACTIVE" if smoke_ok else "NO-OP") if args.smoke else "skipped"
    print(f"\nFINAL: static={verdict_static}   smoke={verdict_smoke}")
    overall_ok = static_ok and (smoke_ok if args.smoke else True)
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
