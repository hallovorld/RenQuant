"""TDD: PROD XGB scorer's TRULY out-of-sample IC must be > 0.

Failure case being pinned: 2026-05-20 audit found `train_production_model.py`
trains on ALL panel rows with valid fwd_60d label (date ≤ panel_max - 60d),
then `panel-ltr.alpha158_fund.json` is "evaluated" on the same date range
during dashboards/eval — all in-sample. Any IC claim under this setup is
fit-quality, not OOS skill.

This test forces a TRULY OOS measurement:
  1. Use a cutoff-trained artifact (artifacts/prod/truly_oos_eval/) trained
     with --train-cutoff 2024-07-01.
  2. Compute IC on dates STRICTLY AFTER cutoff up to last labeled date
     (panel_max - 60d).
  3. Assert mean IC > 0 and at least 55% of days have positive IC.

This test SKIPS gracefully when the truly-OOS artifact doesn't exist yet —
it's a TDD pin, not a CI blocker. The companion script
`scripts/retrain_prod_truly_oos.py` produces the artifact.

Source: doc/research/2026-05-20-prod-signal-tdd.md (this session).
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
ART  = REPO / "backtesting/renquant_104/artifacts/prod/truly_oos_eval"
EVAL_JSON = ART / "eval_truly_oos.json"


def _skip_if_missing() -> None:
    if not EVAL_JSON.exists():
        pytest.skip(
            f"Truly-OOS eval not run yet — produce via\n"
            f"  python scripts/retrain_prod_truly_oos.py --train-cutoff 2024-07-01\n"
            f"  python scripts/eval_truly_oos.py --artifact-dir {ART}\n"
            f"Expected JSON at: {EVAL_JSON}"
        )


def _load() -> dict:
    return json.loads(EVAL_JSON.read_text())


class TestProdSignalTrulyOOS:
    """Pin the truly-OOS skill bound."""

    def test_eval_window_strictly_post_cutoff(self):
        _skip_if_missing()
        e = _load()
        cutoff = e["train_cutoff"]
        first_eval_date = e["eval_dates"][0]
        assert first_eval_date > cutoff, (
            f"eval first date {first_eval_date} must be > train cutoff {cutoff} "
            f"(otherwise IC is in-sample contaminated)"
        )

    def test_mean_ic_positive(self):
        _skip_if_missing()
        e = _load()
        ic = float(e["ic_mean"])
        assert ic > 0, (
            f"Truly-OOS mean IC = {ic:+.4f} ≤ 0. The model has no out-of-sample "
            f"skill on the period {e['eval_dates'][0]} → {e['eval_dates'][-1]} "
            f"(n_dates={len(e['eval_dates'])}). Either the in-sample IC of "
            f"+0.105 was overfitting OR the regime has shifted post-2024-07."
        )

    def test_positive_day_rate_above_55pct(self):
        _skip_if_missing()
        e = _load()
        n_pos = int(e["n_pos_days"])
        n_tot = len(e["eval_dates"])
        rate = n_pos / n_tot
        assert rate >= 0.55, (
            f"Only {n_pos}/{n_tot} ({100*rate:.1f}%) days had positive IC. "
            f"Random guess = 50%; signal of any strength would consistently "
            f"exceed 55%. Suggests the +0.105 in-sample IC was tail-dominated."
        )

    def test_top10_realized_alpha_positive(self):
        """Out-of-sample, top-10 picks should realize > 0 alpha vs universe."""
        _skip_if_missing()
        e = _load()
        alpha = float(e["top10_alpha"])
        assert alpha > 0, (
            f"Truly-OOS top-10 alpha = {alpha:+.4f} ≤ 0. Top picks "
            f"underperformed universe-mean OOS — model selection is "
            f"NOT alpha-positive on unseen data."
        )

    def test_top10_alpha_majority_positive_per_regime(self):
        """Per-regime: at least 3 of {BULL_CALM, BULL_VOL, BULL_STRONG, BEAR,
        CHOPPY} must show positive top-10 alpha. Per PRIME DIRECTIVE
        (regime-conditional strategy), pooled-mean alpha hiding losses in 2+
        regimes is misleading."""
        _skip_if_missing()
        e = _load()
        pr = e.get("per_regime", {})
        pos = [r for r, d in pr.items()
               if d.get("top10_alpha", 0) > 0 and d.get("n", 0) >= 3]
        neg = [r for r, d in pr.items()
               if d.get("top10_alpha", 0) <= 0 and d.get("n", 0) >= 3]
        assert len(pos) >= 3, (
            f"Only {len(pos)} regime(s) had positive OOS top-10 alpha "
            f"(positive: {pos}; negative: {neg}). PRIME DIRECTIVE requires "
            f"regime-conditional value; ≤ 2 winning regimes = strategy "
            f"applies in too few regimes for a global deploy."
        )

    def test_dsr_passes_promotion_gate(self):
        """CLAUDE.md §5.13.4 promotion gate: DSR > 0.5. Pins the run of
        scripts/dsr_pbo_truly_oos.py — if a future retrain produces a
        signal that fails DSR, this test fails and forbids silent promote."""
        _skip_if_missing()
        e = _load()
        if "dsr" not in e:
            pytest.skip("DSR not computed yet — run scripts/dsr_pbo_truly_oos.py")
        dsr = float(e["dsr"])
        assert dsr > 0.5, (
            f"DSR = {dsr:.3f} ≤ 0.5 — signal does not survive selection-"
            f"bias correction (n_trials={e.get('dsr_n_trials','?')}). "
            f"Per CLAUDE.md §5.13.4, this is below Tier 3 promotion gate."
        )

    def test_pbo_flagged_on_artifact(self):
        """PBO is informative for regime-selection overfit. Pin the metric
        so the artifact stays up-to-date even if value > 0.5 (regime-
        selection IS overfit currently — PRIME DIRECTIVE gate addresses
        this by disabling BULL_CALM)."""
        _skip_if_missing()
        e = _load()
        assert "pbo" in e, (
            "PBO must be present in eval_truly_oos.json — run "
            "scripts/dsr_pbo_truly_oos.py to compute"
        )
