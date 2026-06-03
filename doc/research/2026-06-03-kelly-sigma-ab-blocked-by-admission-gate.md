# 2026-06-03 — Kelly σ-horizon A/B is null with golden base + v2 WF manifest

**Status**: experiment-design finding. The first execution of the Kelly
σ-horizon A/B (the experiment prescribed by
[`2026-06-03-kelly-sizing-audit.md`](2026-06-03-kelly-sizing-audit.md), PR
#158) produced a **null result by construction** — not because the
σ-horizon change has no effect, but because the chosen config + manifest
pairing trades zero times across the whole window. Documents the root
cause and the corrected recipe so the next run is valid.

**Per §7.12** (unexpected result → audit before accepting): theory predicts
the σ-horizon fix changes Kelly target sizes; the run showed no difference.
First hypothesis = the experiment is mis-specified, not "the fix does
nothing." That hypothesis is confirmed below.

---

## 1 · What was run

```
scripts/run_kelly_sigma_horizon_ab.py --execute --parallel-seeds \
  --match-base-config-to-manifest \
  --manifest-path artifacts/sim/walkforward_manifest_v2_20260602.json \
  --output-dir doc/research/evidence/2026-06-03-kelly-sigma-horizon-ab
```

- `--match-base-config-to-manifest` selected **`strategy_config.golden.json`**
  (fingerprint `sha256:14586756d4f67691` matches the v2 WF artifacts).
- Control = golden; treatment = golden + `ranking.kelly_sizing.sigma_horizon_days = 60`.
- 5 seeds each + A/A resplit, window 2024-01-02 → 2026-03-28 (27-month OOS).
- Killed after ~1h once the null cause was confirmed (§6.4 — no point
  burning compute on a guaranteed-null run).

## 2 · Symptom

The sim log filled with:

```
RegimeModelAdmissionTask: regime=BULL_CALM decision=BLOCK
    reason=regime_admission:no_trade_monotonicity candidates_blocked=108
NoCandidateAlert: 442 consecutive days with zero candidates surviving
    CandidateJob (limit=15) — ScoreBuyTask rejecting all. Regime=BULL_CALM
NoTradeAlert: 442 consecutive days with zero orders (limit=15)
```

A 442-day zero-candidate streak across a ~565-trading-day window = the
backtest essentially never opens a position.

## 3 · Root cause

`job_panel_scoring.py::_trade_monotonicity_admission` blocks a regime when
the WF artifact lacks `wf_gate_metadata.trade_monotonicity`:

```python
def _trade_monotonicity_admission(metadata, regime):
    wf = metadata.get("wf_gate_metadata") ...
    tm = wf.get("trade_monotonicity") ...
    if not isinstance(tm, dict) or not tm:
        return False, "regime_admission:no_trade_monotonicity", {}
```

Checked all **39** artifacts under
`artifacts/walkforward_v2_20260602/*/panel-ltr.json`:

| Key | Present? |
|---|---|
| `wf_gate_metadata` | **No** (0/39) |
| `trade_monotonicity` | **No** (0/39) |

The v2 walk-forward artifacts were trained WITHOUT the trade-monotonicity
gate metadata that the golden config's `RegimeModelAdmissionTask` requires.
So admission returns BLOCK for every cut, in every regime → every candidate
is vetoed before sizing → zero trades.

## 4 · Why this makes the Kelly A/B null (not just "no effect")

The Kelly σ-horizon change acts **only on position sizing** — it rescales
σ so `f* = μ/σ²` uses a 60-day σ instead of an annualized σ. That only
matters once a position is being opened or topped up.

The admission gate sits **upstream** of Kelly sizing and is **identical in
both arms** (control and treatment share the same golden admission config
and the same WF artifacts). So:

```
A_golden:            admission BLOCK → 0 trades → flat cash curve
B_sigma_horizon_60:  admission BLOCK → 0 trades → flat cash curve  (identical)
ΔSharpe = 0, Δcash% = 0, ΔAPY = 0   — for the wrong reason
```

A null A/B here says nothing about whether matching the σ horizon helps.
The signal is masked by a confound that zeroes out both arms.

## 5 · Corrected recipe (for the re-run)

The fix is to make the A/B trade, while keeping the σ-horizon change as the
ONLY difference between arms. Three options, preferred order:

1. **Disable the regime-model admission gate in the A/B sim config only.**
   The `trade_monotonicity` admission is a model-promotion gate, not part
   of the sizing question under test. It is identical in both arms, so
   removing it for the A/B isolates the Kelly effect without biasing
   control vs treatment. Add to the A/B base+treatment configs:
   `ranking.regime_model_admission.enabled = false` (verify exact key) OR
   point the run at a sim config that doesn't gate admission. **Cheapest,
   methodologically clean.**

2. **Use WF artifacts that carry `wf_gate_metadata.trade_monotonicity`.**
   Either regenerate the v2 artifacts with the gate metadata, or select a
   manifest whose artifacts already include it. Most faithful to prod, but
   requires an artifact rebuild.

3. **Run a full-window point-in-time backtest instead of the WF manifest**
   (no per-cut admission gate), accepting the lower fidelity to the
   prod walk-forward path.

`scripts/run_kelly_sigma_horizon_ab.py` should additionally **fail loud
when total trades across the control arm is ~0**, instead of emitting a
verdict over a no-trade backtest — a no-trade A/B is never a valid Tier-3
input and should be caught before the promotion gate reads it. (Follow-up
hardening, separate PR.)

## 6 · What is NOT concluded

- This does **not** show the σ-horizon fix is ineffective. The audit's
  mechanism argument (annualized σ underweights Kelly ~4× — see
  `2026-06-03-kelly-sizing-audit.md` §2) still stands and is untested.
- This does **not** change any production config. Golden is unchanged.

## 7 · Next action

Re-run the A/B under recipe option 1 (admission gate disabled in the A/B
configs only), confirm the control arm actually trades (non-zero orders),
then read the per-regime ΔSharpe / Δcash% with the §7.2 placebo battery.
That run, if it trades, is the real test of the audit's hypothesis.

---

Agent-Origin: Claude
