# Fix: shadow PatchTST config-fingerprint re-stamp (shadow e2e was dark 3 days)

2026-06-25.

## Symptom
The daily SHADOW e2e run (`strategy_config.shadow.json`, PatchTST primary) fail-closed for
**3 consecutive days (06-23/24/25)**: `P-CONFIG-FP [HARD] fingerprint mismatch` →
`panel_scorer_config_mismatch` cleared ALL ~67 buy candidates → "no trade
(panel_scoring_fail_closed)". So the shadow's no-trade was a **contract failure, not a model
decision** — the shadow comparison leg produced nothing for 3 days. (The PROD leg was
unaffected: it uses the XGB scorer whose fp matches live; PatchTST runs there only via
ApplyShadowScoringTask, which doesn't trip the strict gate.)

## Root cause
The PatchTST shadow artifact's stamped `config_fingerprint` (`sha256:14586756…`) no longer
matched the live config (`sha256:f8fb2259…`). `diff_fields=['sector_map','watchlist']`. The
drift is **purely ADDITIVE**: the watchlist grew 142→145 (+CRWV, +RKLB, +SPCX), and sector_map
gained 18 labels — **no removals, no sector reassignments**. The artifact was simply stamped
against the pre-growth config.

## Why a re-stamp (not a retrain) is correct here
- `asset_embeddings=False` → no per-ticker embeddings, so new tickers don't need a trained
  slot; they're scored cross-sectionally by the same weights.
- watchlist feeds CSRankNorm, but adding 3 names to a ~142-name cross-section shifts each
  name's rank by <1.5% — immaterial for a rank-normalized feature. SPCX (9 bars) self-filters.
- sector_map only feeds downstream fundamental fills / sector caps, not the trained weights;
  and nothing was reassigned. lookahead_days/objective/label unchanged (the canonical tool's
  compatibility check PASSED). A retrain for a +3-name additive change would be overkill.

## Fix
Re-stamped the artifact sidecar with the canonical tool (computes the fp via the SAME
`renquant_common.config_consistency.fingerprint_config` the preflight uses — not a hand-faked
hash):
```
scripts/stamp_patchtst_fingerprint.py \
  --artifact-meta artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt.metadata.json \
  --strategy-config <pinned strategy-104 strategy_config.shadow.json> \
  --expected-label-col fwd_60d_excess --write
```
→ `config_fingerprint = sha256:f8fb2259b2bf1537`, watchlist 145. (Backup kept:
`*.metadata.json.bak.20260625-restamp`.)

## Verified end-to-end (one-shot shadow e2e, readonly, isolated alpaca_shadow)
- `preflight ✓ P-CONFIG-FP [HARD] fingerprint match sha256:f8fb2259b2bf1537`
- `Config-consistency: hf_patchtst…pt OK fp=f8fb2259` — scorer LOADED (no fail-close)
- `AssembleInferenceMatrixTask: X.shape=(71, 172)` — shadow now SCORES (not cleared)

## Follow-ups (separate)
- **Divergent runtime config (footgun):** `backtesting/renquant_104/strategy_config.shadow.json`
  is dirty/uncommitted at the OLD 142-watchlist (fp 14586756) while the pinned config is 145
  (fp f8fb2259). The live run reads the pinned (145), so it's not breaking the run, but a stale
  hand-edited runtime copy is a landmine — reconcile/revert it to the pin.
- This unblocks the analyst-revision shadow model (#193), whose path-to-live targets this same
  shadow leg.
