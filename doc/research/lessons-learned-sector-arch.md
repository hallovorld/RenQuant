# Lessons learned — sector-aware architecture work (2026-04-25 → 2026-05-01)

Unified record of what we've learned across the wl178 expansion saga
and the design-v2 sector-aware architecture work. Captures protocol,
mechanism, bugs, theory, and process — so an independent operator
(or future-Claude) can navigate the same minefield without re-walking
each step.

---

## TL;DR

1. **List expansion alone never works** on a heterogeneous universe with
   the cross-sectional rank-pairwise loss. Confirmed by E5 (B1=227 mutual-fund),
   E17 (wl178=quality-filter), E21 (wl174=ETF removal), and Layer-1-alone
   (2026-05-01). Architecture intervention is required.
2. **Hard-split per-sector is wrong** — Witter 2025 (JMS) ran exactly
   that A/B and global-pooled wins. Audit literature BEFORE designing.
3. **Single-feature IC ≠ ensemble IC.** Layer 1's `_sr` features carried
   IC up to +0.035 individually but the XGBoost ensemble extracted zero.
   Tree models attention-dilute under correlated features (Hastie ESL §10.13.1).
4. **Coverage matters**: Layer 1 was effectively disabled for 42% of
   wl178 tickers because the sector_map was incomplete (104/178 entries).
   Always audit data-pipeline COVERAGE before declaring an architecture
   experiment failed.
5. **Triage discipline** (CLAUDE.md §2b + the protocol the operator
   defined 2026-05-01): (a) pinpoint the failure step, (b) cite theory,
   (c) dig for implementation bugs, (d) decide retry — in that order.

---

## 1 · The diagnostic protocol that worked

When wl178 retrains failed silently with negative IC, we built a
three-check diagnostic before committing to architecture work:

| Check | Question | Tool |
|---|---|---|
| Check 1 | Random A/A split of wl178 — do BOTH halves recover production-like IC? | `audit_wl178_failure.py` emits configs; manual `train_104.py --skip-baseline` per half |
| Check 2 | Are sectors structurally heterogeneous? KS test on per-feature distributions | scipy `stats.ks_2samp` per sector pair × per feature; aggregate by median |
| Check 3 | Is the wl178 build pipeline corrupted? (cache mismatch, normalization drift) | feature consistency cross-check vs wl103 |

**Verdict matrix encoded in the script** — output is a single
"PROCEED / HALT" verdict that gates the architecture work. Don't
write code until the verdict says proceed.

**Cost:** ~5 min for Check 2+3 (read-only); ~60 min for Check 1
(retrains) but those run in background. Total wallclock under 1 hour.

**Generalizes to any architecture decision.** Before investing 1+ week
in NN backends, run a 3-check diagnostic to pinpoint the failure mode.

---

## 2 · Literature audit BEFORE design

Lesson burned in by Witter 2025: we proposed per-sector hard split as
"the obvious fix." Witter ran exactly that A/B and the conclusion was
the OPPOSITE of our intuition. Had we written code first, we'd have
spent 1-2 weeks confirming a known dead end.

**Protocol:**
1. State the hypothesis as a falsifiable claim.
2. Spawn a research agent (or do a focused arXiv/SSRN search) to find
   3+ papers that ran the SAME or similar A/B.
3. Read OSS reference implementations — qlib, alpha-mind,
   FSTGAT, MIGA — for the design pattern they actually use.
4. If literature contradicts: revise design BEFORE writing code.
5. If literature supports: cite specifically in the design doc and
   proceed.

**Red flag tells you literature audit is overdue:**
- "I think this should work because X."
- "The fix is obvious."
- "We just need to..."

**Concrete saves so far:**
- Audit refuted hard-split design → switched to Layer 1+5 helpers + planned graph/MoE.
- Saved approximately 2 weeks of dead-end implementation.

---

## 3 · Mechanism map — why heterogeneity breaks rank-pairwise

```
Heterogeneous universe (mixed sectors)
        ↓
 Per-ticker feature distributions differ on raw scale
   (tech mom_12_1 ∈ [-0.15, 0.40]   vs   utility ∈ [-0.04, 0.06])
        ↓
 Global cross-sectional z-score normalizes by FULL-universe mean+std
   → single tech ticker dominates the upper tail every date
   → utility ticker permanently in the middle
        ↓
 Pairwise rank loss compares (tech_z, utility_z) where the gap
 reflects scale not skill. Gradient signal swamped by scale noise.
        ↓
 1.  Train IC drops (model can't fit; +0.118 → +0.085 on wl178)
 2.  OOS IC collapses (no signal to generalize from; ~+0.04 → ~0)
 3.  Model converges to constant prediction at early-stop guard
```

**The four published papers that anchor this:**
- Witter 2025 (JMS Vol. 10(3)) — direct A/B, pooled wins.
- Gu-Kelly-Xiu 2020 (RFS) — pooled with sector dummies is the canon.
- Poh-Lim-Zohren-Roberts 2020 (arXiv:2012.07149) — the rank:pairwise
  paper itself uses one global model on heterogeneous futures
  (commodities/FX/rates/equity) by feature normalization, not splits.
- Feng et al. 2019 (arXiv:1809.09441) — sector relations encoded as
  graph edges, not as separate models.

Together they tell us: heterogeneity is real, it does break rank loss,
but the canonical fix is **conditioning the pooled model**, not
splitting it.

---

## 4 · Five-layer architecture (design v2)

The post-audit design — each layer flag-gated, independently A/B-able:

| L | Description | Status | Mechanism it adds |
|---|---|---|---|
| L1 | per-`(date, sector)` rank-norm `_sr` columns | ✅ shipped helper + Task wired | Sector-relative percentile so feature scale is comparable across sectors |
| L2 | sector one-hot `sector_<S>` columns | ✅ shipped Task | Explicit sector identity for tree splits to anchor on |
| L3 | Temporal Graph Convolution (NN) | 📋 design doc only | Pair-level relation: same-sector tickers' representations attend to each other |
| L4 | Soft MoE with sector-group gating (NN) | 📋 design doc only | Specialist experts per cluster of sectors; soft routing |
| L5 | Empirical-Bayes shrinkage on per-sector pct | ✅ shipped helper | Smooths small-sector noise toward global benchmark |

**Layering rationale:** each layer is one mechanism. Test independently
(flag-toggle), stack only when a layer passes its own A/B. Don't ship
the whole stack at once — when the result is bad you can't pinpoint
which mechanism failed.

---

## 5 · Implementation bugs we found (and what they teach)

### BUG 1 — sector_map coverage 58% on wl178

The strategy_config carried sector labels for 104 of 178 tickers.
Layer-1 fallback for the other 74 was global percentile, which is
information-equivalent to a rank version of `_z`. Result: 42% of the
universe got Layer 1 effectively disabled, contaminating the ensemble
with redundant signal.

**Lesson:** before declaring an architecture experiment failed, AUDIT
COVERAGE OF EVERY DATA INPUT THE ARCHITECTURE DEPENDS ON. The sector
map looked fine ("106 entries") until we cross-checked against the
watchlist and saw the 42% gap.

**Test for this class of bug:** write a `conftest`-style fixture that
asserts `set(sector_map.keys()) ⊇ set(watchlist)` at config load time.
A new Task `ConfigCoverageGuard` should fire warnings (or hard-fail
in dev) on coverage gaps in any of:
- `sector_map` vs `watchlist`
- `defensive_tickers` vs `watchlist`
- `correlation_matrix` vs `watchlist`
- ticker_sectors derived from sector_map

### BUG 2 — small sectors dropped by `min_sector_size=5`

Energy (3), utility (1), commodity (1) on wl178 fell below the floor
and routed to global percentile. Five more tickers losing sector-
relative treatment.

**Lesson:** parameter defaults pick up implementation assumptions
silently. `min_sector_size=5` came from EB shrinkage math (Robbins
1956 — order-statistic noise dominates below ~5). But for `_sr`
*assignment* (not for `_sr` reliability), 3 is plenty and fewer
tickers get demoted. Two different defaults for two different uses.

### BUG 3 — feature redundancy not designed for

Layer 1 added `_sr` ALONGSIDE `_z`. For the 79 tickers in
fully-covered sectors, `_sr` is genuinely sector-relative. For the
99 tickers without sector-relative treatment (BUGS 1+2), `_sr` ≈
rank version of the same raw value. Tree ensemble sees redundant
info → attention dilutes → train_ic regresses.

**Lesson:** when adding new representations of the same underlying
data, design the substitution path as well as the augmentation path.
Add `replace_z` flag so operators can run "_sr only" experiments and
isolate the augmentation effect from the substitution effect.

### BUG 4 + 4.5 — config-path bug, AND test mocked the same wrong path

Two bugs in series. BUG 4 added a `monotone_constraints` rewrite when
`replace_z=True` drops `_z` columns. BUG 4.5 was that the rewrite
targeted `ctx.config["monotone_constraints"]` (top level) but the
real config nests it at `ctx.config["panel_ltr"]["monotone_constraints"]`.

**The smoking gun**: my own unit test for BUG 4 mocked the SAME wrong
path the bug used. Test passed; production crashed. Both pieces of
code shared the same wrong assumption, so the test confirmed the
buggy code matched the buggy expectation.

**Lesson:** when writing a test for a config-path-dependent fix, mock
with the ACTUAL config structure (load real strategy_config.json,
deepcopy, mutate). Don't mock at a level that abstracts away the
nesting — that's where path bugs hide.

**Process improvement adopted:** for any new code that reads
`ctx.config[...]`, the test must include at least one assertion that
loads the real strategy_config.json and verifies the read produces
the expected value at the expected path. The
`test_replace_z_real_strategy_config_structure` test in
`tests/test_sector_rank_norm_task.py` is the pattern.

---

## 6 · Process patterns that worked

### Branch isolation
Every architecture experiment goes on its own `exp/<name>` branch via
worktree. Main stays clean, multiple experiments run in parallel
without artifact contention, merge only when criteria pass.

```
git worktree add /tmp/renquant-<exp> -b exp/<descriptive-name>
mkdir -p /tmp/renquant-<exp>/data
ln -s $(pwd)/data/ohlcv /tmp/renquant-<exp>/data/ohlcv
```

### Background dispatch + no polling
Long-running training goes via `run_in_background=true`. Don't poll
for status — wait for completion notifications. Saved hours of
context-switching during this work.

### Compare script as the meta-table
`scripts/compare_panel_experiments.py` parses logs from conventional
paths and emits one row per experiment with key metrics. Single
source of truth for "where each architecture variant landed" without
grep-the-logs ceremony.

### Failed-experiments-log as the institutional memory
Per CLAUDE.md §5.7 — every failed experiment, partial result, or
abandoned approach gets a same-day entry. E1 (M2 horizon blender),
E5/E17/E21 (universe expansion), and E_LAYER1_ALONE all landed
there. Future-Claude or Codex reading the repo cold sees both what
we built and what we tried-and-rejected.

---

## 7 · The "explain how it failed" triage protocol (operator-defined 2026-05-01)

Acknowledged as the standard going forward. When an experiment lands
a bad result:

1. **Pinpoint the failure step** — find the EXACT diagnostic that
   first deviated from expectation. (Layer 1: train_ic regressed
   from +0.116 → +0.069 — adding signal made fit harder, the
   smoking gun.)

2. **Cite theory for the failure mode** — pull 1-3 specific papers
   that explain WHY the observed pattern happens. (Hastie ESL on
   correlated features → tree ensemble attention dilution.)

3. **Implementation bugs first, theoretical "feature-not-good-enough"
   second.** Audit data coverage, code paths, integration points
   BEFORE accepting that the architecture is wrong. Three bugs
   surfaced for Layer 1 alone.

4. **Decide retry.** Worth-worthy if (a) the bugs are cheap to fix,
   (b) theoretically the architecture should still work post-fix,
   (c) compute cost is bounded. Layer 1 RETRY scheduled because
   sector_map fix is ~30 min and theory still says Layer 1 should
   help once coverage is complete.

This protocol prevents the failure mode of "architecture experiment
returned bad result → declare architecture wrong" — which is a
specific instance of CLAUDE.md §2b ("audit before accepting
unexpected A/B results").

---

## 8 · Open questions for next session

1. **Does Layer 1 work with full sector_map coverage?** — answer
   pending the v2-config retrain (BUG 1+2 fixed).
2. **Does substitution mode (`_sr` only, no `_z`) outperform
   augmentation?** — answer pending a `replace_z=true` retrain.
3. **Does Layer 2 (one-hot) on TOP of Layer 1 produce the lift Layer
   1 alone couldn't?** — answer pending the v2 + L1+L2 retrain.
4. **If all of L1 / L1+L2 / substitution underperform, escalate to
   Phase C graph attention?** — design doc shipped; trigger gate
   defined.

---

## 9 · Tooling / artifacts produced this round

| Artifact | Purpose |
|---|---|
| `scripts/audit_wl178_failure.py` | 3-check diagnostic — KS sectors + A/A test + feature consistency |
| `scripts/compare_panel_experiments.py` | Meta-table over experiment logs |
| `scripts/build_universe.py` | Russell 1000 ticker pull |
| `scripts/fetch_universe_ohlcv.py` | yfinance backfill of universe OHLCV |
| `scripts/build_sector_map.py` | Pulls IWB sectors → completes wl178 sector_map (BUG 1 fix) |
| `training_panel/factors.py::cross_sectional_rank_within_sector` | Layer 1 helper |
| `training_panel/eb_shrinkage.py` | Layer 5 helper |
| `training_panel/pp_panel_training.py::SectorRankNormalizeTask` | Layer 1 Task |
| `training_panel/pp_panel_training.py::SectorOneHotTask` | Layer 2 Task |
| `doc/research/per-sector-architecture-plan.md` | Design v2 doc |
| `doc/research/phase-c-graph-attention-design.md` | Phase C design doc (NN graph) |
| `doc/research/macro-v3-isolation.md` | Quarantine notes for the WIP that was breaking main CI |
| `doc/research/branch-pointers.md` (on main) | Index of active experimental branches |
| `doc/research/failed-experiments-log.md` | E_LAYER1_ALONE entry |
| **This doc** | Unified lessons across the round |
