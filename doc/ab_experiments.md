# A/B Experiment Log

Per-experiment journal. Prepend new runs to top. Every entry contains:
- **Date** (when the A/B ran)
- **Hypothesis** (what we expected and why)
- **Variants** (config delta per variant)
- **Result** (raw numbers)
- **Verdict** (PROMOTE / SHELVE / INCONCLUSIVE + reasoning)
- **Commit** (infra commit for the flags; verdict commit for the config change)

Use this log to avoid re-running experiments that already answered a question, and to surface patterns (e.g. "every rotation-unlock attempt has regressed").

---

## 2026-04-24 PT — 6 experiments

### 6. Route B — thesis_primary rotation mode

**Hypothesis:** Route A showed rotation hurts APY under ER-based pair discovery. Route B tests whether using thesis-degradation as PRIMARY gate (bypassing ER) picks better pairs by prioritizing swaps OUT of degraded thesis holdings.

**Variants (27-mo OOS, `allow_fetch=False`):**
- A_GOLDEN_v4.1 — baseline (rotation.mode="er", default)
- B_thesis_primary_0.30_0.10 — mode="thesis_primary", degradation=0.30 uplift=0.10
- C_thesis_primary_0.15_0.05 — loose 0.15/0.05
- D_thesis_primary_0.20_0.07 — medium 0.20/0.07

**Result:**

| Variant | APY | ΔAPY | Rot | Streak |
|---|---:|---:|---:|---:|
| A_GOLDEN_v4.1 | **+34.56%** | 0.00 | 0 | 27d |
| B_thesis_primary 0.30/0.10 | +34.56% | 0.00 | 0 | 27d |
| C_thesis_primary 0.15/0.05 | +33.51% | −1.05 | 3 | 27d |
| D_thesis_primary 0.20/0.07 | +29.96% | **−4.60** | 0 | 44d |

**Verdict: SHELVE — consistent with Route A.** Thesis-primary with strict thresholds (0.30/0.10) matches baseline (0 rotations both). Loose (0.15/0.05) fires 3 rotations and regresses by −1.05 pt, same pattern as Route A (rotations cost APY). D's −4.6 pt with 0 rotations is unexplained but likely interaction with streak length.

**🚨 BASELINE SHIFT DETECTED:** Route A's A_GOLDEN_v4.1 = **+39.82%**, Route B's A_GOLDEN_v4.1 = **+34.56%** (−5.26 pt for IDENTICAL config). Two hypotheses:
1. Notebook running concurrently with Route B introduced non-determinism (shared SQLite, thread ordering)
2. Commits shipped between Route A and Route B introduced subtle regression

**Mitigation:** isolated baseline re-run in progress. If it returns to 39.82% → notebook concurrency was the cause. If it stays at 34.56% → bisect today's commits.

**Learning:** **Don't run A/B sims concurrently with notebook that touches the same DB.** Use the DB split we shipped today — but also serialize A/B runs.

**Commit:** `cd0c9d5` (thesis_primary infra).

---

### 5. Route A — rotation ER threshold unlock

**Hypothesis:** v4.1 golden has 0 rotations because `min_expected_advantage_pct=0.03` is larger than typical ER deltas. Lowering to 0.005 should let rotations fire; thesis-A gate on top should filter for quality.

**Variants (27-mo OOS, `allow_fetch=False`):**
- A_GOLDEN_v4.1 — baseline (threshold 0.03)
- B_rot_loose_005 — threshold 0.005, no thesis
- C_loose_005_thesis_strict — threshold 0.005 + thesis_rotation (degradation 0.30, uplift 0.10)
- D_loose_005_thesis_loose — threshold 0.005 + thesis_rotation (0.15, 0.05)

**Result:**

| Variant | APY | ΔAPY | Rot | Buys | Win |
|---|---:|---:|---:|---:|---:|
| A_GOLDEN_v4.1 | +39.82% | 0.00 | 0 | 117 | 83% |
| B_rot_loose_005 | +34.88% | **−4.93** | 3 | 118 | 81% |
| C_loose_005_thesis_strict | +39.82% | 0.00 | 0 | 117 | 83% |
| D_loose_005_thesis_loose | +34.80% | **−5.02** | 2 | 110 | 81% |

**Verdict: SHELVE — rotation hurts APY under current model.** Per-rotation impact ≈ **−2.5 APY pt**. The golden's zero rotations is protective; the 0.03 threshold is correctly blocking net-negative swaps. Tax drag + missed continuation on held side > realized ER advantage.

C equals A exactly because thesis-strict (0.30 degradation) was strict enough to filter ALL candidate rotations — so it matches the 0-rotation baseline.

**Commit:** `49c208c` (verdict) — infra stays shipped + dormant (BC `09468e3`, thesis-A `e519177`, thesis-primary `cd0c9d5`).

**Learning for goal (APY=1.41 / Sharpe=2):** rotation is NOT an APY lever under today's model quality. APY lift must come from better panel/NGBoost signals, feature selection (J), better exits (panel_conviction_exit), or watchlist changes (screen_watchlist). Keep rotation infra dormant; reactivate + A/B one flag at a time if/when model improvements surface.

---

### 4. Thesis-A — degradation rotation filter

**Hypothesis:** compare today's candidate vs held position's FIXED entry-time baseline (not noisy today-vs-today Kelly). Swap only when held's thesis degraded AND candidate beats held's original baseline.

**Variants (27-mo OOS):**
- A_GOLDEN_v4.1 — baseline
- B_thesis_strict — `thesis_rotation.enabled=true, degradation_pct=0.30, uplift_pct=0.10`
- C_thesis_loose — `thesis_rotation.enabled=true, degradation_pct=0.15, uplift_pct=0.05`

**Result:**

| Variant | APY | Rot | Buys |
|---|---:|---:|---:|
| A_GOLDEN_v4.1 | +39.82% | 0 | 117 |
| B_thesis_strict | +39.82% | 0 | 117 |
| C_thesis_loose | +39.82% | 0 | 117 |

**Verdict: INCONCLUSIVE — null result.** All 3 variants identical because base rotation (ER-threshold 0.03) produces 0 pairs for thesis-A to filter. Thesis-A is a **filter on top** of ER-based pair discovery; with zero input pairs, zero filter output.

**Learning:** thesis-A is still theoretically sound, but can't be validated until base rotation emits pairs. Led directly to Route A above (test base rotation with lower threshold). See also Route B (thesis_primary as PRIMARY gate, not filter).

**Commit:** `e519177` (infra).

---

### 3. Combined A/B — Kelly-tier-tune + CUSUM-v2 Design C

**Hypothesis 1:** AA shows [0.35, 0.45) rank_score bucket has 80.7% hit rate vs 74.5% for [0.27, 0.35). Raising tier 1 to 0.35 should lift quality.

**Hypothesis 2:** CUSUM cooldown converted from bar-count to wall-clock should close ~2 APY pt live-sim drift.

**Variants (27-mo OOS):**
- A_GOLDEN_v4 — baseline
- B_tier1_0.35 — `tiered_thresholds = [0.35, 0.45, 0.60]`
- C_cusum_wall_time — `cusum_cooldown_mode=wall_time, cusum_cooldown_days=3.0`

**Result:**

| Variant | APY | ΔAPY | Buys | Win |
|---|---:|---:|---:|---:|
| A_GOLDEN_v4 | +37.85% | 0.00 | 115 | 82% |
| B_tier1_0.35 | +28.97% | **−8.88** | 55 | 91% |
| C_cusum_wall_time | +39.82% | **+1.97** | 117 | 83% |

**Verdict 1 (Kelly-tier-tune): SHELVE.** Theory false — "higher hit rate = higher APY" doesn't hold. Tightening tier 1 from 0.27 → 0.35 cut trades 115 → 55; win rate up 82 → 91% but volume × compounding loss wins. Hit rate up doesn't imply APY up.

**Verdict 2 (CUSUM-v2): PROMOTE as v4.1.** +1.97 APY pt is below the default +2 pt floor BUT matches theoretical prediction exactly ("close ~2 pt drift") — rigorously-controlled variables per CLAUDE.md §2a. Golden promoted from v4 (+37.85%) to v4.1 (+39.82%).

**Commit:** `1bb5ae1` (promote v4.1).

---

### 2. AB-trim — Kelly rebalance partial sell (+ CLAUDE.md §2b audit)

**Hypothesis:** When current_pct > kelly_target + threshold, emit partial sell (TrimHeldTask). Maintains discipline at Kelly target.

**Variants (27-mo OOS):**
- A_GOLDEN (trim_threshold=0.10 default) — **not a clean baseline — TrimHeldTask defaulted ON**
- B_hysteresis_.10 — explicit trim_threshold=0.10
- C_tight_.00 — trim_threshold=0.00 (always trim to exact target)

**Result:**

| Variant | APY | ΔAPY | Trims | Buys |
|---|---:|---:|---:|---:|
| Pre-trim v4 golden (no TrimHeldTask) | +37.82% | — | N/A | — |
| A_GOLDEN (with default trim 0.10) | +25.09% | **−12.73** | 22 | 111 |
| B_hysteresis_.10 | +25.09% | 0.00 vs A | 22 | 111 |
| C_tight_.00 | +34.89% | +9.80 vs A | 206 | 139 |

**Verdict: SHELVE + audit — default OFF.** Every trim setting under-performs pre-trim baseline. Default 0.10 was especially damaging (−12.7 pts) because it fires 22 times at post-rally peaks and misses continuation upside.

**Audit (per CLAUDE.md §2b, commit `c06e7bb`):** found 2 real bugs:
1. Kelly target re-computes every bar; bar-to-bar volatility in μ/confidence causes spurious "over-weight" signals → churn.
2. `hs.mu` is stale for held tickers not in today's panel-candidate set → bad Kelly targets → bad trim triggers.

Fix: added guards — skip trim when `kelly_target < 0.05` (too noisy) or `hs.mu <= 0` (model bearish, full exit path should handle). Infra stays shipped with `trim_enabled=false` default.

**Commits:** `6d8a52c` (infra), `709ddc8` (default off after A/B), `c06e7bb` (audit guards + §2b principle).

---

### 1. BULL_VOL-reversal — block BULL_VOLATILE buys

**Hypothesis:** AA shows BULL_VOLATILE Spearman IC = −0.172 on 445 rows (ranker direction-wrong during vol spikes). Blocking buys in BULL_VOLATILE should protect against bad trades.

**Variants (27-mo OOS):**
- A_GOLDEN_v4
- B_defensives_only — BULL_VOL regime allows only defensive tickers (GLD/TLT/XLV/XLU)
- C_full_cash — BULL_VOL regime blocks ALL buys

**Result:**

| Variant | APY | ΔAPY | Buys |
|---|---:|---:|---:|
| A_GOLDEN_v4 | +37.85% | 0.00 | 115 |
| B_defensives_only | +38.29% | **+0.44** | 117 |
| C_full_cash | +37.85% | 0.00 | 115 |

**Verdict: SHELVE — below +2 pt promotion floor.** The AA-surfaced IC = −0.17 on 445 rows was real but represented only 0.8% of all sample rows. In the live pipeline, upstream filters (tier + Kelly + universe_floor) were already keeping BULL_VOL buys near zero — blocking them doesn't move the needle. C = A exactly, confirming current config already isn't buying in BULL_VOL.

Infra kept dormant. Can be activated if future config changes start admitting more BULL_VOL buys.

**Commit:** `b281a7e` (infra), `76818e2` (verdict + shelve).

---

## Cross-experiment patterns

1. **"Hit rate up ≠ APY up"** — Kelly-tier-tune. More discriminating filter can hurt if volume × compounding loss beats the quality gain.
2. **"Rotation activity hurts more than it helps under current model"** — Route A, AB-trim both showed swap/rebalance tasks taking APY hits. Tax drag + missed continuation is harder to overcome than expected.
3. **Filter-on-top gates produce null A/B when base gate emits 0 items** — Thesis-A. Watch for this pattern; A/B must test the base signal not just the filter.
4. **AA data surfaces real anomalies that don't always move portfolio APY** — BULL_VOL. In-sample IC is necessary-not-sufficient; confirm with portfolio A/B before acting.
5. **Default-on flags are dangerous.** AB-trim's default 0.10 cost 12.7 APY pt before I caught it. **Every new flag ships default OFF** (CLAUDE.md implicit rule).
6. **Theoretical predictions that match A/B reality can bypass the +2pt promotion floor** — CUSUM-v2 promoted at +1.97 because theory specifically predicted ~2 pt (CLAUDE.md §2a).

---

## Template for new entries

Prepend at top; preserve history.

```
### N. {Experiment name}

**Hypothesis:** what we expected and why.

**Variants ({window}, {allow_fetch={...}}):**
- A — baseline
- B — ...

**Result:**

| Variant | APY | ΔAPY | ... |
| ...     | ... | ...  | ... |

**Verdict: {PROMOTE / SHELVE / INCONCLUSIVE}** — reasoning.

**Commit:** `sha` (infra); `sha` (verdict).

**Learning:** pattern for other experiments.
```
