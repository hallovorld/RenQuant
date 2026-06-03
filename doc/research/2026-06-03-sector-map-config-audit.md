# 2026-06-03 — Sector-Map + Per-Sector Cap Config Audit

**Purpose**: §8 Step 4 prerequisite. The Step 4 offline walk-forward A/B
replay compares 5+ allocators (current QP, simplified-QP, Hybrid Option F,
inverse-vol top-K, equal-weight top-K) through the **single
`ConstraintSnapshot` contract**
(`backtesting/renquant_104/kernel/portfolio_qp/constraint_snapshot.py`,
field `sector_indicator` / `sector_cap_vec` / `sector_names`).

Before the per-cut WF replay loader (Step 4e) or the Hybrid Option F
allocator (Step 4d) can populate that snapshot, the offline tooling has
to know:

1. The exact ticker → sector mapping currently in production.
2. The exact per-sector cap currently in force, per regime.
3. Which tickers in the prod universe currently have NO sector row, and
   how the QP guards them.
4. The expected `ConstraintSnapshot.sector_indicator` and
   `sector_cap_vec` shapes the replay loader must emit.

This is **discovery-only**. No code changes. Findings inform the Step 4
loader PR.

References:
- §8 Step plan: `doc/research/2026-06-02-qp-architecture-review-and-alternatives.md` §8 Step 1 (ConstraintSnapshot) + Step 4 (offline A/B).
- Earlier sector audit: `doc/research/2026-05-23-qp-sector-metadata-audit.md` (the missing-metadata guard fix).
- Contract: `kernel/portfolio_qp/constraint_snapshot.py::ConstraintSnapshot`.
- Builder: `kernel/portfolio_qp/tasks.py::BuildSectorConstraintMatrixTask` (lines 1369-1455 of `tasks.py`) + `ApplySectorMetadataGuardTask` (lines 784-846).
- Cap resolver: `kernel/portfolio_qp/tasks.py::_resolve_sector_weight_cap` (lines 1458-1483) + `kernel/regime_resolver.py::resolve_regime_knob`.

---

## 1. Current `sector_map` — structure + counts

Source: `backtesting/renquant_104/strategy_config.json::sector_map` (and
the byte-identical `strategy_config.golden.json` overlay).

- **Total mapped tickers**: 156.
- **Total sectors**: 15.
- **Top-level watchlist** (`config['watchlist']`): 142 names; **0 unmapped**
  (every watchlist ticker has a sector row).
- **Sector-map superset**: 14 tickers carry sector rows but are NOT on the
  current watchlist. They are kept mapped because they have appeared in
  the universe historically (or in defensive / sector-ETF sleeves) and a
  missing sector row would silently exempt them from the QP cap. Listed
  for traceability only:

  `NVTS, NXPI, SHOP, OKTA, DOCU, GTLB, HUBS, PCTY, ESTC, PGR, HOOD, NVO,
  TLT, XLV`.

### Per-sector member counts

| Sector            | Members |
|-------------------|--------:|
| software          |      26 |
| finance           |      20 |
| ai_chip           |      19 |
| industrial        |      19 |
| consumer          |      16 |
| datacenter_hw     |      13 |
| healthcare        |      12 |
| giant_tech        |       9 |
| energy            |       8 |
| utility           |       6 |
| real_estate       |       3 |
| commodity         |       2 |
| benchmark         |       1 |
| defensive_bonds   |       1 |
| telecom           |       1 |
| **Total**         | **156** |

Notes:
- `XLK` is mapped to `giant_tech` (the ETF-of-tech proxy). It is also in
  the `sector_etf_map` as the ETF for the tech-family sectors
  (`giant_tech / ai_chip / datacenter_hw / software`). **Rule (this audit,
  per CLAUDE.md §7.5 single-source-of-truth, conservative interpretation):
  the dual role is INTENDED.** Any direct XLK position counts as
  `giant_tech` exposure under the QP sector cap (it is a member of the
  sector group, not an exempt hedge), AND simultaneously serves as the
  hedging ETF the regime overlay can purchase to gain tech-family beta.
  Counting XLK exposure as risk against the `giant_tech` cap is the
  conservative ledger: a long XLK directly inflates tech-family exposure,
  so the cap must price it. The non-conservative alternative (remove XLK
  from `giant_tech` membership so the cap only governs single-name tech
  positions) would silently exempt ETF-of-sector positions from the same
  cap that governs constituent names — a §7.7-class implicit exemption.
  Treat this rule as the design intent until a research memo
  empirically argues otherwise. Pin: any future refactor that strips
  XLK from `giant_tech` membership must update this audit + add a
  regression test that asserts the XLK cap accounting it removes.
- `benchmark`, `defensive_bonds`, `telecom` are single-member sectors —
  the QP sector cap on those rows is materially just a per-name cap.
- `XLF`, `XLV`, `XLE`, `XLI`, `XLY`, `XLU` are mapped into their
  respective business sectors (finance / healthcare / energy / industrial /
  consumer / utility) rather than a separate "sector_etf" sector. This is
  consistent with `sector_etf_map`'s reverse direction (sector → ETF).

### Defensive / regime-overlay sleeves

- `defensive_tickers`: `["GLD", "TLT", "XLV", "XLU"]`. **All 4 mapped** —
  `commodity` / `defensive_bonds` / `healthcare` / `utility` respectively.
- `bear_offensive_tickers`: not in config (no current BEAR-offensive
  sleeve).
- Live state (`backtesting/renquant_104/live_state.alpaca.json`):
  positions empty at audit time (LIVE PAPER PAUSED). No unmapped-position
  exposure.

---

## 2. Per-sector cap configuration

Source resolution order (per `_resolve_sector_weight_cap` in
`tasks.py:1458-1483`):

```
regime_params.<regime>.max_sector_weight_pct
   > config.max_sector_weight_pct
   > max_positions_per_sector × per_name_cap
```

with `final_cap = min(legacy_count_cap, regime_or_global_cap)` so the
count-based diversification stays a hard ceiling and the regime overlay
can only **tighten** further.

### Active values

- **Global key `max_sector_weight_pct`**: NOT SET (no global override —
  resolution always reads the regime overlay).
- **Global key `max_positions_per_sector`**: `6`.
- **Per-regime overlays** (`regime_params.<R>.max_sector_weight_pct`):

| Regime         | max_position_pct | max_sector_weight_pct | legacy_count_cap (6 × max_position_pct) | **Effective sector cap** |
|----------------|-----------------:|----------------------:|----------------------------------------:|-------------------------:|
| BULL_CALM      |             0.15 |                  0.35 |                                    0.90 |                **0.35** |
| BULL_VOLATILE  |             0.20 |                  0.30 |                                    1.20 |                **0.30** |
| CHOPPY         |             0.15 |                  0.30 |                                    0.90 |                **0.30** |
| BEAR           |             0.00 |                  0.20 |                                    0.00 |                **0.00** |

Reading:

- In every regime, the regime overlay binds — the count × per-name
  product is loose in 3/4 regimes (and trivially zero in BEAR because
  `max_position_pct = 0`).
- BEAR's effective cap is 0 because `max_position_pct = 0` collapses
  the legacy_count_cap to 0; `min(0.0, 0.2) = 0.0`. The QP can therefore
  hold or reduce, but cannot increase, any sector exposure in BEAR. This
  is consistent with `entry_mode = "blocked"` and `cash_reserve_pct = 1.0`
  in BEAR — sector cap is the third hard wall after entry-mode and
  cash-reserve.

### Provenance the QP records

`BuildSectorConstraintMatrixTask` writes
`ctx._qp_sector_cap_source` to one of:

- `"count_x_per_name"` — only the legacy product was applied (no regime
  override seen). With the current config, this branch is unreachable
  for the 4 named regimes — all 4 set `max_sector_weight_pct`.
- `"regime_or_global_max_sector_weight_pct"` — overlay (or global)
  bound the cap. **Current production always lands here.**

`sector_cap_source` is also a `ConstraintSnapshot` provenance field
(`constraint_snapshot.py:91`) and Step 4 replay diagnostics should
stratify by it (cap-source distribution per regime per cut is one of the
"zero hard-constraint regression" gate signals).

---

## 3. Tickers missing a sector mapping

### In the active prod universe

- `watchlist` (142 names): **0** missing.
- `defensive_tickers` (4 names): **0** missing.
- Live positions at audit time: **0** open.
- Sector-map superset (156 names total): **0** missing (every key has a
  string value).

**Conclusion**: at the current cut, no prod-universe ticker is unmapped.
The §7.7 "missing-sector implicit exemption" bug class is closed for the
current config.

### What guards the contract anyway

The QP defends against future drift even when the universe is clean:

- `ApplySectorMetadataGuardTask` (tasks.py:784-846) caps any unmapped
  candidate's QP upper bound at `max(w_current, 0)` (cannot increase
  weight). It logs missing tickers to `ctx._qp_missing_sector_tickers`
  and tags candidates with `blocked_by=missing_sector_map`.
- `BuildSectorConstraintMatrixTask._build_sector_index` (tasks.py:1446-
  1455) silently drops unmapped tickers from the `S` rows — they get no
  sector membership, so they cannot inflate a sector group. The guard
  task above is the complement: ensures the missing row does not become
  a permission to add risk.
- `risk.require_sector_map_for_buys = true` (strategy_config.json:383)
  blocks unmapped tickers at buy-side selection too — earlier in the
  pipeline.

### Implication for the WF replay loader (Step 4e)

The loader **must replay the unmapped-ticker guard**, even though
historically the unmapped count was ≤ small. Two cases the loader has
to support:

1. **Cut-date universe ⊆ today's sector_map**: the historical universe
   is a subset (live universe was smaller in 2020/2021/2022), so the
   today-snapshot map already covers every ticker. No special handling.
2. **Cut-date universe ⊄ today's sector_map**: some ticker that was
   on the universe historically has been removed from the current
   sector_map (delisted, demerged, ticker symbol changed). Today's
   snapshot map would give it no row, which the QP would convert to a
   `missing_sector` weight cap of 0. That would **change the historical
   QP's feasible set at that cut** — a hard-constraint regression vs the
   live decision at that bar.

A snapshot-at-cut-date map (case (b) below) is required to preserve the
"zero hard-constraint regression vs Step 1 ConstraintSnapshot" gate
(Step 4 non-negotiable, addendum §8 Step 4).

---

## 4. Shape the WF replay loader must populate

The Step 4e loader needs to produce, per cut, the same fields
`BuildSectorConstraintMatrixTask` stamps in live:

```python
# Inputs the loader needs:
tickers: list[str]               # universe at cut date (length n)
regime: str                      # detected regime at cut date
sector_map: dict[str, str]       # ticker -> sector_name
max_positions_per_sector: int    # legacy count
max_position_pct: float          # from regime_params[regime]
max_sector_weight_pct: float     # from regime_params[regime] (preferred)
qp_sector_cap_enabled: bool      # gate

# Outputs to feed ConstraintSnapshot:
sector_indicator: np.ndarray     # shape (S, n), dtype float, 0/1
sector_cap_vec:   np.ndarray     # shape (S,),   dtype float
sector_names:     tuple[str,...] # length S, sorted lexicographically
missing_sector_tickers: tuple[str,...]  # any unmapped tickers (informational + guard)
sector_cap_source: str           # "regime_or_global_max_sector_weight_pct" | "count_x_per_name"
```

### Construction (mirrors `BuildSectorConstraintMatrixTask`)

1. Drop unmapped tickers from the indicator (record them in
   `missing_sector_tickers`). They remain in `tickers` of length `n` — the
   indicator matrix simply has no row containing them.
2. Build `sector_to_idx: dict[str, list[int]]` over the remaining tickers
   (the contributing column indices for each sector).
3. Sort sector names lexicographically. Let `S = len(sorted_names)`.
4. `sector_indicator = np.zeros((S, n), dtype=float)`; for each sector
   row, write `1.0` at every contributing column. (Dtype is `float` to
   match `BuildSectorConstraintMatrixTask` and to keep CVXPY constraint
   construction homogeneous; the values are 0/1.)
5. `per_name_cap = max(w_upper_hard[mapped_idx])` if `w_upper_hard` is
   available; else fall back to `max_position_pct` for the regime. The
   replay loader should mirror live's anchoring — see §3 of the
   2026-05-23 audit for why anchoring on the WHOLE-universe max
   inflated mapped sectors and was fixed to anchor only on mapped
   indices.
6. `legacy_cap = max_positions_per_sector × per_name_cap`.
7. `cap = min(legacy_cap, regime_overlay_cap)` if the regime overlay
   exists, else `legacy_cap`.
8. `sector_cap_vec = np.full(S, cap, dtype=float)` — uniform across
   sectors (per current config; future per-sector differentiation would
   widen this to a vector with non-uniform entries).
9. Provenance: `sector_cap_source = "regime_or_global_max_sector_weight_pct"`
   if regime overlay bound; `"count_x_per_name"` otherwise.

### Validation the snapshot enforces

`ConstraintSnapshot._validate` (constraint_snapshot.py:118-194) requires:

- `sector_indicator` and `sector_cap_vec` either both `None` or both set.
- `sector_indicator.ndim == 2` and `sector_indicator.shape[1] == n`.
- `sector_indicator.shape[0] == sector_cap_vec.shape[0]`.
- (Per-asset arrays separately validated for finiteness and
  soft ≤ hard cap.)

The loader must therefore tolerate a "no sectors" cut (e.g. empty
universe at a very early cut) by returning `None` for all three sector
fields — the snapshot accepts this.

### Hybrid Option F (Step 4d) — sector-cap projection

Hybrid Option F's Stage 3 (sector projection) consumes
`ConstraintSnapshot.sector_indicator` + `sector_cap_vec` to compute the
per-sector residual budget and reverse-greedy drop names from saturated
sectors. The loader's `S × n` indicator + `S` cap vector are exactly
the inputs the projection needs — no schema change required for Option F
adoption. (Open question §9.2 in the architecture memo asks reverse-
greedy vs minimum-distance projection; both consume the same snapshot
fields.)

---

## 5. Recommendation: snapshot vs verbatim live config

**Recommendation: hold a per-cut snapshot of the sector_map and the
sector-cap config, not the live config verbatim.**

### Reasons

1. **Hard-constraint regression gate** (§8 Step 4 non-negotiable). The
   gate compares each baseline's daily decisions against the snapshot
   built at the same cut. If the loader reads today's sector_map and
   a 2022-Q2 cut has a ticker that today's map drops, the historical
   QP's feasible set at that bar would differ from the replay's — a
   regression by construction. Per-cut snapshot eliminates this.

2. **Reproducibility of WF artifacts**. WF cuts already pin model and
   calibrator artifacts at the cut date (per the leakage architecture in
   `doc/research/2026-06-01-leakage-architecture.md`). The sector_map
   and per-sector caps are part of the same decision contract; pinning
   model but not constraints leaks today's allocator structure into
   historical decisions.

3. **Cap-policy evolution traceability**. `max_sector_weight_pct` per
   regime has changed over time (e.g. BEAR was tightened to 0.20 in
   the 2026-05-14 PRIME DIRECTIVE wave). A verbatim-live loader would
   apply today's 0.20 to a 2022 cut, overwriting whatever the live
   system actually decided then.

4. **Cheap to implement**. A per-cut snapshot is two scalars
   (`max_positions_per_sector`, regime overlay map) plus a small dict
   (`ticker → sector`). Snapshot as a JSON sidecar alongside the cut's
   model artifact under
   `backtesting/renquant_104/artifacts/prod/wf_cuts/<cut_id>/sector_snapshot.json`,
   or equivalently as a new field in the existing WF manifest.

### Where the historical sector_map comes from (the load-bearing decision)

Three options were considered:

1. **Reconstruct from git history**: walk `git log -p
   backtesting/renquant_104/strategy_config.json` for every commit
   touching the `sector_map` field, materializing a per-commit map
   timeline. Pro: highest fidelity to what the live system actually
   saw at each historical bar. Con: `sector_map` was only introduced
   on **2026-04-21 (commit `657950e`, "feat: renquant_104 panel-LTR
   hybrid extensions + pipeline principle")**. Every WF cut with a
   bar date before 2026-04-21 has no historical sector_map at all —
   the live system simply did not enforce sector caps then. So git
   history is bounded below by that commit, and earlier cuts still
   need a fallback.
2. **Snapshot today's sector_map and document the limitation**: every
   WF cut uses today's `sector_map` + per-regime caps, with a verdict-
   JSON footnote that historical universe drift is NOT modeled. The
   QP's `missing_sector` guard still fires correctly for any cut-date
   ticker that fell off today's map (per §3 above), so the worst case
   is a small number of historical names getting the conservative
   "cannot increase weight" treatment from `ApplySectorMetadataGuardTask`.
3. **Fetch sector membership from a vendor source per date**: most
   faithful, but requires a new artifact under `renquant-base-data`,
   a vendor subscription, and a backfill window that covers WF cut
   dates back to 2020. Out of scope for §8 Step 4.

**Recommendation: Option 2 (snapshot today's map, document the
limitation in the verdict JSON).** Rationale:

- For WF cuts on or after **`657950e` (2026-04-21)**, the live system's
  effective `sector_map` and today's `sector_map` are close enough
  that today-snapshot drift is bounded by config evolution between
  that commit and today (config-evolution drift is captured by the
  per-regime cap snapshot in this audit's §5 schema, and stratified
  by `sector_cap_source` per §2's provenance).
- For WF cuts before `657950e`, the live system enforced **no** sector
  cap at all. Today-snapshot is strictly more restrictive (false
  positives possible: a few historical names might get
  `missing_sector` cap=0 they did not see live). This is the
  conservative direction — the §8 Step 4 gate asks "zero
  hard-constraint regression vs Step 1 ConstraintSnapshot built at the
  same cut from the same snapshot map", which Option 2 satisfies by
  using the same map on both sides of the gate.
- Option 1 is reserved for if the Step 4 verdict reveals
  sector-cap-driven baseline differences large enough to motivate
  per-commit map reconstruction. We can revisit then.
- Option 3 is reserved for if RenQuant ever takes a vendor sector
  feed for production (an unrelated, larger decision).

**The §4e loader (PR #142) consumes today's snapshot as the lower
bound from `657950e`.** Cuts before that commit fall back to today's
snapshot AND get tagged in their verdict JSON with
`sector_snapshot_source = "today_fallback_pre_657950e"`. Cuts on or
after that commit get `sector_snapshot_source = "today_snapshot"` —
indistinguishable from the live config because today's map IS the
live map at this audit's commit, and the per-cut snapshot frozen at
loader-run time pins reproducibility going forward. The Step 4 verdict
JSON MUST surface this field per-cut so a reviewer can immediately
distinguish "the sector cap binding here is the cap the live system
saw at this bar" from "we approximated with today's cap".

### What "verbatim live" is acceptable for

The §8 Step 5 live-shadow phase (NOT a Sharpe gate; operational
telemetry only) can read today's live `strategy_config.json` directly,
because the comparison there is "what would the alternative allocator
have decided TODAY" — today's config IS the correct contract.

The non-acceptable use is Step 4's offline historical replay.

### Suggested snapshot fields (minimum viable)

```json
{
  "cut_date": "2022-06-30",
  "sector_map": { "AAPL": "giant_tech", "...": "..." },
  "max_positions_per_sector": 6,
  "regime_params": {
    "BULL_CALM":     { "max_position_pct": 0.15, "max_sector_weight_pct": 0.35 },
    "BULL_VOLATILE": { "max_position_pct": 0.20, "max_sector_weight_pct": 0.30 },
    "CHOPPY":        { "max_position_pct": 0.15, "max_sector_weight_pct": 0.30 },
    "BEAR":          { "max_position_pct": 0.00, "max_sector_weight_pct": 0.20 }
  },
  "qp_sector_cap_enabled": true,
  "config_sha": "<sha256 of the source strategy_config.json at snapshot time>"
}
```

`config_sha` lets the Step 4 loader verify a snapshot has not been
silently regenerated against a newer live config.

---

## 6. Open items for the Step 4 loader PR

**Loader implementation: PR #142 (merged 2026-06-03, commit
`4d2d198`) — §8 Step 4e WF cut loader for A/B replay (sim DB →
`AllocatorReplayBar`).** This audit memo is the source of truth for
the loader's sector-cap contract; the cross-reference should be
bidirectional (PR #142 description points back to this memo, and any
future loader patch first re-reads this memo's §3 / §4 / §5 rules
before editing the sector path).

These are not asks of this audit PR — they are the implementation
items the loader PR will pick up:

1. **Snapshot generator**: a one-shot script that walks the existing
   WF cuts under `artifacts/prod/wf_cuts/` and emits
   `sector_snapshot.json` for each. For pre-snapshot cuts (no live
   config history), document the policy: use today's map but flag the
   cut in the manifest as `sector_snapshot_source = "today_fallback"`.

2. **Loader function** in the replay tooling: signature
   `load_sector_constraint(cut_dir, tickers, regime) -> (S, cap_vec, names, missing)`.
   Mirrors `BuildSectorConstraintMatrixTask.run` step-for-step against
   the snapshot dict.

3. **Regression test** that pins the loader's output (for a fixed cut
   + fixed regime) byte-identical to
   `BuildSectorConstraintMatrixTask`'s ctx mutations driven by the
   same snapshot. This is the Step 4 "zero hard-constraint
   regression" gate's smallest unit.

4. **Hybrid Option F consumer**: Option F's Stage 3 projection reads
   `ConstraintSnapshot.sector_indicator` + `sector_cap_vec` directly —
   the loader's output IS Option F's input. No additional shaping
   needed there.

5. **Replay diagnostics**: per-cut, per-regime, log
   `sector_cap_source` distribution. If a cut produces
   `"count_x_per_name"` for a regime where today says
   `"regime_or_global_max_sector_weight_pct"`, that's a config-drift
   datapoint (not a bug — expected when the snapshot pre-dates the
   regime-overlay introduction).

---

## 7. Audit verdict

- The current `sector_map` is clean (156 mapped, 15 sectors, 0
  unmapped in prod universe).
- The per-regime sector cap is the binding constraint in every named
  regime — regime overlay always tightens below the legacy count cap.
- `BuildSectorConstraintMatrixTask` already emits the exact
  `(S × n, S, names)` shape `ConstraintSnapshot.sector_indicator /
  sector_cap_vec / sector_names` expects; the Step 4 loader is a
  read-from-snapshot port of the same logic.
- **The WF replay loader should hold per-cut snapshots, NOT read the
  live config verbatim**, to satisfy the Step 4 "zero hard-constraint
  regression" gate.
- No code change required in this PR.

---

Agent-Origin: Claude
