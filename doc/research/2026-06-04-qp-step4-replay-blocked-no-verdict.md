# 2026-06-04 — QP §8 Step 4 A/B replay produces no verdict: data + wiring gaps

**Question** (operator): "QP 昨天做了一堆修改,现在有什么提升吗?" — yesterday's
large batch of QP changes, any measured improvement?

**Answer**: **None measurable, and it currently cannot be measured.**
Yesterday's QP work built the entire §8 Step 4 offline allocator A/B replay
framework — but the framework is non-functional end-to-end: its input data
is unpopulated and its new allocators are unregistered. The replay crashes
on first real invocation with zero bars loaded. No allocator has ever been
compared; there is zero evidence any of the new QP work improves anything.

This is a §7.7-class finding: a lot of machinery shipped (allocators,
loader, CLI driver, gates, fwd-horizon flag), but the track is **decoration
until the evaluation actually runs**. The verdict was never produced.

---

## 1 · What was built yesterday (the "一堆修改")

All scaffolding for the §8 Step 4 offline walk-forward allocator A/B:

- `baseline_allocators.py`: `hybrid_option_f_allocator`, `hard_only_qp_allocator`
- `wf_replay_loader.py`: `load_replay_bars_from_sim_db`
- `allocator_replay.py`: `replay_all`, paired-returns, significance
- `run_ab_replay.py`: CLI driver + verdict assembly + Step 4g gates + `--fwd-horizon-days`
- sector_map / ConstraintSnapshot contract docs, runtime-sanity guards, drift audits

Code is real and tested in isolation. But it has **never produced a verdict
artifact** — confirmed: zero files under `doc/research/evidence/*replay*`
or `artifacts/**/*replay*verdict*`.

## 2 · What happens when you actually run it

```
python -m kernel.portfolio_qp.run_ab_replay \
  --wf-artifact-root data/sim_runs.db \
  --start-cut 2024-01-02 --end-cut 2026-03-27 \
  --out .../verdict.json \
  --allocators equal_weight_top_k,inverse_vol_top_k,fractional_kelly_top_k \
  --incumbent fractional_kelly_top_k --fwd-horizon-days 60
```

→ crashes:

```
ValueError: zero-size array to reduction operation maximum which has no identity
  (np.max(delta) on empty paired-returns; "Mean of empty slice" warnings)
```

Root cause: the loader returns **0 bars** for every horizon (60/20/10/5).

## 3 · Why zero bars — three stacked data/wiring breaks

`load_replay_bars_from_sim_db` requires, per (date, ticker), all of:
`score_distribution.mu IS NOT NULL`, `.sigma IS NOT NULL`, and
`ticker_forward_returns.<fwd_col> IS NOT NULL`, joined on
`score_distribution.date = ticker_forward_returns.as_of_date AND ticker`.

Measured against the live `data/sim_runs.db` (2024-01-02 → 2026-03-27, 561
dates):

| Break | Evidence |
|---|---|
| **B1 — `fwd_60d` is 100% NULL** | `ticker_forward_returns.fwd_60d` non-null = **0 / 31617**. The CLI default `--fwd-horizon-days 60` can never match. (`fwd_1d/5d/10d/20d` ARE populated.) |
| **B2 — `mu`/`sigma` 92% NULL and disjoint from forward returns** | `score_distribution.mu`+`sigma` non-null = **3052 / 37392**, and those rows match `ticker_forward_returns` on (date,ticker) = **0**. The populated mu/sigma rows are off-watchlist tickers (e.g. SHOP) that have no forward-return rows. The full table join (no filter) is 31056, so the tables DO join — but the `mu+sigma+fwd` intersection is empty. |
| **B3 — new allocators unregistered** | `run_ab_replay.py::_ALLOCATOR_REGISTRY` contains only `equal_weight_top_k`, `inverse_vol_top_k`, `fractional_kelly_top_k`. `hybrid_option_f_allocator` and `hard_only_qp_allocator` (built yesterday) are NOT registered, so they can't be named in `--allocators` without extra wiring. |

Even fixing B1 (use `fwd_20d`) leaves B2 — the loader still returns 0 bars
because mu/sigma and forward returns never co-occur on the same (date,ticker).

## 4 · What this means

- **No improvement from yesterday's QP work has been measured** — the
  evaluation that would measure it cannot run. Capability was added; nothing
  was validated.
- The Step 4 track is **blocked on a data-pipeline gap**, not a code gap.
  The replay code is fine; the sim DB it reads was never populated with the
  point-in-time `mu`/`sigma` + `fwd_60d` it needs on the same rows.
- This compounds the Kelly verdict
  ([`2026-06-03-kelly-sigma-horizon-ab-verdict.md`](2026-06-03-kelly-sigma-horizon-ab-verdict.md)):
  we found the cash drag lives in the QP layer, and the QP A/B replay is the
  intended tool to investigate it — but that tool can't run yet.

## 5 · What it takes to get a real verdict (ordered)

1. **Backfill `fwd_60d`** in `ticker_forward_returns` (or run the replay at
   `--fwd-horizon-days 20`, which IS populated, as a stopgap). Owner:
   `scripts/backfill_forward_returns.py`.
2. **Populate `score_distribution.mu`/`sigma` for the watchlist tickers**
   on the same (date,ticker) as the forward returns — the sim that writes
   `score_distribution` must stamp mu/sigma for every scored name, not just
   the 8% it currently does. This is the load-bearing fix: without
   mu/sigma↔fwd overlap there are no bars.
3. **Register the new allocators** (`hybrid_option_f`, `hard_only_qp`) in
   `run_ab_replay.py::_ALLOCATOR_REGISTRY` so they can be compared.
4. **Harden the loader/driver to fail loud on zero bars** instead of
   crashing in `np.max` on an empty array — emit
   `invalid_experiment.json{reason: no_bars_loaded}` (mirrors the Kelly-AB
   no-trade guard, PR #202).

Until at least (1)+(2)+(4), the QP allocator A/B cannot produce a verdict,
and "did the QP changes help?" stays unanswerable.

## 6 · Verdict

The QP Step 4 framework is **built but non-functional end-to-end**. No
allocator comparison exists. Recommend treating the Step 4 track as
**blocked** and prioritizing the sim-DB data backfill (mu/sigma + fwd_60d
co-population) before any further allocator code, per §7.7 (don't add more
machinery to a path that can't execute) and §6.4 (the evaluation tool must
work before we trust any QP "improvement").

---

Agent-Origin: Claude
