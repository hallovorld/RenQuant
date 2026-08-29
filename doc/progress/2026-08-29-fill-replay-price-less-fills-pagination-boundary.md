# 2026-08-29 — Fill replay never drops price-less fills; the pagination boundary is no longer duplicated; the full-exit lot resurrection is fixed (RenQuant#618 class C)

**Bottom line:** the `LIVE-TAX-LOTS ... reconstructed lot qty != broker qty`
warning that fires on every live run (VLO 7 vs 5, PANW 6 vs 3, APH 14 vs 8,
NVDA 14 vs 7 — `logs/daily_104/2026-08-28.log:392-394`, `2026-08-25.log:392-394`)
is reproduced EXACTLY by the pre-fix replay on PRICED fills: after a FULL
sell `HoldingState.total_shares()` falls back to the legacy `shares` field,
the ticker is never popped, and the next BUY's `ensure_lots` re-synthesises
the already-sold lot — `2+5=7`, `7+7=14`, `3+3=6`, `6+8=14` `[VERIFIED: old
module loaded from origin/main, new module side by side; see §2]`. That
resurrection is fixed here, together with the two class-C defects the issue
names: (a) a fill with no `filled_avg_price` was silently dropped (a
price-less SELL never reduced lots) — now applied at qty with a stand-in
basis, flagged `price_missing=True`, warned once per fill and counted; (b)
`get_filled_orders` re-fetched the boundary order on every 500-order page
(the cursor was `oldest`, not `oldest − 1µs`) — now the cursor steps back
1µs AND orders are deduplicated by `id`. The hydration site logs the signed
per-ticker delta with the degraded-fill counts, so the next mismatch names
its cause. Nothing is deployed by this PR (merged ≠ deployed; the live tree
picks it up on the operator's next sync).

Scope: `backtesting/renquant_104/adapters/runner_tax_lots.py`,
`live/alpaca_broker.py`, the 14-line adoption block in
`adapters/runner.py` (import + the `LIVE-TAX-LOTS` block, now a helper
call). The entry-date code in `runner.py` (`:449-471`, `:542-547`,
`:1351-1354`, `:1610`) is untouched — a parallel PR owns class B.

## 1. Defects and fixes (file:line on this branch)

| # | Defect (pre-fix) | Fix |
|---|---|---|
| C-0 (found here) | `runner_tax_lots.py` set `hs.shares = hs.total_shares()` after every fill; `total_shares()` returns the legacy `shares` when `lots` is empty (`kernel/exits.py:173-178`), so a full sell left `shares` at the pre-sell qty, the `<= 1e-9` pop never ran, and the next BUY's `apply_buy_lot → ensure_lots` (`kernel/exits.py:195-219`) synthesised a lot from the stale legacy fields | `_lot_shares(hs)` (`runner_tax_lots.py`) sums the LOTS — never the legacy field — at every update site; a full exit flattens the ticker and a re-entry starts clean |
| C-a | `runner_tax_lots.py:124` `if ... price <= 0 ...: continue` dropped the whole fill | price-less SELL → `apply_sell_lots_detailed` at qty; stand-in price = disposed cost basis, recorded in `stats["degraded_fills"]` with `price_missing=True`, ONE warning per fill. Price-less BUY → lot appended at qty with the ticker's running weighted basis (else `0.0`, never NaN), lot carries `price_missing=True`, warned once, counted. `adopt_live_tax_lots` back-fills flagged lots from the broker average (residual `(avg × qty − Σ known cost) / missing sh`, else the average itself) and REFUSES to attach lots whose basis is still unknown (a 0-basis lot would understate the weighted entry and overstate realized gains) |
| C-a′ | `:140` a SELL before any BUY in the window was silently skipped; an oversell was silently clamped | still not applied / clamped (there is no lot to consume), but counted (`dropped_sell_without_lots`, `oversell_clamped`) and logged at INFO |
| C-b | `live/alpaca_broker.py:765` `until_cursor = oldest` (comment at `:755` promised −1µs) → the boundary order came back on the next page | `next_cursor = oldest - timedelta(microseconds=1)`; orders deduplicated by `str(o.id)` across pages (`seen_ids`); id-less orders are kept; a non-datetime `submitted_at` keeps the inclusive cursor and relies on the dedupe; a cursor that does not move backward breaks the loop |
| invariant | `runner.py` mismatch warning carried only the two quantities | `adopt_live_tax_lots(holding, ticker, lots, broker_qty, broker_avg_price, stats=...)`: same `LIVE-TAX-LOTS: <T> reconstructed lot qty X != broker qty Y; using broker avg_entry_price fallback` prefix (grep-compatible with the issue), plus `delta=±D; replay saw N fill(s) for it, degraded: price_missing_sell= price_missing_buy= sell_without_lots= oversell_clamped=`; a ticker with NO lots but degraded fills also warns (was silent); a clean no-lots ticker stays silent |
| summary | none | one `LIVE-TAX-LOTS replay summary: fills= applied= price_missing_sell= price_missing_buy= dropped_unparseable= dropped_sell_without_lots= dropped_unknown_action= oversell_clamped= tickers_with_lots=` line per reconstruction (WARNING when anything degraded, INFO otherwise) |

`reconstruct_live_tax_lots_from_fills` now returns
`LiveTaxLotReconstruction` — a `dict` subclass (every existing consumer:
`.get`, `==`, iteration unchanged) with `.stats` (`new_replay_stats()`).
Downstream basis consumers stay safe: `compute_disposed_lot_tax`
(`kernel/portfolio.py:113-118`) skips `price <= 0` lots;
`apply_live_sell_lot_accounting` (`runner_tax_lots.py`) returns `False`
unless `proceeds_basis > 0`; and the adoption helper never attaches a lot
whose basis is unknown, so no NaN/0 basis reaches `weighted_avg_entry_price`.

## 2. Evidence

- Old-vs-new on the issue's ledgers, PRICED fills (script: load
  `origin/main:backtesting/renquant_104/adapters/runner_tax_lots.py` as a
  module beside the new one; `[VERIFIED]`):
  `VLO old=7.0 new=5.0 lots_old=[(2.0, 340.53), (5.0, 346.5)]`,
  `NVDA old=14.0 new=7.0 lots_old=[(7.0, 210.0), (7.0, 222.9)]`,
  `PANW old=6.0 new=3.0`, `APH old=14.0 new=8.0`. The resurrected lot even
  carries the NEW price with the OLD qty (VLO `(2.0, 340.53)`): the legacy
  `entry_price` had been overwritten by the re-entry's weighted average
  before `ensure_lots` read it.
- Whether Alpaca actually returned a price-less filled order for these
  names is `[NOT VERIFIED]` (no fill dump exists; the daily logs do not
  print fills). The drop path was real (`:124`) and is now a counted
  degradation, but the observed numbers are fully explained by C-0.
- Pagination duplication needs ≥ 500 closed orders in one walk; whether
  this account crosses that is `[NOT VERIFIED]`. The fix is correct either
  way and is pinned by a mock with an inclusive-`until` server.

## 3. Tests (`.venv` python 3.10, pytest 8.4.2, `-o addopts=''`)

- `tests/test_fill_replay_rq618.py` (new, 45 cases): price-less SELL
  reduces qty (every missing-price shape: `None/0/""/"n/a"/nan/-1`;
  `filled_avg_price` honoured before degrading; HIFO/AVG/FIFO); the issue's
  VLO ledger with price-less sells reconciles to 5; full-exit re-entry with
  PRICED fills reconciles for VLO/NVDA/PANW/APH (the C-0 regression);
  price-less BUY applied with a running-basis or `0.0` (never NaN) stand-in,
  flag survives the return copy; ONE warning per price-less fill; the
  summary line carries every count; unparseable fills still dropped;
  `adopt_live_tax_lots` match / mismatch-with-delta-and-counts /
  residual back-fill (`(110×10 − 5×100)/5 = 120`) / broker-avg fallback /
  unknown-basis refusal / no-lots warn-vs-silent / end-to-end; the runner
  handling site is wired (static pin); pagination: 1250 orders over an
  inclusive AND an exclusive `until` server → 1250 unique ids in 3 requests,
  the 2nd/3rd `until` equal the page's oldest `submitted_at − 1µs`, `after`
  forwarded, id-less orders kept, string `submitted_at` does not crash,
  source pin (`until_cursor = oldest` gone).
- `tests/test_runner_tax_lots_invariants.py` (+2, 1 adjusted): conservation
  over histories with 30 % of prices stripped (counts match exactly);
  conservation over FULL round trips (the generator never sold the full
  position, which is why the resurrection was never caught); the
  `price <= 0` entry left the "malformed → ignored" list because it is
  now applied; `sell_without_prior_buy` asserts the count.
- CI: `.github/workflows/live-broker-fractional-contract.yml` step 4 runs
  `tests/test_fill_replay_rq618.py tests/test_runner_tax_lots.py
  tests/test_runner_tax_lots_invariants.py` (paths added), so the new file
  is named by a workflow (RenQuant "green check that covered nothing" rule).
  All four steps of that workflow pass locally: 42 / 127 / 50+4 skipped / 66.
- Related suites unchanged: `test_wash_sale_economic`,
  `test_runner_sell_attribution`, `test_disposed_lot_tax_netting`,
  `test_hifo_lot_selection`, `test_tax_lots_g7`, `test_runner_ext_sell`,
  `test_broker_sync`, `test_state_ext_sell_fill_attribution`,
  `test_preflight_broker_fill_freshness`, `test_round3_audit_fixes_2026_04_25`,
  `test_runner_state_fixes` — green. `tests/test_adapter_context_contract.py`
  has 7 failures that are IDENTICAL on a clean `origin/main` checkout
  (`git stash -u` → same 7 → `git stash pop`) `[VERIFIED]`; not touched here.
  Broad related run (`tests/test_runner*.py test_*broker*.py test_*lot*.py
  test_*fill*.py test_*tax*.py test_*alpaca*.py`, xdist): 877 passed, 2
  skipped, 4 failed — the 4 are `tests/test_runner_artifacts.py::
  TestLoadContextArtifacts::*`, identical on clean `origin/main` `[VERIFIED]`.

## 4. Not done / follow-ups

- Class A (clamp-before-buy-leg) and class B (entry-date lifecycle) of #618
  are not in this PR. Note for class B: `first_fill_map` (`runner.py:449-471`)
  reads the same `get_filled_orders` list, so the pagination dedupe removes
  one duplicate source from it, but the "oldest BUY ever" seed logic is
  unchanged.
- `stats["degraded_fills"]` is in-memory only; nothing persists it yet. A
  tax report that wants to flag `price_missing` lots reads the flag off the
  `TaxLot` (attribute, not a dataclass field — `kernel.exits.TaxLot` is
  shared with sim/LEAN and was not widened here).
- Deployment: the live tree (`/Users/renhao/git/github/RenQuant`) is a
  separate checkout; this branch writes nothing there. The next daily run
  after the operator's sync will print the replay summary and, if any
  mismatch remains, the delta + degraded counts that name the cause.
