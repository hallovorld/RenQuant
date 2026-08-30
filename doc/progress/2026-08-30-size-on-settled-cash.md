# 2026-08-30 — Buys are sized on settled cash; `execution.buying_power_mode` is read and honoured (default `settled_cash`, never margin)

**Bottom line:** the live buy budget is no longer the broker's hard-wired
`non_marginable_buying_power`. `RunnerAdapter.make_context` now reads
`execution.buying_power_mode` from the strategy config, asks the broker for
ONE account snapshot, sizes buys on the balance that mode names — default
(key absent) is `settled_cash` = `account.cash` floored at 0 — and logs
`cash=… nmbp=… buying_power=… mode=…` on every run. A negative settled
balance (the account is on margin) sizes to $0, records `no_settled_cash`
as the skip / no-trade reason, and credits NO same-bar sell proceeds.
Sell paths are untouched.

**Decision still open for the operator (read this before expecting the live
number to change):** the config the daily run loads is the PINNED
renquant-strategy-104 `configs/strategy_config.json` (d3c8026a), and it
declares `"buying_power_mode": "non_marginable_buying_power"` (line 691;
every shadow config says the same) `[VERIFIED git show d3c8026a:…]`.
Because this PR HONOURS the key, the live run will resolve `mode=nmbp` and
size on the same number it uses today — now with settled cash printed
beside it. Sizing on settled cash on the live book requires ONE of:
(a) a one-line strategy-104 PR setting `"buying_power_mode": "settled_cash"`
+ pin advance (recommended: the sim reads the same key with the same
vocabulary, so sim and live stay in one mode), or (b) removing the key
(the default is then settled cash live; note the sim's absent-default is
nmbp — see §4). Neither is done here: `strategy_config.json` is a
production path this PR does not write.

## 1. Incident (operator findings, 2026-08-30 — REPORTED, not re-read from the broker here)

- 08-27: HPE $1,034 bought with settled cash of $33.
- 08-28: WELL $1,904 + NET bought with settled cash of −$1,140.
- 08-28 close: `account.cash` −$1,139.70 on a ~$10.8k book → 1.11× on margin.

Mechanism `[VERIFIED in source]`: `backtesting/renquant_104/adapters/runner.py`
called `broker.get_cash()`; `live/alpaca_broker.py::get_cash` (P0-9, 2026-05-20)
returned `account.non_marginable_buying_power` unconditionally (settled cash +
UNSETTLED sell proceeds). On a margin account Alpaca lets a buy clear against
proceeds that have not settled and fronts the difference as a margin debit.
The strategy config had declared `execution.buying_power_mode` for sim/live
parity; the sim read it (`adapters/sim_order_helpers.py`,
`adapters/lean_account.py`, `kernel/pipeline/task_benchmark_sleeve.py`);
the live adapter never did — the key was decorative on the path that trades.

## 2. Which tree trades (resolved the runner's way, not assumed)

`[VERIFIED]` `scripts/daily_104.sh` runs `-m renquant_orchestrator daily-bridge`
(RQ_DAILY_RUNNER=multirepo default). `live_bridge.bootstrap_multirepo` puts
the umbrella repo root and `backtesting/renquant_104` FIRST on `sys.path`,
appends the pinned `src/` roots, and aliases only `kernel.<stem>` modules to
`renquant_pipeline.kernel`. No pinned repo ships an `adapters` package
(`find .subrepo_runtime/repos -path '*adapters*' -name runner.py` → none;
the only `task_selection.py` is `renquant_pipeline/kernel/pipeline/`).
`live/runner.py` does `from .alpaca_broker import AlpacaBroker`. Therefore
the modules on the order path are the UMBRELLA's `live/alpaca_broker.py` and
`backtesting/renquant_104/adapters/runner.py` — the two this PR changes.
The renquant-execution twin (`src/renquant_execution/alpaca_broker.py:244-249`,
pin 91c7bf88) is ALSO hard-wired to nmbp but is not on the order path
(the pair is a recorded `diverged_pin`); it needs its own PR in that repo
(§7). The pinned `kernel.pipeline.task_selection` needs no change: it sizes
against whatever `ctx.cash` the adapter injects.

## 3. Change

### 3.1 `live/broker.py` — the vocabulary and the resolver (broker layer owns "buying power")

- `BUYING_POWER_MODE_SETTLED="settled_cash"` (DEFAULT), `…_NMBP=
  "non_marginable_buying_power"`, `…_MARGIN="buying_power"`, with an alias
  table that is a SUPERSET of the sim's (`cash`/`settled` → settled;
  `cash_plus_unsettled`/`unsettled` → nmbp; `margin` → buying_power).
- `normalize_buying_power_mode(raw)`: absent/empty → settled; unknown →
  `ValueError` (a typo in a deployed config fail-closes to $0, never picks a
  mode).
- `resolve_sizing_cash(mode, settled_cash=, non_marginable_buying_power=,
  buying_power=)`: pure. Picks the named field, floors at 0, names the
  reason (`no_settled_cash` / `no_buying_power` / `cash_unreadable`). A
  requested field the SDK does not expose degrades to SETTLED cash — never
  to a larger figure — and says so in `sizing_source`.
- `BaseBroker.get_buying_power_snapshot(mode=None)`: default reads
  `get_cash()` as settled cash (PaperBroker, IBKR: conservative).

### 3.2 `live/alpaca_broker.py`

- `AlpacaBroker(..., buying_power_mode=None)`; `_account_balances(account)`
  reads `cash` / `non_marginable_buying_power` / `buying_power` raw (missing
  attribute → `None`, never 0).
- `get_buying_power_snapshot(mode=None)`: one `get_account()` read →
  `resolve_sizing_cash`. `buying_power` (margin) logs a WARNING on every
  read; an unavailable field logs a WARNING.
- `get_cash()` = `get_buying_power_snapshot()["sizing_cash"]` in the
  constructor mode (default settled). The unconditional
  `return float(nmbp)` is gone (pinned by test).

### 3.3 `live/broker_readonly.py`

Explicit forward of `get_buying_power_snapshot` (not left to `__getattr__`)
so every shadow lane sizes on exactly what the underlying broker reports.

### 3.4 `backtesting/renquant_104/adapters/runner_execmath.py`

- `resolve_buy_sizing_cash(broker, config)`: reads the key, calls the
  snapshot (or, for a broker without it, treats `get_cash()` as settled
  cash), stamps `configured_mode` + `broker_api`, logs
  `runner: buy-sizing cash=… nmbp=… buying_power=… mode=… -> sizing_cash=$…`
  (INFO) and `runner: BUY budget is $0 this bar — <reason>` (WARNING).
- `unsettled_proceeds_spendable(mode)`: True only for nmbp / buying_power;
  `None`/unknown → False (conservative default).

### 3.5 `backtesting/renquant_104/adapters/runner.py`

- `make_context`: `buy_sizing = resolve_buy_sizing_cash(broker, self._config)`;
  `cash = buy_sizing["sizing_cash"]`. The audit-Issue-36 fail-SAFE is kept
  verbatim (`cash = 0.0` + `log.error`) and now also covers an unrecognised
  mode; `ctx.buy_sizing_cash = dict(buy_sizing)` is recorded on the context.
- `commit`: the same-bar sell credit is gated —
  `if sell_credit > 0 and not unsettled_proceeds_spendable(mode): sell_credit = 0.0`
  (logged) BEFORE `buy_cash_remaining += sell_credit`. Same-bar exit
  proceeds settle T+1; crediting them under settled sizing would be the
  margin exposure again.
- `commit`: a $0 budget with a named cause skips every remaining BUY intent
  with `skip_reason = <reason>` (`no_settled_cash`, …) instead of the
  generic `cash_budget_exhausted`.

### 3.6 `live/runner.py::_no_trade_reason`

After `bear_only` / `transition_window`, a named `sizing_reason` on
`ctx.buy_sizing_cash` is returned verbatim (`no_settled_cash`,
`no_buying_power`, `cash_unreadable`, `cash_read_failed`) ahead of the
counter rollups — the binding constraint for every candidate is the
account state, and the operator should read that, not `qp_zero_shares(n)`.
Missing attribute → unchanged behaviour (pinned).

## 4. Sim/live parity statement

- Same key, same canonical names, same aliases: pinned by
  `TestVocabularyMatchesSim` (live table ⊇ `sim_order_helpers` and
  `lean_account` tables; every `backtesting/renquant_104/strategy_config*.json`
  value normalises identically on both sides).
- `settled_cash`: sim `available_buying_power` = settled cash only (no
  pending T+N), benchmark sleeve treats sell proceeds as non-fundable; live
  = `account.cash` floored at 0, no same-bar credit. Same regime.
- `non_marginable_buying_power`: sim = settled + pending queue; live =
  Alpaca's field. Same regime (today's pinned value).
- `buying_power` (margin): LIVE ONLY. The sim normalisers reject it
  (`ValueError`, pinned by `test_sim_rejects_the_margin_mode_live_accepts`),
  so setting it breaks parity by construction; live logs a WARNING on every
  read. Explicit opt-in, never a default.
- Absent-key default DIFFERS: live → settled; sim → nmbp
  (`sim_order_helpers._normalize_buying_power_mode`). Every config in this
  repo and in the pin declares the key explicitly, so the default never
  binds today; not changed here (the sim default is the sim's contract).

## 5. Other `get_cash()` callers (behaviour change is the settled default)

- `adapters/runner_execmath.live_post_execution_snapshot` (post-run "cash"
  diagnostic) → now settled cash by default. More honest for a field named
  `cash`.
- `renquant-orchestrator native_live_snapshots.py:93 "cash": broker.get_cash()`
  → settled when handed the umbrella broker; the execution twin still
  returns nmbp (§7).
- `backtesting/renquant_103/adapters/runner.py:66` — not on any scheduled
  path.

## 6. Tests and CI

- `tests/test_buy_sizing_settled_cash.py` (new, 65 cases): vocabulary;
  sim-parity pins; the pure resolver (negative → 0 + `no_settled_cash`,
  `$33` settled is NOT topped up by `$1,034` nmbp, nmbp/margin modes
  honoured, unavailable-field degradation never exceeds settled cash,
  unparseable → `cash_unreadable`); `AlpacaBroker` with a stubbed trading
  client (08-27 / 08-28 shapes: `get_cash()` → 33.0 / 0.0 where the old code
  returned 1034 / 1904; one account read per snapshot; margin WARNING; old
  text gone); readonly forward; PaperBroker default; the adapter helper
  (absent key → settled + the `cash=33.00 nmbp=1034.00 mode=settled_cash`
  log line; explicit nmbp → 1904; `cash` alias; negative → $0 + WARNING;
  garbage → raises; legacy double); `unsettled_proceeds_spendable`; source
  pins on the runner wiring (old `cash = broker.get_cash()` gone; gate
  precedes the credit; named skip reason; rollup branch).
- `tests/test_no_trade_priority.py` (+5, `notification_contract` marker):
  `no_settled_cash` beats the counter rollups; a positive budget changes
  nothing; missing attribute is neutral; `cash_read_failed` named;
  `transition_window` still outranks.
- `tests/test_p0_fixes_regression_guards.py`: the P0-9 source-regex guard
  ("get_cash must use nmbp") is replaced by three behavioural pins of the
  new contract (default settled; nmbp available when configured; nmbp
  missing → settled).
- CI: `.github/workflows/live-broker-fractional-contract.yml` step 5 runs
  the new file (paths extended with `adapters/runner_execmath.py` and the
  test). Proven in a throwaway venv with exactly what the workflow installs
  (python3.10 + pytest, no pandas/numpy/alpaca): 65 passed, 0 skipped; the
  four existing steps unchanged (41/122/47/66 passed).
- Local: the affected suites (14 files) pass with the repo venv — 631
  passed; the only failures are 2 `TestEMA50GateFailSafe` cases in
  `tests/test_audit_2026_05_04_fixes.py` that exercise
  `kernel/pipeline/task_gates.py` (untouched here; their log shows
  `renquant_pipeline` not importable in the worktree). Full `-m "not slow"`
  suite (xdist): 15,661 passed / 69 failed / 2 collection errors on this
  branch. Controlled against clean `origin/main` in the SAME worktree
  (stash → rerun the 41 failing files → pop): the failure sets differ
  only by xdist ordering noise; every branch-only id rerun serially on
  both sides gives identical results (2 failed / 8 passed each, same
  ids; the 2 collection errors are `No module named 'renquant_pipeline'`
  — the scratchpad worktree has no pinned sibling beside it). No failure
  is attributable to this change.

## 7. Follow-ups (not in this PR)

1. **Operator decision** (top of this doc): strategy-104 config →
   `"buying_power_mode": "settled_cash"` + pin advance, if the live book is
   to size on settled cash. Until then the deployed number is unchanged and
   the log line is the evidence of what the account actually holds.
2. renquant-execution `alpaca_broker.py::get_cash` — port the same
   mode-driven contract (twin, `diverged_pin`); otherwise any consumer that
   is handed the execution broker still reads nmbp as "cash".
3. Sim absent-key default (`nmbp`) vs live (`settled`): align if the key is
   ever removed from a config.

## 8. Deploy (NOT done here — merged is not deployed)

The daily run executes the umbrella checkout at `/Users/renhao/git/github/RenQuant`
(REPO_DIR); after merge the operator fast-forwards it to `origin/main`
(ask-first landing action). First live evidence of the change = the
`runner: buy-sizing cash=… nmbp=… mode=non_marginable_buying_power` line in
the next daily log (mode = the pinned config's value until follow-up 1 lands).

**Rollback:** revert the merge commit. No config, state or artifact is
written by this PR; with the pinned config's `non_marginable_buying_power`
the sizing number is byte-identical to pre-fix, so the only observable
delta before follow-up 1 is logging + the post-run snapshot's `cash` field.
