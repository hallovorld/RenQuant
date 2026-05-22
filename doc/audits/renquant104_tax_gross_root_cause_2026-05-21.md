# renquant_104 WF Tax/Gross Root Cause

Date: 2026-05-21

## Finding

The 172-feature WL200 walk-forward evidence did not fail because `tax` was
arithmetically greater than profitable trade gross PnL. It failed because the
sim tax model taxes every realized winner immediately and gives no immediate
benefit for realized losers.

That makes total tax compare against **positive gross only**, while the reported
`gross_pnl` line is **winners plus losers netted together**.

Across the three 172-feature WF cuts:

| cut | positive gross | negative gross | net gross | current per-trade tax | net after current tax |
|---|---:|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | 48,651.87 | -29,568.48 | 19,083.39 | 24,202.48 | -5,119.09 |
| 2024-07-01 to 2025-06-30 | 42,776.17 | -30,236.15 | 12,540.02 | 21,387.21 | -8,847.20 |
| 2025-04-01 to 2026-03-28 | 65,580.87 | -25,403.42 | 40,177.45 | 32,790.44 | 7,387.01 |
| total | 157,008.92 | -85,208.06 | 71,800.86 | 78,380.13 | -6,579.27 |

`78,380.13 / 157,008.92 = 49.92%`, matching the configured short-term tax rate.
So the ledger is internally consistent under its current model.

## Why This Looks Impossible

It looks impossible because humans read "gross PnL" as the tax base. In this
report, it is not. The effective tax base is winner-only gross PnL:

```text
tax ~= 50% * sum(max(trade_gross_pnl, 0))
net_gross = sum(winners) + sum(losers)
```

When losers are large enough, `net_gross < tax` is possible.

## Is This Tax Model Scientific?

Not as a faithful taxable-account return model.

The project currently models tax as an immediate per-trade cash debit:

- `kernel/portfolio.py::compute_trade_tax()` taxes only positive realized PnL.
- `adapters/sim.py::_apply_sell()` subtracts that tax from cash on trade date.
- Loss trades produce no immediate tax credit.

This is a valid stress-test assumption, but it is not a faithful U.S. Schedule D
capital-gains model. Official IRS materials describe capital gains/losses as
being summarized on Schedule D; Pub. 544 says capital gains are taxable when
total gains exceed total losses, and capital losses are deductible subject to
limits/carryover. See:

- IRS Topic 409, Capital gains and losses:
  https://www.irs.gov/taxtopics/tc409
- IRS Publication 544, Sales and Other Dispositions of Assets:
  https://www.irs.gov/publications/p544
- IRS Schedule D instructions:
  https://www.irs.gov/instructions/i1040sd

Approximate same-period annual-netting tax, ignoring wash-sale deferrals and
assuming all closed trades are short-term, would be:

| cut | net gross | current per-trade tax | annual-net ST tax approx | after-tax if annual-net |
|---|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | 19,083.39 | 24,202.48 | 9,541.69 | 9,541.69 |
| 2024-07-01 to 2025-06-30 | 12,540.02 | 21,387.21 | 6,270.01 | 6,270.01 |
| 2025-04-01 to 2026-03-28 | 40,177.45 | 32,790.44 | 20,088.73 | 20,088.73 |

This does not prove the strategy is good. It proves the current after-tax WF
gate is mixing a trading-performance question with an overly punitive tax-cash
timing assumption.

## Trading Root Cause

Even before fixing the tax model, the trading tree is weak:

- All 509 closed WF round trips entered in `BULL_CALM`. The regime router is not
  creating meaningful exposure diversity in these WF cuts.
- Stop-loss exits are the largest economic damage:
  - 75 trades
  - gross/net PnL: -58,740.15
  - median hold: 15 days
  - win rate: 0%
- Single-day-loss exits add another -11,163.99 after tax.
- The rank score is not economically monotonic. The top score decile is net
  negative after tax, so "higher score = better trade" is not validated.
- Median hold is 24 days and 99th percentile hold is 90 days, so almost all
  winners are short-term taxable.

## Required Fix

Do not archive the project on this evidence. Archive or demote the current
production candidate.

Required engineering repair:

1. Split performance reporting into pre-tax, per-trade-tax stress, and
   annual-net-tax approximations.
2. Add a tax-ledger mode that nets realized short-term and long-term gains/losses
   by tax year, applies wash-sale deferrals, then debits estimated tax at year
   end or configured payment dates.
3. Keep the current per-trade immediate-tax model only as a stress mode, clearly
   labeled.
4. Add acceptance gates that separately require:
   - positive gross edge,
   - score/PnL monotonicity,
   - annual-net after-tax viability,
   - SPY benchmark outperformance,
   - no single exit family dominating losses.
5. Fix WF config drift: the default `strategy_config.sim_wl200.json` still points
   at the old 169-feature manifest, while current prod is a 172-feature sentiment
   artifact. The 172-feature config validates recipe scope correctly but is not
   the default promotion path.

## Bottom Line

`tax > net gross` is explainable under the current sim code, but the model is not
scientifically adequate as the only promotion metric for a taxable account. The
current 172-feature XGB/panel-LTR candidate should remain rejected; the project
should not be archived solely from this result.
