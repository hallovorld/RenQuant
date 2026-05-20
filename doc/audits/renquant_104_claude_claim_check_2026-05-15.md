# RenQuant 104 Claude Claim Check - 2026-05-15


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

Scope: verify the Claude Code status update against the local checkout and running processes.

This is a claim check, not a replacement for `doc/audits/renquant_104_deep_audit_2026-05-15.md`.

## Executive Read

Claude's update is partially true:

- The three audit-mandated regression tests exist locally and pass.
- The p0activated simulation batch is genuinely running.
- NGBoost training is genuinely running.

But the update is not sufficient to close the audit:

- The cited commit `7e1f80c` does not exist in this local checkout.
- The equivalent local commit appears to be `e65877a`.
- Two `python scripts/train_ngboost_proper.py --help` processes are consuming near-full CPU, because the script has no argparse/help path and runs training even when called with `--help`.
- The p0activated partial logs already show important new red flags: Kelly zeroing candidates, QP infeasibility, clipped calibrator targets, feature-health warnings, top-up orders after QP handling, and insufficient-cash warnings.
- The three new tests cover only the vol-target / drawdown-Kelly exposure-scaling issue. They do not cover the deeper P0/P1 audit findings.

## Commands Run

```bash
git show --stat --oneline --decorate --name-only 7e1f80c
```

Result: failed. Local git reports `unknown revision or path not in the working tree`.

```bash
git log --all --oneline --decorate | rg '7e1f80c|e65877a|555f5b1'
```

Result:

```text
e65877a test: ship 3 audit-mandated regression tests + p0activated sim configs
555f5b1 fix(qp): hoist vol-target + DD-Kelly out of dormant Kelly path
```

```bash
/Users/renhao/miniconda3/envs/renquant/bin/python -m pytest \
  tests/test_vol_target_scales_qp_upper.py \
  tests/test_dd_kelly_scales_qp_upper.py \
  tests/test_vol_target_independent_of_ngb.py \
  -q
```

Result:

```text
11 passed in 1.49s
```

## Claim 1: "3 Regression Tests Fixed"

Verdict: materially true, commit hash wrong.

The files exist:

- `tests/test_vol_target_scales_qp_upper.py`
- `tests/test_dd_kelly_scales_qp_upper.py`
- `tests/test_vol_target_independent_of_ngb.py`

The passing result is real: 11/11 passed locally.

The cited commit `7e1f80c` is not present. The local commit containing these tests is:

```text
e65877a test: ship 3 audit-mandated regression tests + p0activated sim configs
```

Important limitation: these tests only validate that vol-target and drawdown-Kelly exposure scaling hit QP upper bounds and do not depend on NGBoost. They do not validate:

- data freshness semantics after market close;
- covariance artifact path resolution;
- QP regime enablement;
- score saturation / calibrator pairing;
- Kelly behavior when NGBoost is off;
- timeout enforcement;
- future-dated training config;
- production artifact metadata health.

## Claim 2: "NGBoost SUSPECT Running"

Verdict: running, and Claude's first-seed number is verified; the full 5-seed conclusion is still incomplete.

Evidence file:

```text
logs/ngb_proper/2026-05-15.log
```

Observed results so far:

```text
seed=42   val_ic=+0.0372  sigma-calib=+0.275  mu_xs_std=0.01637  best_iter=36
seed=7    val_ic=+0.0365  sigma-calib=+0.269  mu_xs_std=0.01563  best_iter=8
seed=123  val_ic=+0.0293  sigma-calib=+0.273  mu_xs_std=0.01712  best_iter=33
seed=2024 running at time of check
```

Claude's specific first-seed statement is true:

- `val_IC=+0.0372`
- XGB baseline quoted by the script is `+0.0294 +/- 0.0029`
- `sigma-calib=+0.275`

But the statistical conclusion is not yet available because only 3/5 seeds had landed at the time of this check. Seed 123 is essentially equal to the XGB baseline, so the final t-test still matters.

Process-control issue:

Observed running processes include:

```text
.venv/bin/python scripts/train_ngboost_proper.py
python scripts/train_ngboost_proper.py --help
python scripts/train_ngboost_proper.py --help
```

The two `--help` processes are consuming roughly a full CPU core each. Inspection of `scripts/train_ngboost_proper.py` shows no argparse path; `if __name__ == "__main__": sys.exit(main())` runs full training unconditionally. Therefore a diagnostic help command accidentally starts the expensive 5-seed training loop.

This does not invalidate the NGBoost experiment, but it is operationally sloppy and can corrupt runtime expectations:

- duplicated heavy training jobs;
- misleading shell commands;
- wasted CPU while simulations are also running;
- hard-to-reproduce wall-clock estimates.

## Claim 3: "p0activated 16-Window Sim Running"

Verdict: running, not yet validated.

Observed active `scripts/run_sim_104.py` processes for the p0activated config. Partial equity outputs currently exist for:

```text
Q01.json
Q02.json
Q03.json
Q04.json
Q05.json
Q06.json
```

Partial outcomes seen so far:

```text
Q01 final=95697.22  return=-4.3%
Q02 final=96179.74  return=-3.8%
Q03 final=98521.81  return=-1.5%
Q04 final=99714.88  return=-0.3%
Q05 final=102850.20 return=+2.9%
Q06 final=94661.11  return=-5.3%
```

This is not yet an A/B validation. A valid conclusion needs all 16 windows, the baseline comparator, regime breakdown, drawdown profile, turnover/tax profile, and a failure-count table.

## New Red Flags From p0activated Logs

The running logs expose several issues that should be investigated before treating p0activated as "fixed":

1. Kelly still zeros trades in defensive / weak-edge contexts.

Example from Q01:

```text
ApplyKellySizingTask: ... cands=0 non-zero ... zero_reasons[mu_le_min_edge=1]
SizeAndEmitTask: GLD Kelly=0 - skip
NoTradeAlert: 41 consecutive days with zero orders
```

This overlaps with the original audit concern that sizing/expected-return plumbing can turn apparently valid candidates into no-trade events.

2. QP infeasibility appears repeatedly.

Example from Q02:

```text
QP infeasible: status=infeasible ... min_invested_pct=0.700
QP still infeasible after relax - dropping C2 caps for this bar
EmitOrdersFromQPSolutionTask: status=infeasible:infeasible - skip
```

This suggests the new flags may have moved risk from one subsystem into the optimizer constraints rather than resolved it.

3. Global calibrator target clipping happens at runtime.

Example:

```text
expected_return.y has max|y|=0.9540 > 0.20 sanity bound ... clipping to +/-0.20
Until the calibrator is retrained with clipped targets, Kelly sizing on this signal is suspect.
```

Runtime clipping is not a clean fix. The calibrator should be trained on the intended target transform and validated as an artifact contract.

4. Feature-health warnings still appear.

Example:

```text
ALL 5 fund features are 0 across 1 tickers
ALL 3 PEAD features are 0 across 1 tickers
ALL 3 SUE features are 0 across 1 tickers
```

Some one-ticker defensive cases can make this less severe, but these warnings are still important because they are exactly the kind of silent feature collapse that can make a panel ranker look valid while using degraded input.

5. QP and top-up logic may be emitting conflicting actions.

Example from Q04:

```text
JointActionTask: solver=qp - already handled by QP task
TopUpHeldTask: TXN +53 shares
TopUpHeldTask: AAPL +60 shares
...
SimAdapter: insufficient cash for TXN
SimAdapter: insufficient cash for AAPL
```

If QP is the joint allocator, post-QP top-up should be audited carefully. This can invalidate the optimizer's intended weights and produce cash-allocation artifacts.

## Bottom Line

Claude made real progress on one narrow implementation gap. The regression tests are useful and passing.

However, the claims do not close the major audit findings. The system still needs hard evidence for:

- correct market-close data freshness behavior;
- correct production covariance loading;
- QP regime gating parity with rotation gating;
- unsaturated live score calibration;
- Kelly/NGBoost/offline fallback semantics;
- valid expected-return units for QP;
- timeout enforcement;
- artifact-contract checks before live trading;
- completed p0activated 16-window A/B results.

Recommendation: do not mark RenQuant 104 as production-trustworthy based on this Claude update alone.
