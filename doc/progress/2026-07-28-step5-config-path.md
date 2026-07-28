# daily_104 Step 5: pass the resolved pinned config PATH, not the name

## STATUS
delivered

## WHAT
Step 5 now invokes live-bridge with `--strategy-config-path
"$BLEND_STRATEGY_CONFIG"` (the pinned-repo path its own gate already
resolved) instead of `--strategy-config-name`.

## WHY/DIR
Full-lane rehearsal (2026-07-28 pre-market, operator directive) hit it:
the runner resolves --strategy-config-name against the umbrella strategy
dir (backtesting/renquant_104/), where legacy shadow configs live as
tracked files — but strategy_config.shadow_blend.json exists ONLY in the
pinned strategy repo configs/. As merged, Step 5 would fail its first real
session ("Strategy config not found") despite its gate having found the
profile. Step 4 never hit this because shadow.json exists umbrella-side.

## EVIDENCE
Rehearsal log: gate resolved the pinned path, runner then errored
"Strategy config not found: .../backtesting/renquant_104/strategy_config
.shadow_blend.json", rc=1. Rerun with --strategy-config-path (same
resolved file) proceeds into the funnel. bash -n clean.

## NEXT
Merge before today's 13:55 session; the lane's first live Step-5 then
runs the pinned profile directly.
