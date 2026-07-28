# Umbrella shadow-scoring copy: strategy_dir-first artifact resolution

## STATUS
delivered

## WHAT
`backtesting/renquant_104/kernel/panel_pipeline/shadow_scoring.py`: relative
shadow artifact paths now resolve strategy_dir-first (then repo root),
mirroring the #211 canonical resolver. One conditional; behavior for
absolute paths and repo-root-resident artifacts unchanged.

## WHY/DIR
The 2026-07-27 evening preflight probe surfaced the divergence: this
umbrella-local copy resolved ONLY against the repo root, so the clf shadow
(strategy_dir-relative artifacts/shadow/panel-clf...) reported "artifact
not found" on every consumer that executes this copy — while the pinned
pipeline copy loaded the identical config fine (today's 13:55 session:
loaded, 84 scored). Duplicated-kernel divergence class (see the
triple-impl playbook); the probe/sentinel surfaces run this copy.

## EVIDENCE
Repro: repo-root candidate /Users/.../RenQuant/artifacts/shadow/... does
not exist; strategy_dir candidate exists (content 6101a9fe). Post-fix the
strategy_dir candidate wins; ast/syntax verified; e2e probe rerun follows
this PR (operator directive: e2e complete and perfect).

## NEXT
Probe rerun must show the clf shadow LOADED in probe mode (zero
load-failure warnings); longer-term the umbrella copy should delegate to
renquant_pipeline's shadow task outright (fork retirement, F-2 class).
