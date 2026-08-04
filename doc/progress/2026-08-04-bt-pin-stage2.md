# 2026-08-04 — backtesting pin advance: the Stage-2 stamp goes live

STATUS:    single-pin advance carrying bt#104
WHAT:      renquant-backtesting ea7b014a -> 8c2c4456 (read back from
           main): bt#104's Stage-2 scoring-stamp wiring, executed under
           the operator's same-day sign-off (record on bt#94). After
           deployment, the next gate run stamps lineage_stage2 as a
           SIBLING beside lineage_stage1 — admission byte-identical
           (the wiring's own integration test executes that property).
           No other pin moves; the RFC#210 provider (bt#102) is already
           in the from-pin, unchanged.
WHY/DIR:   operator directive ("不等！干！") — the wiring merged today;
           holding it for a future batch is deferral without a reason.
EVIDENCE:  bt#104 merged 8c2c4456 with the execution-level runner/helper
           integration (real stamp + unmutated admission inputs);
           full bt suite 624 passed at the wiring PR with the same 2
           machine-local failures as clean main.
NEXT:      merge -> live pull + runtime backtesting sync -> the next
           weekly gate run (possibly today's post-close rerun if this
           lands first; otherwise Saturday) carries the first stamp.
