# Pin bump: orchestrator namespace-alias fix (#524)

Date: 2026-07-16

`renquant-orchestrator` `511edfa9` → the #524 merge (1 commit): bootstrap
now force-aliases non-owned kernel stems under BOTH `kernel.<stem>` and
`renquant_pipeline.kernel.<stem>`. The first post-F-8 full daily died at
MetaLabelVetoTask (`renquant_pipeline.kernel.meta_label.task_meta_label_veto`
exists only in the authoritative backtesting copy). Reproduced pre-fix,
proven post-fix in the live env; regression test added. Last blocker for
the 07-16 qualifying daily run.
