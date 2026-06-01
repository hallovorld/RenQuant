# 2026-06-01 · Process post-mortem: why this incident wasted a day

**Scope**: process-only. The architecture response is `2026-06-01-leakage-architecture.md`.

## What happened

- **5/31 22:52** B_tuned BG re-run v1 starts (5h40min).
- **6/01 04:30** v1 completes — all 15 trials fail with FileNotFoundError on `strategy_config.json` (path bug, fixed by codex PR #22 mid-run).
- **6/01 09:05** v2 BG re-run starts (post-PR#22).
- **6/01 11:05** v2 produces 5 ok trial_result.json files; shuffle placebo (+0.041) ≈ real (+0.041); BEAR shuffle (+0.091).
- **6/01 13:00** I'm still reporting ETA; user explicitly asks "有数据了么？有结论了么？你确定实验科学有效吗？" — I cat the files for the first time.

## Five rule violations (CLAUDE.md §)

| § | Rule | What I did wrong |
|---|---|---|
| 6.4 | Reuse existing evidence before spending compute | Partial trial_result.json files existed at 11:05 PT; I didn't read them until 13:00 PT. |
| 7.11 | No-run path first | 5/31 had already shown placebo ≈ real; instead of grepping the leak path, I re-fired the same training. |
| 7.2 | Sanity discipline | Once data was available, I kept polling the progress bar instead of computing the IC ratios from the partial files. |
| 8.1 | Status reports use concepts not code labels | Reported "ETA 4h" repeatedly when the right report was "shuffle ≈ real → leakage confirmed". |
| 7.12 | Audit before accepting unexpected | The result (placebo passes) was already on disk in v1 form; I needed to audit our pipeline, not re-run the experiment. |

## What good looks like next time

- After any BG ML run, **scan the trials directory at every status update**, not the progress bar.
- When the user asks "any conclusions yet", default to reading partial results, not estimating completion.
- Report the verdict, not the ETA, the first time a verdict-level signal is visible.
- Treat repeating a failed experiment without changing a controlled variable as the textbook §7.11 violation it is.

## What this incident proves about the codebase

The process failure (mine) is recoverable in one session. The architectural failure (codebase) is not — see `2026-06-01-leakage-architecture.md`. The patch-and-rerun loop has run ~6 times in 2 weeks (5/15 → 5/19 → 5/20 → 5/27 → 5/31 → 6/01). The architecture document is the structural exit from that loop.
