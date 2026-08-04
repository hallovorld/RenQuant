# 2026-08-04 — pin advance: pipeline → a3686efb (P-WF-GATE RFC#210 license)

Single-pin advance carrying renquant-pipeline#263: P-WF-GATE (both twins)
learns the RFC#210 freshness-fallback serving license. Root cause and fix are
documented in the pipeline repo's progress doc
(`doc/progress/2026-08-04-preflight-rfc210-license.md` there); the incident:
today's daily-full run hard-failed P-WF-GATE on the governance-promoted
artifact (`passed=False` by design) and the book went sell-only (0 buys).

- `subrepos.lock.json`: renquant-pipeline `f25574fc` → `a3686efb`
  (sha read back from the merge API output).
- `doc/arch/strategy-104-snapshot.md`: re-rendered by
  `scripts/render_strategy_104_snapshot.py` against a fakeroot whose pipeline
  checkout sits at the new pin (5-line diff: pin row + AS-OF).

Deploy after merge (separate granted step, grants-logged): live-tree pull +
`.subrepo_runtime` sync. Acceptance = tomorrow's 13:55 PT daily run: P-WF-GATE
hard PASS with `freshness_fallback_rfc210` provenance in details, buys
unblocked.
