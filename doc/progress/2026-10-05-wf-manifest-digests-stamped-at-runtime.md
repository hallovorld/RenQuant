# 2026-10-05 — WF manifest digests are stamped at runtime (supersedes RenQuant#639)

STATUS:   delivered, awaiting review — zero reviews at head. Supersedes #639
          (closed with a pointer here), which codex rejected on 2026-10-03 for
          committing regenerated walk-forward corpus bytes.
WHAT:     `scripts/weekly_wf_promote.sh` gains Step 3.5b: after the
          fingerprint stamp (Step 3.5) and before the gate (Step 4) it runs
          `scripts/stamp_wf_manifest_digests.py --manifest <strategy-dir
          manifest>` and then the same script with `--check`; any failure
          alarms WEEKLY-FAIL and exits 1. No corpus bytes, no manifest bytes
          are committed. The weekly-promote fixture gains a stub for the new
          script; a new test file pins ordering, fail-closed shape and the
          stamp→check sequence on a corpus whose bytes are rewritten first.
WHY/DIR:  root-cause, not a hand-stamp — the per-cut artifacts/calibrators
          are REWRITTEN by Step 3.5 on first contact (compact JSON +
          fingerprint), so their digests can only be known after that step,
          on the machine that runs it. Committing those bytes (#639) made the
          corpus a production-path write; hand-stamping the live manifest
          (2026-09-13 containment) made it drift. Stamping in the job that
          rewrites the bytes removes both.
EVIDENCE: see §4(b) below.
  artifact:      scripts/weekly_wf_promote.sh (Step 3.5b), tests/_weekly_promote_fixture.py (stub), tests/test_weekly_wf_promote_digest_stamp_step.py (3 cases).
  prod or exp:   exp — script change; effective on the next weekly run after merge + live sync. On the live tree the manifest already carries exactly these digests (containment), so the first run is a no-op stamp + a passing `--check`.
  existing data: stamping origin/main's UNSTAMPED manifest copy against the live corpus bytes reproduced the live contained manifest byte-for-byte (`cmp` identical, 43 entries / 86 digests) and `--check` on the live manifest reports 0 problems [VERIFIED 2026-10-05 21:4x PDT, scratch copy, read-only on the live tree]. The live per-cut files equal #639's committed copies byte-for-byte (two sampled, `cmp`) — i.e. #639 committed what Step 3.5 writes.
  best-known?:   yes for the mechanism; the stamp script refuses to overwrite a stamp whose bytes changed, so a recipe/corpus regeneration must also regenerate the manifest (that is the designed loud failure, not a silent re-stamp).
  scope:         scripts/weekly_wf_promote.sh, tests/ only. No data, config, artifact or manifest bytes touched.
NEXT:     after merge + live sync, the 2026-09-13 manifest containment can be
          closed as "legitimized by code" (its bytes are what Step 3.5b now
          produces); the committed manifest stays unstamped on purpose and goes
          dirty at runtime exactly like the per-cut artifacts already do.

## Conclusion

The WF gate has needed `artifact_sha256`/`calibrator_sha256` on every manifest
entry since the resolver's compatibility window closed on 2026-09-01; without
them every weekly retrain's simulation crashed (3/3 cuts failed execution,
2026-09-01..09-13). The digests describe the bytes Step 3.5 writes, so the
only place they can be stamped honestly is right after Step 3.5, every run.
Step 3.5b does that and self-verifies with `--check`.

## §4(b) evidence block

- `stamp_wf_manifest_digests.py --manifest <scratch copy of origin/main manifest> --resolve-as <live manifest path>` → `stamped 43 entries (86 file digests)`; `cmp` against the live contained manifest: identical [VERIFIED 2026-10-05].
- `--check` on the live manifest: `43 entries checked, 0 problem(s)` [VERIFIED 2026-10-05].
- Focused suites on this branch: test_weekly_wf_promote_digest_stamp_step (3), rfc210_fallback, snapshot_backstop, wrapper_guard, recipe_guard, wf_gate_sim_ran, stamp_wf_manifest_digests → 54 passed, 1 failed. The failure (`wrapper_guard::test_layer3_cuts_match_candidate_artifact_recipe`, config_fingerprint `14586756…` vs `f8fb2259…`) fails identically with this branch's changes stashed — it compares committed cut bytes to the committed candidate and is pre-existing on origin/main, not introduced here [VERIFIED: stash/pop run].
- Path contract: the fingerprint script resolves a relative manifest against the strategy dir; the digest script takes the path literally — Step 3.5b therefore passes `$REPO_DIR/backtesting/renquant_104/$WF_MANIFEST`, and the source test pins that.
