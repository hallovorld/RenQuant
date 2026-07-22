# Snapshot provenance fixes for the strategy-104 pin advance   (PR #524)

STATUS:    in-progress
WHAT:      Advances the strategy-104 subrepo pin (082dccd -> 1e840cd, PatchTST
           shadow-path fix #62 + shadow watchlist #58) and fixes the
           pre-deploy snapshot renderer
           (`scripts/render_strategy_104_snapshot.py`) so it cannot go green
           over a candidate whose scorer provenance is silently `unknown` or
           whose fingerprint depends on mutable live state:
           1. One resolver, same as runtime (was a second, drifting
              resolver): `_resolve_artifact_path` now delegates to
              renquant-pipeline's canonical `kernel.artifact_resolver.locate_artifact`
              (imported from the pinned checkout) — the same
              `absolute -> strategy_dir -> repo_root` order the loader and
              the pre-deploy CI gate (#525) use. Declaration-only CI (no
              `.subrepo_runtime`) falls back to a display-only strategy_dir
              join.
           2. Required-scorer provenance can no longer be silently
              `unknown`: a CONFIGURED scorer (active / in-run shadow /
              shadow-e2e) that fails to resolve to a metadata-bearing file
              now emits a `SCORER PROVENANCE UNRESOLVED` warning.
           3. Re-fit calibrators are runtime observations, not part of the
              candidate fingerprint: no longer folded into the `sources` set
              that feeds the Source fingerprint (digests still render,
              under an explicit "runtime observation, excluded" note), and
              `--source-pin` renders the snapshot as a pre-deploy
              declaration from a candidate lock checkout rather than the
              live runtime HEAD.
WHY/DIR:   Codex round-2 CR on this PR found the snapshot could report a
           pin as pre-deploy-clean while (a) the PatchTST shadow scorer's
           provenance rendered `unknown` (resolver joined only
           `strategy_dir/ref`, missing the umbrella-root ref the daily
           loader actually resolves) and (b) the fingerprint absorbed the
           mutable live-refit calibrator instead of the pinned candidate —
           both would let a pin advance past the snapshot gate without
           proving the deployed scorer is the one actually traceable.
EVIDENCE:  n/a (CI/snapshot-renderer contract change, no model/data
           performance claim). `scripts/render_strategy_104_snapshot.py
           --selftest` -> SELFTEST OK.
           `python3 -m pytest tests/test_render_strategy_104_snapshot.py`
           -> 16 passed. [VERIFIED]
NEXT:      Two review threads remain open and are NOT resolved by this
           fix-pass (both need actions outside this PR's code):
           1. Codex's latest review says this pin candidate still has no
              passing candidate-assembly artifact gate — #525 must merge
              first, then the #525 gate must run against this exact lock
              candidate and the result attached here before reconsidering
              the pin.
           2. The committed `doc/arch/strategy-104-snapshot.md` was
              rendered by the OLD renderer; a faithful regeneration needs
              the live `.subrepo_runtime` + artifacts and is an
              operator-gated step (`make snapshot` on the live machine as
              part of the pin deploy), not something a hosted/worktree
              checkout can produce. Do not merge until both are closed.
