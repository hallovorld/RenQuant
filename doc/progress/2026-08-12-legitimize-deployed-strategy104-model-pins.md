# 2026-08-12 — legitimize two already-deployed subrepo pins in the reviewed surface

STATUS:   RECORDING DEPLOYED REALITY (2026-08-12). Origin's `subrepos.lock.json`
          lagged behind the operator machine, which advanced both pins on
          2026-08-12 (market closed, operator-authorized). This PR syncs the
          reviewed surface so the daily run-surface-drift scan stops flagging
          them. CI `verify-pinned-declaration` reproduced locally on the edited
          tree: **OK (exit 0)** `[VERIFIED — python3 scripts/render_strategy_104_snapshot.py
          --verify-pinned-declaration --configs-dir <s104@e00d9356>/configs
          --lock-file subrepos.lock.json --output doc/arch/strategy-104-snapshot.md]`.

WHAT:     Two `commit` values in `subrepos.lock.json`, nothing else:
          - `renquant-strategy-104`:
            `c8bba9c9b30c960f85daa2529ee422a0997ad607`
            -> `e00d9356ac620426df031e0c08ce66301c50c22e`
          - `renquant-model`:
            `96fe2d3daeb33d4083df40594366ec2066f0070f`
            -> `bd0fa488d2164121d30ccda9d9e781fcd73637b3`
          Plus the machine-generated `doc/arch/strategy-104-snapshot.md`, whose
          machine block records the strategy-104 pin: regenerated for the new
          pin (see WHY/DIR — the snapshot-fresh gate requires it).

WHY/DIR:  These are code/config pins that are ALREADY SERVING; the PR closes the
          origin-vs-machine drift, it does not decide anything new.

          renquant-strategy-104 `e00d9356` is the 2026-08-06 config commit #94
          "feat(risk): per-name cap 12% -> 30%, slots stay 8 (operator
          directive)" — an ancestor of strategy-104 `origin/main`, running live
          since 2026-08-07 `[VERIFIED — git merge-base --is-ancestor c8bba9c9
          e00d9356 (forward) AND e00d9356 ancestor of s104 origin/main]`. The
          only functional delta from the prior pin is two fields, in the active
          config and every shadow variant: `regime_params.BULL_CALM.max_position_pct`
          0.12 -> 0.3 and `kelly_sizing.max_concentration` 0.12 -> 0.3
          `[VERIFIED — git diff c8bba9c9..e00d9356 -- configs/]`. Reverting this
          pin would drop the live per-name cap back to 12% and could force-sell
          live positions, so it MUST stay `e00d9356`.

          renquant-model `bd0fa488` is renquant-model `origin/main` tip
          `[VERIFIED — git ls-remote https://github.com/hallovorld/renquant-model
          HEAD]`, reached from the prior pin by 21 first-parent merges (62
          commits total), every one merged/reviewed on model `main`
          `[VERIFIED — git merge-base --is-ancestor 96fe2d3 bd0fa488; git
          rev-list --count --first-parent 96fe2d3..bd0fa488 = 21]`. The
          operative reason to deploy it now: the momentum-train receipt-writer
          (#221) + its test-isolation fix (#225, "make receipt dir injectable so
          tests stop writing the prod path") so the scorer-identity monitor
          stops raising false "silent scorer swap" CRITICALs on legitimate
          weekly shadow rebuilds. renquant-model is the training factory, NOT
          live serving (role in the lock: "MODEL FACTORY"); blast radius is
          training/shadow-side.

          Snapshot regen — why it is in this PR and why it is a hand-verified
          delta, not a fresh-clone render: the `strategy-104-snapshot-fresh`
          CI job's `verify-pinned-declaration` clones strategy-104 at the lock
          pin and fails if the committed snapshot's machine block was generated
          from a different strategy-104 pin. It failed on the stale snapshot
          `[VERIFIED — "VERIFY FAIL: snapshot was generated from strategy-104
          pin c8bba9c9... but subrepos.lock.json pins e00d9356..."]`. A naive
          `render` in this fresh clone is WRONG: the snapshot references a LIVE
          shadow artifact (`.../hf_patchtst_all_seed44_model.pt.metadata.json`)
          that is not committed to the umbrella and is absent from any clone, so
          a full render DEGRADES real operator-captured provenance
          (trained_date, config_fingerprint, feature count, digest) to
          "unknown (file missing)". Instead I edited only the fields the two
          pins genuinely drive — the recorded pins, the two cap knobs (0.1200 ->
          0.3000), the two changed source-file hashes — and recomputed the
          `Source fingerprint` with the renderer's OWN `_source_fingerprint`
          over the committed source-hash map with exactly those two
          substitutions. Reconstruction validated: the recomputed OLD
          fingerprint reproduces the committed `3a31f41b...` byte-for-byte, and
          the two new hashes are the renderer's own `_sha256_file` output
          `[VERIFIED — strategy_config.json sha256:43cbb9b2021a1c68,
          subrepos.lock.json sha256:a58b263618fa3dc6; new fingerprint
          e20d5ac6...; edited machine block re-hashes to that same fingerprint]`.

EVIDENCE:
  artifact:      `subrepos.lock.json` (2 `commit` lines) and
                 `doc/arch/strategy-104-snapshot.md` (machine-generated
                 production snapshot; machine block + the two pin-driven knobs +
                 the two changed source hashes + recomputed fingerprint).
  prod or exp:   prod-adjacent reviewed surface. Both pins are ALREADY deployed
                 and serving on the operator machine (advanced 2026-08-12 under
                 authorization); this PR only makes origin match. No live state,
                 config, artifact, or `.subrepo_runtime` checkout was written —
                 the work happened in an isolated fresh clone.
  existing data: yes — resolved every sha READ-ONLY against the public/subrepo
                 remotes (ls-remote / bare clones) and against the committed
                 config diff. Nothing generated on any live tree.
  best-known?:   yes — the candidate-pin artifact gate + snapshot regen that a
                 pin bump normally runs are satisfied here without new
                 artifacts: NO artifact_path changed at `e00d9356` (commit #94
                 touched only the two cap scalars), and renquant-model publishes
                 to renquant-artifacts whose pin is UNCHANGED. So the artifacts
                 both pins reference are byte-identical to what already serves;
                 the only regeneration owed is the snapshot's recorded
                 pin/knobs/hashes, done above.
  scope:         "this is the umbrella reviewed surface (`subrepos.lock.json` +
                 the strategy-104 snapshot it drives), vs existing best =
                 origin lagging the operator machine by these two pins. It does
                 NOT re-derive whether either pin is a good idea (both are
                 operator-authorized, already live), and it does NOT touch the
                 pre-existing execution/pipeline snapshot-table drift, which is
                 out of scope for this PR."

NEXT: none blocking. This PR IS the reviewed-surface record the CONTAINMENT
      PROTOCOL / drift scan asks for. The operator machine already ran its
      byte-exact `make snapshot-check` / `promote_pin` snapshot regen at deploy
      on 2026-08-12; this PR carries the equivalent semantic + machine-block
      update the hosted CI gate enforces. Pre-existing observation, deliberately
      NOT bundled: the snapshot's human "Subrepo pins" table still shows an
      older execution (`5724dc74`) and pipeline (`e13cd3eb`) pin than the
      committed lock (`91c7bf88` / `4aec0e35`) — that drift predates this PR,
      does not affect any CI gate (the fingerprint is over source hashes, not
      the pins table), and should be closed by whichever pin bump legitimizes
      those two.
