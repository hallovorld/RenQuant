# Atomic, verified, reversible pin promotion (roadmap eng #2)

STATUS:   PR. New manual deploy tool; DRY-RUN by default — touches the live pin ONLY on
          --apply, which the operator runs deliberately. No automatic/scheduled use.
WHAT:     `scripts/promote_pin.py` — replaces the manual "hand-edit subrepos.lock.json + run
          subrepo_assemble + hope" dance (the 6-step process behind the 2026-06-23 deploy
          fragility). `bump --subrepo X --commit <sha> [--apply]` and `revert [--apply]`.
WHY-DIR:  the postmortem's (#189) root-cause #2 — deploy fragility — and the direct de-risker
          for the imminent mu启动 pin bump: once pipeline #146/#147 merge, bumping the
          renquant-pipeline pin becomes one safe command instead of six manual ones.
SAFETY:   --apply: backup subrepos.lock.json (timestamped) → atomic write (temp + os.replace,
          never a half-written lock) → subrepo_assemble --sync → optional --verify-cmd (e.g.
          the bundle self-consistency check #188) → AUTO-REVERT (restore backup + re-sync) if
          sync or verify fails. Always prints the one-command revert. Rejects no-op bumps,
          non-sha commits, unknown subrepos.
EVIDENCE: 6 unit tests (bump purity, no-op/bad-input rejection, atomic roundtrip, CLI dry-run
          no-write, --apply write+backup+revert). Dry-run verified against the real lock (it
          reads + previews, writes nothing without --apply). `[VERIFIED — pytest + dry-run]`
NEXT:     compose with a readonly daily-full assert-buys as the standard --verify-cmd, so a
          promote that would stop trading auto-reverts. Pairs with #188 (verify half) to make
          the build→deploy atomic + reversible end-to-end.
