# `make doctor` — one-command live-system health check

STATUS:   PR. New read-only health tool; no runtime change. Caught a real drift on first run.
WHAT:     `scripts/system_doctor.py` + `make doctor`. Composes the checks whose ABSENCE
          produced the 2026-06-23 deploy fragility: (1) PIN/RUNTIME DRIFT — every
          .subrepo_runtime/repos/<name> is at EXACTLY its subrepos.lock.json pin AND clean;
          (2) LOCK INTEGRITY — parses, source_repo.never_delete, every pin a full 40-hex sha;
          (3) BUNDLE CONSISTENCY — best-effort shell-out to the orchestrator's pre-deploy
          bundle check (#188), SKIP (not RED) if absent; (4) PROMOTE-BACKUP HYGIENE.
          Exit 0 green / 1 any RED; --json.
WHY-DIR:  the operator's #1 stability ask. Nothing previously verified the LIVE runtime still
          matches the audited pins — a hand-edit or half-applied promote drifts it silently.
EVIDENCE: 5 unit tests (lock integrity, backup pile-up, real tmp-git drift+dirty, unmaterialized
          skip). Run against the LIVE tree it (a) VALIDATED today's two deploys —
          runtime_at_pin[strategy-104]=a15a64b (demean live) and [pipeline]=42d6205 (mu-fix
          live) both match — and (b) CAUGHT a real drift nothing else flagged:
          runtime_clean[renquant-model]=dirty (training code edits in the model runtime).
          `[VERIFIED — pytest + live run]`
NEXT:     compose a readonly daily-full assert-buys check (heavy; reuse the promote --verify-cmd)
          + a data-freshness summary; wire `make doctor` into the daily-full preflight + the
          promote verify so a drift/failed deploy is caught before it reaches production.
