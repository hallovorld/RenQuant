# Regenerate the strategy-104 snapshot after the A4-T1 promotion   (PR TBD)

STATUS:    delivered — snapshot-only follow-up to RenQuant#632.
WHAT:      `doc/arch/strategy-104-snapshot.md` re-rendered from the live tree
           (`render_strategy_104_snapshot.py --output <scratch>`, read-only on
           the tree) after `weekly_wf_promote.sh --promote-staged 20260831T141820Z`
           swapped the active pair. No code, no lock change.
WHY/DIR:   `render --check` in the live tree reported STALE after the
           promotion (the active artifact's metadata hash is one of the
           snapshot's sources), so `make doctor` would alarm until the
           committed snapshot matches. The diff is exactly the promoted
           artifact's provenance: `panel-ltr.alpha158_fund.json`
           sha256 6461b827… → f1b1c132…, trained_date 2026-08-02 → 2026-08-31,
           binding cutoff effective_train_cutoff_date=2026-06-03, train_run_id
           b43751be → 8b9b8093, oos_mean_ic (stamped) +0.0448 → +0.0534, WF gate
           stamped passed=false run_at 2026-08-31T14:26:19; calibrator artifact
           bce257d1… → 43e859b4…, trained 2026-08-31, pool_ic +0.1192 → +0.1250.
EVIDENCE:  artifact:      `doc/arch/strategy-104-snapshot.md`; sources listed in its own machine block
           prod or exp:   prod (documentation of the served pair; nothing executed)
           existing data: 26 differing lines vs the committed snapshot, all in the active-model / active-calibrator blocks + source fingerprint [VERIFIED — diff 2026-09-03 09:11 PDT]; promotion itself recorded on RenQuant#632 (PROMOTED, receipt 2cd9d27b…)
           best-known?:   n/a — the promoted candidate is a zero-trade artifact admitted under the time-limited A4-T1 authorization, not a quality claim; the stamped oos_mean_ic is the trainer's own number
           scope:         "this is the snapshot of the pair served since 2026-09-03 09:09 PDT — a rendering, not a verdict"
NEXT:      none — `make doctor` / `render --check` green once merged and the live tree is ff-pulled.
