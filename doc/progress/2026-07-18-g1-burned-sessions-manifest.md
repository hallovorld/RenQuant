# 2026-07-18 — G1 burned-sessions manifest (v5 prereg §4.7 rule 1 prep)

STATUS: prep artifact — pilot registration NOT authorized (turnkey after the
epoch-5 refreeze); this PR only enumerates.

WHAT: `doc/experiments/g1-pilot/burned-sessions-manifest.json` (+ md
companion) — the committed enumeration §4.7 rule 1 requires at the
pilot-registration commit. Every two-arm paired session to date, compiled
read-only from the harness home (`~/renquant-shadow-ab`: session records,
per-date bundles, 3 epoch archives + EPOCH-NOTEs, pre-arming archive,
counters, freezes, logs, run manifest). 14 entries, ALL burned: 3 valid
pairs (all Saturday 2026-07-11 — the epoch-1 preflight whose record was
overwritten, the intact epoch-2 pair, and the epoch-3-minting 09:43 PT
pair whose discrete record was overwritten but whose sealed inputs + full
per-arm output trees survive unblinded in the live root), 8 failed
attempts, 3 no-record scheduler failures (2026-07-14..16 stall). Epoch-4
has ZERO sessions. No DB-backed session records exist (no `*.db` under
the harness root); the JSON records + logs are the complete set.

REVIEW CORRECTION (Codex, 2026-07-18): the first compilation MISSED the
09:43 PT epoch-3-minting valid pair (session-log start 16:43:06Z, exit=0
at line 819; freeze frozen_at 16:43:08Z = start+2s; sealed digest
36bffe24, universe 145) because its discrete record had been overwritten
by the 14:35 PT abort — and consequently mis-attributed that abort's
same_world collision to the 07:20 PT pair (whose digest is f215af9b and
whose bundle was already archived). Both fixed; totals now 14/3-valid;
the first compilation's by_epoch also miscounted its own epoch-3 entries
(stated 6, enumerated 7; corrected mechanical count = 8). Scope note
added: the manifest (log-sweeping) — not the record-only PART-B telemetry
— is the completeness surface for the burn set.

WHY/DIR: G1 v5 (#494) two-stage start: the registration commit must freeze
this manifest — "the manifest, not prose, is the auditable object". The
known analytic exposure is pinned to citations: #485's "~4 paired sessions"
DGP caveat (+ its committed results JSON + progress doc), the
write-containment memo's direct read of the 02:15 pair's byproducts, the
weekday-gate memo's analysis of the Saturday world, and renquant-model#60's
PIT schedule-mismatch over the epoch-3 era. Honesty note kept explicit:
#485's sigma_d was PROVISIONAL PARAMETRIC, not fitted from the pairs — they
are burned as analyzed-in-a-memo evidence, per the rule's own standard.

NEXT: (a) PART-B telemetry PR in renquant-orchestrator (per-epoch/per-role
n_paired_sessions, safe default all-burned) consumes this manifest's
format; (b) at registration, append any sessions recorded after 2026-07-18,
stamp `pilot_registration_commit`, freeze.
