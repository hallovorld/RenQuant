# Progress: monthly meta-label retrain redesign RFC (task #73)

Date: 2026-07-18

## What

`doc/design/2026-07-18-metalabel-monthly-retrain-redesign.md` — RFC
making the chronically failing monthly meta-label job leakage-correct
and consumer-gated.

## Why

The job has failed every firing: the leakage guard CORRECTLY refuses a
snapshot sim that replays a 12-month window with the current scorer
(anchor ~2026-06-21 vs window start 2025-05-18). Investigation found
the as-of machinery already exists (WF corpus: 39 point-in-time
vintages, cutoffs 2024-01-01→2026-03-09; `model_as_of` keys on
cutoff_date) and was the deployed classifier's ORIGINAL 2026-05-11
validation methodology — the monthly job silently dropped it. Deeper:
the consumer is doubly dark (`ranking.meta_label.enabled=false` AND the
artifact was removed 2026-05-11), so even a successful retrain feeds
nothing.

## Design core

Step 0 consumer gate (dark → exit 0 "skipped by design", killing the
monthly alarm honestly); walk-forward snapshot config with explicit
manifest override (the prod pointer is a dead reference); fail-closed
corpus-staleness assertion (newest cutoff ≥ TRAIN_END − 35d). Labels,
trainer, health gate, schedule, and the guard itself unchanged. The
consumer re-arm decision (exit veto vs entry filter vs retire) is a
separate design PR.

## Status

RFC only — no script changes, no config changes, no deployment.
