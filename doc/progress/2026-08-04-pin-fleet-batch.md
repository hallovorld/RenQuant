# 2026-08-04 — fleet pin batch: all four blend shadows land (GOAL-9 AC2 complete)

Two pins in one reviewed batch, completing the fleet's serving surface:

- renquant-strategy-104 `2358b56b` → `b99101d5`: carries s104#89 (F2 profile,
  fast-blend, pending-marker dormant) + s104#90 (F1/F3 3-component profiles
  with mechanical semantic guards).
- renquant-pipeline `ab5db5ab` → `e13cd3eb`: carries pipeline#267
  (BlendPanelScorer ≥2 components — the F1/F3 enabler).

Rails already merged on the umbrella main (Step 5c RQ#575, Step 5d/5e RQ#576
incl. the per-lane success-echo audit fix that also repaired the latent 5c
instance). Snapshot re-rendered at the new pins.

After deploy (batch 7): F1 serves its FIRST real decision at the next daily
(both components exist); F2/F3 stay dormant until the 2026-08-08 fast genesis
batch (orch#795 playbook). Fleet-relative daily comparison (orch#794 AC4)
becomes 4-lane from tomorrow.
