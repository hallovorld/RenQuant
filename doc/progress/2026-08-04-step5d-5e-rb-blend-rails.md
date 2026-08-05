# 2026-08-04 — Step 5d/5e: the F1/F3 fleet rails (GOAL-9 AC2 complete on the rail side)

Clones the Step-5c pattern for the two 3-component lanes (pipeline#267
N-generalization; profiles s104#90; tags registered at birth pipeline#265):

- Step 5d — F1 `shadow_blend_rb_mom` (z(prod)+z(clf)+z(slow)); dormant until
  the F1 profile reaches the pinned configs.
- Step 5e — F3 `shadow_blend_rb_fast`; additionally stays fail-closed on the
  fast leg until the 2026-08-08 genesis batch (the F2 dormancy semantics).

Guards FIRST this time (the RQ#575 lesson): 12 static guards (6 per lane —
profile gate + INFO ordering, tag/log/config threading, own timeout env,
distinct FAIL/TIMEOUT titles with alert-default pinning, preflight-block
suppression, 5c→5d→5e ordering + non-fatal) were generated alongside the
blocks and caught one real generator slip (double-braced `${DATE}`/env
expansions in the assertions) before any review round. `bash -n` clean;
shadow-notify suite 32 passed.
