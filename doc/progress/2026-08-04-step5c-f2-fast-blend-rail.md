# 2026-08-04 — Step 5c: the F2 fast-blend daily rail (GOAL-9 AC2)

Clones Step 5b (the maintained pattern) for the F2 lane —
zblend(reversal + FAST momentum), orch#794:

- profile gate: `strategy_config.shadow_blend_momentum_fast.json` (landed
  s104#89 with the bounded pending-first-artifact marker) — absent profile =
  loud INFO skip, the lands-before-the-rail order both prior lanes used.
- lane isolation: tag `alpaca_shadow_blend_mom_fast` (registered AT BIRTH,
  pipeline#265), own log (`*_shadow_blend_mom_fast.log`), own
  timeout env (`RENQUANT_SHADOW_BLEND_MOM_FAST_TIMEOUT_SEC`), distinct ntfy
  titles (`SHADOW-BLEND-MOM-FAST-*`).
- dormancy semantics: until the first fast artifact publishes (Saturday
  2026-08-08 genesis), the blend loader fail-closes on the absent
  component[1] — the DESIGNED daily record, non-fatal in this wrapper (the
  preflight-block detection and failure ntfy branches are inherited from
  Step 5b unchanged).

The clone is a systematic rename of the Step 5b block (profile/log/tag/
timeout/rc/pattern variable names + operator-visible strings); `bash -n`
clean; `tests/test_daily_104_shadow_notify.py` 14 passed.
