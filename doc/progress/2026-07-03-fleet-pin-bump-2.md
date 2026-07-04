# Fleet pin bump 2 — materializes the sprint-D1 merge wave (all flags OFF)

STATUS: chore (lock-only). Second bump of 2026-07-03, following #438; authored by the leader
loop under the sprint landing grant.

## Old → new

| Subrepo | Old | New | Delta highlights |
|---|---|---|---|
| renquant-common | 19cba70f | 74b728c4 | M6 stage-1 0.9.2 OPERATIONAL_KEYS (#22) |
| renquant-pipeline | df7bc073 | 1dafc3c7 | fingerprint dispatch (#164, flag default-true=today's behavior), software-stops kernel mirror (#165, flag OFF), S-FRAC stage-2 sizing (#166, flag OFF) |
| renquant-execution | bad04155 | 8fd788cf | Alpaca BrokerPort (#21, constructed only post-arming), S-FRAC stage-1 fractional orders (#22, consumed only under flags) |
| renquant-base-data | bb69f5e2 | d98e0d2a | C1/PIT revision-drift feature pipeline (#32, additive research lake) |
| renquant-orchestrator | 4b8af94e | 7f852224 | expkit (#287), M4-b harness (#288, run-gated), entry-timing policy (#289, shadow-only), C1 scheduling (#290), Stage-2 live executor (#291, quadruple-gated dark), attribution engine (#292), M6 census (#286) |

## Safety

Same argument class as #438: every behavior-bearing addition is flag-OFF, shadow-only,
run-gated, or authorization-gated; the fingerprint dispatch default reproduces today's
behavior byte-identically (regression-pinned in #164); no strategy-104/model/backtesting
changes in this wave. Monday's runs see identical trading behavior; the new surfaces are
observe/dark until their pre-registered enablements.
