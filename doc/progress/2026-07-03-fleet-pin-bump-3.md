# Fleet pin bump 3 — sprint tail (st104 M6 flag config #42, orchestrator risk-budget #294)

STATUS: chore (lock-only), third and final bump of 2026-07-03. base-data/pipeline pins were
already advanced on main by the co-maintaining loops (rawlabel #33, provenance #34, stops
mirror — thank you); this bump adds the last two. Safety class identical to #438/#441:
the strategy-104 delta is an explicit-equals-default fingerprint key (merge-inert, pinned
by test); the orchestrator delta is the observe-only risk-budget ledger + sprint docs.
After this: the machine at pin-align runs the complete sprint output.
