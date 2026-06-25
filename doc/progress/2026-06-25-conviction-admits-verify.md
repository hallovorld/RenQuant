# Reusable deploy guard — would the conviction gate still BUY?

STATUS:   PR. New read-only deploy guard; no runtime change.
WHAT:     `scripts/check_conviction_admits.py` — replays the LIVE conviction_gate config
          against the latest recorded candidate scores and asserts admits >= --min-admits
          (mirrors ConvictionGateTask incl. the #147 full-cross-section demean). The
          generalized, deterministic, OFFLINE form of the verify that protected the demean
          go-live. Exit 0 / 1 (would-not-buy) / 2 (cannot-eval).
WHY-DIR:  a config/pin change that silently zeroes admissions is the sell-only footgun (the
          demean-over-subset bug, caught 2026-06-24). Wire this as the standard
          `promote_pin.py --verify-cmd` so a deploy that would stop trading AUTO-REVERTS;
          also a fast pre-deploy and `make doctor` check.
EVIDENCE: 4 unit tests (absolute vs demean admit counts, would-not-buy trigger, ok-when-admits,
          empty-db). Run against the LIVE config (demean now ON): `OK: admits=4 / n=79
          (demean=True)` — confirms demean still buys (MU/CRWD/PANW/CSCO), exit 0.
          `[VERIFIED — pytest + live run]`
NEXT:     set it as the default --verify-cmd in promote_pin.py; add to `make doctor`; replaces
          the ad-hoc /tmp/verify_demean.py used for the demean bump.
