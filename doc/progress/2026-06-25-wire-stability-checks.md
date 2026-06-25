# Wire the stability tools into the live flow

STATUS:   PR. No trading-logic change; both wirings are DEFENSIVE (skip if the referenced
          script isn't merged yet) and NON-FATAL on the daily path.
WHAT:     (A) promote_pin.py — when no explicit --verify-cmd, DEFAULT to
          check_conviction_admits.py (#405): every pin promote now auto-verifies the system
          still buys and AUTO-REVERTS on the sell-only footgun. (B) daily_104.sh — a NON-FATAL
          system_doctor (#404) health heartbeat after the config-drift guard: ntfy on RED
          (pin/runtime drift, lock integrity, bundle consistency, backup hygiene) so drift is
          caught the day it happens, without ever halting the book.
WHY-DIR:  turns the two manual stability tools into continuous, automatic protection — the
          operator's "ensure all components stay stable". Non-redundant with the existing
          preflight_pin_align (which fail-closes a broken runtime) — this adds the broader
          report + the deploy-time still-buys guard.
SAFETY:   both skip cleanly if system_doctor.py / check_conviction_admits.py are absent (until
          #404/#405 merge); the daily check is non-fatal (alert-only); promote's auto-revert
          only triggers on a real would-not-buy verdict.
EVIDENCE: bash -n daily_104 OK; py_compile promote_pin OK; 6 promote_pin tests pass (defensive
          default-verify leaves them green). `[VERIFIED — bash -n + pytest]`
NEXT:     once #404/#405 merge, both are fully active with zero further change.
