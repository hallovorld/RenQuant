# G1 materiality rationale: MEE = 3 bps/session, PE = 6 bps/session

Date: 2026-07-18
Status: pre-registration artifact required by the v5 prereg (§4.1
"written economic rationale ... recorded before the calibration pilot");
frozen at the pilot-registration commit. Drafted personally.

## MEE = 3 bps/session (minimum economically material effect)

3 bps/session ≈ 7.5% annualized (250 sessions, arithmetic) on the
equity book. The honest dollar arithmetic at TODAY'S scale: on the
~$10.7k book this is ~$3.2/session (~$800/yr) — small in dollars. MEE
is nonetheless set at 3 bps because the deployment's cost is not
dollar-denominated but operational and epistemic:

1. **Regime-switch cost.** Replacing conviction sizing with equal
   weights changes the live sizing path, its monitoring baselines
   (turnover, concentration, per-name exposure), and every downstream
   diagnostic calibrated on the current regime. Re-baselining is
   person-hours plus a transition window of degraded anomaly
   detection — the incident class GOAL-5 exists to prevent. Below
   ~7.5%/yr of expected improvement this risk is not worth carrying.
2. **Optionality cost.** The conviction path embeds the model's
   ranking information in sizing; discarding it is only justified when
   the measured penalty of doing so is decisively negative (the
   D6 +9.3% finding is the hypothesis; this experiment is its test).
   A sub-3bps edge would leave the decision inside estimation noise of
   the D6 magnitude and reverse on the next re-estimation.
3. **Scale-forward value.** The methodology is intended to survive
   book growth; at 10× book the same 3 bps is ~$8k/yr against
   unchanged operational cost. Setting MEE lower than 3 bps would
   authorize deployments whose value never clears the fixed
   operational cost at any plausible near-term scale.

Not derived from the paired data (frozen before the blinded pilot).

## PE = 6 bps/session (planning effect for power)

Strictly above MEE by design: the study is sized to detect an effect
WORTH ACTING ON, not the bare materiality line (sizing power at the
pass line was the v3 structural error — power there is ≈ α by
coverage). 6 bps ≈ 15% annualized is the D6-magnitude effect
(equal-weight arm's +9.3% annualized over governor arms, adjusted down
for the exposure-matched construction and cost symmetry of the v5
arms): if the D6 signal is real at even ~2/3 strength, the experiment
should find it with power ≥ 0.80; if the true effect is materially
below PE, the experiment may correctly fail to confirm — and a
LB > MEE outcome remains decisive on its own terms.

## Consistency checks

- MEE < PE (3 < 6), both frozen before pilot; neither may be revised
  except by a NEW amendment with fresh rationale (v5 §4.6 step 4).
- The deployment rule (LB90 > MEE) and the power target (≥ 0.80 at PE)
  reference exactly these constants; the verdict schema carries both.
- If the pilot-calibrated noise makes power ≥ 0.80 at PE infeasible
  within T ≤ 480, the recorded outcome is INFEASIBLE AT CONSERVATIVE
  PILOT NOISE — MEE/PE are not walked to force feasibility.
