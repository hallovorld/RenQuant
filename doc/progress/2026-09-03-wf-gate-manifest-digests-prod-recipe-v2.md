# WF gate dark since 2026-09-01: stamp digests on the manifest the gate actually resolves   (PR #639)

STATUS:    delivered — the weekly promote gate's WF simulation has crashed on
           every retrain since 2026-09-01; this PR restores it and pins the
           naming so the gap cannot silently re-open.
WHAT:      (1) `backtesting/renquant_104/artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.calibrated.json`
           gains `artifact_sha256` + `calibrator_sha256` on all 43 entries
           (86 digests; added lines only, digest key directly after its
           `*_uri` key) via the existing `scripts/stamp_wf_manifest_digests.py`.
           (2) The 43 corpus scorers (`artifacts/walkforward_gbdt_prod_recipe_v2/*/panel-ltr.json`)
           and 43 calibrators (`artifacts/sim/walkforward_calibrators/*/panel-rank-calibration.json`)
           are committed in the bytes the gate's own Step 3.5
           (`scripts/stamp_walkforward_fingerprints.py`) produces — the
           fingerprint-stamped, compact-JSON state the live tree has carried
           since the first promote run after the 06-16 corpus commit. The
           digests are computed over exactly these bytes, so the committed
           corpus, the manifest, and what the gate resolves are one object.
           (3) `tests/test_wf_gate_manifest_digests_prod_recipe_v2.py` (6
           tests): the promote script's `WF_MANIFEST=` must name a stamped
           manifest; all 86 digests present and well-formed; the exact
           post-window loader call (`require_digest=digest_required(now=window+1)`)
           resolves all 86; the missing-digest failure reproduced
           deterministically with the "compatibility window closed" text;
           the committed corpus IS the Step-3.5 state (every scorer carries
           `config_fingerprint`; every calibrator's `scorer_artifact_sha256`
           equals the sha256 of its committed scorer and the manifest's
           `artifact_sha256`); added-lines-only shape.
           (4) `.github/workflows/wf-manifest-digest-contract.yml`: runs both
           digest test files + the stamper's `--check` on the two live-format
           manifests. `tests/test_stamp_wf_manifest_digests.py` (07-18) was
           named by no workflow.
WHY/DIR:   `kernel/manifest_uri_resolver.py` closes its digest compatibility
           window at `ARTIFACT_DIGEST_REQUIRED_AFTER = 2026-09-01`; past it
           `WalkForwardModelLoader._resolve_model_uri` passes
           `require_digest=True` and an unstamped entry fails closed. The
           07-18 stamping pass (`doc/progress/2026-07-18-wf-manifest-digest-stamp.md`)
           stamped `walkforward_manifest.json` and
           `walkforward_manifest_v2_20260602.json` — but the weekly promote
           gate had switched on 2026-06-16 to
           `walkforward_manifest_gbdt_prod_recipe_v2.calibrated.json`
           (`scripts/weekly_wf_promote.sh` line 497), which was 0/43 stamped.
           The guard validated the wrong object. From 2026-09-01 every
           retrain's 3 WF cuts died in `sim.make_context → model_as_of →
           _resolve_model_uri` with `ManifestUriResolutionError`, `run_wf_gate`
           summarised it as "3/3 sim cuts failed execution", and
           `weekly_wf_promote.sh` classified that as an ordinary reject:
           "Reject disposition: prod FRESH (trained 2026-08-31, 3d <= 28d SLA)
           — governance nominal, calm notify, exit 0". Three days of daily
           candidates (09-01, 09-02, 09-03) were discarded by an
           infrastructure crash reported as a calm governance outcome. This is
           G-C's refresh path: no candidate can reach an honest verdict until
           the simulation runs.
           Why commit the Step-3.5 bytes rather than the pretty HEAD bytes:
           the gate rewrites any referenced scorer whose `config_fingerprint`
           differs from the pinned served config's and any calibrator whose
           scorer binding differs, in compact JSON. Digests over the HEAD
           bytes would mismatch on the first run after any checkout (the
           08-31 07:17 live-tree pull reset all 86 files to HEAD; the 07:18
           retrain's Step 3.5 re-stamped them at 07:21:41 — mtimes). Digests
           over the stamped bytes, with those bytes committed, are stable:
           Step 3.5 is a no-op on them.
EVIDENCE:  artifact:      `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.weekly_{20260901T201006Z,20260902T201008Z,20260903T201006Z}.staging.json` → `metadata.wf_gate_metadata.wf_reason = "3/3 sim cuts failed execution"`, `cuts[*].returncode = 1`, `cuts[*].error_tail` ending in `ManifestUriResolutionError: manifest URI 'artifacts/walkforward_gbdt_prod_recipe_v2/2023-10-02/panel-ltr.json' … requires a stamped artifact_sha256 digest (compatibility window closed 2026-09-01)`; `logs/weekly_wf_promote/2026-09-0{1,2,3}.log` carry the same traceback and "Reject disposition: prod FRESH … calm notify, exit 0" [VERIFIED — read 2026-09-03 18:14–18:20 PDT]
           prod or exp:   prod — the weekly/anomaly promote gate's WF simulation (runs daily at 13:10 PT via conditional_retrain_104); no served artifact, config, or pin changes
           existing data: live tree read-only: manifest 43 entries, 0/43 `artifact_sha256`, 0/43 `calibrator_sha256`, `json.dumps(indent=2)` shape; 43/43 scorers and 43/43 calibrators differ from HEAD (compact rewrite + fingerprint/binding, e.g. calibrator 2023-10-02: 1 insertion / 473 deletions), mtimes 2026-08-31 07:21:41 [VERIFIED — `git status --porcelain`, `git diff --stat`, `stat`]; stamper on a scratch copy of the live bytes: "stamped 43 entries (86 file digests)"; `--check` 0 problems against the scratch roots AND against the live manifest path (`--resolve-as`, read-only); diff vs the live manifest = 86 added lines, 0 removed [VERIFIED — 2026-09-03 18:18–18:21 PDT]; the 87 files in this PR are byte-identical to the live tree (86) and the stamped scratch manifest (1): `cmp` on all 87 [VERIFIED — 18:21 PDT]; tests in the scratch tree with the full corpus: `test_wf_gate_manifest_digests_prod_recipe_v2.py` 6 passed + `test_stamp_wf_manifest_digests.py` 9 passed = 15 passed [VERIFIED — 2026-09-03 18:22 PDT]
           best-known?:   n/a — provenance repair; no model claim. The candidates themselves remain weak (genuine_ic 0.0011 / 0.0011 / 0.0002 for 09-01/02/03) — this PR lets the gate SAY so instead of crashing
           scope:         "this PR changes one manifest, the committed bytes of the 86 corpus files it references, one test file and one workflow; it does not change the resolver, the loader, the window date, the promote script, or any served artifact"
NEXT:      (a) live tree: read-only checks → `git pull --ff-only` (86 tracked
           files currently modified live will match HEAD byte-for-byte after
           the merge — verify with `git status --porcelain` before AND after;
           if any of them changed live between now and landing, STOP and
           re-stamp, do not overwrite) → the next 13:10 PT retrain's WF gate
           runs its 3 cuts (expect `WF result:` with real Sharpe/APY, not
           "failed execution"). (b) Ops-truth follow-up (separate PR):
           `weekly_wf_promote.sh` must classify `cuts[*].returncode != 0`
           ("N/3 sim cuts failed execution") as an ALARM disposition, not
           "governance nominal" — the gate crashing is not the gate
           declining. (c) Pre-existing, not widened: calibrator bindings
           embed the live tree's absolute `scorer_artifact` path (same
           machine-specificity as the v1 manifest); a served-config change
           that moves a model-relevant field will re-stamp the corpus at
           Step 3.5 and mismatch these digests — that is fail-closed by
           design and needs a re-stamp PR (`stamp_wf_manifest_digests.py`
           then this test file) in the same batch as such a config change.
           (d) Size: 33 MB of corpus blobs re-committed (compact rewrite of
           files already tracked at similar size).
