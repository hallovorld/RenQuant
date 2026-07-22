# Pre-deploy gate: configured artifact paths must resolve + identify   (PR #525)

STATUS:    delivered
WHAT:      `scripts/check_config_artifact_paths.py` — a static, read-only
           pre-deploy CI gate. Resolution + content identity are a SINGLE
           injected dependency (`ArtifactContract`); `default_contract()`
           calls the canonical renquant-pipeline #211 contract
           (`shadow_health.resolve_artifact_identity`, which delegates to
           `kernel.artifact_resolver` — the one runtime authority
           `absolute -> strategy_dir -> repo_root`) and HARD-FAILS
           (`CanonicalContractUnavailable`) if it is unavailable — no silent
           in-repo fallback, no second drifting resolver. A declared registry
           (`scripts/config_artifact_gate_registry.json`) enumerates every
           profile the scheduled live/shadow/A-B paths can load
           (`strategy_config.json`, `.shadow.json`, `.shadow_a.json`,
           `.shadow_b.json`, and the production artifact manifest), each
           with a `shape` (`strategy_config` | `artifact_manifest`) and
           `required` flag. FAIL on `resolved==false` OR missing required
           identity (scorers: `trained_date` + `config_fingerprint`;
           calibrators: `trained_date`; `content_sha256` is the
           swap-detection anchor, pinned-digest mismatch FAILS). A `../../`
           relative-path segment is a hard error even when the resolver
           happens to resolve it (topology-dependent otherwise — the exact
           incident class this gate exists to catch). CI wiring
           (`.github/workflows/config-artifact-path-gate.yml`): `unit`
           always runs the fixture suite; `verify-pinned-paths` (only on
           `subrepos.lock.json` changes) checks out `renquant-strategy-104`
           + `renquant-pipeline` at their lock pins and runs the gate
           registry-wide against the real deploy topology
           (`--strategy-dir backtesting/renquant_104 --data-root .`).
WHY/DIR:   Closes a silent-failure class: a strategy-104 config carried a
           PatchTST checkpoint `artifact_path` of
           `../../artifacts/patchtst_shadow/.../hf_patchtst_all_seed44_model.pt`
           that, after the pinned-subrepo migration, resolved OUTSIDE the
           repo tree (`RenQuant/../../artifacts` = `/Users/renhao/git/artifacts`).
           The file was never found and the scorer silently failed to load
           for a long time — nothing statically verified that configured
           artifact paths point at real, identified artifacts before deploy.
EVIDENCE:  n/a (CI gate + unit tests only, no model/data performance claim).
           `python3 -m pytest -q tests/test_check_config_artifact_paths.py`
           -> 20 passed, 1 skipped bare (the real-#211 test skips without
           the pipeline); 21 passed with the pinned pipeline importable
           (`PYTHONPATH=<renquant-pipeline>/src`). Fixtures cover: broken
           `../../` FAILS / all-resolving+identified PASSES; fail-closed on
           metadata-less scorer/calibrator; pinned `content_sha256`
           mismatch FAILS / match PASSES; `../../` lint fails even when the
           resolver resolves it; registry-driven multi-profile/multi-shape
           with optional-skip and required-absent failure. Registry run
           through the REAL #211 resolver against the REAL pinned profiles
           (real umbrella topology): 5 profiles / 19 paths; all
           primary/calibrator paths resolve AND identify; the `../../`
           PatchTST checkpoint FAILS in 5 positions (live shadow, the
           PRIMARY of shadow/shadow_a/shadow_b, and the manifest
           `readonly_shadow`) — exit 1. [VERIFIED]
NEXT:      The gate flags a REAL defect still open: the strategy-104
           profiles still use the `../../` escape for the PatchTST
           checkpoint. On the next pin bump, `verify-pinned-paths` fails
           closed until those paths are re-authored to a repo-relative or
           absolute form that resolves — fixing the configs lives in
           renquant-strategy-104, out of scope for this umbrella CI change.
           After #525 merges, umbrella #524 must rebase and demonstrate it
           runs against the proposed pin (tracked separately).
