# WF loader: bind calibrator_sha256 + fix the manifest round-trip drop

STATUS: delivered
WHAT: closed the two calibrator-leg gaps left open by the #499 digest-stamp
pass (session task #82). (1) ENFORCEMENT — `RetrainEntry` gains
`calibrator_sha256`; `WalkForwardModelLoader.calibrator_as_of` now resolves
through a new `_resolve_calibrator_uri` that mirrors `_resolve_model_uri`
exactly: the stamped digest is passed to the bounded resolver
(`expected_digest` — a present digest is ALWAYS verified) with
`require_digest=digest_required()` under the SAME
`ARTIFACT_DIGEST_REQUIRED_AFTER=2026-09-01` window (missing digest warns
before the cutoff, fails closed on/after it — fail-closed keeps the prior
model pinned, the safe direction). Resolution now happens BEFORE the
calibrator-cache lookup (cache re-keyed to the resolved path), so a warm
cache can never bypass the digest check — matching `model_as_of` ordering.
`resolve_manifest_uri` gains an optional `digest_field` (default
`artifact_sha256`) used ONLY to name the right key in the missing-digest
remedy; enforcement semantics are identical for both legs. (2) ROUND-TRIP —
`_parse_entry` (loader), `_validate_entry` + `_entry_to_dict` (manifest I/O)
carry `calibrator_sha256` through read → write, fixing the silent drop that
would have un-stamped the corpus calibrator digests on the next manifest
rewrite (`_entry_to_dict` previously preserved only `artifact_sha256`).
WHY/DIR: merged #499 review: both corpus manifests are 39/39 stamped for
artifacts AND calibrators, but `calibrator_as_of` resolved with NO digest and
a manifest round-trip dropped the calibrator stamp — the calibrator leg of
the fold identity contract (the 4×-recurring calibrator/scorer binding
incident class) was provenance-only. Binding it makes a tampered/wrong
calibrator file a hard refusal instead of a silent foreign calibration
surface.
EVIDENCE: `tests/test_walkforward_loader.py::TestCalibratorDigestBinding`
(5 new): round-trip preservation across write → load → read → rewrite;
one-byte calibrator flip REFUSED (`ManifestUriResolutionError`, incl. on the
warm-cache path, with an untampered positive control); pre-window missing
digest tolerated with the re-stamp warning; post-window (date-injected via
the real `digest_required(now=...)`) missing digest fails closed naming
`calibrator_sha256`; the committed 39/39-stamped v2 corpus manifest parses
with both digests populated and every calibrator resolves + digest-verifies
under the closed window. Targeted suites green: loader/resolver/stamp files
59 passed (was 54 baseline); consumer set (sim_artifacts, wf_replay,
placebo_replay, fingerprint_dispatch, regen_oos, stamp_walkforward,
kernel_parity) 80 passed; strategy-104-snapshot-fresh CI job set 148 passed
+ renderer selftest OK; kernel-parity STRICT pass against the pinned
pipeline d32f7017 (loader/manifest are allowlisted drift;
`manifest_uri_resolver.py` is umbrella-only — verified absent at the pin).
`test_stamp_wf_manifest_digests.py` updated: v2 enforcement test now
requires BOTH digest fields, exactly as the loader wires.
NEXT: none for the binding. Pre-existing (NOT introduced or widened here,
verified on main): bare `pytest tests/` over the whole umbrella suite is
nondeterministically polluted (~258-263 failures on main incl. this loader
test file wholesale flapping — sys.modules poisoning of
`renquant_pipeline`-dependent WF test files); CI's curated subsets and
targeted runs are the stable signal.
