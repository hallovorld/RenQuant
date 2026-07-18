# WF corpus manifests: per-entry digest stamps before the 2026-09-01 resolver enforcement

STATUS: delivered
WHAT: stamped `artifact_sha256` into every entry of both committed WF corpus
manifests under `backtesting/renquant_104/artifacts/sim/` —
`walkforward_manifest.json` (39 entries, absolute deployed-tree URIs) and
`walkforward_manifest_v2_20260602.json` (39 entries; also stamped
`calibrator_sha256` per entry as forward-compatible provenance — the
resolver/loader enforce only `artifact_sha256` today, `calibrator_as_of`
resolves calibrators without digest binding). 117 file digests total
(39 v1 artifacts + 39 v2 artifacts + 39 v2 calibrators; no overlap), each
computed as sha256 of the referenced file's raw bytes — byte-for-byte the
resolver's `_digest` — via the REAL bounded resolver
(`kernel.manifest_uri_resolver.resolve_manifest_uri`) for root/containment/
ambiguity parity with enforcement. Stamper committed as
`scripts/stamp_wf_manifest_digests.py` (deterministic; refuses non-
`json.dumps(indent=2)` inputs so the diff is ADDED LINES ONLY — digest key
directly after its `*_uri` key; `--check` verify mode; `--resolve-as` derives
resolver roots from the deployed manifest path so v1's absolute URIs hash the
live files while the committed copy is what gets stamped). Referenced
artifact files were verified byte-identical between the live operator tree
and origin/main HEAD for all 117 files before stamping; the live tree was
only read, never written (manifests deploy via normal checkout sync).
WHY/DIR: PR #497 review finding (metalabel retrain redesign, design doc
§"digest-stamp prerequisite"): `kernel/manifest_uri_resolver.py` closes its
digest compatibility window at `ARTIFACT_DIGEST_REQUIRED_AFTER =
2026-09-01`; on/after it `WalkForwardModelLoader._resolve_model_uri` passes
`require_digest=True` and an unstamped entry fails closed — with 0/39
stamped, the weekly WF promote gate would go dark on the deadline (fail-
closed keeps the prior model pinned, but every future promote would reject).
EVIDENCE: `tests/test_stamp_wf_manifest_digests.py` (9/9, stdlib-only):
enforcement-date semantics via `digest_required(now=...)` injection; both
manifests fully stamped (64-hex); AC-5 — resolver ACCEPTS every stamped
entry with the enforcement date forced one day past 2026-09-01
(`require_digest=True`, mirroring the loader wiring); missing digest under
enforcement fails closed with the `artifact_sha256` re-stamp error; a
one-byte flip in a referenced artifact is REFUSED with a digest mismatch
(`ManifestUriResolutionError`); manifests keep the `json.dumps(indent=2)`
byte shape with digest keys adjacent to their URIs. `--check` passes 0
problems against both the deployed tree roots and an in-repo checkout;
existing `tests/test_manifest_uri_resolver.py` stays green (22 passed,
1 pre-existing skip).
NEXT: none for the stamp itself. Open (pre-existing) gaps, not widened here:
calibrator resolution is digest-unbound in the loader (stamps are now in
place if binding is added); v1's absolute machine URIs remain
machine-specific (untouched — URI rewriting is out of scope for a stamping
pass).
