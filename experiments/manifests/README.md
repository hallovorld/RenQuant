# Experiment manifest registry

This directory is the ONLY registered location for `scripts/run_sim_104.py
--experiment-manifest <path>` manifests (F-7, RenQuant#471). It exists so
that experiment mode — which deliberately bypasses the strict-pinned
`renquant-strategy-104` subrepo resolution used by every other run — cannot
be pointed at an arbitrary, caller-supplied file and have its output mistaken
for reproducible, promotable evidence.

Two things must both be true for a manifest to be accepted:

1. **Location.** The manifest file must resolve to a path under this
   directory (`experiments/manifests/`). This alone is necessary but **not**
   sufficient — see the incident this closes: Codex's 2026-07-14 review of
   round 5 found "any arbitrary absolute `--experiment-manifest` path is
   accepted."
2. **Registration.** The manifest's own file digest (`sha256:` + hex of its
   raw bytes) must match an entry in [`INDEX.json`](INDEX.json) keyed by the
   manifest's `experiment_id`. Registering an experiment is a deliberate,
   auditable act: add the manifest file here **and** append its
   `{"digest": ..., "path": ...}` entry to `INDEX.json` in the *same*
   commit/PR. A manifest that merely lives in this directory, with no
   matching registry entry (or a digest that no longer matches — e.g. the
   file was edited after being registered), is refused.

Both checks are enforced by
`renquant_artifacts.experiment_registry.verify_manifest_registered`, called
from `scripts/run_sim_104.py::load_experiment_manifest` — the check fails
closed (`sys.exit(1)`) if the manifest is missing from `INDEX.json` or if the
digests disagree.

## Manifest schema

Each manifest is a JSON file with these required top-level keys (see
`renquant_artifacts.EXPERIMENT_MANIFEST_REQUIRED_KEYS`):

| Key | Meaning |
|---|---|
| `experiment_id` | Unique identifier, matches the `INDEX.json` key |
| `config_path` | Path (repo-root-relative or absolute) to the strategy config JSON under test |
| `config_digest` | `sha256:` digest of that config file's bytes at registration time |
| `status` | One of `ACTIVE`, `COMPLETED`, `RETIRED` (`RETIRED` refuses to run) |
| `pins` | Dict of the 5 required pin categories (below) |
| `data_manifest_path` | Path to a data manifest JSON (schema: `renquant_base_data.validate_data_manifest`) whose `fingerprint` the `pins.data_snapshot` value must match |
| `model_artifact_path` | Path to the model artifact file whose content hash (`renquant_common.model_fingerprint.artifact_sha256`) the `pins.model_artifact` value must match |

### `pins` (all 5 required, all verified against the live environment)

| Pin | Verified against |
|---|---|
| `strategy_config` | The `renquant-strategy-104` entry's `commit`/`remote` in `subrepos.lock.json`, cross-checked against the actual checkout (HEAD/dirty/remote) |
| `pipeline_version` | Same, for the `renquant-pipeline` entry |
| `data_snapshot` | `fingerprint` field of the data manifest at `data_manifest_path` |
| `model_artifact` | Full-file hash of `model_artifact_path` |
| `calendar_universe` | Stable hash (`renquant_artifacts.hash_jsonable`) of the sorted, de-duplicated `watchlist` the resolved strategy config declares |

All 5 are verified by `renquant_artifacts.verify_experiment_pins`, called
from `run_sim_104.py::main()` immediately after the config is loaded and
before any simulation output is produced. A category whose supporting
evidence is missing is an ERROR, not a silently-skipped pass — see the
docstring on `verify_experiment_pins` for why "pins is optional and not
verified" (Codex round-5 finding) is exactly the failure mode this closes.

## Why any of this matters: the promotion boundary

Every experiment-mode run writes a durable
`_experiment_classification.json` marker (`classification:
"EXPLORATORY_ONLY"`) into its output directory, atomically, before
`run_backtest()` is called. That marker is not just a log line: if a
candidate artifact manifest later declares
`"provenance_dir": "<that output directory>"`,
`renquant_artifacts.validation.ValidateArtifactManifestTask` — the function
every real promotion/admission caller across the multirepo funnels through
(`renquant_pipeline.inference.ValidateRuntimeInputsTask` for live/shadow/sim
runtime, and `renquant_artifacts.registry.{load,resolve}_artifact_manifest`
for registry resolution) — refuses to validate it. See
`renquant-artifacts/tests/test_experiment_registry.py::TestPromotionBoundaryIntegration`
for an end-to-end proof against the real (non-mocked) entrypoints.
