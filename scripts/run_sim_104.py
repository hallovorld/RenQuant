#!/usr/bin/env python
"""Run a 27-month OOS sim for renquant_104 with a named strategy config.

Usage::

    python scripts/run_sim_104.py
    python scripts/run_sim_104.py --strategy-config-name strategy_config.shadow.json
    python scripts/run_sim_104.py --start 2024-01-01 --end 2026-03-28

    # Experiment mode — requires a REGISTERED experiment manifest (must live
    # under experiments/manifests/ AND have a matching digest entry in
    # experiments/manifests/INDEX.json — see that file for how to register
    # a new experiment):
    python scripts/run_sim_104.py \\
        --experiment-manifest experiments/manifests/sweep-2026-07-14.json

Outputs APY, Sharpe, MaxDD, n_trades, and compares to the golden config.

Experiment-mode governance (F-7, RenQuant#471) is implemented against the
canonical, shared contract in ``renquant_artifacts.experiment_registry`` --
this script is a CALLER of that contract, not a second implementation of it.
Any other producer of non-production sim/backtest/score-backfill output
(e.g. renquant-model's score-backfill tooling) should reuse the same
``renquant_artifacts`` functions rather than hand-rolling an equivalent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

from renquant_artifacts import (
    EXPERIMENT_MANIFEST_REQUIRED_KEYS,
    EXPERIMENT_MANIFEST_VALID_STATUSES,
    EXPERIMENT_PINS_REQUIRED_KEYS,
    build_experiment_provenance_reference,
    reject_exploratory_promotion,
    verify_experiment_pins,
    verify_manifest_registered,
    write_experiment_classification,
)
from renquant_common.model_fingerprint import artifact_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("run-sim-104")

STRATEGY   = "renquant_104"
SIM_START  = "2024-01-02"
SIM_END    = "2026-03-28"   # ~27 months

# ``reject_exploratory_promotion`` is re-exported (not redefined) from
# renquant_artifacts.experiment_registry so ``from run_sim_104 import
# reject_exploratory_promotion`` keeps working for existing callers/tests,
# even though THIS script never calls it directly -- the real caller is
# renquant_artifacts.validation.ValidateArtifactManifestTask, the actual
# promotion boundary (see doc/progress/2026-07-14-f7-r7-promotion-gate.md).
# There is exactly one implementation; this name only re-exports it.
reject_exploratory_promotion = reject_exploratory_promotion  # noqa: PLW0127 (re-export)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(repo), *args), text=True, stderr=subprocess.DEVNULL,
    ).strip()


def _normalize_remote(url: str) -> str:
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.lower()


def _verify_pin(
    repo_path: Path, expected_commit: str, expected_remote: str,
) -> list[str]:
    """Check HEAD, dirty state, and remote URL against a lock-file pin.

    Returns a list of error strings (empty = clean).
    """
    errors: list[str] = []

    if not expected_commit:
        errors.append("lock entry has no commit hash")
    if not expected_remote:
        errors.append("lock entry has no remote URL")
    if errors:
        return errors

    try:
        head = _git(repo_path, "log", "-1", "--format=%H")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        errors.append(f"git metadata failed: {exc}")
        return errors

    if not head.startswith(expected_commit):
        errors.append(
            f"HEAD {head[:12]} does not match lock commit "
            f"{expected_commit[:12]}"
        )

    try:
        dirty = bool(_git(repo_path, "status", "--porcelain"))
    except subprocess.CalledProcessError as exc:
        errors.append(f"git dirty check failed: {exc}")
        return errors

    if dirty:
        errors.append("working tree is dirty")

    try:
        actual_remote = _git(repo_path, "remote", "get-url", "origin")
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        errors.append(f"could not read remote URL: {exc}")
        return errors

    if _normalize_remote(actual_remote) != _normalize_remote(expected_remote):
        errors.append(
            f"remote URL mismatch: lock={expected_remote} "
            f"vs local={actual_remote}"
        )

    return errors


EXPERIMENT_MANIFESTS_DIR = "experiments/manifests"
#: Immutable registry index: a small git-tracked JSON file mapping
#: experiment_id -> {"digest": <sha256 of the manifest file>, "path": ...}.
#: Registering a manifest is a deliberate, auditable act (append an entry
#: here in the SAME commit/PR that adds the manifest file) -- living under
#: EXPERIMENT_MANIFESTS_DIR is necessary but not sufficient; the manifest's
#: content must also match a registered digest (Codex review 2026-07-14,
#: finding 3). See experiments/manifests/INDEX.json and its README.
EXPERIMENT_MANIFEST_INDEX_PATH = "experiments/manifests/INDEX.json"


def load_experiment_manifest(
    manifest_path: Path, *, repo_root: Path,
) -> dict:
    """Load and validate an experiment manifest JSON file.

    Required fields: experiment_id, config_path, config_digest, status,
    pins, data_manifest_path, model_artifact_path (see
    ``renquant_artifacts.EXPERIMENT_MANIFEST_REQUIRED_KEYS``).

    Beyond schema validation, this function requires the manifest to be a
    REGISTERED record: its own file digest must match an entry in
    ``experiments/manifests/INDEX.json`` keyed by ``experiment_id`` (see
    ``renquant_artifacts.verify_manifest_registered``). This is what makes
    "registered experiment" mean more than "someone pointed --experiment-
    manifest at a JSON file under the right directory."

    The returned dict carries four additional internal keys used by the
    caller (``main()``) so paths/digests are resolved exactly once:
    ``_manifest_digest``, ``_data_manifest_path``, ``_model_artifact_path``,
    ``_registry_index_path``.
    """
    if not manifest_path.exists():
        log.error("Experiment manifest not found: %s", manifest_path)
        sys.exit(1)

    raw = json.loads(manifest_path.read_text())
    missing = EXPERIMENT_MANIFEST_REQUIRED_KEYS - raw.keys()
    if missing:
        log.error("Experiment manifest %s missing required keys: %s",
                  manifest_path, sorted(missing))
        sys.exit(1)

    if raw["status"] not in EXPERIMENT_MANIFEST_VALID_STATUSES:
        log.error("Experiment manifest %s has invalid status %r "
                  "(expected one of %s)",
                  manifest_path, raw["status"],
                  sorted(EXPERIMENT_MANIFEST_VALID_STATUSES))
        sys.exit(1)

    if raw["status"] == "RETIRED":
        log.error("Experiment manifest %s has status RETIRED — "
                  "cannot run a retired experiment", manifest_path)
        sys.exit(1)

    pins = raw.get("pins")
    if not isinstance(pins, dict):
        log.error("Experiment manifest %s: 'pins' must be a dict, got %s",
                  manifest_path, type(pins).__name__)
        sys.exit(1)
    missing_pins = EXPERIMENT_PINS_REQUIRED_KEYS - pins.keys()
    if missing_pins:
        log.error("Experiment manifest %s pins missing required keys: %s",
                  manifest_path, sorted(missing_pins))
        sys.exit(1)

    cfg_path = Path(raw["config_path"])
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    if not cfg_path.exists():
        log.error("Experiment config %s (from manifest %s) not found",
                  cfg_path, manifest_path)
        sys.exit(1)

    actual_digest = "sha256:" + hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    if actual_digest != raw["config_digest"]:
        log.error(
            "Experiment config digest mismatch: manifest says %s "
            "but file is %s — config has been modified since manifest "
            "was registered", raw["config_digest"], actual_digest,
        )
        sys.exit(1)

    # The 5 pin categories require concrete evidence to verify against
    # (data_snapshot -> a data manifest, model_artifact -> an artifact
    # file). A registered experiment must point at real files, not merely
    # declare pin values with nothing behind them (Codex review 2026-07-14,
    # finding 2).
    data_manifest_path = Path(raw["data_manifest_path"])
    if not data_manifest_path.is_absolute():
        data_manifest_path = (repo_root / data_manifest_path).resolve()
    if not data_manifest_path.exists():
        log.error("Experiment data_manifest_path %s (from manifest %s) not found",
                  data_manifest_path, manifest_path)
        sys.exit(1)

    model_artifact_path = Path(raw["model_artifact_path"])
    if not model_artifact_path.is_absolute():
        model_artifact_path = (repo_root / model_artifact_path).resolve()
    if not model_artifact_path.exists():
        log.error("Experiment model_artifact_path %s (from manifest %s) not found",
                  model_artifact_path, manifest_path)
        sys.exit(1)

    manifest_digest = "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    index_path = repo_root / EXPERIMENT_MANIFEST_INDEX_PATH
    registry_errors = verify_manifest_registered(
        manifest_digest, raw["experiment_id"], index_path,
    )
    if registry_errors:
        for err in registry_errors:
            log.error("MANIFEST NOT REGISTERED: %s", err)
        log.error(
            "Experiment manifest %s is not a registered record. Register it "
            "by adding a matching {\"digest\": ..., \"path\": ...} entry to "
            "%s in the same commit/PR that adds the manifest file — an "
            "unregistered manifest file is not accepted no matter where it "
            "lives.", manifest_path, index_path,
        )
        sys.exit(1)

    raw["_manifest_digest"] = manifest_digest
    raw["_data_manifest_path"] = str(data_manifest_path)
    raw["_model_artifact_path"] = str(model_artifact_path)
    raw["_registry_index_path"] = str(index_path)
    return raw


def verify_and_classify_experiment(
    manifest_data: dict,
    config: dict,
    *,
    repo_root: Path,
    strategy_dir: Path,
    experiment_manifest_arg: str,
    config_digest: str,
) -> Path:
    """Verify all 5 experiment pins, then atomically write the EXPLORATORY_ONLY
    classification marker. Exits(1) if any pin fails to verify.

    Split out of ``main()`` so the F-7 r7 pin-verification + classification
    contract is directly unit-testable without needing the full sim runtime
    (kernel/sim imports) that ``main()`` pulls in.

    Returns the output directory the classification marker was written to
    -- the directory embedded in the canonical ``provenance`` reference (see
    :func:`renquant_artifacts.build_experiment_provenance_reference`) any
    later artifact manifest must carry for
    ``renquant_artifacts.validate_artifact_manifest`` to enforce the
    promotion-boundary rejection. As of the renquant-artifacts#24 follow-up
    fix (Codex 2026-07-14: "the promotion guard is bypassable because
    provenance is optional and self-declared"), a bare directory string is
    no longer sufficient on its own -- the manifest must set the full
    ``provenance`` record this function logs below, built by the SAME
    shared helper the artifacts-side validator trusts, from values this
    function already resolved (the run's own output dir + the registered
    manifest's own index path) rather than accepting them as arbitrary
    caller input.
    """
    from renquant_base_data import validate_data_manifest  # noqa: PLC0415
    data_manifest_raw = json.loads(
        Path(manifest_data["_data_manifest_path"]).read_text(),
    )
    data_manifest_report = validate_data_manifest(data_manifest_raw)

    pin_errors = verify_experiment_pins(
        manifest_data["pins"],
        repo_root=repo_root,
        data_manifest=data_manifest_report,
        model_artifact_path=manifest_data["_model_artifact_path"],
        universe=config.get("watchlist", []),
    )
    if pin_errors:
        for err in pin_errors:
            log.error("PIN VERIFICATION FAILED: %s", err)
        log.error(
            "Experiment manifest %s pins do not verify against the actual "
            "environment — refusing to run. Fix the checkout/data/model/"
            "universe drift, or re-register a manifest whose pins match "
            "reality.", experiment_manifest_arg,
        )
        sys.exit(1)
    log.info("All 5 experiment pins verified against the actual "
             "environment (strategy_config, pipeline_version, "
             "data_snapshot, model_artifact, calendar_universe).")

    # Classification is written BEFORE run_backtest() so a crash mid-run
    # never leaves real output with no EXPLORATORY_ONLY marker next to it
    # (Codex review 2026-07-14, finding 3). Atomic tmp+rename is
    # implemented once in renquant_artifacts.experiment_registry.
    output_dir = strategy_dir / "artifacts" / "experiments" / manifest_data["experiment_id"]
    cls_file = write_experiment_classification(
        output_dir,
        experiment_id=manifest_data["experiment_id"],
        manifest_path=experiment_manifest_arg,
        manifest_digest=manifest_data["_manifest_digest"],
        config_digest=config_digest,
    )
    log.info("Wrote EXPLORATORY_ONLY classification: %s", cls_file)
    provenance_reference = build_experiment_provenance_reference(
        output_dir, manifest_data["_registry_index_path"],
    )
    log.info(
        "Any artifact manifest built from this run's output MUST set "
        "provenance=%s (this exact reference -- see "
        "renquant_artifacts.build_experiment_provenance_reference; do not "
        "hand-build an equivalent) so "
        "renquant_artifacts.validate_artifact_manifest refuses to promote "
        "it (see renquant_artifacts.validation.ValidateArtifactManifestTask). "
        "provenance is now a REQUIRED manifest field -- omitting it, or any "
        "other lineage claim that does not resolve to this SAME registered "
        "record, is rejected, not silently accepted.",
        provenance_reference,
    )
    return output_dir


def write_candidate_artifact_manifest(
    output_dir: Path,
    *,
    manifest_data: dict,
    sim_metrics: dict,
) -> Path:
    """Emit the candidate-artifact manifest for this experiment run's
    output, with ``provenance`` baked in by THIS producer -- not left for a
    separate, disconnected caller to assert later.

    This closes the gap Codex's round-3 follow-up review found in the log
    line above: "run_sim_104.py only logs the reference returned by
    build_experiment_provenance_reference(); it does not emit an artifact
    manifest or bind that reference into the registry publication path."
    Before this function existed, nothing in this script actually wrote a
    manifest at all -- the ``provenance`` reference above was informational
    only, and any real manifest for this run's output had to be hand-built
    by a later, separate caller (exactly the disconnect that let a
    dishonest caller declare ``provenance={"kind": "none"}`` instead of
    reusing this reference).

    Any candidate artifact built from this run's output should read this
    exact file (or reconstruct it byte-for-byte via the same call) rather
    than hand-roll an equivalent manifest -- mirroring the
    ``model_content_sha256`` triple-impl-avoidance idiom
    ``renquant_artifacts.experiment_registry`` already documents.

    Because every registered experiment's classification is
    EXPLORATORY_ONLY by construction, ``renquant_artifacts.
    validate_artifact_manifest`` (and therefore every real promotion/
    admission caller across the multirepo --
    ``renquant_pipeline.inference.ValidateRuntimeInputsTask`` and
    ``renquant_artifacts.registry.{load,resolve}_artifact_manifest``)
    unconditionally rejects THIS manifest for promotion, which is the
    correct outcome for exploratory output -- see
    ``reject_exploratory_promotion``. Writing that rejection-bound manifest
    here, with provenance baked in at the source, is what makes a later
    dishonest ``kind="none"`` substitution a detectable divergence from the
    real record instead of an unfalsifiable, separately hand-built claim.
    """
    provenance = build_experiment_provenance_reference(
        output_dir, manifest_data["_registry_index_path"],
    )
    model_artifact_path = Path(manifest_data["_model_artifact_path"])
    manifest = {
        "artifact_id": f"{manifest_data['experiment_id']}-candidate",
        "model_family": manifest_data.get("model_family", "experiment-sim-output"),
        "strategy": STRATEGY,
        "fingerprint": artifact_sha256(model_artifact_path),
        "uri": f"file://{model_artifact_path}",
        "local_artifact_path": str(model_artifact_path),
        "promotion_status": "diagnostic",
        "metrics": {"accepted": False, **sim_metrics},
        "provenance": provenance,
    }
    out = output_dir / "candidate_artifact_manifest.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.rename(out)
    log.info(
        "Wrote candidate artifact manifest with baked-in provenance=%s -> "
        "%s -- renquant_artifacts.validate_artifact_manifest rejects any "
        "promotion attempt against this manifest (EXPLORATORY_ONLY, "
        "registered experiment) because provenance is set by THIS "
        "producer at the source, not left for a separate caller to assert.",
        provenance, out,
    )
    return out


def _resolve_strategy_config(
    repo_root: Path,
    strategy_dir: Path,
    config_name: str,
    *,
    experiment_manifest: str | None = None,
) -> tuple[Path, str, dict | None]:
    """Resolve strategy config from pinned subrepo or experiment manifest.

    Returns (cfg_path, source, manifest_data) where source is "PINNED" or
    "EXPLORATORY_ONLY", and manifest_data is the parsed manifest dict (or
    None for PINNED mode).

    Default mode (strict-pinned): ALL configs resolve from the pinned
    renquant-strategy-104 subrepo. The checkout HEAD, clean state, and remote
    URL are verified against subrepos.lock.json. No filename-based routing.

    Experiment mode (--experiment-manifest): resolves the config through an
    immutable manifest containing experiment_id, config_digest, pins, and
    status. Outputs are classified EXPLORATORY_ONLY.
    """
    if experiment_manifest:
        manifest_path = Path(experiment_manifest)
        if not manifest_path.is_absolute():
            manifest_path = (repo_root / manifest_path).resolve()
        allowed_root = (repo_root / EXPERIMENT_MANIFESTS_DIR).resolve()
        try:
            manifest_path.resolve().relative_to(allowed_root)
        except ValueError:
            log.error(
                "Experiment manifest %s is outside the registered location "
                "%s/ — manifests must be registered under that directory",
                manifest_path, EXPERIMENT_MANIFESTS_DIR,
            )
            sys.exit(1)
        manifest = load_experiment_manifest(manifest_path, repo_root=repo_root)
        cfg_path = Path(manifest["config_path"])
        if not cfg_path.is_absolute():
            cfg_path = (repo_root / cfg_path).resolve()
        log.warning(
            "EXPLORATORY_ONLY [experiment_id=%s, manifest=%s]: using %s — "
            "results CANNOT be used for promotion or live deployment",
            manifest["experiment_id"], manifest_path, cfg_path,
        )
        return cfg_path, "EXPLORATORY_ONLY", manifest

    lock_path = repo_root / "subrepos.lock.json"
    if not lock_path.exists():
        log.error(
            "subrepos.lock.json not found at %s — cannot resolve pinned "
            "config. Use --experiment-config for research experiments.",
            lock_path,
        )
        sys.exit(1)

    lock = json.loads(lock_path.read_text())
    strat_entry = None
    for entry in lock.get("subrepos", []):
        if entry.get("name") == "renquant-strategy-104":
            strat_entry = entry
            break

    if strat_entry is None:
        log.error("renquant-strategy-104 not found in subrepos.lock.json")
        sys.exit(1)

    raw_local = strat_entry["local_path"]
    local_path = Path(raw_local)
    if not local_path.is_absolute():
        local_path = (repo_root / local_path).resolve()

    expected_commit = strat_entry.get("commit", "")
    expected_remote = strat_entry.get("remote", "")
    pin_errors = _verify_pin(local_path, expected_commit, expected_remote)
    if pin_errors:
        for err in pin_errors:
            log.error("PIN DRIFT [renquant-strategy-104]: %s", err)
        log.error(
            "Strategy subrepo checkout does not match lock pin. "
            "Fix the checkout or use --experiment-config for experiments.",
        )
        sys.exit(1)

    cfg_path = local_path / "configs" / config_name
    if not cfg_path.exists():
        log.error(
            "Config %s not found in pinned subrepo: %s (commit %s). "
            "Use --experiment-config to load from an explicit path.",
            config_name, cfg_path, expected_commit[:12],
        )
        sys.exit(1)

    log.info("Using PINNED config: %s (commit %s, verified)",
             cfg_path, expected_commit[:12])
    return cfg_path, "PINNED", None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy-config-name", default="strategy_config.json",
                   help="Config filename (default: strategy_config.json)")
    p.add_argument("--start", default=SIM_START)
    p.add_argument("--end",   default=SIM_END)
    p.add_argument("--compare-to", default="strategy_config.golden.json",
                   help="Golden config to compare against (default: strategy_config.golden.json)")
    p.add_argument("--initial-cash", type=float, default=100_000)
    # 2026-05-09 audit FIX-G: per-seed isolation for parallel multi-seed runs.
    # Without these, multiple sims clobber data/sim_runs.db (single SQLite
    # writer) → race + TRUNCATE conflicts.
    p.add_argument("--sim-db-path", default=None,
                   help="Override persistence.sim_db_path so parallel "
                        "multi-seed runs use isolated DBs.")
    p.add_argument("--no-persist", action="store_true",
                   help="Disable persistence entirely (fastest; no DB writes).")
    p.add_argument("--equity-json", default=None,
                   help="Write daily equity curve to JSON (for paired-returns analysis)")
    p.add_argument("--trade-log-json", default=None,
                   help="Write raw SimResult.trade_log events to JSON.")
    p.add_argument("--trade-log-csv", default=None,
                   help="Write raw SimResult.trade_log events to CSV.")
    p.add_argument("--round-trips-csv", default=None,
                   help="Write FIFO-matched round trips to CSV.")
    p.add_argument("--trade-report-md", default=None,
                   help="Write a Markdown trade-forensics report.")
    p.add_argument("--no-compare", action="store_true",
                   help="Skip the golden-config comparison run.")
    p.add_argument("--skip-preflight", action="store_true",
                   help="Skip the static-path preflight on side configs. "
                        "ONLY use when intentionally running a no-op (e.g. "
                        "re-baselining against new artifacts).")
    p.add_argument("--allow-raw-qp-mu", action="store_true",
                   help="Emergency/debug override: allow QP configs that do "
                        "not have a strict expected-return μ contract.")
    p.add_argument("--experiment-manifest", default=None,
                   help="Path to an experiment manifest JSON file. The "
                        "manifest must contain experiment_id, config_path, "
                        "config_digest (sha256), and status. All outputs "
                        "are classified EXPLORATORY_ONLY.")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / STRATEGY
    sys.path.insert(0, str(strategy_dir))

    cfg_path, config_source, manifest_data = _resolve_strategy_config(
        REPO_ROOT, strategy_dir, args.strategy_config_name,
        experiment_manifest=args.experiment_manifest,
    )
    cfg_bytes = cfg_path.read_bytes()
    config = json.loads(cfg_bytes)
    # Full-file audit fingerprint, "sha256:"-prefixed to match the convention
    # used elsewhere for content fingerprints (e.g.
    # renquant_common.model_fingerprint.artifact_sha256) rather than a
    # truncated, unprefixed hex digest.
    config_digest = "sha256:" + hashlib.sha256(cfg_bytes).hexdigest()
    log.info("Config fingerprint: %s  source=%s  path=%s",
             config_digest, config_source, cfg_path)

    from qp_contracts import validate_qp_contract_config  # noqa: PLC0415
    qp_contract = validate_qp_contract_config(config)
    if not qp_contract.passed and not args.allow_raw_qp_mu:
        log.error(qp_contract.summary())
        log.error("QP contract evidence: %s", qp_contract.evidence)
        sys.exit(3)
    if qp_contract.qp_enabled:
        log.info("QP contract: %s  evidence=%s",
                 qp_contract.summary(), qp_contract.evidence)
    # Historical sims/WF cuts are not live inference. The live freshness
    # guard correctly requires every symbol to include the latest completed
    # NYSE close, but old windows can contain IPO/new-listing gaps and would
    # be falsely rejected. Live runner keeps the default enabled.
    data_freshness = config.setdefault("data_freshness", {})
    if "enabled" not in data_freshness:
        data_freshness["enabled"] = False
        log.info("data_freshness.enabled=false by default for historical sim")

    # 2026-05-16: gate on static-path preflight for any side config to
    # prevent the recurrence of the 5/15 no-op build script bug (5h of
    # compute on configs whose knobs didn't reach the kernel). See
    # scripts/validate_sim_config_active.py and the 2026-05-16 entry
    # in doc/research/failed-experiments-log.md.
    SIDE_CFG_BASELINE = "strategy_config.sim_baseline_hmm.json"
    is_side = (args.strategy_config_name.startswith("strategy_config.sim_")
               and not args.strategy_config_name.startswith("strategy_config.sim_baseline"))
    if is_side and not args.skip_preflight:
        import subprocess
        validator = REPO_ROOT / "scripts" / "validate_sim_config_active.py"
        if validator.exists():
            log.info("preflight: static-path validator vs %s", SIDE_CFG_BASELINE)
            r = subprocess.run(
                [sys.executable, str(validator),
                 "--baseline", SIDE_CFG_BASELINE,
                 "--candidate", args.strategy_config_name],
                cwd=str(strategy_dir), capture_output=True, text=True,
            )
            if r.returncode != 0:
                log.error("PREFLIGHT FAILED for %s — config writes to a path "
                          "the kernel does not read (NO-OP). Aborting to "
                          "prevent wasted compute. Pass --skip-preflight to "
                          "override.", args.strategy_config_name)
                log.error("validator output:\n%s", r.stdout)
                sys.exit(2)
            log.info("preflight: ACTIVE — knob reaches kernel")
        else:
            log.warning("preflight skipped — validator not found at %s", validator)

    config["_strategy_dir"]         = str(strategy_dir)
    config["_strategy_config_name"] = args.strategy_config_name
    config["_strategy_config_path"] = str(cfg_path)
    config["_strategy_config_source"] = config_source
    config["_strategy_config_digest"] = config_digest
    if config_source == "EXPLORATORY_ONLY" and manifest_data is not None:
        config["_exploratory_only"] = True
        config["_experiment_id"]    = manifest_data["experiment_id"]
        config["_experiment_manifest"] = args.experiment_manifest
    config["initial_cash"]          = args.initial_cash
    config["backtest_start"]        = args.start
    config["backtest_end"]          = args.end

    experiment_output_dir = None
    if config_source == "EXPLORATORY_ONLY" and manifest_data is not None:
        # F-7 r7 (Codex review 2026-07-14, findings 2+3): verify all 5
        # required pin categories against the ACTUAL environment, then
        # atomically write the EXPLORATORY_ONLY classification BEFORE any
        # simulation output is produced. See verify_and_classify_experiment.
        experiment_output_dir = verify_and_classify_experiment(
            manifest_data, config,
            repo_root=REPO_ROOT,
            strategy_dir=strategy_dir,
            experiment_manifest_arg=args.experiment_manifest,
            config_digest=config_digest,
        )

    # Per-seed DB isolation
    if args.no_persist:
        config["persistence"] = {"enabled": False}
    elif args.sim_db_path:
        config.setdefault("persistence", {})["sim_db_path"] = args.sim_db_path

    from kernel.data import fetch_ohlcv  # noqa: PLC0415
    from sim.runner import run_backtest   # noqa: PLC0415

    # Load benchmark + sector ETFs
    log.info("Fetching SPY + sector ETFs …")
    benchmark = config.get("benchmark", "SPY")
    spy_df    = fetch_ohlcv(benchmark)
    etf_map   = config.get("sector_etf_map", {})
    ohlcv: dict = {benchmark: spy_df}
    for sym in sorted(set(config.get("watchlist", [])) | set(etf_map.values())):
        try:
            ohlcv[sym] = fetch_ohlcv(sym)
        except Exception as exc:
            log.warning("  %s: %s", sym, exc)

    log.info("Running sim: %s → %s  config=%s",
             args.start, args.end, args.strategy_config_name)
    result = run_backtest(
        config        = config,
        strategy_dir  = strategy_dir,
        ohlcv         = ohlcv,
        spy_df        = spy_df,
        sector_etf_map = etf_map,
        snapshot      = False,
    )
    result.print_summary()

    if config_source == "EXPLORATORY_ONLY" and manifest_data is not None:
        # F-7 follow-up (Codex review 2026-07-14, round 3): connect this
        # run's output to artifact publication -- see
        # write_candidate_artifact_manifest's docstring for why this call
        # site (not a later, disconnected caller) must be the one to bind
        # provenance into the manifest.
        write_candidate_artifact_manifest(
            experiment_output_dir,
            manifest_data=manifest_data,
            sim_metrics={
                "apy": float(result.apy),
                "sharpe": float(result.sharpe) if result.sharpe == result.sharpe else None,
                "max_dd": float(result.max_dd) if result.max_dd == result.max_dd else None,
                "n_trades": len(result.buys),
            },
        )

    # Emit daily equity curve for paired-returns analysis (industry-standard
    # eval per doc/research/evaluation-protocol.md). Records date + nav so
    # downstream paired t-test + Newey-West HAC + block-bootstrap have the
    # raw daily P&L stream rather than the noisy per-window APY estimate.
    if args.equity_json:
        from pathlib import Path as _P
        eq = result.equity_df.copy()
        eq.index = eq.index.astype(str)
        payload = {
            "config":        args.strategy_config_name,
            "start":         args.start,
            "end":           args.end,
            "initial_cash":  args.initial_cash,
            "final_value":   float(result.final_value),
            "total_return":  float(result.total_return),
            "apy":           float(result.apy),
            "sharpe":        float(result.sharpe) if result.sharpe == result.sharpe else None,
            "event_level_apy": float(result.apy),
            "event_level_sharpe": (
                float(result.sharpe) if result.sharpe == result.sharpe else None
            ),
            "event_level_tax_debited": float(result.event_level_tax_debited),
            "event_level_tax_estimate": float(result.event_level_tax_estimate),
            "tax_cash_debited": float(result.tax_cash_debited),
            "tax_cash_debit_mode": str(result.tax_cash_debit_mode),
            "annual_net_tax_estimate": float(result.annual_net_tax_estimate),
            "tax_overstatement_vs_annual_net": (
                float(result.tax_overstatement_vs_annual_net)
            ),
            "annual_net_final_value": (
                float(result.annual_net_final_value_estimate)
                if result.annual_net_final_value_estimate
                == result.annual_net_final_value_estimate else None
            ),
            "annual_net_total_return": (
                float(result.annual_net_total_return_estimate)
                if result.annual_net_total_return_estimate
                == result.annual_net_total_return_estimate else None
            ),
            "annual_net_apy": (
                float(result.annual_net_apy_estimate)
                if result.annual_net_apy_estimate
                == result.annual_net_apy_estimate else None
            ),
            "annual_net_sharpe": (
                float(result.annual_net_sharpe_estimate)
                if result.annual_net_sharpe_estimate
                == result.annual_net_sharpe_estimate else None
            ),
            "annual_net_ann_vol": (
                float(result.annual_net_ann_vol_estimate)
                if result.annual_net_ann_vol_estimate
                == result.annual_net_ann_vol_estimate else None
            ),
            "annual_net_max_dd": (
                float(result.annual_net_max_dd_estimate)
                if result.annual_net_max_dd_estimate
                == result.annual_net_max_dd_estimate else None
            ),
            "ann_vol":       float(result.ann_vol) if result.ann_vol == result.ann_vol else None,
            "max_dd":        float(result.max_dd) if result.max_dd == result.max_dd else None,
            "equity":        eq["portfolio"].astype(float).to_dict(),
        }
        annual_eq = result.annual_net_equity_df_estimate.copy()
        if (not annual_eq.empty and "portfolio" in annual_eq.columns):
            annual_eq.index = annual_eq.index.astype(str)
            payload["annual_net_equity"] = (
                annual_eq["portfolio"].astype(float).to_dict()
            )
        _P(args.equity_json).parent.mkdir(parents=True, exist_ok=True)
        _P(args.equity_json).write_text(json.dumps(payload, indent=2))
        log.info("Wrote daily equity → %s (%d days)", args.equity_json, len(eq))

    if any([args.trade_log_json, args.trade_log_csv,
            args.round_trips_csv, args.trade_report_md]):
        from sim_trade_ledger import write_trade_outputs  # noqa: PLC0415
        end_prices = {}
        for sym, df in ohlcv.items():
            try:
                hist = df.loc[:args.end]
                if not hist.empty and "close" in hist.columns:
                    end_prices[sym] = float(hist["close"].iloc[-1])
            except Exception:  # noqa: BLE001
                pass
        written = write_trade_outputs(
            result           = result,
            config           = config,
            trade_json       = args.trade_log_json,
            trade_csv        = args.trade_log_csv,
            round_trips_csv  = args.round_trips_csv,
            report_md        = args.trade_report_md,
            end_prices       = end_prices,
            title            = (
                f"renquant_104 sim trade forensics "
                f"({args.strategy_config_name}, {args.start} to {args.end})"
            ),
            extra_metrics    = {
                "config": args.strategy_config_name,
                "start": args.start,
                "end": args.end,
            },
        )
        for kind, path in sorted(written.items()):
            log.info("Wrote %s → %s", kind, path)

    # Compare to golden if available (skip with --no-compare to halve runtime)
    if args.no_compare:
        return
    golden_path = strategy_dir / args.compare_to
    if golden_path.exists() and args.compare_to != args.strategy_config_name:
        log.info("Running golden comparison: %s", args.compare_to)
        golden_cfg = json.loads(golden_path.read_text())
        golden_cfg["_strategy_dir"]  = str(strategy_dir)
        golden_cfg["initial_cash"]   = args.initial_cash
        golden_cfg["backtest_start"] = args.start
        golden_cfg["backtest_end"]   = args.end
        golden = run_backtest(
            config        = golden_cfg,
            strategy_dir  = strategy_dir,
            ohlcv         = ohlcv,
            spy_df        = spy_df,
            sector_etf_map = etf_map,
            snapshot      = False,
        )
        r_apy = result.apy * 100
        g_apy = golden.apy  * 100
        delta = r_apy - g_apy
        print()
        print("=" * 50)
        print(f"  {args.strategy_config_name:<35} APY={r_apy:+.2f}%  WR={result.win_rate:.0%}  trades={len(result.buys)}")
        print(f"  {args.compare_to:<35} APY={g_apy:+.2f}%  WR={golden.win_rate:.0%}  trades={len(golden.buys)}")
        print(f"  Delta vs golden                         APY={delta:+.2f} pp")
        verdict = "PROMOTE ✓" if delta >= 0 else "REJECT ✗"
        print(f"  Verdict: {verdict}")
        print("=" * 50)


if __name__ == "__main__":
    main()
