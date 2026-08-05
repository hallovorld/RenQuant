"""3-layer regression guard for the weekly WF promote wrapper.

Source incident: 2026-06-02 weekly WF promote wrapper rot.

For ~6 days (2026-05-27 → 2026-06-02) every weekly retrain was silently
rejected because the `scripts/weekly_wf_promote.sh` Step 3.5 + Step 4
referenced manifest + strategy-config paths whose downstream artifacts
were inconsistent across three layers:

* **Layer 1 (file existence).** The shell variable `WF_MANIFEST` and the
  `--strategy-config` arg pointed at files that did not exist on disk
  (or had been retired by a prior rename). Fixed surface-level in merged
  PR #89.
* **Layer 2 (manifest → per-cut artifacts).** The manifest the wrapper
  pointed at contained 43 entries referencing per-cut artifact files
  inside ``walkforward_172_sentiment/``, but that directory never got
  written by the 5/30 rebuild. The wrapper passed Layer 1 (manifest
  exists) but the WF gate fail-closed every bar because individual
  ``artifact_uri`` paths were missing.
* **Layer 3 (recipe fingerprint drift).** Even after a manifest pointing
  at populated paths was substituted in, the per-cut artifacts under
  those paths had been trained from a different recipe than the
  candidate model the gate was scoring. The panel scorer's
  ``assert_consistent`` then fail-closed EVERY bar
  (``panel_scorer_config_mismatch``) → zero trades → the gate could
  never pass any model.

Codex's closed PR #90 proposed a 1-layer guard (only Layer 1). The
incident proved we need all three: each layer catches a distinct class
of rot. This test makes the rotted state unrepresentable on ``main``.

Cross-refs:
  - ``memory/project_wf_infra_rot_2026-06-02.md`` (incident memo)
  - ``doc/research/2026-06-02-placebo-gate-overstrict-for-long-horizon.md``
  - merged PR #89 ``fix/weekly-wf-promote-stale-config-paths`` (surface fix)
  - closed PR #90 (1-layer guard idea this expands on)

Test design (per CLAUDE.md §7.1):
  - Parses the bash wrapper with regex; does NOT execute it.
  - Reads manifest + per-cut artifacts as JSON, side-effect-free.
  - Each rot class lives in its own ``TestWeeklyWrapperRegressionGuard``
    method, named after the rot it pins.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
WRAPPER = REPO / "scripts" / "weekly_wf_promote.sh"
STRATEGY_ROOT = REPO / "backtesting" / "renquant_104"


# ---------------------------------------------------------------------------
# Wrapper parsing helpers (regex / shlex; never execute the script).
# ---------------------------------------------------------------------------

_SHELL_VAR_ASSIGN = re.compile(
    r'^\s*([A-Z_][A-Z0-9_]*)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))\s*(?:#.*)?$',
    re.MULTILINE,
)


def _read_wrapper() -> str:
    assert WRAPPER.exists(), f"weekly wrapper missing: {WRAPPER}"
    return WRAPPER.read_text()


def _shell_assignments(src: str) -> dict[str, str]:
    """Collect all top-level ``KEY=value`` assignments in the script.

    Quote-stripping covers the three forms used by the wrapper:
    ``KEY="value"``, ``KEY='value'``, ``KEY=value``. Returns the LAST
    assignment per key (matching shell precedence).
    """
    out: dict[str, str] = {}
    for m in _SHELL_VAR_ASSIGN.finditer(src):
        key = m.group(1)
        val = next((g for g in (m.group(2), m.group(3), m.group(4)) if g is not None), "")
        out[key] = val
    return out


def _extract_strategy_config_arg(src: str) -> str:
    """Pull the literal token following ``--strategy-config`` in the
    ``run_wf_gate`` invocation.

    The wrapper writes the call as a multi-line backslash-continued
    sequence ``run_wf_gate \\\n    --artifact "$STAGING_ART" \\\n
    --strategy-config strategy_config.foo.json ...``. We match the next
    non-empty token after the flag, stripping quotes.
    """
    m = re.search(
        r"--strategy-config\s+(?:\"([^\"]+)\"|'([^']+)'|(\S+))",
        src,
    )
    assert m, "wrapper does not invoke run_wf_gate with --strategy-config"
    token = next(g for g in m.groups() if g is not None)
    return token


# ---------------------------------------------------------------------------
# Test class.
# ---------------------------------------------------------------------------


class TestWeeklyWrapperRegressionGuard:
    """Three layered guards against the 2026-06-02 wrapper-rot class.

    Each test pins a SINGLE rot mechanism so a future regression names
    itself when it fires.
    """

    # --- Layer 1: wrapper-referenced files must EXIST on disk -------------

    def test_layer1_wf_manifest_path_exists(self) -> None:
        """Layer 1: WF_MANIFEST assignment must resolve to an existing file.

        2026-06-02 rot mode: wrapper carried a stale relative path that
        no longer pointed at any committed manifest after the v2 rebuild
        renamed everything.
        """
        src = _read_wrapper()
        assignments = _shell_assignments(src)
        assert "WF_MANIFEST" in assignments, (
            "wrapper no longer declares WF_MANIFEST — the Step 3.5 stamp call "
            "now points at an unaudited path. Restore an explicit assignment."
        )
        manifest_rel = assignments["WF_MANIFEST"]
        # Wrapper `cd "$REPO_DIR"` (line 131) → working dir is the strategy
        # subtree's parent. Step 3.5 invokes from REPO; --manifest is
        # interpreted relative to REPO.
        manifest_path = (STRATEGY_ROOT / manifest_rel).resolve()
        assert manifest_path.exists(), (
            f"WF_MANIFEST={manifest_rel!r} does not exist at "
            f"{manifest_path} — this is the exact Step 3.5 failure mode "
            "from the 2026-06-02 incident."
        )

    def test_layer1_strategy_config_path_exists(self) -> None:
        """Layer 1: --strategy-config arg must resolve to an existing file.

        2026-06-02 rot mode: wrapper pointed at
        ``strategy_config.sim_wl200_gbdt_prod_recipe_calibrated.json``
        before the file had been committed; the gate aborted with
        FileNotFoundError before scoring a single bar.
        """
        src = _read_wrapper()
        cfg_rel = _extract_strategy_config_arg(src)
        # run_wf_gate runs from REPO. Strategy configs live under the
        # strategy subtree; the gate resolves relative paths there.
        cfg_path = (STRATEGY_ROOT / cfg_rel).resolve()
        assert cfg_path.exists(), (
            f"--strategy-config {cfg_rel!r} does not exist at {cfg_path}. "
            "PR #89 fixed this by committing the file; future renames must "
            "land config + wrapper in the same PR."
        )

    # --- Layer 2: manifest payload must point at populated per-cut files --

    def test_layer2_manifest_retrains_populated_and_artifacts_exist(self) -> None:
        """Layer 2: every retrain entry's per-cut files must exist.

        2026-06-02 rot mode: manifest had 43 entries referencing
        ``walkforward_172_sentiment/`` — directory was never created by
        the 5/30 rebuild. ``run_wf_gate`` fail-closed on FileNotFoundError
        for every cut, blocking promotion silently. A passing Layer 1
        does NOT imply a populated manifest.
        """
        src = _read_wrapper()
        manifest_rel = _shell_assignments(src)["WF_MANIFEST"]
        manifest_path = (STRATEGY_ROOT / manifest_rel).resolve()
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())

        retrains = manifest.get("retrains", [])
        assert isinstance(retrains, list) and len(retrains) > 0, (
            f"manifest {manifest_path} has empty/missing retrains[] — "
            "WF gate has nothing to score. This was the 2026-06-02 "
            "Layer 2 failure mode in a separate guise."
        )

        missing: list[str] = []
        for i, entry in enumerate(retrains):
            artifact_uri = entry.get("artifact_uri")
            assert artifact_uri, f"retrains[{i}] missing artifact_uri"
            artifact_path = (STRATEGY_ROOT / artifact_uri).resolve()
            if not artifact_path.exists():
                missing.append(f"retrains[{i}].artifact_uri -> {artifact_uri}")

            # calibrator_uri is OPTIONAL in the schema. Only enforce when
            # the manifest declares one.
            cal_uri = entry.get("calibrator_uri")
            if cal_uri:
                cal_path = (STRATEGY_ROOT / cal_uri).resolve()
                if not cal_path.exists():
                    missing.append(f"retrains[{i}].calibrator_uri -> {cal_uri}")

        assert not missing, (
            "manifest references per-cut files that do not exist on disk:\n  - "
            + "\n  - ".join(missing[:10])
            + (f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else "")
            + "\nThis is the exact 2026-06-02 Layer 2 failure: a manifest "
            "renamed without re-writing the per-cut artifacts."
        )

    # --- Layer 3: per-cut artifacts must agree on recipe ------------------

    def test_layer3_per_cut_artifacts_share_recipe(self) -> None:
        """Layer 3: all cuts share recipe fingerprint + features + kind.

        2026-06-02 rot mode: per-cut artifacts existed (Layer 2 ok), BUT
        they had been trained from a different recipe than the candidate
        the gate was scoring. The panel scorer's ``assert_consistent``
        fail-closed every bar → zero trades. Internal-consistency check:
        any cross-cut drift here means the gate is scoring an apples /
        oranges WF panel.

        ``recipe_fingerprint`` may be None on all cuts (today's state —
        the WF v2 cuts predate fingerprint stamping); ``all None`` is
        still uniform and accepted. ``config_fingerprint`` is the
        fallback signal and MUST also be uniform.
        """
        src = _read_wrapper()
        manifest_rel = _shell_assignments(src)["WF_MANIFEST"]
        manifest_path = (STRATEGY_ROOT / manifest_rel).resolve()
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        retrains = manifest.get("retrains", [])

        recipe_fps: set[Any] = set()
        config_fps: set[Any] = set()
        kinds: set[Any] = set()
        feature_sigs: set[tuple[str, ...]] = set()

        for i, entry in enumerate(retrains):
            artifact_path = (STRATEGY_ROOT / entry["artifact_uri"]).resolve()
            # Layer 2 already asserts existence; this test reads the file.
            payload: dict[str, Any] = json.loads(artifact_path.read_text())
            recipe_fps.add(payload.get("recipe_fingerprint"))
            config_fps.add(payload.get("config_fingerprint"))
            kinds.add(payload.get("kind"))
            fc = payload.get("feature_cols") or []
            assert isinstance(fc, list), (
                f"retrains[{i}] feature_cols not a list in {artifact_path}"
            )
            feature_sigs.add(tuple(sorted(fc)))

        # recipe_fingerprint: all-None is acceptable (legacy WF v2 cuts);
        # any mix of None + non-None is a stamp regression.
        assert len(recipe_fps) == 1, (
            "per-cut recipe_fingerprint values disagree across cuts: "
            f"{sorted(map(repr, recipe_fps))}. Stamping ran on a subset "
            "of cuts → cross-cut apples/oranges. This is the 2026-06-02 "
            "Layer 3 failure mode (config_mismatch fail-closed)."
        )

        # config_fingerprint is the fallback when recipe_fingerprint is
        # None on all cuts (the current WF v2 state).
        if recipe_fps == {None}:
            assert len(config_fps) == 1 and None not in config_fps, (
                "recipe_fingerprint is None on all cuts AND "
                f"config_fingerprint is not uniform: {sorted(map(repr, config_fps))}. "
                "Need at least one fingerprint axis to be uniform; otherwise "
                "the panel scorer cannot prove cross-cut consistency."
            )

        assert len(kinds) == 1, (
            f"per-cut kinds disagree: {sorted(map(repr, kinds))}. "
            "Mixing model families (e.g. panel_ltr_xgboost + ngboost) "
            "across a WF manifest is incoherent; the gate can't score."
        )

        assert len(feature_sigs) == 1, (
            f"per-cut feature_cols disagree across {len(feature_sigs)} "
            "distinct shapes. A recipe rename must regenerate ALL cuts; "
            "partial regeneration is the rot."
        )

    # --- Layer 3 (cross-artifact): manifest cuts MUST match candidate ----

    def test_layer3_cuts_match_candidate_artifact_recipe(self) -> None:
        """Layer 3 (cross-artifact): manifest cuts share the candidate's recipe.

        Codex review on PR #103 (2026-06-02) caught that the
        intra-manifest check above is necessary but NOT sufficient: a
        manifest whose cuts are uniformly 169-feature can pass that
        check while the weekly candidate / prod artifact is 172-feature,
        and the real gate still fail-closes with
        ``panel_scorer_config_mismatch`` — the exact 2026-06-02 incident.

        Resolution: the candidate is the artifact at
        ``ranking.panel_scoring.artifact_path`` in
        ``strategy_config.shadow.json`` (the GBDT production scoring
        artifact used by ``scripts/weekly_wf_promote.sh``).
        Compare its ``kind`` + ``feature_cols`` + ``recipe_fingerprint``
        / ``config_fingerprint`` against the (already-uniform-per the
        prior test) manifest cuts. Any axis drift here means a future
        weekly retrain into this manifest will scoring-fail-closed.
        """
        src = _read_wrapper()
        manifest_rel = _shell_assignments(src)["WF_MANIFEST"]
        manifest_path = (STRATEGY_ROOT / manifest_rel).resolve()
        manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        retrains = manifest.get("retrains", [])
        assert retrains, "Layer 2 should have caught an empty manifest"

        gbdt_cfg = json.loads(
            (STRATEGY_ROOT / "strategy_config.shadow.json").read_text()
        )
        candidate_rel = (
            gbdt_cfg.get("ranking", {})
            .get("panel_scoring", {})
            .get("artifact_path")
        )
        assert candidate_rel, (
            "strategy_config.shadow.json missing ranking.panel_scoring."
            "artifact_path; cannot resolve weekly wrapper's GBDT scoring "
            "artifact for cross-check"
        )
        candidate_path = (STRATEGY_ROOT / candidate_rel).resolve()
        assert candidate_path.exists(), (
            f"candidate artifact missing on disk: {candidate_path}. "
            "Resolved via strategy_config.shadow.json's "
            "ranking.panel_scoring.artifact_path."
        )

        candidate: dict[str, Any] = json.loads(candidate_path.read_text())
        cand_kind = candidate.get("kind")
        cand_features = tuple(sorted(candidate.get("feature_cols") or []))
        cand_recipe_fp = candidate.get("recipe_fingerprint")
        cand_config_fp = candidate.get("config_fingerprint")

        # Reuse cut[0] as a representative; Layer 3 intra-manifest test
        # already pinned cut-uniformity, so any single cut speaks for all.
        cut_path = (STRATEGY_ROOT / retrains[0]["artifact_uri"]).resolve()
        cut: dict[str, Any] = json.loads(cut_path.read_text())

        # kind: must match (model family agreement)
        assert cut.get("kind") == cand_kind, (
            f"kind mismatch: manifest cuts are {cut.get('kind')!r}, "
            f"candidate scoring artifact at {candidate_rel} is "
            f"{cand_kind!r}. The gate will fail-closed."
        )

        # feature_cols: must match exactly (this is the 169-vs-172 trap)
        cut_features = tuple(sorted(cut.get("feature_cols") or []))
        if cut_features != cand_features:
            cut_only = set(cut_features) - set(cand_features)
            cand_only = set(cand_features) - set(cut_features)
            raise AssertionError(
                f"feature_cols mismatch between manifest cuts and "
                f"candidate at {candidate_rel}:\n"
                f"  manifest cut shape: n={len(cut_features)}\n"
                f"  candidate shape:    n={len(cand_features)}\n"
                f"  cut-only feats:    {sorted(cut_only)[:8]}\n"
                f"  candidate-only feats: {sorted(cand_only)[:8]}\n"
                "This is the exact 2026-06-02 169-vs-172 incident: the "
                "WF gate's recipe-match check would fire and 3/3 cuts "
                "would fail at panel_scorer_config_mismatch."
            )

        # fingerprint: prefer recipe_fingerprint; fall back to
        # config_fingerprint (since WF v2 cuts predate recipe-fp stamping
        # — both candidate + cuts may be None on that axis legitimately).
        if cand_recipe_fp is not None or cut.get("recipe_fingerprint") is not None:
            assert cut.get("recipe_fingerprint") == cand_recipe_fp, (
                "recipe_fingerprint mismatch: cuts="
                f"{cut.get('recipe_fingerprint')!r}, candidate="
                f"{cand_recipe_fp!r}. Stamp the cuts (or re-train) so the "
                "WF gate's recipe-validate step passes against this "
                "candidate."
            )
        else:
            # Both None → the config_fingerprint fallback is NOT a recipe
            # comparison and must not pretend to be one.
            #
            # MEASURED 2026-08-04 (orch#799 follow-up): this fallback had been
            # red on clean main. Diagnosis — `config_fingerprint_fields` on
            # both artifacts is exactly {watchlist, sector_map}: cuts carry a
            # 142-name watchlist, the candidate 145 (CRWV/RKLB/SPCX added
            # later). So the fingerprints differ because the WATCHLIST GREW,
            # not because the recipe drifted. Asserting equality here makes
            # the guard structurally red forever after ANY watchlist addition,
            # while saying nothing about recipe compatibility — and the real
            # gate agrees with that reading: its manifest matching keys on the
            # RECIPE fingerprint (`sha256:cfdd6cb8e950da0f`) and passed 43/43
            # rows on the same artifacts this test was failing.
            #
            # The recipe axes that DO decide fail-closed (kind, feature_cols)
            # are asserted above and stay binding. Here we assert only what
            # the fallback can honestly support: the identity difference is
            # confined to the watchlist/sector-map axis, and any OTHER field
            # differing is a real drift that fails.
            cut_fields = cut.get("config_fingerprint_fields") or {}
            cand_fields = candidate.get("config_fingerprint_fields") or {}
            if cut.get("config_fingerprint") != cand_config_fp:
                differing = sorted(
                    k for k in set(cut_fields) | set(cand_fields)
                    if cut_fields.get(k) != cand_fields.get(k)
                )
                assert differing, (
                    "config_fingerprint differs but no field differs — the "
                    "fingerprint recipe itself changed; regenerate the cuts."
                )
                UNIVERSE_AXES = {"watchlist", "sector_map"}
                unexpected = [k for k in differing if k not in UNIVERSE_AXES]
                assert not unexpected, (
                    "config_fingerprint mismatch on a RECIPE-BEARING field "
                    f"{unexpected}: cuts={cut.get('config_fingerprint')!r}, "
                    f"candidate={cand_config_fp!r}. Universe growth "
                    "(watchlist/sector_map) is expected drift for a corpus "
                    "that predates it; anything else means the cuts were "
                    "trained from a different recipe and the gate will "
                    "fail-closed."
                )
                # Universe-only drift: record the direction so a SHRINKING
                # universe (names the corpus has that production dropped)
                # is still visible — that direction can leave the corpus
                # evaluating tickers the live book no longer trades.
                cut_wl = set(cut_fields.get("watchlist") or [])
                cand_wl = set(cand_fields.get("watchlist") or [])
                assert not (cut_wl - cand_wl), (
                    "the WF corpus carries tickers production has DROPPED "
                    f"{sorted(cut_wl - cand_wl)[:8]}; regenerate the cuts so "
                    "the evaluation universe is a subset of production's."
                )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
