"""Property/invariant tests for sim artifact-metadata helpers.

Eng plan S2 item 6 (test-ladder rebalance). adapters/sim_artifacts.py (sim.py
decomposition slice 2) had no direct test. Its centerpiece,
_drop_inference_forbidden_cols, is a LEAKAGE GUARD: it strips label /
split_label / fwd_* columns out of a history frame before it reaches an
inference feature matrix. A column that slips through is future information in
the features — lookahead bias, an invalid backtest. So the guard gets
invariants, not just an example.

No `hypothesis` dependency (hermetic requirements.lock.txt lacks it): column
sets are swept over a deterministic seeded grid.

Invariants pinned:
- the output NEVER contains a forbidden column (label, split_label, any fwd_*)
  — the absolute leakage property.
- every non-forbidden column survives with its data intact.
- the input frame is not mutated; the op is idempotent.
- _resolve_manifest_uri: absolute uri kept when its realpath is under an
  allowed root, rejected otherwise (PR #421 round 3 — bounded resolver);
  relative resolved under the manifest's parent.
- _artifact_kind: metadata.kind preferred over top-level kind; None on
  unreadable / non-dict / missing.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.sim_artifacts import (  # noqa: E402
    _artifact_kind,
    _drop_inference_forbidden_cols,
    _history_seq_len_from_artifact,
    _resolve_manifest_uri,
)
from kernel.manifest_uri_resolver import ManifestUriResolutionError  # noqa: E402

SEED = 0x5117
N = 2000


def _is_forbidden(col) -> bool:
    return col in ("label", "split_label") or str(col).startswith("fwd_")


def _rand_columns(rng):
    pool = [
        "feat1", "mom", "rsi", "alpha158_x", "sector", "vol",
        "label", "split_label", "fwd_1d", "fwd_60d", "fwd_excess",
        "labelled",  # NOT forbidden — only exact "label"
        "fwding",    # forbidden (starts with fwd_? no: "fwding" doesn't start with "fwd_")
    ]
    k = rng.randint(0, len(pool))
    cols = rng.sample(pool, k)
    return cols


class TestLeakageGuard:

    def test_output_never_contains_forbidden_column(self):
        """THE leakage invariant: no label / split_label / fwd_* column may
        survive into the inference frame, for any input column set."""
        rng = random.Random(SEED)
        for _ in range(N):
            cols = _rand_columns(rng)
            if not cols:
                continue
            df = pd.DataFrame({c: [1, 2] for c in cols})
            out = _drop_inference_forbidden_cols(df)
            for c in out.columns:
                assert not _is_forbidden(c), (c, list(df.columns))

    def test_every_non_forbidden_column_survives_with_data(self):
        rng = random.Random(SEED + 1)
        for _ in range(N):
            cols = list(dict.fromkeys(_rand_columns(rng)))
            if not cols:
                continue
            df = pd.DataFrame({c: [hash(c) % 7, 1] for c in cols})
            out = _drop_inference_forbidden_cols(df)
            for c in cols:
                if not _is_forbidden(c):
                    assert c in out.columns, (c, list(out.columns))
                    assert list(out[c]) == list(df[c])

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"feat": [1], "label": [2], "fwd_60d": [3]})
        before = list(df.columns)
        _drop_inference_forbidden_cols(df)
        assert list(df.columns) == before  # original untouched

    def test_idempotent(self):
        rng = random.Random(SEED + 2)
        for _ in range(500):
            cols = _rand_columns(rng)
            if not cols:
                continue
            df = pd.DataFrame({c: [1] for c in cols})
            once = _drop_inference_forbidden_cols(df)
            twice = _drop_inference_forbidden_cols(once)
            assert list(once.columns) == list(twice.columns)

    def test_exact_label_match_only(self):
        # "labelled" / "labels" are real features, not the forbidden "label".
        df = pd.DataFrame({"labelled": [1], "labels": [2], "label": [3]})
        out = _drop_inference_forbidden_cols(df)
        assert "labelled" in out.columns and "labels" in out.columns
        assert "label" not in out.columns

    def test_no_forbidden_columns_returns_same_frame(self):
        df = pd.DataFrame({"feat1": [1], "mom": [2]})
        out = _drop_inference_forbidden_cols(df)
        assert out is df  # no-copy fast path when nothing to drop


class TestResolveManifestUri:
    def test_absolute_uri_within_root_kept(self, tmp_path):
        # Round 3 (PR #421 review): an absolute URI is no longer returned
        # unconditionally — only when its realpath is under an allowed root.
        manifest = tmp_path / "m" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}")
        abs_uri = str(tmp_path / "m" / "model.pt")
        assert _resolve_manifest_uri(manifest, abs_uri) == Path(abs_uri)

    def test_absolute_uri_outside_roots_rejected(self, tmp_path):
        # The bounded contract never returns an external absolute path
        # blindly (PR #421 review, blocking point 2).
        manifest = tmp_path / "m" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}")
        outside = tmp_path / "elsewhere" / "model.pt"
        with pytest.raises(ManifestUriResolutionError):
            _resolve_manifest_uri(manifest, str(outside))

    def test_relative_resolved_under_manifest_parent(self):
        assert _resolve_manifest_uri(Path("/data/m/manifest.json"),
                                     "sub/model.pt") == Path("/data/m/sub/model.pt")

    def test_result_absolute_or_rejected(self, tmp_path):
        """Property: for any (manifest, uri) pair the bounded resolver either
        raises ManifestUriResolutionError (absolute-outside-root or ``..``
        traversal escaping every allowed root) or returns an absolute path.

        Pre-round-3 this asserted the old unconditional-passthrough contract;
        the bounded resolver (PR #421 review) rejects escapes instead of
        returning them, so escaping choices are expected to raise here.
        """
        rng = random.Random(SEED + 3)
        manifest = tmp_path / "root" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}")
        for _ in range(500):
            uri = rng.choice(["/abs/x", "rel/x", "x.json", "../y/z"])
            try:
                out = _resolve_manifest_uri(manifest, uri)
            except ManifestUriResolutionError:
                # "/abs/x" (outside root) and "../y/z" (traversal) are
                # expected to escape every allowed root and raise.
                assert uri in ("/abs/x", "../y/z"), uri
                continue
            assert out.is_absolute()  # manifest is absolute → result always is


class TestArtifactKind:
    def test_metadata_kind_preferred(self, tmp_path):
        p = tmp_path / "a.json"
        p.write_text(json.dumps({"kind": "top", "metadata": {"kind": "meta"}}))
        assert _artifact_kind(p) == "meta"

    def test_top_level_kind_fallback(self, tmp_path):
        p = tmp_path / "a.json"
        p.write_text(json.dumps({"kind": "gmm"}))
        assert _artifact_kind(p) == "gmm"

    def test_none_on_unreadable_or_kindless(self, tmp_path):
        missing = tmp_path / "nope.json"
        assert _artifact_kind(missing) is None
        bad = tmp_path / "bad.json"
        bad.write_text("not json{")
        assert _artifact_kind(bad) is None
        nokind = tmp_path / "nokind.json"
        nokind.write_text(json.dumps({"other": 1}))
        assert _artifact_kind(nokind) is None


class TestHistorySeqLen:
    def test_reads_top_level_seq_len_from_metadata_sidecar(self, tmp_path):
        model = tmp_path / "model.pt"
        (tmp_path / "model.pt.metadata.json").write_text(json.dumps({"seq_len": 24}))
        assert _history_seq_len_from_artifact(model) == 24

    def test_reads_seq_len_from_hyperparameters(self, tmp_path):
        model = tmp_path / "m.pt"
        (tmp_path / "m_metadata.json").write_text(
            json.dumps({"training_contract": {"hyperparameters": {"seq_len": 16}}}))
        assert _history_seq_len_from_artifact(model) == 16

    def test_none_when_no_sidecar(self, tmp_path):
        assert _history_seq_len_from_artifact(tmp_path / "lonely.pt") is None
