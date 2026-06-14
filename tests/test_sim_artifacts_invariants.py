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
- _resolve_manifest_uri: absolute uri kept; relative resolved under the
  manifest's parent.
- _artifact_kind: metadata.kind preferred over top-level kind; None on
  unreadable / non-dict / missing.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

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
    def test_absolute_uri_kept(self):
        assert _resolve_manifest_uri(Path("/data/m/manifest.json"),
                                     "/abs/model.pt") == Path("/abs/model.pt")

    def test_relative_resolved_under_manifest_parent(self):
        assert _resolve_manifest_uri(Path("/data/m/manifest.json"),
                                     "sub/model.pt") == Path("/data/m/sub/model.pt")

    def test_result_absolute_iff_inputs_make_it_so(self):
        rng = random.Random(SEED + 3)
        for _ in range(500):
            manifest = Path(f"/root/{rng.randint(0,9)}/manifest.json")
            uri = rng.choice(["/abs/x", "rel/x", "x.json", "../y/z"])
            out = _resolve_manifest_uri(manifest, uri)
            assert out.is_absolute()  # manifest is absolute → result always is
            if Path(uri).is_absolute():
                assert out == Path(uri)


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
