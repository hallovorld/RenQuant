"""sim.py decomposition slice 2 — sim_artifacts pure-helper tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from adapters.sim_artifacts import (  # noqa: E402
    _drop_inference_forbidden_cols,
    _resolve_manifest_uri,
)


class TestLeakageGuard:
    def test_drops_label_and_fwd_cols(self):
        df = pd.DataFrame({"close": [1.0], "label": [0], "fwd_60d_excess": [0.1],
                           "MA5": [1.0]})
        out = _drop_inference_forbidden_cols(df)
        assert list(out.columns) == ["close", "MA5"]

    def test_drops_split_label(self):
        df = pd.DataFrame({"feat": [1.0], "split_label": ["train"]})
        assert list(_drop_inference_forbidden_cols(df).columns) == ["feat"]

    def test_no_forbidden_unchanged(self):
        df = pd.DataFrame({"close": [1.0], "MA5": [1.0]})
        out = _drop_inference_forbidden_cols(df)
        assert list(out.columns) == ["close", "MA5"]

    def test_any_fwd_prefix_dropped(self):
        df = pd.DataFrame({"fwd_1d": [0], "fwd_20d": [0], "x": [1]})
        assert list(_drop_inference_forbidden_cols(df).columns) == ["x"]


class TestResolveManifestUri:
    def test_absolute_uri_passthrough(self, tmp_path):
        abs_uri = str(tmp_path / "model.pt")
        out = _resolve_manifest_uri(tmp_path / "manifest.json", abs_uri)
        assert Path(out).is_absolute()

    def test_relative_uri_resolved_against_manifest(self, tmp_path):
        manifest = tmp_path / "sub" / "manifest.json"
        out = _resolve_manifest_uri(manifest, "model.pt")
        assert out == manifest.with_name("model.pt")

    def test_strategy_dir_relative_uri_resolves_to_existing_corpus(self, tmp_path):
        """WF-gate regression: orchestrator-built manifests live under
        ``<strategy>/artifacts/sim/`` but emit strategy-dir-relative URIs
        (``artifacts/walkforward_.../panel-ltr.json``). Naive manifest-parent
        joining doubled the prefix into ``artifacts/sim/artifacts/...`` which
        does not exist → FileNotFoundError fail-closed the gate. The resolver
        must walk up ancestors and find the real corpus.
        """
        strategy = tmp_path
        manifest = strategy / "artifacts" / "sim" / "walkforward_manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}")
        uri = "artifacts/walkforward_gbdt_prod_recipe_v2/2023-10-02/panel-ltr.json"
        real = strategy / uri
        real.parent.mkdir(parents=True, exist_ok=True)
        real.write_text("{}")

        out = _resolve_manifest_uri(manifest, uri)

        assert out == real, out
        assert out.exists()
        # And explicitly NOT the doubled manifest-parent path.
        assert out != manifest.parent / uri

    def test_missing_relative_uri_falls_back_to_manifest_parent(self, tmp_path):
        """When no candidate exists, keep the manifest-parent join so the
        downstream not-found error stays meaningful (no silent surprise path).
        """
        manifest = tmp_path / "artifacts" / "sim" / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}")
        out = _resolve_manifest_uri(manifest, "artifacts/nope/panel-ltr.json")
        assert out == manifest.parent / "artifacts/nope/panel-ltr.json"
