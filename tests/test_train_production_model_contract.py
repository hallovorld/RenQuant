"""Regression guards for production panel artifact contract stamping."""
from __future__ import annotations

import numpy as np

from scripts.train_production_model import attach_inference_smoke


class _FakeBooster:
    def predict(self, _dmatrix):
        return np.linspace(-0.25, 0.25, 32)


def test_attach_inference_smoke_stamps_acceptance_metadata():
    artifact: dict = {}

    attach_inference_smoke(artifact, _FakeBooster(), ["f1", "f2", "f3"])

    md = artifact["metadata"]
    assert md["score_sample_range"] == [-0.25, 0.25]
    assert md["inference_smoke_test"] == {
        "n": 32,
        "all_finite": True,
        "n_unique": 32,
    }
