from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel import pipeline as p  # noqa: E402


def _cfg(tmp_path: Path, *, enabled: bool = True) -> dict:
    return {
        "_strategy_dir": str(tmp_path),
        "panel_ltr": {
            "inference_frame_cache": {
                "enabled": enabled,
                "cache_dir": "cache/inference_frames",
            }
        },
        "ranking": {"panel_scoring": {"kind": "hf_patchtst"}},
    }


def test_inference_frame_cache_disabled_is_noop(tmp_path):
    cfg = _cfg(tmp_path, enabled=False)

    assert p._load_inference_frame_cache(cfg, "missing-key") is None


def test_inference_frame_cache_key_changes_with_asof_date(tmp_path):
    cfg = _cfg(tmp_path)
    one = pd.DataFrame(
        {"close": [10.0, 11.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )
    two = pd.DataFrame(
        {"close": [10.0, 11.0, 12.0]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
    )

    key_one = p._inference_frame_cache_key(["AAA"], {"AAA": one}, {"AAA": "Tech"}, cfg)
    key_two = p._inference_frame_cache_key(["AAA"], {"AAA": two}, {"AAA": "Tech"}, cfg)

    assert key_one != key_two


def test_inference_frame_cache_round_trip(tmp_path):
    cfg = _cfg(tmp_path)
    idx = pd.to_datetime(["2026-01-02", "2026-01-05"])
    ff = {"AAA": pd.DataFrame({"f1": [1.0, 2.0]}, index=idx)}
    fac = {"AAA": pd.DataFrame({"g1": [3.0, 4.0]}, index=idx)}
    macro = pd.DataFrame({"m1": [5.0, 6.0]}, index=idx)
    emb = {"AAA": [0.1, 0.2]}

    p._write_inference_frame_cache(cfg, "unit-key", ff, fac, macro, emb)
    loaded = p._load_inference_frame_cache(cfg, "unit-key")

    assert loaded is not None
    loaded_ff, loaded_fac, loaded_macro, loaded_emb = loaded
    pd.testing.assert_frame_equal(loaded_ff["AAA"], ff["AAA"])
    pd.testing.assert_frame_equal(loaded_fac["AAA"], fac["AAA"])
    pd.testing.assert_frame_equal(loaded_macro, macro)
    assert loaded_emb == emb
