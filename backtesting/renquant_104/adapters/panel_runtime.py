"""Shared panel-runtime helpers for sim, live runner, and LEAN adapters.

Inference adapters should not hand-roll panel frame preparation. A divergence
here means sim validates one feature surface while live/LEAN trade another.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PanelFrameBundle:
    feature_frames: dict
    factor_frames: dict
    macro_frame: Any
    asset_embeddings: dict | None


def panel_scoring_enabled(config: dict) -> bool:
    return bool(
        (config.get("ranking", {}) or {})
        .get("panel_scoring", {})
        .get("enabled", False)
    )


def prepare_panel_runtime_frames(
    *,
    config: dict,
    ohlcv: dict,
    spy_df: Any | None = None,
    watchlist: list[str] | None = None,
    ticker_sectors: dict[str, str] | None = None,
) -> PanelFrameBundle:
    """Prepare the panel feature bundle through the single training helper.

    The helper normalizes common adapter inputs:
      - uses config.watchlist unless a caller supplies an explicit watchlist
      - injects benchmark/SPY data when an adapter carries it separately
      - passes only available sector metadata for the configured watchlist
    """
    from training_panel.pipeline import prepare_inference_panel_frames  # noqa: PLC0415

    wl = list(watchlist if watchlist is not None else config.get("watchlist", []))
    benchmark = str(config.get("benchmark", "SPY"))
    ohlcv_panel = dict(ohlcv or {})
    if spy_df is not None and benchmark not in ohlcv_panel:
        ohlcv_panel[benchmark] = spy_df
    if ticker_sectors is None:
        sector_map = config.get("sector_map", {}) or {}
        ticker_sectors = {
            t: sector_map[t]
            for t in wl
            if isinstance(sector_map.get(t), str) and sector_map.get(t)
        }
    ff, fac, macro, emb = prepare_inference_panel_frames(
        watchlist=wl,
        ohlcv=ohlcv_panel,
        ticker_sectors=ticker_sectors,
        config=config,
    )
    return PanelFrameBundle(
        feature_frames=ff,
        factor_frames=fac,
        macro_frame=macro,
        asset_embeddings=emb,
    )


def attach_panel_runtime_frames(ctx: Any, bundle: PanelFrameBundle) -> None:
    """Attach a prepared bundle to an InferenceContext."""
    ctx._panel_feature_frames = bundle.feature_frames  # noqa: SLF001
    ctx._panel_factor_frames = bundle.factor_frames  # noqa: SLF001
    ctx._panel_macro_frame = bundle.macro_frame  # noqa: SLF001
    ctx._panel_asset_embeddings = bundle.asset_embeddings  # noqa: SLF001


def describe_panel_frame_bundle(bundle: PanelFrameBundle) -> tuple[int, int, str, int]:
    macro_desc = (
        "None"
        if bundle.macro_frame is None
        else f"{len(bundle.macro_frame.columns)}cols"
    )
    return (
        len(bundle.feature_frames),
        len(bundle.factor_frames),
        macro_desc,
        len(bundle.asset_embeddings) if bundle.asset_embeddings else 0,
    )
