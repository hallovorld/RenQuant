import math

import pandas as pd

from scripts.build_alpha158_qlib import _compute_excess_label_frame


def test_spy_label_alignment_drops_stale_ffill_gap():
    ticker_close = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2026-01-02", "2026-01-12", "2026-01-13"]),
    )
    spy_close = pd.Series(
        [200.0, 210.0],
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )

    out = _compute_excess_label_frame(
        "AAA",
        ticker_close,
        spy_close,
        horizons=(1,),
        max_spy_ffill_days=5,
    )

    stale_row = out.loc[out["date"] == pd.Timestamp("2026-01-12")].iloc[0]
    assert math.isnan(stale_row["fwd_1d_excess"])


def test_spy_label_alignment_allows_short_holiday_gap():
    ticker_close = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
    )
    spy_close = pd.Series(
        [200.0, 202.0],
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )

    out = _compute_excess_label_frame(
        "AAA",
        ticker_close,
        spy_close,
        horizons=(1,),
        max_spy_ffill_days=5,
    )

    valid_row = out.loc[out["date"] == pd.Timestamp("2026-01-02")].iloc[0]
    assert not math.isnan(valid_row["fwd_1d_excess"])
