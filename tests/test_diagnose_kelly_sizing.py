from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO / "scripts" / "diagnose_kelly_sizing.py"
    spec = importlib.util.spec_from_file_location("diagnose_kelly_sizing", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_apply_kelly_sizing_log_summary(tmp_path: Path) -> None:
    mod = _load_module()
    log_path = tmp_path / "live.log"
    log_path.write_text(
        "2026-06-01 16:00:00 ApplyKellySizingTask: fractional=0.50 "
        "max_conc=0.35  cands=2/40 non-zero (avg=3.5%)  "
        "holdings=1/5 non-zero (avg=4.0%)  "
        "zero_reasons[mu_none=10 sigma_none=3]\n"
        "2026-06-02 16:00:00 ApplyKellySizingTask: fractional=0.50 "
        "max_conc=0.35  cands=4/20 non-zero (avg=5.0%)  "
        "holdings=3/5 non-zero (avg=6.0%)  "
        "zero_reasons[mu_le_min_edge=7]\n",
        encoding="utf-8",
    )

    result = mod.run_diagnostic(
        root=tmp_path,
        log_specs=[str(log_path)],
        data_specs=[],
        state_specs=[],
        use_defaults=False,
    )

    assert result["ok"] is True
    logs = result["logs"]
    assert logs["records"] == 2
    assert logs["candidates"]["nonzero_sum"] == 6
    assert logs["candidates"]["total_sum"] == 60
    assert logs["candidate_avg_kelly_pct"]["median"] == 4.25
    assert logs["zero_reasons"] == {
        "mu_le_min_edge": 7,
        "mu_none": 10,
        "sigma_none": 3,
    }


def test_csv_kelly_targets_are_summarized_per_regime(tmp_path: Path) -> None:
    mod = _load_module()
    data_path = tmp_path / "decision_trace.csv"
    data_path.write_text(
        "run_date,regime,kelly_target_pct,sigma\n"
        "2026-06-01,BULL_CALM,0.00,0.30\n"
        "2026-06-01,BULL_CALM,0.03,0.50\n"
        "2026-06-02,BULL_CALM,0.05,0.40\n"
        "2026-06-02,CHOPPY,0.12,0.20\n",
        encoding="utf-8",
    )

    result = mod.run_diagnostic(
        root=tmp_path,
        log_specs=[],
        data_specs=[str(data_path)],
        state_specs=[],
        use_defaults=False,
    )

    kelly = result["metrics"]["kelly"]
    assert kelly["summary_pct"]["n"] == 4
    assert kelly["summary_pct"]["median"] == 4.0
    assert kelly["nonzero"] == 3
    bull = kelly["by_regime"]["BULL_CALM"]
    assert bull["summary_pct"]["median"] == 3.0
    assert bull["histogram_pct"]["0"] == 1
    assert bull["histogram_pct"]["2..5"] == 1
    assert result["metrics"]["sigma"]["summary_pct"]["median"] == 35.0
    horizon = result["metrics"]["sigma_horizon"]
    assert horizon["annualized_horizon_days"] == 252.0
    assert horizon["target_horizon_days"] == 60.0
    assert horizon["sigma_rescale"] == math.sqrt(60.0 / 252.0)
    assert horizon["kelly_multiplier"] == 4.2
    assert "4.20x" in horizon["interpretation"]


def test_sigma_horizon_diagnostic_accepts_custom_horizons(tmp_path: Path) -> None:
    mod = _load_module()
    data_path = tmp_path / "decision_trace.csv"
    data_path.write_text(
        "run_date,kelly_target_pct,sigma\n"
        "2026-06-01,0.06,0.40\n",
        encoding="utf-8",
    )

    result = mod.run_diagnostic(
        root=tmp_path,
        log_specs=[],
        data_specs=[str(data_path)],
        state_specs=[],
        use_defaults=False,
        annualized_sigma_horizon_days=120,
        target_sigma_horizon_days=30,
    )

    horizon = result["metrics"]["sigma_horizon"]
    assert horizon == {
        "annualized_horizon_days": 120.0,
        "target_horizon_days": 30.0,
        "sigma_rescale": 0.5,
        "kelly_multiplier": 4.0,
        "interpretation": (
            "Same-period Kelly using sigma rescaled from 120d to 30d "
            "multiplies targets by about 4.00x versus leaving sigma "
            "on the 120d horizon."
        ),
    }


def test_sigma_horizon_text_render_includes_multiplier(tmp_path: Path) -> None:
    mod = _load_module()
    data_path = tmp_path / "decision_trace.csv"
    data_path.write_text(
        "run_date,kelly_target_pct,sigma\n"
        "2026-06-01,0.06,0.40\n",
        encoding="utf-8",
    )

    result = mod.run_diagnostic(
        root=tmp_path,
        log_specs=[],
        data_specs=[str(data_path)],
        state_specs=[],
        use_defaults=False,
    )

    rendered = mod.render_text(result)
    assert "same-period horizon: annualized_days=252 target_days=60" in rendered
    assert "kelly_multiplier=4.2000" in rendered


def test_json_state_cash_time_series_uses_recent_rows(tmp_path: Path) -> None:
    mod = _load_module()
    state_path = tmp_path / "live_state_snapshots.json"
    state_path.write_text(
        json.dumps({
            "rows": [
                {"run_date": "2026-05-30", "regime": "BULL_CALM", "cash": 50, "portfolio_value": 100},
                {"run_date": "2026-05-31", "regime": "BULL_CALM", "cash": 45, "portfolio_value": 100},
                {"run_date": "2026-06-01", "regime": "BULL_CALM", "cash_pct": 0.40},
            ]
        }),
        encoding="utf-8",
    )

    result = mod.run_diagnostic(
        root=tmp_path,
        log_specs=[],
        data_specs=[],
        state_specs=[str(state_path)],
        use_defaults=False,
        recent_bars=2,
    )

    cash = result["metrics"]["cash"]
    assert cash["rows"] == 3
    assert cash["recent_rows"] == 2
    assert cash["summary_pct"]["median"] == 42.5
    assert cash["latest"] == {
        "date": "2026-06-01",
        "regime": "BULL_CALM",
        "cash_pct": 40.0,
    }


def test_missing_inputs_fail_closed(tmp_path: Path) -> None:
    mod = _load_module()

    result = mod.run_diagnostic(
        root=tmp_path,
        log_specs=[str(tmp_path / "missing.log")],
        data_specs=[],
        state_specs=[],
        use_defaults=False,
    )

    assert result["ok"] is False
    assert any("input did not match any files" in warning for warning in result["warnings"])
    assert any("no usable Kelly/cash diagnostics found" in warning for warning in result["warnings"])
