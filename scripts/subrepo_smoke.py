#!/usr/bin/env python3
"""Run a tiny cross-repo smoke through train, infer, execute, and backtest contracts."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "subrepos.lock.json").read_text())


for entry in LOCK["subrepos"]:
    src = Path(entry["local_path"]) / "src"
    if src.exists():
        sys.path.insert(0, str(src))


from renquant_execution import PaperBroker  # noqa: E402
from renquant_orchestrator import DailyRunContext, DailyRunPipeline  # noqa: E402
from renquant_pipeline import InferenceContext  # noqa: E402
from renquant_strategy_104 import load_strategy_config, strategy_manifest  # noqa: E402
from renquant_common import Task  # noqa: E402


def _entry(name: str) -> dict[str, Any]:
    for entry in LOCK["subrepos"]:
        if entry["name"] == name:
            return entry
    raise KeyError(name)


def _data_manifest() -> dict[str, Any]:
    return {
        "dataset_id": "subrepo-smoke-daily-panel",
        "schema_version": "smoke-v1",
        "fingerprint": "sha256:smoke-data",
        "uri": "object://renquant-data/subrepo-smoke-daily-panel.parquet",
        "asset_class": "equity",
        "retention_class": "fixture",
    }


class SmokeScoreTask(Task):
    def run(self, ctx: InferenceContext) -> bool | None:
        ticker = ctx.strategy_config["watchlist"][0]
        ctx.scores[ticker] = 0.42
        ctx.decision_trace.append({"stage": "score", "ticker": ticker, "score": 0.42})
        return True


class SmokeSelectTask(Task):
    def run(self, ctx: InferenceContext) -> bool | None:
        ticker = max(ctx.scores, key=ctx.scores.get)
        ctx.order_intents.append({"ticker": ticker, "action": "buy", "quantity": 1})
        ctx.decision_trace.append({"stage": "select", "ticker": ticker})
        return True


def main() -> int:
    strategy_path = Path(_entry("renquant-strategy-104")["local_path"]) / "configs" / "strategy_config.json"
    strategy_config = load_strategy_config(strategy_path)
    strategy_ref = strategy_manifest(strategy_path)
    data_manifest = _data_manifest()

    calls: list[str] = []

    def loader(manifest: dict[str, Any]) -> dict[str, Any]:
        calls.append("load")
        return {"manifest": manifest, "rows": [1, 2, 3]}

    def trainer(dataset: Any, config: dict[str, Any], output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        calls.append("train")
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "artifact_id": "subrepo-smoke-gbdt",
            "model_family": "gbdt-panel-ltr",
            "fingerprint": "sha256:smoke-model",
            "uri": "object://renquant-artifacts/subrepo-smoke-gbdt.json",
            "promotion_status": "candidate",
            "feature_cols": ["alpha_1", "alpha_2"],
            "trained_date": "2026-05-25",
            "config_fingerprint": config["config_fingerprint"],
            "panel_shape": {"rows": len(dataset["rows"]), "cols": 2},
            "lookahead_days": 5,
            "train_run_id": "subrepo-smoke-run",
            "oos_mean_ic": 0.0,
            "oos_std_ic": 0.0,
            "oos_per_fold_ic": [0.0, 0.0],
            "cv_method": "purged-walk-forward-smoke",
            "cv_embargo_days": 5,
        }, {"artifact_id": "subrepo-smoke-calibrator"}

    def validator(artifact: dict[str, Any], dataset: Any, config: dict[str, Any]) -> dict[str, Any]:
        calls.append("validate")
        return {"accepted": False, "oos_mean_ic": 0.0, "smoke": True}

    with tempfile.TemporaryDirectory(prefix="renquant-subrepo-smoke-") as tmp:
        paper_broker = PaperBroker(initial_cash=100000.0)
        daily_ctx = DailyRunContext(
            run_id="subrepo-smoke-daily-full",
            run_type="daily_full",
            strategy_config=strategy_config,
            strategy_manifest=strategy_ref,
            data_manifest=data_manifest,
            model_config={
                "strategy": "renquant_104",
                "objective": "rank:pairwise",
                "config_fingerprint": strategy_ref["fingerprint"],
                "code_commit": _entry("renquant-model-gbdt")["commit"],
            },
            market_snapshot={"as_of": "2026-05-25"},
            account_snapshot={"cash": 100000.0},
            output_dir=Path(tmp) / "daily-run",
            broker=paper_broker,
            runtime_stages=[SmokeScoreTask(), SmokeSelectTask()],
            price_map={"AAPL": 100.0},
            dry_run=True,
        )
        paper_broker.broker_name = "paper-smoke"
        DailyRunPipeline(
            loader,
            trainer,
            validator,
            backtest_runner=lambda ctx: {"ok": True, "n_orders": len(daily_ctx.inference_context.order_intents)},
        ).run(daily_ctx)
        if daily_ctx.training_context is None or daily_ctx.training_context.artifact_manifest is None:
            raise RuntimeError("GBDT smoke did not produce artifact_manifest")
        if not daily_ctx.run_bundle:
            raise RuntimeError("daily smoke did not produce run_bundle")

    summary = {
        "ok": True,
        "training_calls": calls,
        "artifact_id": daily_ctx.training_context.artifact_manifest["artifact_id"],
        "order_intents": daily_ctx.inference_context.order_intents,
        "submitted_orders": daily_ctx.execution_context.submitted_orders,
        "backtest_report": daily_ctx.backtest_context.report,
        "run_bundle_keys": sorted(daily_ctx.run_bundle.keys()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
