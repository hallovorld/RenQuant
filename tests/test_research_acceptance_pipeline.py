"""Contracts for the renquant_104 research acceptance pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))
sys.path.insert(0, str(REPO))

from kernel.pipeline.pp_research_acceptance import (  # noqa: E402
    CommandResult,
    ResearchAcceptanceContext,
    ResearchAcceptancePipeline,
    normalize_targets,
)


def _dry_ctx(**kwargs) -> ResearchAcceptanceContext:
    return ResearchAcceptanceContext(repo=REPO, dry_run=True, python="python", **kwargs)


def test_normalize_targets_expands_all_once():
    assert normalize_targets(["contracts", "all", "contracts"]) == (
        "contracts", "true-oos", "wf-gate",
    )


def test_true_oos_chain_uses_one_artifact_dir_contract():
    ctx = _dry_ctx(
        targets=("true-oos",),
        artifact_dir=Path("tmp/walkforward_oos"),
        eval_json_path=Path("tmp/prod_eval/eval_truly_oos.json"),
    )
    ResearchAcceptancePipeline(("true-oos",)).run(ctx)

    commands = {spec.name: spec.argv for spec in ctx.executed}
    assert commands["retrain_true_oos"][-2:] == (
        "--output-dir", str(REPO / "tmp/walkforward_oos"),
    )
    assert commands["eval_true_oos"][-4:] == (
        "--artifact-dir", str(REPO / "tmp/walkforward_oos"),
        "--out", str(REPO / "tmp/prod_eval/eval_truly_oos.json"),
    )
    assert commands["stamp_dsr_pbo"][-2:] == (
        "--eval-json", str(REPO / "tmp/prod_eval/eval_truly_oos.json"),
    )


def test_true_oos_skip_retrain_keeps_eval_and_dsr_order():
    ctx = _dry_ctx(targets=("true-oos",), skip_retrain=True)
    ResearchAcceptancePipeline(("true-oos",)).run(ctx)
    assert [spec.name for spec in ctx.executed] == ["eval_true_oos", "stamp_dsr_pbo"]


def test_wf_gate_target_skips_without_artifact():
    ctx = _dry_ctx(targets=("wf-gate",))
    ResearchAcceptancePipeline(("wf-gate",)).run(ctx)
    assert ctx.executed == []


def test_wf_gate_command_uses_parallel_cut_jobs_and_strict():
    ctx = _dry_ctx(
        targets=("wf-gate",),
        artifact=Path("backtesting/renquant_104/artifacts/staging.json"),
        wf_jobs=3,
    )
    ResearchAcceptancePipeline(("wf-gate",)).run(ctx)
    argv = ctx.executed[0].argv
    assert "--jobs" in argv and argv[argv.index("--jobs") + 1] == "3"
    assert "--strategy-config" in argv
    assert argv[argv.index("--strategy-config") + 1] == "strategy_config.json"
    assert "--strict" in argv


def test_runner_failure_raises_pipeline_error():
    def fail_runner(spec):
        return CommandResult(spec=spec, returncode=7)

    ctx = ResearchAcceptanceContext(
        repo=REPO,
        python="python",
        targets=("contracts",),
        runner=fail_runner,
    )
    try:
        ResearchAcceptancePipeline(("contracts",)).run(ctx)
    except RuntimeError as exc:
        assert "failed" in str(exc)
    else:
        raise AssertionError("pipeline should raise on non-zero command result")
