"""Regression guards for 2026-05-20 P0 fixes.

Each test pins ONE invariant from doc/audits/2026-05-20-deep-code-audit.md.
The test would have FAILED before the fix landed → would have prevented
the bug from shipping. Per §5.13.3 ("every fix = regression-guard test
class").

Added 2026-05-20 evening after user observed: my P0-11 shadow config
fix broke daily shadow cron (caught only by next-day cron run, not by
any test). User: "能不能靠谱点！另外这些没有 test guard 吗？！"
"""
from __future__ import annotations
import json
import plistlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent


class TestP0_3_4_PlistWeekday:
    """Plist Weekday convention: macOS launchd: 0 AND 7 both = Sunday,
    Mon=1..Sat=6. Pin the weekday integers per cron intent."""

    def test_weekly_wf_promote_runs_saturday_not_sunday(self):
        p = REPO / "scripts/launchd/com.renquant.weekly-wf-promote.plist"
        text = p.read_text()
        # Saturday = 6 (NOT 7 which is Sunday alias)
        match = re.search(r"<key>Weekday</key>\s*<integer>(\d+)</integer>", text)
        assert match is not None, "no Weekday key in weekly-wf-promote.plist"
        weekday = int(match.group(1))
        assert weekday == 6, \
            f"weekly_wf_promote.plist Weekday={weekday}; must be 6 (Saturday). " \
            f"7 = Sunday on macOS launchd → 24h delay vs intended Saturday."

    def test_conditional_retrain_runs_monday_to_friday(self):
        p = REPO / "scripts/launchd/com.renquant.conditional-retrain104.plist"
        text = p.read_text()
        # Mon=1..Fri=5 — extract all Weekday integers
        weekdays = sorted(int(m) for m in re.findall(
            r"<key>Weekday</key>\s*<integer>(\d+)</integer>", text))
        assert weekdays == [1, 2, 3, 4, 5], \
            f"conditional-retrain104 fires on Weekdays {weekdays}; must be " \
            f"[1, 2, 3, 4, 5] (Mon-Fri). 6 = Saturday (no fresh data), " \
            f"0 = Sunday."


class TestP0_5_DashboardArtifactPath:
    """Dashboard must resolve panel-LTR path via golden config, not hardcode."""

    def test_dashboard_uses_canonical_resolver(self):
        src = (REPO / "scripts/build_dashboard.py").read_text()
        assert "_resolve_prod_panel_path" in src, \
            "dashboard must use _resolve_prod_panel_path (canonical "\
            "config-driven resolution per §5.13.14), not hardcoded filename"

    def test_dashboard_does_not_hardcode_legacy_data_path(self):
        """build_dashboard.py must NOT hardcode the legacy
        `data/panel-ltr-prod-alpha158-fund-fwd60d.json` path as the
        primary read target (only as a fallback)."""
        src = (REPO / "scripts/build_dashboard.py").read_text()
        # Find non-fallback occurrences (i.e., not inside try/except FallbackError block)
        # Simpler heuristic: count uses; must NOT be the FIRST `panel_path = REPO_ROOT / ...`
        # statement in section_model_health.
        sec = src.split("def section_model_health")[1].split("def ")[0]
        # The first line that assigns panel_path must call the resolver
        m = re.search(r"panel_path\s*=\s*([^\n]+)", sec)
        assert m is not None
        assert "_resolve_prod_panel_path" in m.group(1), \
            "first panel_path assignment in section_model_health must use resolver"


class TestP0_6_CalibratorMethodDefault:
    """3-site coherence: fit script default + golden + live config all 'platt'."""

    def test_fit_script_default_platt(self):
        src = (REPO / "scripts/fit_panel_calibrator.py").read_text()
        # Find the default in panel_cfg.get
        match = re.search(r'panel_cfg\.get\(\s*"calibration_method"\s*,\s*"(\w+)"',
                          src)
        assert match is not None, "no calibration_method default found"
        assert match.group(1) == "platt", \
            f"script default = {match.group(1)}; must be 'platt' " \
            f"(switched from isotonic 2026-05-18)"

    def test_golden_config_platt(self):
        cfg = json.loads((REPO / "backtesting/renquant_104/strategy_config.golden.json").read_text())
        method = cfg.get("panel_ltr", {}).get("calibration_method")
        assert method == "platt", \
            f"golden calibration_method = {method}; must be 'platt'"

    def test_live_config_platt(self):
        cfg = json.loads((REPO / "backtesting/renquant_104/strategy_config.json").read_text())
        method = cfg.get("panel_ltr", {}).get("calibration_method")
        assert method == "platt", \
            f"live calibration_method = {method}; must be 'platt'"


class TestP0_9_BugDSettledCash:
    """Live broker.get_cash() must use non_marginable_buying_power (cash + T+N),
    not account.cash (settled only). Pre-fix: live under-trades vs sim post-sell."""

    def test_get_cash_uses_non_marginable_buying_power(self):
        src = (REPO / "live/alpaca_broker.py").read_text()
        # The get_cash method must reference non_marginable_buying_power
        m = re.search(r"def get_cash[^}]*?(?=def |\Z)", src, re.DOTALL)
        assert m is not None
        body = m.group(0)
        assert "non_marginable_buying_power" in body, \
            "get_cash must use non_marginable_buying_power (BUG D fix)"

    def test_get_cash_falls_back_when_field_missing(self):
        src = (REPO / "live/alpaca_broker.py").read_text()
        m = re.search(r"def get_cash[^}]*?(?=def |\Z)", src, re.DOTALL)
        body = m.group(0)
        # Fallback path for older alpaca-py without the field
        assert "account.cash" in body, \
            "get_cash must fall back to account.cash if non_marginable_buying_power missing"


class TestP0_10_LiveAccountAssertion:
    """LIVE broker connect must assert account_number matches expected
    (RENQUANT_EXPECTED_LIVE_ACCOUNT env). Per 2026-05-17 e2e mandate."""

    def test_alpaca_connect_logs_account_number(self):
        src = (REPO / "live/alpaca_broker.py").read_text()
        assert "account.account_number" in src, \
            "connect() must log account.account_number for auditability"

    def test_alpaca_connect_asserts_expected_live_account_when_env_set(self):
        src = (REPO / "live/alpaca_broker.py").read_text()
        assert "RENQUANT_EXPECTED_LIVE_ACCOUNT" in src, \
            "must read RENQUANT_EXPECTED_LIVE_ACCOUNT env to verify LIVE account id"
        assert "ALPACA LIVE-ACCOUNT MISMATCH" in src, \
            "must raise on account number mismatch (don't silently connect to wrong account)"

    def test_alpaca_connect_requires_expected_live_account_env(self):
        src = (REPO / "live/alpaca_broker.py").read_text()
        assert "RENQUANT_EXPECTED_LIVE_ACCOUNT is required for LIVE Alpaca" in src, \
            "LIVE connect must fail closed when account pin env is missing"


class TestP0_11_ShadowConfigPathsExistOnDisk:
    """If shadow config artifact_path doesn't exist on disk, the shadow
    daily cron preflight HARD-fails. This regression bit on 2026-05-20
    when I renamed paths without creating the files."""

    def test_shadow_config_artifact_paths_resolve_to_existing_files(self):
        """Every artifact_path key in shadow config must point to a
        readable file on disk. (Side configs whose paths don't exist
        will silently fail daily shadow cron preflight.)"""
        shadow_cfg_path = REPO / "backtesting/renquant_104/strategy_config.shadow.json"
        if not shadow_cfg_path.exists():
            pytest.skip("shadow config not present")
        cfg = json.loads(shadow_cfg_path.read_text())
        strategy_root = shadow_cfg_path.parent

        # Walk the config tree for artifact_path keys
        missing: list[tuple[str, str]] = []
        def walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new = f"{path}.{k}" if path else k
                    if k == "artifact_path" and isinstance(v, str):
                        # Resolve path relative to strategy dir
                        full = (strategy_root / v).resolve()
                        if not full.exists():
                            missing.append((new, v))
                    walk(v, new)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")
        walk(cfg)

        assert not missing, (
            f"Shadow config has {len(missing)} artifact_path(s) pointing to "
            f"non-existent files (would break shadow cron preflight):\n  " +
            "\n  ".join(f"{k} = {v}" for k, v in missing)
        )

    def test_shadow_config_keeps_production_risk_contracts(self):
        """Shadow should only swap the scorer and isolated artifact/state paths.

        If risk knobs drift, prod-vs-shadow daily decisions stop being an
        apples-to-apples model comparison.
        """
        strategy_dir = REPO / "backtesting/renquant_104"
        prod = json.loads((strategy_dir / "strategy_config.json").read_text())
        shadow = json.loads((strategy_dir / "strategy_config.shadow.json").read_text())

        prod_panel = prod["ranking"]["panel_scoring"]
        shadow_panel = shadow["ranking"]["panel_scoring"]
        for key in ("buy_floor", "buy_floor_min", "buy_floor_std_mult"):
            assert shadow_panel.get(key) == prod_panel.get(key)

        prod_joint = prod["rotation"]["joint_actions"]
        shadow_joint = shadow["rotation"]["joint_actions"]
        for key in ("qp_mu_contract", "qp_tax_lot_method",
                    "qp_min_dw_pct", "qp_no_trade_band_cap"):
            assert shadow_joint.get(key) == prod_joint.get(key)

        for regime in ("BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR"):
            assert (
                shadow["regime_params"][regime].get("max_sector_weight_pct")
                == prod["regime_params"][regime].get("max_sector_weight_pct")
            )

    def test_patchtst_shadow_keeps_runtime_regime_admission_diagnostic(self):
        """Readonly PatchTST shadow must generate decision traces.

        Production full/buy stays fail-closed through preflight/runtime
        evidence gates. Shadow is different: it is a no-order diagnostic
        challenger whose current HF checkpoint sidecar does not yet carry
        strict WF regime metadata. If the runtime regime-admission gate
        defaults on here, it clears every candidate after non-strict
        preflight and the shadow run becomes a no-signal no-op.
        """
        strategy_dir = REPO / "backtesting/renquant_104"
        shadow = json.loads((strategy_dir / "strategy_config.shadow.json").read_text())
        panel = shadow["ranking"]["panel_scoring"]

        assert panel["kind"] == "hf_patchtst"
        assert panel["regime_admission"]["enabled"] is False
        assert "shadow" in panel["regime_admission"]["_shadow_reason"].lower()


class TestP0_12_DryRunHardFailuresFailClosed:
    """Training dry-run must not return success when any HARD preflight fails."""

    def test_train_dry_run_enforces_hard_preflight_even_without_strict_contract(self):
        src = (REPO / "scripts/train_104.py").read_text()
        dry_run_block = src.split("if args.dry_run:")[1].split("from kernel.pipeline")[0]

        assert "strict=True" in dry_run_block
        assert "strict=args.strict_contract" not in dry_run_block


class TestP0_13_DDVAttrName:
    """DDV reads sue_signal / pead_signal (NOT sue_score / pead_score) —
    must match what ApplyScoresTask actually writes."""

    def test_ddv_reads_correct_attrs(self):
        src = (REPO / "backtesting/renquant_104/kernel/pipeline/task_buy_quality_gates.py").read_text()
        # Should use sue_signal / pead_signal (post-fix names)
        assert '"sue_signal"' in src or "'sue_signal'" in src, \
            "DDV must read 'sue_signal' attr (matches ApplyScoresTask write)"
        assert '"pead_signal"' in src or "'pead_signal'" in src, \
            "DDV must read 'pead_signal' attr"
        # Should NOT use the old wrong names
        assert '"sue_score"' not in src and "'sue_score'" not in src, \
            "DDV must NOT use 'sue_score' (old wrong name); use 'sue_signal'"
        assert '"pead_score"' not in src and "'pead_score'" not in src, \
            "DDV must NOT use 'pead_score' (old wrong name); use 'pead_signal'"


class TestP0_16_AtomicCp:
    """monthly_calibrator_refresh.sh rollback cp must be atomic (cp .tmp + mv)."""

    def test_monthly_calibrator_uses_atomic_cp(self):
        sh = (REPO / "scripts/monthly_calibrator_refresh.sh").read_text()
        # Look for `cp X Y` without `.tmp && mv` for PROD_CAL writes.
        # Any line that does `cp "$ROLLBACK_CAL" "$PROD_CAL"` direct (no .tmp)
        # is the bug. Atomic pattern: `cp ... "$PROD_CAL.tmp" && mv "$PROD_CAL.tmp" "$PROD_CAL"`
        # OR `cp "$PROD_CAL" "$ROLLBACK_CAL.tmp" && mv "$ROLLBACK_CAL.tmp" "$ROLLBACK_CAL"`
        # Find all cp ... "$PROD_CAL" lines:
        bad = []
        for i, line in enumerate(sh.split("\n"), 1):
            if 'cp ' in line and '"$PROD_CAL"' in line and '.tmp' not in line:
                # exclude `cp X.tmp $PROD_CAL && mv` pattern handled separately
                # exclude comments
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Allow if next-line continuation or pattern handled
                bad.append((i, stripped))
        # Also reject cp Y Z (writing to ROLLBACK) without atomic
        for i, line in enumerate(sh.split("\n"), 1):
            if 'cp "$PROD_CAL" "$ROLLBACK_CAL"' in line and '.tmp' not in line:
                stripped = line.strip()
                if not stripped.startswith("#"):
                    bad.append((i, stripped))
        assert not bad, (
            f"monthly_calibrator_refresh.sh has {len(bad)} non-atomic cp "
            f"to/from PROD_CAL (P0-16 regression):\n  " +
            "\n  ".join(f"line {i}: {ln}" for i, ln in bad)
        )


class TestP0_17_BackupSizeGuard:
    """backup_to_github.sh must check file sizes before push (GitHub 100MB limit)."""

    def test_backup_has_100mb_guard(self):
        sh = (REPO / "scripts/backup_to_github.sh").read_text()
        # Look for the find -size +XXM check
        assert "find" in sh and "-size" in sh, \
            "backup script must `find -size` check before push (GitHub 100MB)"
        assert "TOO_LARGE_FILES" in sh and "99M" in sh, \
            "backup script must hard-fail near GitHub's 100MB file limit"
        assert "90M" in sh, \
            "backup script must retain early warning threshold before the hard fail"

    def test_backup_checks_push_exit_code(self):
        sh = (REPO / "scripts/backup_to_github.sh").read_text()
        # The push must capture rc directly; `if ! git push | tail` makes
        # `$?` equal 0 inside the failure branch because of shell negation.
        assert "if ! git push" not in sh, \
            "backup script must not use `if ! git push`; it masks PUSH_RC"
        assert "PUSH_LOG" in sh and "if git push origin main >\"$PUSH_LOG\" 2>&1" in sh, \
            "backup script must capture git push output and exit code directly"

    def test_backup_defaults_to_multirepo_orchestrator_pipeline(self):
        sh = (REPO / "scripts/backup_to_github.sh").read_text()
        assert 'RQ_STATE_BACKUP_RUNNER:-multirepo' in sh
        assert "renquant_orchestrator.state_backup" in sh
        assert "RQ_STATE_BACKUP_STRICT" in sh
        assert "scripts/subrepo_env.sh" in sh
        assert 'renquant_load_subrepo_env "$REPO_ROOT"' in sh
        assert 'export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"' in sh
        assert 'renquant_subrepo_src "$SUBREPO_ROOT" renquant-orchestrator' in sh


class TestP0_18_WeeklyApyMonitor:
    """Weekly APY monitor must be a subrepo-backed wrapper."""

    def test_weekly_apy_defaults_to_orchestrator_pipeline(self):
        src = (REPO / "scripts/weekly_apy_check.py").read_text()
        assert 'os.environ.get("RQ_WEEKLY_APY_RUNNER", "multirepo")' in src
        assert "renquant_orchestrator.weekly_apy_monitor" in src
        assert "from subrepo_paths import resolve_subrepo_root" in src
        assert "resolve_subrepo_root(REPO_ROOT)" in src
        assert "RQ_WEEKLY_APY_STRICT" in src

    def test_weekly_apy_plist_uses_project_venv(self):
        plist = (REPO / "scripts/launchd/com.renquant.weekly-apy104.plist").read_text()
        assert "/Users/renhao/git/github/RenQuant/.venv/bin/python" in plist
        assert "/Users/renhao/miniconda3" not in plist


class TestP0_19_LaunchdInventory:
    """Active 104 LaunchAgents must be tracked in repo, not only in ~/Library."""

    def test_daily_intraday_and_retrain_panel_plists_are_tracked(self):
        launchd = REPO / "scripts" / "launchd"
        for name in (
            "com.renquant.daily104.plist",
            "com.renquant.intraday104.plist",
            "com.renquant.retrain-panel104.plist",
        ):
            payload = plistlib.loads((launchd / name).read_bytes())
            assert payload["Label"] == name.removesuffix(".plist")
            assert "/Users/renhao/miniconda3" not in (launchd / name).read_text()

    def test_intraday104_plist_keeps_12_minute_market_cadence(self):
        payload = plistlib.loads(
            (REPO / "scripts/launchd/com.renquant.intraday104.plist").read_bytes()
        )
        intervals = payload["StartCalendarInterval"]
        assert len(intervals) >= 100
        assert {"Weekday": 1, "Hour": 6, "Minute": 30} in intervals
        assert {"Weekday": 5, "Hour": 13, "Minute": 0} in intervals

    def test_install_launchagents_includes_backup_plist(self):
        src = (REPO / "scripts/install_launchagents.sh").read_text()
        assert 'scripts/com.renquant.backup.plist' in src

    def test_install_launchagents_check_runs_full_ops_readiness(self):
        src = (REPO / "scripts/install_launchagents.sh").read_text()
        assert "scripts/check_ops_deployment_ready.py" in src
        assert "--launchagents-dir" in src

    def test_install_launchagents_preflights_runtime_readiness_before_install(self):
        src = (REPO / "scripts/install_launchagents.sh").read_text()
        assert "scripts/check_ops_deployment_ready.py" in src
        assert "--skip-launchagents" in src
        assert "RENQUANT_SKIP_OPS_PRECHECK" in src


class TestP0_19_QPProductionPath:
    """QP must use the same production gates/artifacts as rotation."""

    def test_joint_qp_respects_rotation_enabled_regimes(self):
        sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))
        try:
            from kernel.portfolio_qp.job_qp import JointPortfolioQPJob  # noqa: PLC0415
        finally:
            sys.path.remove(str(REPO / "backtesting" / "renquant_104"))
        job = JointPortfolioQPJob()
        ctx = SimpleNamespace(
            config={"rotation": {
                "enabled_regimes": ["BULL_CALM"],
                "joint_actions": {"enabled": True, "solver": "qp"},
            }},
            regime="BULL_VOLATILE",
            bear_only=False,
        )
        assert job.should_skip(ctx) is True
        ctx.regime = "BULL_CALM"
        assert job.should_skip(ctx) is False

    def test_compute_full_sigma_uses_loaded_corr_matrix(self):
        import numpy as np

        sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))
        try:
            from kernel.portfolio_qp.tasks import ComputeFullSigmaTask  # noqa: PLC0415
        finally:
            sys.path.remove(str(REPO / "backtesting" / "renquant_104"))
        ctx = SimpleNamespace(
            config={"rotation": {"joint_actions": {"qp_use_full_sigma": True}}},
            corr_matrix={"AAA": {"BBB": 0.5}},
            _qp_tickers=["AAA", "BBB"],
            _qp_sigma=np.array([0.10, 0.20]),
        )
        ComputeFullSigmaTask().run(ctx)
        assert ctx._qp_Sigma_full is not None
        assert ctx._qp_Sigma_full.shape == (2, 2)
        assert ctx._qp_Sigma_full[0, 1] == pytest.approx(0.01)

    def test_compute_full_sigma_resolves_configured_prod_artifact_path(self):
        src = (REPO / "backtesting/renquant_104/kernel/portfolio_qp/tasks.py").read_text()
        assert "correlation_artifact" in src, \
            "ComputeFullSigmaTask must read regime.correlation_artifact"
        assert "prod/watchlist-correlation.json" in src, \
            "ComputeFullSigmaTask default must match production artifact layout"


class TestP0_18_PanelScorerHFPatchTSTDispatch:
    """PanelScorer.load(.pt) must HF-detect and route to HFPatchTSTPanelScorer
    if the checkpoint has 'config_dict'+'feature_cols' marker keys.

    Pre-fix (commit before 2026-05-20): all .pt files fell through to legacy
    TransformerPanelScorer.load() which expects a sidecar JSON. Loading a
    sidecar-less HF PatchTST checkpoint (e.g. shadow seed44) raised
    'PanelTransformerModel.load: sidecar JSON not found' and SimAdapter
    silently dropped the panel scorer (returned None from
    _try_load_panel_scorer). The model_registry kind='hf_patchtst' path
    (registered 2026-05-18 in model_registry.py) was HF-aware but
    PanelScorer.load() was the split-brain twin that wasn't — §1c violation.
    """

    def test_panel_scorer_loads_hf_patchtst_pt_checkpoint(self):
        import sys
        sys.path.insert(0, str(REPO / "backtesting/renquant_104"))
        try:
            from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
            from kernel.panel_pipeline.hf_patchtst_scorer import HFPatchTSTPanelScorer  # noqa: PLC0415
        finally:
            sys.path.remove(str(REPO / "backtesting/renquant_104"))
        pt = REPO / "artifacts/patchtst_shadow/canonical_5seed_mps/seed_44/hf_patchtst_all_seed44_model.pt"
        if not pt.exists():
            pytest.skip("shadow PatchTST seed44 artifact not present")
        s = PanelScorer.load(pt)
        assert isinstance(s, HFPatchTSTPanelScorer), (
            f"PanelScorer.load returned {type(s).__name__}; expected "
            f"HFPatchTSTPanelScorer (HF-format .pt detection broken)"
        )
        assert s.requires_history is True
        assert len(s.feature_cols) == 172
        assert s.seq_len == 32
