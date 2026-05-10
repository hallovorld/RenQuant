"""Regression test for NGBoostHead.train UnboundLocalError logging shadow.

Pre-fix (2026-05-10 audit): two `import logging` statements were nested
INSIDE `NGBoostHead.train` (inside `if n_dropped:` and
`if n_imputed_train > 0:` branches). Python's compile-time scope analysis
sees any inner `import` and marks that name as LOCAL for the whole
function. When the unconditional `logging.getLogger("ngboost").warning(...)`
at the X_train-nonfinite/extreme-cell guard fired BEFORE either inner
import had run, the result was:

    UnboundLocalError: local variable 'logging' referenced before assignment

NGBoost is OFF in production per CLAUDE.md, so the surrounding
NGBoostFitTask `except Exception` swallowed the error and emitted the
"NGBoost will be skipped this run" message that has spammed every retrain
log. The fix removes the redundant inner imports so all logging references
inside `train` resolve to the module-level `import logging` at line 23.

Per CLAUDE.md §5.13.3: every fix names its class-of-bug invariant + ships
an AUDIT REGRESSION GUARD that pins the invariant. The invariant here is
"no inner shadow of the `logging` module inside `NGBoostHead.train` (or
any other function in `ngboost_head.py`)". Per §5.13.5: single source of
truth — module-level `import logging` only.
"""
from __future__ import annotations

import logging as _stdlib_logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from training_panel.ngboost_head import NGBoostHead  # noqa: E402

_NGB_HEAD_SRC = (
    _STRATEGY_DIR / "training_panel" / "ngboost_head.py"
).read_text()


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _trigger_extreme_panel(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """Build a panel that forces the `|x|>_CLIP` warning branch (line 230)
    to fire while keeping n_dropped == 0 AND n_imputed_train == 0.

    That combination — extreme cells present BUT no dropped rows and no
    imputed cells — is the exact path that triggered the
    UnboundLocalError pre-fix.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Insert a single extreme value > _CLIP=10.0 → triggers the warning
    x1[0] = 50.0
    y = 1.0 * x1 + rng.normal(0.0, 0.2, size=n)
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "residual_return_raw": y,
        "weight": 1.0,
    })
    # All rows finite, label finite, weight non-negative → n_dropped == 0
    assert df.notna().all().all()
    return df, ["x1", "x2"]


# ──────────────────────────────────────────────────────────────────────
# 1. AUDIT REGRESSION GUARD (§5.13.3)
# ──────────────────────────────────────────────────────────────────────

class TestNGBoostLoggingRegressionGuard:
    """Pin the invariant: no UnboundLocalError on `logging` in train()."""

    def test_train_runs_without_unbound_local_error_on_extreme_features(self):
        """The exact code path that raised pre-fix must now complete."""
        df, feats = _trigger_extreme_panel(n=120, seed=1)
        head = NGBoostHead({
            "n_estimators": 20,
            "learning_rate": 0.02,
            "verbose": False,
        })
        try:
            info = head.train(
                df,
                feature_cols=feats,
                label_col="residual_return_raw",
                sample_weight_col="weight",
                # disable val split so we don't add unrelated complexity
                val_fraction=0.0,
                early_stopping_rounds=None,
            )
        except UnboundLocalError as exc:  # pragma: no cover — regression
            pytest.fail(f"NGBoostHead.train raised UnboundLocalError: {exc}")
        # Sanity: fit returned a structured dict
        assert isinstance(info, dict)
        assert info["n_rows"] == len(df)
        assert info["n_features"] == 2

    def test_train_emits_extreme_cell_warning_via_module_logger(self, caplog):
        """The warning at the extreme-cell guard must reach the logger,
        proving the bare `logging.getLogger(...)` call now resolves."""
        df, feats = _trigger_extreme_panel(n=120, seed=2)
        head = NGBoostHead({
            "n_estimators": 20,
            "learning_rate": 0.02,
            "verbose": False,
        })
        with caplog.at_level(_stdlib_logging.WARNING, logger="ngboost"):
            head.train(
                df,
                feature_cols=feats,
                label_col="residual_return_raw",
                sample_weight_col="weight",
                val_fraction=0.0,
                early_stopping_rounds=None,
            )
        # At least one warning from the ngboost logger about extreme cells
        ngb_warnings = [
            r for r in caplog.records
            if r.name == "ngboost" and "extreme" in r.getMessage()
        ]
        assert ngb_warnings, (
            "Expected at least one 'extreme' warning from the ngboost "
            "logger; got none. If logging is broken the warning silently "
            "disappears."
        )


# ──────────────────────────────────────────────────────────────────────
# 2. Graceful degradation when no prior artifact exists
# ──────────────────────────────────────────────────────────────────────

class TestNGBoostFitTaskGracefulDegradation:
    """The wrapping NGBoostFitTask should emit the 'will be skipped this
    run' ERROR via the working logger (not via raise) when fit fails and
    no previous artifact exists. After the fix this path is reachable
    via genuine fit failures, not via the spurious UnboundLocalError."""

    def test_fit_failure_no_prior_artifact_logs_error_does_not_raise(
        self, tmp_path, caplog,
    ):
        # Import lazily — heavyweight module
        from training_panel.pp_panel_training import (  # noqa: PLC0415
            NGBoostFitTask, PanelTrainingContext,
        )

        # Minimal config: ngboost enabled, artifact path under tmp_path
        # so we exercise the "no previous artifact" branch.
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        config = {
            "_strategy_dir": str(tmp_path),
            "panel_ltr": {
                "lookahead_days": 5,
                "ngboost": {
                    "enabled": True,
                    "artifact_path": "ngboost-head.json",
                    "params": {
                        "n_estimators": 5,
                        "learning_rate": 0.01,
                    },
                    "val_fraction": 0.0,
                    "early_stopping_rounds": 0,
                    "cv": {"enabled": False},
                },
            },
        }

        # Build a panel that's guaranteed to fail train: < 10 clean rows
        # forces NGBoostHead.train to raise ValueError. NGBoostFitTask's
        # `except Exception` catches it and routes to the "no previous
        # artifact" log.error path.
        panel = pd.DataFrame({
            "x1": [0.1, 0.2, 0.3],
            "residual_return_raw": [0.01, 0.02, 0.03],
            "weight": [1.0, 1.0, 1.0],
        })

        ctx = PanelTrainingContext(config=config)
        ctx.panel = panel
        ctx.feature_cols = ["x1"]
        # strategy_dir is a @property that reads config["_strategy_dir"]
        assert ctx.strategy_dir == tmp_path

        task = NGBoostFitTask()

        # Confirm no prior artifact exists
        assert not (art_dir / "ngboost-head.json").exists()

        with caplog.at_level(_stdlib_logging.ERROR,
                             logger="training_panel.pipeline"):
            # Must NOT raise — the except block swallows the error.
            task.run(ctx)

        # The error log line is the "will be skipped this run" message,
        # proving the logger works (was broken pre-fix only because the
        # underlying fit raised UnboundLocalError — symptom not cause).
        err_msgs = [
            r.getMessage() for r in caplog.records
            if r.levelno >= _stdlib_logging.ERROR
        ]
        assert any("NGBoost will be skipped this run" in m for m in err_msgs), (
            f"Expected 'NGBoost will be skipped this run' error log; "
            f"got: {err_msgs}"
        )
        # ctx.ngboost_head must remain None (fit didn't succeed)
        assert ctx.ngboost_head is None


# ──────────────────────────────────────────────────────────────────────
# 3. Source-text invariant (§5.13.5 — no parallel shim)
# ──────────────────────────────────────────────────────────────────────

class TestNoLoggingModuleShadowInSource:
    """Static check on the source file: no inner `import logging` or
    `logging = ...` assignment may shadow the module-level import.
    Pinning this textually prevents future-Claude from reintroducing
    the same bug class via a different code path.
    """

    def test_no_inner_import_logging_in_ngboost_head_module(self):
        """The phrase `import logging` must appear EXACTLY ONCE — at the
        module-level import block at line 23. Any inner import shadows."""
        # Count standalone `import logging` (not `import logging as ...`
        # or `from logging import ...`). Compile-time scope analysis
        # treats either form identically as making `logging` local, so
        # we forbid both.
        pattern = re.compile(r"^\s*import\s+logging(\s|$|#)", re.MULTILINE)
        matches = pattern.findall(_NGB_HEAD_SRC)
        assert len(matches) == 1, (
            f"Expected exactly 1 `import logging` (module-level, line 23) "
            f"in ngboost_head.py; found {len(matches)}. Inner imports "
            f"shadow the module name and resurface the UnboundLocalError."
        )

    def test_no_logging_assignment_in_ngboost_head_module(self):
        """Reject any `logging = ...` rebinding that would also shadow."""
        # `logging = something` or `logging: T = something` at any indent.
        pattern = re.compile(
            r"^\s*logging\s*(:\s*\S+\s*)?=\s*(?!=)", re.MULTILINE,
        )
        matches = pattern.findall(_NGB_HEAD_SRC)
        assert not matches, (
            f"Found `logging = ...` assignment(s) in ngboost_head.py "
            f"({len(matches)} matches). Per §5.13.5 the `logging` name "
            f"must only refer to the stdlib module; rebinding it shadows."
        )
