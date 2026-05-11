"""Regression guard for the canonical exit_types module.

Catalogues what each existing task currently uses, then asserts the new
canonical module produces the same sets. Prevents accidental drift
during the refactor (CLAUDE.md §5.13.5 — one business decision, one
source-of-truth).
"""
from __future__ import annotations

import sys
from pathlib import Path

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


class TestCanonicalBaseSets:
    """The base sets — atomic groups that compose the larger sets."""

    def test_path_rule_core_set(self):
        from kernel.exit_types import PATH_RULE_CORE
        assert isinstance(PATH_RULE_CORE, frozenset)
        assert PATH_RULE_CORE == frozenset({
            "stop_loss", "trailing_stop", "single_day_loss", "max_hold",
        })

    def test_path_rule_synonyms_set(self):
        from kernel.exit_types import PATH_RULE_SYNONYMS
        assert isinstance(PATH_RULE_SYNONYMS, frozenset)
        # Historical / legacy / variant names for the path-rule exits
        assert "trailing_stop_loss" in PATH_RULE_SYNONYMS
        assert "sdl" in PATH_RULE_SYNONYMS
        assert "gap_down" in PATH_RULE_SYNONYMS
        assert "max_hold_days" in PATH_RULE_SYNONYMS

    def test_portfolio_risk_set(self):
        from kernel.exit_types import PORTFOLIO_RISK
        assert PORTFOLIO_RISK == frozenset({
            "rotation", "kelly_trim", "joint_sell", "joint_rotation",
        })

    def test_model_driven_set(self):
        from kernel.exit_types import MODEL_DRIVEN
        assert MODEL_DRIVEN == frozenset({"model_sell", "panel_conviction"})


class TestDerivedSets:
    """Derived sets — unions of bases, used by specific tasks."""

    def test_meta_label_veto_eligible(self):
        """MetaLabelVetoTask: only mechanical price-rule exits (PATH_RULE_CORE).
        NOT including synonyms — meta-label was trained on core names."""
        from kernel.exit_types import META_LABEL_VETO_ELIGIBLE, PATH_RULE_CORE
        assert META_LABEL_VETO_ELIGIBLE == PATH_RULE_CORE

    def test_panel_veto_bypass(self):
        """task_panel_veto.RISK_EXIT_TYPES: PATH + PORTFOLIO + synonyms."""
        from kernel.exit_types import (PANEL_VETO_BYPASS, PATH_RULE_CORE,
                                       PORTFOLIO_RISK, PATH_RULE_SYNONYMS)
        # Must include the original set members
        expected = PATH_RULE_CORE | PORTFOLIO_RISK | PATH_RULE_SYNONYMS
        assert PANEL_VETO_BYPASS == expected

    def test_per_bar_cap_exempt(self):
        """task_limit_sells._RISK_EXIT_TYPES: PATH + PORTFOLIO + synonyms."""
        from kernel.exit_types import (PER_BAR_CAP_EXEMPT, PATH_RULE_CORE,
                                       PORTFOLIO_RISK, PATH_RULE_SYNONYMS)
        expected = PATH_RULE_CORE | PORTFOLIO_RISK | PATH_RULE_SYNONYMS
        assert PER_BAR_CAP_EXEMPT == expected

    def test_per_bar_cap_subject(self):
        """task_limit_sells._SOFT_SELL_TYPES: model-driven only."""
        from kernel.exit_types import PER_BAR_CAP_SUBJECT, MODEL_DRIVEN
        assert PER_BAR_CAP_SUBJECT == MODEL_DRIVEN

    def test_path_driven_legacy(self):
        """task_sell.PATH_DRIVEN_EXIT_TYPES: core + portfolio-risk (no synonyms)."""
        from kernel.exit_types import (PATH_DRIVEN_LEGACY, PATH_RULE_CORE,
                                       PORTFOLIO_RISK)
        assert PATH_DRIVEN_LEGACY == PATH_RULE_CORE | {"kelly_trim", "rotation"}

    def test_post_stop_cooldown_triggers(self):
        """task_post_stop_cooldown.DEFAULT_STOP_EXIT_TYPES: path-rule kernel
        but only the price-rule subset (no max_hold; max_hold is a time exit
        not a price stop)."""
        from kernel.exit_types import POST_STOP_COOLDOWN_TRIGGERS
        assert POST_STOP_COOLDOWN_TRIGGERS == frozenset({
            "trailing_stop", "trailing_stop_loss",
            "stop_loss", "single_day_loss", "sdl", "gap_down",
        })


class TestPrincipleCompliance:
    """CLAUDE.md §5.13.5 — one business decision, one function."""

    def test_module_exports_only_frozensets(self):
        from kernel import exit_types
        # Every exported symbol should be a frozenset (no parallel impl)
        for name in exit_types.__all__:
            val = getattr(exit_types, name)
            assert isinstance(val, frozenset), \
                f"{name} should be frozenset, got {type(val).__name__}"

    def test_all_referenced_names_documented(self):
        """Every name in __all__ should have a docstring entry naming it."""
        from kernel import exit_types
        doc = exit_types.__doc__ or ""
        for name in exit_types.__all__:
            assert name in doc, f"{name} missing from module docstring"


class TestBackwardCompatibility:
    """Old import sites continue to work post-refactor (no breakage)."""

    def test_task_meta_label_veto_PATH_RULE_EXITS(self):
        # Old symbol stays — points at canonical
        from kernel.meta_label.task_meta_label_veto import _PATH_RULE_EXITS
        from kernel.exit_types import META_LABEL_VETO_ELIGIBLE
        assert _PATH_RULE_EXITS == META_LABEL_VETO_ELIGIBLE
