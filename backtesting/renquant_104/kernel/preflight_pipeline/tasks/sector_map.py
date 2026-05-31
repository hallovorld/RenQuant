"""P-SECTOR-MAP — every buyable ticker must have sector metadata.

Migrated from kernel.preflight._check_sector_map_coverage.
"""
from __future__ import annotations

from kernel.preflight import PreflightCheck  # noqa: PLC0415 (legacy bridge)

from ..base import PreflightTask
from ..ctx import PreflightContext


class SectorMapCoverageTask(PreflightTask):
    """P-SECTOR-MAP — panel-LTR uses sector metadata for sector-neutralized
    features, relative-strength vs sector ETF, and QP sector caps. Missing
    entries silently turn a stock into "no sector" and let it avoid
    sector-aware controls. Sell-only runs are exempt; this check protects new
    entries, not risk exits.
    """

    check_name = "P-SECTOR-MAP"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        if not self._coverage_required(ctx.config):
            return PreflightCheck(
                self.check_name, "soft", True,
                "sector-map coverage not required for this strategy/config",
            )
        return self._evaluate_coverage(ctx)

    @staticmethod
    def _coverage_required(config: dict) -> bool:
        panel_enabled = bool(
            config.get("ranking", {})
            .get("panel_scoring", {})
            .get("enabled", False)
        )
        return bool(
            config.get("risk", {}).get("require_sector_map_for_buys", panel_enabled)
        )

    def _evaluate_coverage(self, ctx: PreflightContext) -> PreflightCheck:
        config = ctx.config
        normalized_mode = str(ctx.run_mode or "").lower().replace("_", "-")
        watchlist = list(config.get("watchlist") or [])
        sector_map = config.get("sector_map", {}) or {}
        benchmark = config.get("benchmark", "SPY")
        buyable = [t for t in watchlist if t != benchmark]
        missing = sorted(
            t for t in buyable
            if not isinstance(sector_map.get(t), str) or not sector_map.get(t)
        )
        sectors = sorted({v for v in sector_map.values()
                          if isinstance(v, str) and v})
        sector_etfs = config.get("sector_etf_map", {}) or {}
        unmapped_sectors = sorted(s for s in sectors if s not in sector_etfs)
        details = {
            "watchlist_size": len(watchlist),
            "buyable_size": len(buyable),
            "missing_count": len(missing),
            "missing_sample": missing[:20],
            "unmapped_sectors": unmapped_sectors[:20],
            "run_mode": ctx.run_mode,
        }
        if missing or unmapped_sectors:
            msg = (
                f"sector metadata incomplete: {len(missing)}/{len(buyable)} "
                f"buyable watchlist tickers missing sector_map entries "
                f"(sample={missing[:10]}), {len(unmapped_sectors)} sector(s) "
                f"missing sector_etf_map entries (sample={unmapped_sectors[:10]}). "
                "Missing sector metadata disables relative-strength context and "
                "QP sector caps for those names."
            )
            if normalized_mode.startswith("sell-only"):
                return PreflightCheck(
                    self.check_name, "soft", True,
                    msg + " Sell-only risk exits are allowed; new buys remain blocked.",
                    details=details,
                )
            return PreflightCheck(
                self.check_name, "hard", False, msg, details=details,
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"sector coverage OK ({len(buyable)} buyable tickers, "
            f"{len(sectors)} sectors mapped)",
            details=details,
        )
