"""Runner context-artifact loading — runner.py decomposition (make_context).

EXTRACTED 2026-06-13 from adapters/runner.py make_context() (eng plan
S2 item 5 god-method decomposition). Loads the GMM regime artifact,
the correlation artifact (v2-schema aware, malformed→None), and the
earnings calendar from the strategy artifacts dir. Pure-ish (filesystem
reads only); returns (gmm, corr, earnings). DRPH-gated.
"""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("adapters.runner")


def load_context_artifacts(strategy_dir, config: dict) -> tuple[Any, Any, Any]:
    """Return (gmm, corr, earnings) loaded from the strategy artifacts dir.
    Malformed corr/earnings JSON is logged and treated as missing (None) —
    the live trade must not abort on a bad artifact file."""
    from kernel.regime import load_gmm_artifact  # noqa: PLC0415
    from kernel.config import artifact_path       # noqa: PLC0415

    regime_cfg = config.get("regime", {})
    artifacts_dir = strategy_dir / "artifacts"
    if not artifacts_dir.exists():
        artifacts_dir = strategy_dir

    # 2026-05-11 sim/prod isolation: defaults relocated to prod/.
    # Sim configs override these keys to sim/<file>.
    gmm_path  = artifacts_dir / regime_cfg.get("gmm_artifact", "prod/spy-gmm-regime.json")
    gmm       = load_gmm_artifact(gmm_path)

    corr_path = artifacts_dir / regime_cfg.get("correlation_artifact", "prod/watchlist-correlation.json")
    # 2026-05-09 audit fix (RU-JSON-MALFORMED): pre-fix, malformed JSON
    # in corr/earnings artifacts raised JSONDecodeError straight up
    # → adapter __init__ crashed → live trade aborted with cryptic
    # traceback. Now: malformed file logged + treated as missing
    # (downstream tasks already handle None gracefully).
    #
    # 2026-05-10 audit §5.13.5: also unwrap v2-schema correlation
    # artifact (matrix + as_of_date). Live runner is in live mode,
    # so the leakage guard is a no-op — but we still parse the v2
    # schema correctly so downstream `corr.get(t, {})` keeps working.
    from kernel.walk_forward import parse_correlation_artifact  # noqa: PLC0415
    try:
        corr_raw = json.loads(corr_path.read_text()) if corr_path.exists() else None
        corr, _corr_as_of = parse_correlation_artifact(corr_raw)
        if not corr:
            corr = None
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("corr artifact %s malformed (%s) — treating as missing", corr_path, exc)
        corr = None

    # 2026-05-11 sim/prod isolation + audit fix (was hardcoded, now config-driven).
    earn_path = artifacts_dir / regime_cfg.get("earnings_artifact", "prod/earnings-calendar.json")
    try:
        earnings = json.loads(earn_path.read_text()) if earn_path.exists() else None
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("earnings artifact %s malformed (%s) — treating as missing", earn_path, exc)
        earnings = None

    return gmm, corr, earnings
