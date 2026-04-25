"""Strategy config loader — self-contained, no external dependencies."""
import json
from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parent.parent

BULL_CALM     = "BULL_CALM"
BULL_VOLATILE = "BULL_VOLATILE"
CHOPPY        = "CHOPPY"
BEAR          = "BEAR"
REGIMES       = [BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR]


def load_config(path: Path | None = None) -> dict:
    p = path or (STRATEGY_DIR / "strategy_config.json")
    with open(p) as f:
        return json.load(f)


def split_date_parts(date_text: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in date_text.split("-"))


def artifact_path(filename: str) -> Path:
    """Return the canonical path for a strategy artifact (artifacts/ subdir).

    Audit #89: tolerate filenames that already include an `artifacts/`
    prefix (some configs supply "artifacts/spy-gmm-regime.json"). Strip
    the prefix before joining so we don't end up with `…/artifacts/artifacts/…`.
    """
    fn = str(filename)
    if fn.startswith("artifacts/") or fn.startswith("artifacts\\"):
        fn = fn[len("artifacts/"):]
    return STRATEGY_DIR / "artifacts" / fn


# ── Aliases for callers migrating from common/ ────────────────────────────────

def load_strategy_config(path: Path | None = None) -> dict:
    """Alias for load_config — compatible with common.config.load_strategy_config."""
    return load_config(path)


def build_model_path(strategy_dir: Path, symbol: str, filename: str) -> Path:
    """Return the canonical model artifact path for a symbol."""
    return strategy_dir / "models" / symbol / filename


def universe_floor_spec(config: dict) -> tuple[str, float]:
    """Return (floor_type, threshold) for universe admission.

    Config shape::

        ranking:
          universe_floor:
            type:      "none" | "sharpe" | "ic"   # default "none"
            threshold: 0.0                          # numeric floor

    Returns ("none", 0.0) if absent.  Unknown types fall back to "none"
    with a runtime warning logged by FilterUniverseFloorTask.
    """
    block = (
        config.get("ranking", {})
              .get("universe_floor", {})
    )
    floor_type = str(block.get("type", "none")).lower()
    threshold  = float(block.get("threshold", 0.0))
    return floor_type, threshold

