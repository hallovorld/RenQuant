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
    """Return the canonical path for a strategy artifact (artifacts/ subdir)."""
    return STRATEGY_DIR / "artifacts" / filename
