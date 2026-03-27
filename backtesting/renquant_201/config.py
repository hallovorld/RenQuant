"""LEAN-local config helpers — self-contained for Docker compatibility."""
import json
from pathlib import Path

_STRATEGY_DIR = Path(__file__).resolve().parent


def load_strategy_config(config_path: Path | None = None) -> dict:
	path = config_path or (_STRATEGY_DIR / "strategy_config.json")
	with path.open() as f:
		return json.load(f)


def split_date_parts(date_text: str) -> tuple[int, int, int]:
	return tuple(int(part) for part in date_text.split("-"))
