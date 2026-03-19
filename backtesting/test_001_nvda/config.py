import json
from pathlib import Path


STRATEGY_DIR = Path(__file__).resolve().parent
CONFIG_PATH = STRATEGY_DIR / "strategy_config.json"


def load_strategy_config(config_path: Path | None = None) -> dict:
	path = config_path or CONFIG_PATH
	with path.open() as config_file:
		return json.load(config_file)


def split_date_parts(date_text: str) -> tuple[int, int, int]:
	return tuple(int(part) for part in date_text.split("-"))


def build_model_path(repo_root: Path, model_name: str) -> Path:
	return repo_root / "backtesting" / "test_001_nvda" / f"{model_name}.json"