import json
from pathlib import Path


def load_strategy_config(config_path: Path) -> dict:
    """Load a strategy_config.json file."""
    with config_path.open() as f:
        return json.load(f)


def split_date_parts(date_text: str) -> tuple[int, int, int]:
    """Split 'YYYY-MM-DD' into (year, month, day) integers."""
    return tuple(int(part) for part in date_text.split("-"))


def build_model_path(repo_root: Path, strategy_name: str, model_name: str) -> Path:
    """Return the path to a model artifact: repo_root/backtesting/<strategy>/<model>.json"""
    return repo_root / "backtesting" / strategy_name / f"{model_name}.json"
