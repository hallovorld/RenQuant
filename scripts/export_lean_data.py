#!/usr/bin/env python
"""Export cached OHLCV parquet data into LEAN daily equity files.

Usage::

    python scripts/export_lean_data.py --symbol NVDA
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent


def build_daily_lines(df: pd.DataFrame) -> list[str]:
	lines = []
	for timestamp, row in df.iterrows():
		date_text = pd.Timestamp(timestamp).strftime("%Y%m%d 00:00")
		open_price = int(round(float(row["open"]) * 10000))
		high_price = int(round(float(row["high"]) * 10000))
		low_price = int(round(float(row["low"]) * 10000))
		close_price = int(round(float(row["close"]) * 10000))
		volume = int(round(float(row["volume"])))
		lines.append(
			f"{date_text},{open_price},{high_price},{low_price},{close_price},{volume}"
		)
	return lines


def export_symbol(symbol: str) -> tuple[Path, Path, Path]:
	symbol_lower = symbol.lower()
	parquet_path = REPO_ROOT / "data" / "ohlcv" / symbol.upper() / "1d.parquet"
	if not parquet_path.exists():
		# Fallback: notebook working directory caches to Notebooks/data/ohlcv/
		notebook_path = REPO_ROOT / "Notebooks" / "data" / "ohlcv" / symbol.upper() / "1d.parquet"
		if notebook_path.exists():
			parquet_path = notebook_path
		else:
			raise FileNotFoundError(f"Cached parquet not found: {parquet_path}")

	df = pd.read_parquet(parquet_path).copy()
	if not isinstance(df.index, pd.DatetimeIndex):
		df.index = pd.to_datetime(df.index)
	df = df.sort_index()
	df = df.dropna(subset=["open", "high", "low", "close", "volume"])
	if df.empty:
		raise RuntimeError(f"No rows found in {parquet_path}")

	data_dir = REPO_ROOT / "backtesting" / "data" / "equity" / "usa"
	daily_zip_path = data_dir / "daily" / f"{symbol_lower}.zip"
	map_file_path = data_dir / "map_files" / f"{symbol_lower}.csv"
	factor_file_path = data_dir / "factor_files" / f"{symbol_lower}.csv"

	daily_zip_path.parent.mkdir(parents=True, exist_ok=True)
	map_file_path.parent.mkdir(parents=True, exist_ok=True)
	factor_file_path.parent.mkdir(parents=True, exist_ok=True)

	with zipfile.ZipFile(daily_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
		zf.writestr(f"{symbol_lower}.csv", "\n".join(build_daily_lines(df)) + "\n")

	start_date = df.index.min().strftime("%Y%m%d")
	map_file_path.write_text(f"{start_date},{symbol_lower},Q\n20501231,{symbol_lower},Q\n")
	first_close = float(df["close"].iloc[0])
	factor_file_path.write_text(f"{start_date},1,1,{first_close}\n")

	return daily_zip_path, map_file_path, factor_file_path


def main() -> int:
	parser = argparse.ArgumentParser(description="Export cached parquet OHLCV into LEAN equity daily files")
	parser.add_argument("--symbol", required=True, help="Ticker symbol, for example NVDA")
	args = parser.parse_args()

	daily_zip_path, map_file_path, factor_file_path = export_symbol(args.symbol)
	print(f"Daily zip   : {daily_zip_path}")
	print(f"Map file    : {map_file_path}")
	print(f"Factor file : {factor_file_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())