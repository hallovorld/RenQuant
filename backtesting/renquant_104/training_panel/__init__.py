"""Panel-LTR training pipeline (Stage 1+ of the scoring redesign).

Ships alongside per-stock `training/` — no existing code is modified. The
panel pipeline is switched on at inference via `ranking.model_type` in
strategy_config.json. See doc/research/scoring-research.md for the full design.
"""
