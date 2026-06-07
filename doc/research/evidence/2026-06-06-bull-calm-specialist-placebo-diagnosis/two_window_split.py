import argparse
import sys
from pathlib import Path
import pandas as pd


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
parser.add_argument("--github-root", type=Path, default=None)
args = parser.parse_args()

REPO = args.repo_root.resolve()
GITHUB_ROOT = args.github_root.resolve() if args.github_root else REPO.parent
STRAT = REPO / "backtesting/renquant_104"
for p in (
    REPO,
    REPO / "scripts",
    STRAT,
    GITHUB_ROOT / "renquant-pipeline/src",
    GITHUB_ROOT / "renquant-backtesting/src",
    GITHUB_ROOT / "renquant-common/src",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from scripts.analyze_manifest_sanity_placebo import (
    _load_artifact_payload, _load_sanity_panel, _score_manifest_sanity,
    _sanity_model_label_col, summarize_ic, shift_diagnostics, build_regime_series,
)
ART=STRAT/"artifacts/walkforward_bull_calm_specialist/bull_calm/2025-11-24/panel-ltr.json"
MAN=STRAT/"artifacts/sim/walkforward_manifest_v2_20260606_per_regime_bull_calm.json"
art=_load_artifact_payload(ART)
label=_sanity_model_label_col(art); feat=list(art["feature_cols"])
panel,_=_load_sanity_panel(feat,label)
panel=panel.dropna(subset=[label]).copy(); panel["date"]=pd.to_datetime(panel["date"])
distinct=sorted(panel["date"].unique())
val_cut=pd.Timestamp(distinct[int(len(distinct)*0.8)])
val=panel[panel["date"]>val_cut].copy()
mu,_=_score_manifest_sanity(val,feat,MAN,ART,art,panel_history=panel)
val=val.loc[mu.index].copy(); mu=mu.loc[val.index]
reg=build_regime_series(val["date"].unique(), strategy_dir=STRAT)
reg_map=dict(zip(pd.to_datetime(reg["date"]).dt.normalize(), reg["regime"]))
val["regime"]=pd.to_datetime(val["date"]).dt.normalize().map(reg_map)
bc=val[val["regime"]=="BULL_CALM"].copy(); mu_bc=mu.loc[bc.index]
dts=sorted(bc["date"].unique()); mid=dts[len(dts)//2]
print(f"BULL_CALM OOS dates: {len(dts)}  ({pd.Timestamp(dts[0]).date()} → {pd.Timestamp(dts[-1]).date()})  split @ {pd.Timestamp(mid).date()}")
def half(name, sub_idx):
    sub=bc.loc[sub_idx]; m=mu_bc.loc[sub_idx]
    real=summarize_ic(m, sub[label].clip(-0.5,0.5), sub["date"])
    sh=shift_diagnostics(panel, sub, m, label, shifts=[60])
    row=sh[0] if isinstance(sh,list) and sh else (sh.get("60") if isinstance(sh,dict) else {})
    ar=row.get("aligned_real_ic"); pb=row.get("model_placebo_ic")
    net=(ar-pb) if (ar is not None and pb is not None) else None
    print(f"  {name}: n_dates={real['n_dates']:>3}  real_meanIC={real['mean_ic']:+.4f}  aligned60={ar:+.4f}  placebo60={pb:+.4f}  NET={net:+.4f}" if ar is not None else f"  {name}: real={real['mean_ic']}")
h1=bc[bc["date"]<=mid].index; h2=bc[bc["date"]>mid].index
print("=== specialist BULL_CALM tilt: two-window stability ===")
half("H1", h1); half("H2", h2)
half("FULL", bc.index)
