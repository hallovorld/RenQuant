import sys, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
REPO=Path("/Users/renhao/git/github/RenQuant"); STRAT=REPO/"backtesting/renquant_104"
for p in (REPO, REPO/"scripts", STRAT):
    if str(p) not in sys.path: sys.path.insert(0,str(p))

# 1. Build sentiment-velocity features (point-in-time, per ticker)
rows=[]
for f in glob.glob(str(REPO/"data/news_sentiment_alpaca/*.parquet")):
    d=pd.read_parquet(f).sort_values("date")
    if d.empty: continue
    d=d.rename(columns={"symbol":"ticker"})
    d["date"]=pd.to_datetime(d["date"])
    d["log_art"]=np.log1p(d["n_articles"])
    d["sent_vel_5"]   = d["mean_sentiment"] - d["mean_sentiment"].shift(5)
    roll = d["mean_sentiment"].rolling(20, min_periods=10)
    d["sent_surprise"]= (d["mean_sentiment"]-roll.mean())/(roll.std()+1e-9)
    d["news_accel_5"] = d["log_art"] - d["log_art"].shift(5)
    d["posshare_vel_5"]= d["sentiment_pos_share"] - d["sentiment_pos_share"].shift(5)
    rows.append(d[["ticker","date","sent_vel_5","sent_surprise","news_accel_5","posshare_vel_5"]])
sv=pd.concat(rows, ignore_index=True)
print(f"sentiment-velocity rows: {len(sv)}  tickers: {sv['ticker'].nunique()}")

# 2. Panel: keys + forward labels
pan=pd.read_parquet(REPO/"data/transformer_v4_wl200_clean.parquet",
                    columns=["ticker","date","fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"])
pan["date"]=pd.to_datetime(pan["date"])
oos=pan[(pan["date"]>="2024-02-01")&(pan["date"]<="2026-02-10")].copy()

# 3. Regimes for OOS dates -> BULL_CALM
from scripts.analyze_manifest_sanity_placebo import build_regime_series
reg=build_regime_series(oos["date"].unique(), strategy_dir=STRAT)
rmap=dict(zip(pd.to_datetime(reg["date"]).dt.normalize(), reg["regime"]))
oos["regime"]=oos["date"].dt.normalize().map(rmap)
bc=oos[oos["regime"]=="BULL_CALM"].copy()
print(f"BULL_CALM OOS rows: {len(bc)}  dates: {bc['date'].nunique()}")

# 4. Merge velocity features
m=bc.merge(sv, on=["ticker","date"], how="inner")
print(f"merged (have sentiment) rows: {len(m)}  coverage: {len(m)/len(bc):.0%}")

feats=["sent_vel_5","sent_surprise","news_accel_5","posshare_vel_5"]
# combined z-score
for c in feats:
    m[c+"_z"]=m.groupby("date")[c].transform(lambda s:(s-s.mean())/(s.std()+1e-9))
m["combo"]=m[[c+"_z" for c in feats]].mean(axis=1)

def cs_ic(df, fcol, ycol):
    ics=[]
    for _,g in df.groupby("date"):
        g=g.dropna(subset=[fcol,ycol])
        if len(g)>=5:
            r=spearmanr(g[fcol],g[ycol]).correlation
            if np.isfinite(r): ics.append(r)
    return np.mean(ics) if ics else np.nan, len(ics)

def placebo_ic(df, fcol, ycol):
    # shift feature +60 trading rows within ticker -> misalign
    df=df.sort_values(["ticker","date"]).copy()
    df["_lag"]=df.groupby("ticker")[fcol].shift(60)
    return cs_ic(df, "_lag", ycol)[0]

mid=m["date"].median()
print("\n=== sentiment-velocity range-finder (BULL_CALM OOS) ===")
print(f"{'feature':16} {'horizon':8} {'realIC':>8} {'placebo':>8} {'NET':>8} {'H1':>8} {'H2':>8}")
for fcol in feats+["combo"]:
    for y in ["fwd_5d_excess","fwd_20d_excess","fwd_60d_excess"]:
        real,nd=cs_ic(m,fcol,y); pb=placebo_ic(m,fcol,y)
        h1,_=cs_ic(m[m["date"]<=mid],fcol,y); h2,_=cs_ic(m[m["date"]>mid],fcol,y)
        net=real-pb if np.isfinite(real) and np.isfinite(pb) else np.nan
        print(f"{fcol:16} {y[:7]:8} {real:+8.4f} {pb:+8.4f} {net:+8.4f} {h1:+8.4f} {h2:+8.4f}")
