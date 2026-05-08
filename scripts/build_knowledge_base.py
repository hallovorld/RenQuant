#!/usr/bin/env python
"""Build a local ChromaDB knowledge base from project docs + external research.

Usage:
    python scripts/build_knowledge_base.py          # build / rebuild
    python scripts/build_knowledge_base.py --query "transformer IC US stocks"
    python scripts/build_knowledge_base.py --query "XGBoost walk-forward" --n 5

Collections:
    project_docs    — all doc/**/*.md + CLAUDE.md (internal knowledge)
    experiments     — failed-experiments-log.md parsed per experiment
    external        — research findings added manually or via --add-source

Storage: knowledge/chromadb/   (gitignored, rebuild anytime)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR    = REPO_ROOT / "knowledge" / "chromadb"
KB_DIR.mkdir(parents=True, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────────

def chunk_markdown(text: str, source: str, max_chars: int = 1200) -> list[dict]:
    """Split markdown into chunks at section boundaries."""
    chunks = []
    current_title = "intro"
    current_buf   = []

    def flush():
        body = "\n".join(current_buf).strip()
        if len(body) > 80:
            chunks.append({"text": f"# {current_title}\n\n{body}",
                            "source": source, "section": current_title})
        current_buf.clear()

    for line in text.splitlines():
        if re.match(r"^#{1,3} ", line):
            flush()
            current_title = line.lstrip("#").strip()
        else:
            current_buf.append(line)
        # Hard split on very long sections
        joined = "\n".join(current_buf)
        if len(joined) > max_chars:
            flush()

    flush()
    return chunks


def doc_id(text: str, source: str) -> str:
    return hashlib.md5(f"{source}::{text[:120]}".encode()).hexdigest()


# ── external research knowledge (from today's research session) ────────────

EXTERNAL_ENTRIES = [
    {
        "id": "ext_gu_kelly_xiu_2020",
        "text": (
            "Gu, Kelly, Xiu (2020 RFS) 'Empirical Asset Pricing via Machine Learning'. "
            "Universe: ~30,000 US stocks 1957-2016, MONTHLY predictions. "
            "94 firm characteristics (fundamental + momentum + technical). "
            "Best models: neural net OOS R²≈0.35%, long-short decile Sharpe≈1.35. "
            "OLS outperformed by nonlinear ML BUT requires 94 features + 60 years. "
            "Pure OHLCV at daily frequency gives much lower IC (~0.02-0.03). "
            "Top features by importance: short-term reversal, 3/6/9/12m momentum, "
            "earnings-to-price, book-to-market, market cap, max daily return, idiosyncratic vol. "
            "Conclusion: fundamental features are essential for US stock IC above 0.03."
        ),
        "tags": "benchmark,us_stocks,ic,fundamental,ols,neural_net,monthly",
    },
    {
        "id": "ext_cakici_2023",
        "text": (
            "Cakici et al. (2023) 'Machine Learning Goes Global', Journal of Economic Dynamics. "
            "46 countries tested. Key finding: OLS outperforms 8 ML methods on US LARGE-CAP stocks. "
            "ML advantage concentrates in SMALL-CAPS: alpha 2.37% vs 0.99% annualized (small vs large). "
            "Estimation error in limited samples degrades nonlinear models faster than linear ones. "
            "Implication: for top-500 US stocks, linear model is hard to beat with ML. "
            "ML worth it only for Russell 2000 / small-cap universe."
        ),
        "tags": "benchmark,us_stocks,large_cap,ols_beats_ml,small_cap,ml",
    },
    {
        "id": "ext_master_2024",
        "text": (
            "MASTER: Market-Guided Stock Transformer (Li et al., AAAI 2024). "
            "GitHub: SJTU-DMTai/MASTER. Cross-stock attention with market guidance. "
            "Results on Chinese A-shares: CSI300 IC=0.064, CSI800 IC=0.052. "
            "Outperforms XGBoost by 25-30% on A-shares. "
            "NOT tested on US stocks. A-shares are less efficient = higher IC ceiling. "
            "Architecture: LSTM backbone + cross-stock attention + market-level feature selection. "
            "Requires 1000+ stocks for stable cross-stock correlation learning. "
            "With <500 US stocks the cross-stock attention provides little benefit."
        ),
        "tags": "transformer,cross_stock_attention,a_shares,csi300,ic=0.064,master,lstm",
    },
    {
        "id": "ext_itransformer_2024",
        "text": (
            "iTransformer (Liu et al., ICLR 2024). GitHub: thuml/iTransformer. "
            "Inverts attention: each VARIATE's full T-step history = one token, attend across variates. "
            "ETTh1 benchmark (7 sensor variates): MSE=0.454 pred_len=96. "
            "Our reproduction: MSE=0.386 (better, early stopping). "
            "Our ETTh1 cross-variate IC = 0.77 (very high because temperature is predictable). "
            "Stock application: val_ic ≈ +0.018-0.025 (816 tickers), train_ic up to 0.10 = overfitting. "
            "Root cause: cross-ticker attention memorizes training-period inter-stock correlations "
            "that don't persist across market regimes. N=816 still too small for stable patterns. "
            "Conclusion: iTransformer needs 3000+ stocks and 10+ years to work for stock ranking."
        ),
        "tags": "itransformer,etth1,transformer,cross_variate,overfitting,ic=0.025,architecture",
    },
    {
        "id": "ext_patchtst_2023",
        "text": (
            "PatchTST (Nie et al., ICLR 2023). GitHub: yuqinie98/PatchTST. "
            "Key idea: divide time series into overlapping patches, attend across patches (temporal axis). "
            "ETTh1 benchmark: MSE=0.370 pred_len=96 (better than iTransformer 0.454). "
            "Our reproduction: MSE=0.377 (matches paper within 2%). Architecture verified correct. "
            "Stock application (816 tickers, seq_len=60): val_ic ≈ +0.020 (seed42). "
            "Train_ic up to 0.07 vs val_ic 0.020 — less overfitting than iTransformer (gap=0.05 vs 0.12). "
            "Patch mechanism helps regularization vs plain transformer. "
            "Still overfit: stocks change regimes faster than patches can generalize."
        ),
        "tags": "patchtst,etth1,patch,temporal,overfitting,ic=0.020,architecture",
    },
    {
        "id": "ext_portfolio_master_sp500",
        "text": (
            "PortfolioMASTER Transformer on S&P 500 (arXiv 2510.14156, 2025). "
            "Tested on S&P 500 with multiple loss functions. "
            "Spearman IC range: 0.073-0.077 (RankNet loss best at 0.0767). "
            "This is much higher than typical US large-cap IC — may reflect favorable test period "
            "or lack of walk-forward validation. "
            "URL: https://arxiv.org/html/2510.14156v1 "
            "Worth investigating: what universe, data period, and evaluation methodology they use."
        ),
        "tags": "transformer,sp500,ic=0.077,ranknet,benchmark,us_stocks",
    },
    {
        "id": "ext_dlinear_2023",
        "text": (
            "DLinear / NLinear (Zeng et al., AAAI 2023) 'Are Transformers Effective for Time Series?' "
            "Simple linear decomposition (trend + residual, one Linear layer each) beats transformer "
            "on 6/7 standard forecasting benchmarks. "
            "Key finding: plain transformer is NOT necessary for time series prediction at this scale. "
            "Implication for stocks: our Linear baseline IC=+0.032 is hard to beat with complexity. "
            "Paper shows that on short-horizon time series, attention's inductive bias is often harmful."
        ),
        "tags": "linear,dlinear,transformer_effectiveness,benchmark,time_series",
    },
    {
        "id": "ext_chen_design_choices",
        "text": (
            "Chen, Hanauer, Kalsbach 'Design Choices, ML, and Cross-Section of Stock Returns' (2024). "
            "1,056 model variants tested. Monthly top-minus-bottom return range: 0.13% to 1.98%. "
            "Non-standard error from design choices exceeds statistical bootstrap error by 59%. "
            "Nonlinear ML beats OLS ONLY under: expanding window + continuous target + post-pub features. "
            "Conclusion: implementation details matter more than model architecture choice. "
            "Source: https://quantpedia.com/design-choices-in-ml-and-the-cross-section-of-stock-returns/"
        ),
        "tags": "design_choices,ols,ml,us_stocks,monthly,benchmark",
    },
    {
        "id": "ext_analyst_revisions_ic",
        "text": (
            "Mill Street Research: Analyst estimate revisions IC = 0.23 (global, monthly). "
            "6000+ global stocks, 40% US. Monthly IC 0.23 is extremely high. "
            "Source: https://www.millstreetresearch.com/ "
            "For daily cross-sectional models: earnings revision signals add ~5-15bp daily IC "
            "when combined with price/volume features. "
            "Features: FY1 EPS change direction, breadth of upward vs downward revisions. "
            "Data source: IBES estimates (paid), or approximated from earnings surprise post-announcement. "
            "Key implication: adding analyst revisions likely most impactful single feature addition "
            "to get from IC=0.03 to IC=0.05+ on US stocks."
        ),
        "tags": "analyst_revisions,ic=0.23,fundamental,us_stocks,monthly,feature_importance",
    },
    {
        "id": "ext_qlib_benchmark",
        "text": (
            "Qlib (microsoft/qlib) benchmark results on Chinese A-shares (CSI300/CSI500). "
            "Alpha158: 158 features from OHLCV. Alpha360: 360 features. "
            "IC on CSI300: Linear≈0.034, XGBoost≈0.050, DoubleEnsemble≈0.052, TRA≈0.044. "
            "All results are for CHINESE A-SHARES, NOT US stocks. "
            "US stocks expected to be 25-40% lower IC due to market efficiency. "
            "For US stocks with 103-816 tickers: expect IC=0.022-0.037 with Alpha158. "
            "Our actual result: 291-ticker OLS IC=+0.031 matches this prediction exactly. "
            "Source: https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md"
        ),
        "tags": "qlib,alpha158,csi300,a_shares,benchmark,ic,chinese_stocks",
    },
    {
        "id": "ext_renquant_e34_universe",
        "text": (
            "RenQuant E34 (2026-05-07): Phase 1 universe expansion 103→816 tickers (R1K Stage1 filter). "
            "Result: Ridge OLS val_ic=+0.0010, test_ic=+0.0038. Near zero. "
            "Previous 291-ticker baseline: val_ic=+0.031. "
            "Root cause: alpha158 OHLCV features have strong signal ONLY for tech/growth stocks. "
            "Adding financials/industrials/REITs/consumer staples dilutes signal. "
            "Transfer coefficient collapse: breadth × 8 but IC × 0.1 = net WORSE. "
            "Fix: cluster-based admission (select tickers where alpha158 actually has IC), "
            "or add fundamental features that work across sectors."
        ),
        "tags": "renquant,e34,universe_expansion,ic=0,transfer_coefficient,alpha158_limitation",
    },
    {
        "id": "ext_renquant_walk_forward",
        "text": (
            "RenQuant walk-forward results (E27, E29, 2026-05-07). "
            "Production 27-feat XGB: mean alpha vs SPY = -15.62% (E27, 3-cut). "
            "alpha158 Linear: mean alpha vs SPY = -2.0 pts (5-cut). "
            "alpha158 XGB: mean alpha vs SPY = -2.56 pts (5-cut, fixed dispatch bug). "
            "All models lose to SPY long-only at 103-ticker breadth. "
            "Single-cut 27-mo Sharpe 0.68 is regime-smoothing artifact. "
            "T4 (2025 tariff crash) is only cut where alpha158 outperforms SPY (+9 pts). "
            "Models are defensive low-vol factors, not true alpha generators."
        ),
        "tags": "renquant,walk_forward,sharpe,alpha,spy,no_go,alpha158",
    },
]


def build(force: bool = False):
    import chromadb
    from sentence_transformers import SentenceTransformer

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=str(KB_DIR))

    # ── Collection 1: project docs ──────────────────────────────────────────
    col_docs = client.get_or_create_collection("project_docs")
    existing_ids = set(col_docs.get()["ids"])

    doc_files = list((REPO_ROOT / "doc").rglob("*.md")) + [REPO_ROOT / "CLAUDE.md"]
    added_docs = 0
    for path in doc_files:
        rel = str(path.relative_to(REPO_ROOT))
        text = path.read_text(errors="ignore")
        chunks = chunk_markdown(text, rel)
        for chunk in chunks:
            cid = doc_id(chunk["text"], rel)
            if cid not in existing_ids or force:
                emb = embedder.encode(chunk["text"]).tolist()
                col_docs.upsert(
                    ids=[cid],
                    embeddings=[emb],
                    documents=[chunk["text"]],
                    metadatas=[{"source": rel, "section": chunk["section"]}],
                )
                added_docs += 1

    print(f"project_docs: {col_docs.count()} chunks total (+{added_docs} new)")

    # ── Collection 2: external research ────────────────────────────────────
    col_ext = client.get_or_create_collection("external_research")
    existing_ext = set(col_ext.get()["ids"])
    added_ext = 0
    for entry in EXTERNAL_ENTRIES:
        if entry["id"] not in existing_ext or force:
            emb = embedder.encode(entry["text"]).tolist()
            col_ext.upsert(
                ids=[entry["id"]],
                embeddings=[emb],
                documents=[entry["text"]],
                metadatas=[{"tags": entry["tags"]}],
            )
            added_ext += 1

    print(f"external_research: {col_ext.count()} entries total (+{added_ext} new)")
    print(f"\nKnowledge base built at: {KB_DIR}")
    print("Query with: python scripts/build_knowledge_base.py --query 'your question'")


def query(q: str, n: int = 5):
    import chromadb
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client   = chromadb.PersistentClient(path=str(KB_DIR))
    q_emb    = embedder.encode(q).tolist()

    print(f"\n🔍 Query: {q!r}\n{'─'*60}")

    for col_name in ("external_research", "project_docs"):
        col = client.get_or_create_collection(col_name)
        if col.count() == 0:
            continue
        results = col.query(query_embeddings=[q_emb], n_results=min(n, col.count()))
        docs  = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        print(f"\n[{col_name}]")
        for doc, meta, dist in zip(docs, metas, dists):
            score = round(1 - dist, 3)
            src   = meta.get("source") or meta.get("tags", "")
            print(f"  score={score:.3f}  {src}")
            print(f"  {doc[:280].strip()}")
            print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--query", "-q", help="Search the knowledge base")
    p.add_argument("--n",     type=int, default=5, help="Number of results")
    p.add_argument("--force", action="store_true", help="Rebuild all embeddings")
    args = p.parse_args()

    if args.query:
        query(args.query, args.n)
    else:
        build(force=args.force)
