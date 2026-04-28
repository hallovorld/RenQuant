#!/usr/bin/env python
"""Compare panel-LTR ablation artifacts (A: no-emb, B: emb, C: emb+macro)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ART_DIR = REPO_ROOT / "backtesting" / "renquant_104" / "artifacts"


def load(path: Path) -> dict | None:
    if not path.exists():
        print(f"  ⚠ missing: {path.name}")
        return None
    return json.loads(path.read_text())


def paired_t(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) != len(b) or len(a) <= 1:
        return float("nan"), float("nan")
    diffs = [bi - ai for ai, bi in zip(a, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return mean, math.inf if mean != 0 else 0.0
    return mean, mean / se


def show(label: str, art: dict | None) -> None:
    if art is None:
        print(f"\n── {label}: NOT TRAINED ─────────")
        return
    fc = art.get("feature_cols") or []
    emb = sum(1 for c in fc if c.startswith("emb_"))
    print(f"\n── {label} ─────────────────")
    print(f"  feature_cols : {len(fc)}  (emb cols: {emb})")
    print(f"  panel_shape  : {art.get('panel_shape')}")
    print(f"  oos_mean_ic  : {art.get('oos_mean_ic')}")
    print(f"  oos_std_ic   : {art.get('oos_std_ic')}")
    folds = art.get("oos_per_fold_ic") or []
    if folds:
        print(f"  fold ICs     : {[round(x,4) for x in folds]}")
    print(f"  best_iter    : {art.get('best_iter')}")
    print(f"  params.objective: {(art.get('params') or {}).get('objective')}")


def compare(label: str, a: dict, b: dict, alpha_t=1.5, rel_floor=0.03) -> None:
    fa = a.get("oos_per_fold_ic") or []
    fb = b.get("oos_per_fold_ic") or []
    if len(fa) != len(fb) or len(fa) == 0:
        print(f"\n  {label}: fold count mismatch (A={len(fa)} B={len(fb)})")
        return
    mean_a = sum(fa) / len(fa)
    mean_b = sum(fb) / len(fb)
    diff, t = paired_t(fa, fb)
    rel = (mean_b - mean_a) / mean_a if mean_a else float("nan")
    pass_rel = mean_b > mean_a * (1 + rel_floor)
    pass_t = abs(t) > alpha_t
    print(f"\n  ── {label} ──")
    print(f"    mean IC A    = {mean_a:+.5f}")
    print(f"    mean IC B    = {mean_b:+.5f}")
    print(f"    Δ (B-A)      = {mean_b-mean_a:+.5f}  ({rel*100:+.1f}%)")
    print(f"    paired t     = {t:+.3f}  (n={len(fa)}, threshold |t| > {alpha_t})")
    print(f"    decision     = {'PROMOTE' if (pass_rel and pass_t) else 'NO-GO'}  (rel>{rel_floor*100:.0f}%: {pass_rel}, t-stat: {pass_t})")


def main() -> int:
    arms = {
        "A — no embeddings"            : ART_DIR / "panel-ltr.ablation-no-emb-pairwise.json",
        "B — with embeddings"          : ART_DIR / "panel-ltr.ablation-with-emb-pairwise.json",
        "C — embeddings + 8 macros"    : ART_DIR / "panel-ltr.ablation-emb-macro.json",
        "Reference (production today)" : ART_DIR / "panel-ltr.json",
    }
    arts = {k: load(v) for k, v in arms.items()}
    for k, a in arts.items():
        show(k, a)

    print("\n══ A/B comparisons (paired t on 15 CPCV folds) ══")
    a = arts["A — no embeddings"]
    b = arts["B — with embeddings"]
    c = arts["C — embeddings + 8 macros"]
    if a and b: compare("B vs A: does embedding help?", a, b)
    if b and c: compare("C vs B: do macros-as-panel help on top of embedding?", b, c)
    if a and c: compare("C vs A: combined emb+macro vs naked baseline", a, c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
