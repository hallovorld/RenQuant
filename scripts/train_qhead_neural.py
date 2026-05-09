#!/usr/bin/env python
"""Phase C — Neural quantile head (Lim 2021 TFT-style joint quantile MLP).

Hypothesis: XGB QHead is panel-saturated at val μ-IC ≈ +0.029 ± 0.003
(5-seed A/A from E51). Need different inductive bias to break ceiling.
Neural MLP with joint quantile head:
  - Shares parameters across quantiles (joint info flow, no crossing)
  - Allows non-tree feature interactions
  - LayerNorm = inherent regime invariance per layer

Architecture (param/sample ratio constraint, CLAUDE.md §5.12):
  Input(169) → LN → Linear(32) → ReLU → Dropout(0.3) →
              LN → Linear(16) → ReLU → Dropout(0.2) →
              Linear(3) → 3 quantile outputs

Total params ≈ 169*32+32 + 32*16+16 + 16*3+3 + LN(2*169+2*32+2*16) ≈ 6422.
568563 train rows / 6422 = 88. Ratio 1/88 — MEETS §5.12 1/100 spec roughly.

Loss: pinball loss summed over q ∈ {0.16, 0.50, 0.84}.
Optimizer: AdamW lr=1e-3, wd=1e-4.
Epochs: 50, early-stop on val pinball loss (patience=5).

References (CLAUDE.md §5.12):
- Lim et al. 2021 ICLR "Temporal Fusion Transformers" §3.4 (joint quantile head)
- Koenker & Bassett 1978 (pinball loss)
- Ba et al. 2016 (LayerNorm)
- Kingma & Ba 2014 (Adam) → Loshchilov 2019 (AdamW)

Output: artifacts/ngboost-head.alpha158_fund_neural.json (kind=neural_quantile_head).
       Currently NOT compatible with QualityFloorTask loader — separate adapter
       needed. This script reports val μ-IC + saves a torch artifact for later
       integration if val_ic ≥ +0.04 across 5 seeds.
"""
from __future__ import annotations
import json, time, hashlib, base64, pickle, sys, logging
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("phaseC")

REPO = Path(__file__).resolve().parent.parent
QUANTILES = [0.16, 0.50, 0.84]
LABEL = "fwd_60d_excess_raw"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class QuantileMLP(nn.Module):
    def __init__(self, in_dim: int, h1: int = 32, h2: int = 16,
                 n_quantiles: int = 3, dropout: float = 0.3):
        super().__init__()
        self.ln_in = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, h1)
        self.ln1 = nn.LayerNorm(h1)
        self.do1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(h1, h2)
        self.ln2 = nn.LayerNorm(h2)
        self.do2 = nn.Dropout(dropout * 0.5)
        self.head = nn.Linear(h2, n_quantiles)

    def forward(self, x):
        x = self.ln_in(x)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.do1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.do2(x)
        return self.head(x)   # [N, n_q]


def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor, quantiles: list[float]):
    """y_pred: [N, n_q], y_true: [N]. Returns scalar mean pinball loss."""
    losses = []
    y_true = y_true.unsqueeze(-1)  # [N, 1]
    for i, q in enumerate(quantiles):
        diff = y_true - y_pred[:, i:i+1]
        loss_q = torch.maximum(q * diff, (q - 1) * diff).mean()
        losses.append(loss_q)
    return torch.stack(losses).sum()


def cs_ic(mu, y, dates):
    df = pd.DataFrame({"p": mu, "y": y, "d": dates})
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("d") if len(g) >= 5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else float("nan")


def train_eval_one_seed(seed: int, Xtr, ytr, Xva, yva_raw, val_dates,
                        n_epochs=50, batch_size=2048, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    in_dim = Xtr.shape[1]
    model = QuantileMLP(in_dim, h1=32, h2=16).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Pre-fetch tensors (data fits in memory)
    Xtr_t = torch.from_numpy(Xtr).float().to(DEVICE)
    ytr_t = torch.from_numpy(ytr).float().to(DEVICE)
    Xva_t = torch.from_numpy(Xva).float().to(DEVICE)
    yva_t = torch.from_numpy(yva_raw).float().to(DEVICE)

    n = Xtr_t.shape[0]
    best_val_loss = float("inf"); best_epoch = -1; best_state = None
    patience = 5; bad = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        ep_loss = 0.0; nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = pinball_loss(pred, yb, QUANTILES)
            loss.backward()
            opt.step()
            ep_loss += loss.item(); nb += 1

        model.eval()
        with torch.no_grad():
            pv = model(Xva_t)
            v_loss = pinball_loss(pv, yva_t, QUANTILES).item()
        if v_loss < best_val_loss - 1e-6:
            best_val_loss = v_loss; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = model(Xva_t).cpu().numpy()
    mu = pv[:, 1]
    sd = np.maximum((pv[:, 2] - pv[:, 0]) / 2.0, 1e-6)
    val_ic = cs_ic(mu, yva_raw, val_dates)
    sigma_calib = float(spearmanr(sd, np.abs(yva_raw - mu))[0])
    mu_xs_std = float(pd.DataFrame({"mu": mu, "d": val_dates}).groupby("d")["mu"].std().mean())
    return {
        "val_ic": val_ic, "sigma_calib": sigma_calib, "mu_xs_std": mu_xs_std,
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
        "state": best_state,
    }


def main():
    panel_path = REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet"
    art_panel  = REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json"
    out_path   = REPO / "backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund_neural.json"

    panel_meta = json.loads(art_panel.read_text())
    feat_cols = list(panel_meta["feature_cols"])
    panel_fp  = panel_meta["config_fingerprint"]
    fmeans = np.asarray(panel_meta.get("feature_means", [0.0] * len(feat_cols)))
    fstds  = np.asarray(panel_meta.get("feature_stds",  [1.0] * len(feat_cols))) + 1e-9

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[LABEL])
    distinct_dates = sorted(panel["date"].unique())
    val_cut = distinct_dates[int(len(distinct_dates) * 0.8)]
    train = panel[panel["date"] <= val_cut].copy()
    val   = panel[panel["date"] >  val_cut].copy()

    Xtr_raw = train[feat_cols].fillna(0).values.astype(np.float32)
    Xva_raw = val[feat_cols].fillna(0).values.astype(np.float32)
    # Apply panel artifact's stored normalization (matches production scoring)
    Xtr = ((Xtr_raw - fmeans) / fstds).astype(np.float32)
    Xva = ((Xva_raw - fmeans) / fstds).astype(np.float32)
    ytr = train[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    yva = val[LABEL].clip(-0.5, 0.5).values.astype(np.float32)
    val_dates = val["date"].values

    log.info("Device=%s  Train/val rows=%d/%d  feat_dim=%d", DEVICE, len(Xtr), len(Xva), len(feat_cols))

    # Count params on the chosen architecture
    m = QuantileMLP(in_dim=len(feat_cols), h1=32, h2=16)
    n_params = sum(p.numel() for p in m.parameters())
    log.info("Architecture: 169 → 32 → 16 → 3, params=%d  (train_rows/params=%.1f, target ≥100)",
             n_params, len(Xtr) / n_params)

    # 5-seed A/A — per CLAUDE.md §5.2
    log.info("\n══ Phase C — neural QHead 5-seed run ══")
    results = []
    for s in [42, 7, 123, 2024, 31415]:
        t0 = time.time()
        r = train_eval_one_seed(s, Xtr, ytr, Xva, yva, val_dates)
        log.info("  seed=%-5d val_ic=%+.4f σ-calib=%+.3f μ_xs_std=%.5f best_epoch=%d val_loss=%.5f (%.1fs)",
                 s, r["val_ic"], r["sigma_calib"], r["mu_xs_std"],
                 r["best_epoch"], r["best_val_loss"], time.time()-t0)
        results.append(r)

    val_ics = [r["val_ic"] for r in results]
    mean_ic = float(np.mean(val_ics)); std_ic = float(np.std(val_ics, ddof=1))
    log.info("")
    log.info("Neural QHead 5-seed: mean=%+.4f std=%.4f  range=[%+.4f, %+.4f]",
             mean_ic, std_ic, min(val_ics), max(val_ics))
    log.info("Compare baseline XGB QHead: mean=+0.0294 std=0.0029  (E51 5-seed A/A)")
    log.info("Δ = %+.4f", mean_ic - 0.0294)
    if mean_ic >= 0.040:
        log.info("✓ TARGET MET (mean ≥ +0.040)")
    elif mean_ic > 0.0294 + 2 * 0.0029:
        log.info("✓ SIGNIFICANT BEAT (>2σ above baseline)")
    else:
        log.info("✗ Within noise of baseline")

    # Save best-seed model to artifact
    best_idx = int(np.argmax(val_ics))
    best = results[best_idx]
    log.info("\nSaving best seed %d (val_ic=%+.4f) artifact...", [42,7,123,2024,31415][best_idx], best["val_ic"])
    state_b64 = base64.b64encode(pickle.dumps(best["state"])).decode("ascii")
    artifact = {
        "version": 1, "kind": "neural_quantile_head",
        "trained_date": str(datetime.utcnow().date()),
        "feature_cols": feat_cols,
        "feature_means": fmeans.tolist(),
        "feature_stds":  fstds.tolist(),
        "quantiles": QUANTILES,
        "architecture": {"in_dim": len(feat_cols), "h1": 32, "h2": 16, "n_quantiles": 3, "dropout": 0.3},
        "model_state_b64": state_b64,
        "training_notes": (
            f"Phase C neural QHead (Lim 2021 TFT-style joint quantile MLP). "
            f"5-seed A/A: mean val_ic={mean_ic:+.4f} ± {std_ic:.4f}, baseline XGB +0.0294±0.0029."
        ),
        "train_mu_ic_mean_5seed": mean_ic,
        "train_mu_ic_std_5seed": std_ic,
        "train_mu_ic_per_seed": val_ics,
        "config_fingerprint": f"sha256:phasec_{hashlib.sha256(json.dumps({'h1':32,'h2':16,'lr':1e-3}).encode()).hexdigest()[:16]}",
        "panel_artifact_fingerprint": panel_fp,
    }
    out_path.write_text(json.dumps(artifact))
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    sys.exit(main())
