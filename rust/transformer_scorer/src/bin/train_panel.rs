//! `train-panel` — read a Python-exported panel CSV, train a transformer
//! end-to-end via candle, save the resulting safetensors. Closes the
//! Rust-training loop:
//!
//!     Python BuildPanelTask  →  panel.csv
//!         │
//!         ▼
//!     Rust train-panel  ──────  panel-transformer.safetensors
//!         │
//!         ▼
//!     Rust score-panel  (or PanelTransformer::load on Python side)
//!
//! Usage:
//!     train-panel \
//!         --panel    /path/to/panel.csv \
//!         --output   /tmp/rust_panel \
//!         --epochs   10 \
//!         --batch    16 \
//!         --lr       0.001 \
//!         --weight-decay 0.0001 \
//!         --d-model  64 \
//!         --n-heads  4 \
//!         --n-layers 2 \
//!         --ff-dim   128 \
//!         --dropout  0.2 \
//!         --device   cpu
//!
//! Writes:
//!   <output>.safetensors — trained weights
//!   <output>.json        — sidecar with feature_cols + TransformerParams
//!                          (matches the Python artifact format so
//!                          PanelScorer::load can read either)

use anyhow::{anyhow, Context, Result};
use candle_core::Device;
use clap::Parser;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;

use transformer_scorer::config::TransformerParams;
use transformer_scorer::dataset::Panel;
use transformer_scorer::metrics::pooled_ic_owned;
use transformer_scorer::trainer::{LossKind, Trainer};

#[derive(Parser, Debug)]
#[command(version, about = "Rust panel-transformer trainer")]
struct Args {
    /// Input panel CSV (date,ticker,<features>,label).
    #[arg(long)]
    panel: PathBuf,

    /// Artifact stem (writes <stem>.safetensors + <stem>.json).
    #[arg(long)]
    output: PathBuf,

    #[arg(long, default_value_t = 10)]
    epochs: usize,

    #[arg(long, default_value_t = 16)]
    batch: usize,

    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    #[arg(long = "weight-decay", default_value_t = 1e-4)]
    weight_decay: f64,

    /// Hyperparameters that mirror Python's TransformerParams. Defaults
    /// match a SMALL panel — bigger d_model/layers tend to overfit on
    /// 1256 dates × 99 tickers.
    #[arg(long = "d-model", default_value_t = 32)]
    d_model: usize,

    #[arg(long = "n-heads", default_value_t = 4)]
    n_heads: usize,

    #[arg(long = "n-layers", default_value_t = 2)]
    n_layers: usize,

    #[arg(long = "ff-dim", default_value_t = 64)]
    ff_dim: usize,

    #[arg(long, default_value_t = 0.2)]
    dropout: f32,

    #[arg(long, default_value = "cpu")]
    device: String,

    /// Fraction of date-groups held out as validation (final block,
    /// preserves chronological order — no shuffling). Used for early
    /// stopping + best-checkpoint selection.
    #[arg(long = "val-frac", default_value_t = 0.2)]
    val_frac: f64,

    /// Stop early if val-IC hasn't improved for this many epochs.
    /// 0 = disabled.
    #[arg(long = "patience", default_value_t = 5)]
    patience: usize,

    /// Loss function. "listnet" (top-1 listwise CE, matches Python) or
    /// "ranknet" (Burges 2005 pairwise — Poh-Lim-Zohren 2020 alternative
    /// for cross-sectional ranking, often gives stronger middle-of-pack
    /// gradients).
    #[arg(long, default_value = "listnet")]
    loss: String,

    /// Comma-separated list of feature columns to drop after CSV load.
    /// Use this to remove distribution-shifted cols (e.g. hourly +
    /// minute features that are +60% NaN in train but populated in val
    /// — see DAT-RUST-DROP-DISTSHIFT audit fix). Matches semantics of
    /// Python `panel_ltr.drop_cols` in strategy_config.json.
    #[arg(long = "drop-cols", default_value = "")]
    drop_cols: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "metal")]
        "metal" => Device::new_metal(0)
            .context("creating Metal device")?,
        #[cfg(not(feature = "metal"))]
        "metal" => return Err(anyhow!(
            "metal feature not enabled — rebuild with --features metal",
        )),
        other => return Err(anyhow!("unknown device '{}', use cpu or metal", other)),
    };

    eprintln!("[train-panel] loading {}", args.panel.display());
    let t0 = Instant::now();
    let mut panel = Panel::load_csv(&args.panel)
        .with_context(|| format!("loading panel from {}", args.panel.display()))?;
    eprintln!(
        "[train-panel] panel: {} dates × ~{} tickers/date  features={}  rows={}  ({:.1}s)",
        panel.n_dates(),
        if panel.n_dates() > 0 { panel.n_rows() / panel.n_dates() } else { 0 },
        panel.n_features(),
        panel.n_rows(),
        t0.elapsed().as_secs_f64(),
    );

    // Audit fix DAT-RUST-DROP-DISTSHIFT (Round 3, 2026-04-25): drop
    // distribution-shifted feature cols at load time. Cf. dataset.rs
    // doc-comment for full rationale.
    let drop_list: Vec<String> = args.drop_cols.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if !drop_list.is_empty() {
        let n = panel.drop_columns(&drop_list)
            .context("applying --drop-cols")?;
        eprintln!(
            "[train-panel] dropped {} cols → {} features remain (cols: {:?})",
            n, panel.n_features(), drop_list,
        );
    }

    let groups = panel.to_grouped_tensors(&device)
        .context("packing panel into per-date tensor groups")?;
    let n_features = panel.n_features();

    // Chronological train/val split (no shuffle). val_frac is the
    // tail fraction; preserves the temporal contract that we evaluate
    // on dates AFTER the training set.
    let n_groups = groups.len();
    let n_val = ((n_groups as f64) * args.val_frac).floor() as usize;
    let n_val = n_val.max(1).min(n_groups.saturating_sub(1));
    let split = n_groups - n_val;
    let (train_groups, val_groups) = groups.split_at(split);
    eprintln!(
        "[train-panel] split: train={} groups, val={} groups (last {:.0}%)",
        train_groups.len(), val_groups.len(), args.val_frac * 100.0,
    );

    let params = TransformerParams {
        d_model:         args.d_model,
        n_heads:         args.n_heads,
        n_layers:        args.n_layers,
        feedforward_dim: args.ff_dim,
        dropout:         args.dropout,
        feature_dropout: 0.0,
    };
    eprintln!(
        "[train-panel] arch: d_model={} heads={} layers={} ff={} dropout={} \
         lr={} weight_decay={}",
        params.d_model, params.n_heads, params.n_layers, params.feedforward_dim,
        params.dropout, args.lr, args.weight_decay,
    );

    let mut trainer = Trainer::new(
        n_features, params.clone(), args.lr, args.weight_decay, device,
    )?;
    let loss_kind = LossKind::from_str_lossy(&args.loss);
    trainer.set_loss(loss_kind);
    eprintln!("[train-panel] loss = {:?}", loss_kind);

    eprintln!(
        "[train-panel] training {} epoch(s), batch={}, patience={}",
        args.epochs, args.batch, args.patience,
    );
    let t_train = Instant::now();
    let mut best_val_ic: f32 = f32::NEG_INFINITY;
    let mut best_epoch: usize = 0;
    let mut epochs_since_improve: usize = 0;
    let best_stem = args.output.with_file_name(format!(
        "{}.best",
        args.output.file_name().and_then(|s| s.to_str()).unwrap_or("model"),
    ));
    for epoch in 0..args.epochs {
        let t_ep = Instant::now();
        let losses = trainer.train_epoch(train_groups, args.batch)?;
        let mean_loss: f32 = if losses.is_empty() {
            f32::NAN
        } else {
            losses.iter().sum::<f32>() / losses.len() as f32
        };
        // Eval on the chronological hold-out.
        let val_preds = trainer.predict_groups(val_groups)?;
        let (val_ic, n_valid_groups) = pooled_ic_owned(&val_preds);
        eprintln!(
            "[train-panel] epoch {:>3}/{}  steps={}  mean_loss={:.6}  val_IC={:+.4} (n={})  ({:.2}s)",
            epoch + 1, args.epochs, losses.len(),
            mean_loss, val_ic, n_valid_groups, t_ep.elapsed().as_secs_f64(),
        );
        if val_ic.is_finite() && val_ic > best_val_ic {
            best_val_ic = val_ic;
            best_epoch = epoch + 1;
            epochs_since_improve = 0;
            // Save the best-so-far checkpoint distinctly so the final
            // artifact is the best, not the last.
            trainer.save_safetensors(&best_stem)?;
        } else {
            epochs_since_improve += 1;
            if args.patience > 0 && epochs_since_improve >= args.patience {
                eprintln!(
                    "[train-panel] early stop — no val_IC improvement for {} epochs (best={:+.4} @ epoch {})",
                    args.patience, best_val_ic, best_epoch,
                );
                break;
            }
        }
    }
    eprintln!(
        "[train-panel] training done — total {:.1}s. best val_IC={:+.4} @ epoch {}",
        t_train.elapsed().as_secs_f64(), best_val_ic, best_epoch,
    );

    // Use the best-IC checkpoint as the canonical artifact.
    let best_safetensors = best_stem.with_extension("safetensors");
    if best_safetensors.exists() {
        let final_safetensors = args.output.with_extension("safetensors");
        std::fs::copy(&best_safetensors, &final_safetensors)
            .with_context(|| format!("copying {} → {}",
                best_safetensors.display(), final_safetensors.display()))?;
        let _ = std::fs::remove_file(&best_safetensors);
        eprintln!("[train-panel] promoted best checkpoint as final artifact");
    } else {
        // No improvement ever fired — fall back to the last weights.
        trainer.save_safetensors(&args.output)?;
    }
    let sidecar_path = args.output.with_extension("json");
    let sidecar = json!({
        "feature_cols": panel.feature_cols,
        "params": {
            "d_model":          params.d_model,
            "n_heads":          params.n_heads,
            "n_layers":         params.n_layers,
            "feedforward_dim":  params.feedforward_dim,
            "dropout":          params.dropout,
            "feature_dropout":  params.feature_dropout,
        },
        "trained_on": chrono_today(),
        "n_features": n_features,
        "n_dates":    panel.n_dates(),
        "backend":    "transformer-rust",
    });
    std::fs::write(&sidecar_path, serde_json::to_string_pretty(&sidecar)?)?;

    eprintln!(
        "[train-panel] artifact saved:\n  {}.safetensors\n  {}",
        args.output.display(), sidecar_path.display(),
    );
    Ok(())
}

/// YYYY-MM-DD of today UTC, without pulling in chrono just for this.
fn chrono_today() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs / 86_400;
    // Days since 1970-01-01 → calendar via Hinnant inverse. Same algo
    // as dataset.rs::civil_to_days run backwards.
    let z = days + 719_468;
    let era = if z >= 0 { z / 146_097 } else { (z - 146_096) / 146_097 };
    let doe = (z - era * 146_097) as i64;          // [0, 146_097)
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;   // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);                     // [0, 365]
    let mp = (5 * doy + 2) / 153;                                           // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = (y + if m <= 2 { 1 } else { 0 }) as i32;
    format!("{:04}-{:02}-{:02}", y, m, d)
}
