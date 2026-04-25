//! `score-panel` CLI — load a transformer artifact and score a feature
//! matrix CSV. Mirrors `TransformerPanelScorer.score(feature_matrix)`
//! on the Python side so adapters can shell out for offline / batch use.
//!
//! Usage::
//!
//!     score-panel \
//!         --artifact backtesting/renquant_104/artifacts/panel-transformer \
//!         --input    /tmp/panel_features.csv \
//!         --output   /tmp/panel_scores.csv
//!
//! Input CSV format: first column = ticker, remaining columns = features
//! (header row gives feature names, must include every entry of the
//! artifact's `feature_cols`).

use anyhow::{anyhow, Context, Result};
use candle_core::Device;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(version, about = "Panel transformer scorer (Rust)")]
struct Args {
    /// Artifact stem path: looks for `<stem>.safetensors` + `<stem>.json`.
    #[arg(long)]
    artifact: PathBuf,

    /// Input CSV: header = ticker,<feature1>,<feature2>,...; one row per ticker.
    #[arg(long)]
    input: PathBuf,

    /// Output CSV: ticker,score
    #[arg(long)]
    output: PathBuf,

    /// Compute device. "cpu" everywhere; "metal" on macOS w/ MPS-equivalent.
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        #[cfg(feature = "metal")]
        "metal" => Device::new_metal(0)
            .context("creating Metal device — does this Mac have an Apple GPU?")?,
        #[cfg(not(feature = "metal"))]
        "metal" => return Err(anyhow!("metal feature not enabled — rebuild with --features metal")),
        other => return Err(anyhow!("unknown --device {} (use cpu or metal)", other)),
    };

    let scorer = transformer_scorer::PanelScorer::load(&args.artifact, device)
        .with_context(|| format!("loading artifact at {}", args.artifact.display()))?;

    // ── Read input CSV ──────────────────────────────────────────────────
    let mut rdr = csv::Reader::from_path(&args.input)
        .with_context(|| format!("opening input {}", args.input.display()))?;
    let headers = rdr.headers()?.clone();
    if headers.len() < 2 || headers.get(0) != Some("ticker") {
        return Err(anyhow!(
            "input CSV must start with column 'ticker' followed by feature cols"
        ));
    }
    let feature_names: Vec<String> = headers
        .iter()
        .skip(1)
        .map(|s| s.to_string())
        .collect();

    let mut tickers: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for rec in rdr.records() {
        let rec = rec?;
        if rec.len() != headers.len() {
            return Err(anyhow!(
                "row width {} != header width {}",
                rec.len(),
                headers.len()
            ));
        }
        let ticker = rec.get(0).unwrap_or("").to_string();
        tickers.push(ticker);
        let row: Result<Vec<f32>> = rec
            .iter()
            .skip(1)
            .map(|s| s.parse::<f32>().map_err(|e| anyhow!(e)))
            .collect();
        rows.push(row?);
    }

    if tickers.is_empty() {
        eprintln!("warning: no rows in input — writing empty output");
        std::fs::write(&args.output, "ticker,score\n")?;
        return Ok(());
    }

    // ── Build matrix, score, emit ───────────────────────────────────────
    let t = tickers.len();
    let f = feature_names.len();
    let mut matrix = ndarray::Array2::<f32>::zeros((t, f));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            matrix[[i, j]] = v;
        }
    }

    let scores = scorer.score(&matrix, &feature_names)?;

    let mut wtr = csv::Writer::from_path(&args.output)
        .with_context(|| format!("opening output {}", args.output.display()))?;
    wtr.write_record(&["ticker", "score"])?;
    for (ticker, score) in tickers.iter().zip(scores.iter()) {
        wtr.write_record(&[ticker.as_str(), &score.to_string()])?;
    }
    wtr.flush()?;
    Ok(())
}
