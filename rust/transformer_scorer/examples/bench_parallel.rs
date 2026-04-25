//! Parallel-scoring benchmark — quick demonstration that the Rust path
//! saturates all CPU cores cleanly without GIL-style contention.
//!
//! Compares serial scoring vs `rayon::par_iter` of N batches, where each
//! "batch" is one date-group of T tickers. Real production use is
//! one bar's worth of tickers, but scoring 5000 historical bars in a
//! sim is the natural place to want this.
//!
//! Run from rust/:
//!   cargo run --release --example bench_parallel -- 5000
//!
//! Argument is the number of batches; default 1000.

use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use transformer_scorer::PanelScorer;

fn main() -> anyhow::Result<()> {
    // Args.
    let n_batches: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let mut fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fixtures.push("tests");
    fixtures.push("fixtures");

    let stem = fixtures.join("poc_panel");
    let scorer = Arc::new(PanelScorer::load(&stem, candle_core::Device::Cpu)?);

    // Read input matrix once.
    let csv_path = fixtures.join("poc_features.csv");
    let mut rdr = csv::Reader::from_path(&csv_path)?;
    let headers = rdr.headers()?.clone();
    let feature_names: Vec<String> = headers.iter().skip(1).map(|s| s.to_string()).collect();

    let mut rows: Vec<Vec<f32>> = Vec::new();
    for r in rdr.records() {
        let r = r?;
        let row: Vec<f32> = r
            .iter()
            .skip(1)
            .map(|s| s.parse::<f32>().unwrap_or(0.0))
            .collect();
        rows.push(row);
    }
    let t = rows.len();
    let f = feature_names.len();
    let mut matrix = ndarray::Array2::<f32>::zeros((t, f));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            matrix[[i, j]] = v;
        }
    }

    let cpus = num_cpus_or_default();
    eprintln!(
        "Bench: scoring {} batches of ({} × {}) features on {} cores",
        n_batches, t, f, cpus,
    );

    // ── Serial baseline ─────────────────────────────────────────────
    let t0 = Instant::now();
    let mut serial_acc = 0.0_f64;
    for _ in 0..n_batches {
        let s = scorer.score(&matrix, &feature_names)?;
        serial_acc += s.iter().map(|&v| v as f64).sum::<f64>();
    }
    let serial_dt = t0.elapsed();
    eprintln!(
        "  serial : {:>7.2} ms total, {:>5.0} batches/s, sum={:.4}",
        serial_dt.as_secs_f64() * 1000.0,
        n_batches as f64 / serial_dt.as_secs_f64(),
        serial_acc,
    );

    // ── Parallel via rayon ──────────────────────────────────────────
    // Configure rayon threadpool size = #cpus by default. We could clamp
    // explicitly via a ThreadPoolBuilder, but the default already picks
    // physical+SMT cores which is what we want for a CPU-bound job.
    let t0 = Instant::now();
    let par_acc: f64 = (0..n_batches)
        .into_par_iter()
        .map(|_| {
            let s = scorer.score(&matrix, &feature_names).expect("score");
            s.iter().map(|&v| v as f64).sum::<f64>()
        })
        .sum();
    let par_dt = t0.elapsed();
    eprintln!(
        "  rayon  : {:>7.2} ms total, {:>5.0} batches/s, sum={:.4}",
        par_dt.as_secs_f64() * 1000.0,
        n_batches as f64 / par_dt.as_secs_f64(),
        par_acc,
    );

    // Sums must match — parallelism doesn't change determinism.
    let drift = (par_acc - serial_acc).abs();
    eprintln!("  drift  : {:.2e} (must be 0 modulo float assoc)", drift);

    let speedup = serial_dt.as_secs_f64() / par_dt.as_secs_f64();
    eprintln!("  speedup: {:.2}× ({} cores; ideal ≤ {})", speedup, cpus, cpus);
    Ok(())
}

fn num_cpus_or_default() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}
