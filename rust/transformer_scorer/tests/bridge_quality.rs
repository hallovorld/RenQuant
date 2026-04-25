//! Bridge-quality tests — Python ↔ Rust data sanitization at the boundary.
//!
//! Companion to poc_parity.rs; this file focuses on _what happens when
//! someone hands the loader bad data_. The principle: fail loud, never
//! silently propagate NaN/inf into model output.

use candle_core::Device;
use std::path::PathBuf;
use transformer_scorer::PanelScorer;

fn fixtures() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p
}

#[test]
fn nan_feature_value_in_score_call_does_not_crash() {
    // The library's `score()` accepts an ndarray::Array2<f32>; if NaN
    // somehow slips into the matrix (caller bug), forward pass will
    // produce NaN output. The CLI is the boundary that REJECTS NaN
    // (see main.rs BRIDGE-6); the library is the lower-level path
    // and we accept that NaN-in → NaN-out for compatibility, but we
    // verify the call doesn't panic.
    let dir = fixtures();
    let stem = dir.join("poc_panel");
    let scorer = PanelScorer::load(&stem, Device::Cpu).expect("load fixture");

    let cols = scorer.feature_cols().to_vec();
    let f = cols.len();
    let mut m = ndarray::Array2::<f32>::zeros((1, f));
    m[[0, 0]] = f32::NAN;

    // Library accepts the call (no panic). The output may be NaN —
    // that's OK at this layer; the CLI / adapters validate inputs.
    let result = scorer.score(&m, &cols);
    match result {
        Ok(v) => {
            // Length is correct; whether the score is NaN is fine here.
            assert_eq!(v.len(), 1);
        }
        Err(_) => {
            // Also acceptable — some candle paths may fail loud on NaN.
        }
    }
}

#[test]
fn library_score_with_inf_does_not_panic() {
    let dir = fixtures();
    let stem = dir.join("poc_panel");
    let scorer = PanelScorer::load(&stem, Device::Cpu).expect("load fixture");

    let cols = scorer.feature_cols().to_vec();
    let f = cols.len();
    let mut m = ndarray::Array2::<f32>::zeros((2, f));
    m[[0, 0]] = f32::INFINITY;
    m[[1, 0]] = f32::NEG_INFINITY;

    let result = scorer.score(&m, &cols);
    // Don't crash. CLI is the layer that should reject; lib forwards.
    let _ = result;
}

#[test]
fn cli_binary_rejects_nan_csv_input() {
    // Write a CSV with a "nan" cell, run score-panel, expect non-zero exit.
    use std::io::Write;
    let dir = fixtures();
    let stem = dir.join("poc_panel");

    // Build the binary path from the test runner's location.
    let mut bin = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    bin.pop();
    bin.push("target");
    bin.push("release");
    bin.push("score-panel");
    if !bin.exists() {
        // Fall back to debug build if release isn't built (CI default).
        bin.set_file_name("score-panel");
        bin.pop();
        bin.push("debug");
        bin.push("score-panel");
        if !bin.exists() {
            // Skip: test only meaningful when a binary has been built.
            eprintln!("score-panel binary not found at {}; skipping CLI test", bin.display());
            return;
        }
    }

    let tmp_dir = std::env::temp_dir();
    let bad_csv = tmp_dir.join("rust_test_nan_input.csv");
    let bad_out = tmp_dir.join("rust_test_nan_out.csv");

    let mut f = std::fs::File::create(&bad_csv).expect("create tmp csv");
    writeln!(f, "ticker,f0,f1,f2,f3,f4,f5").unwrap();
    writeln!(f, "T0,0.1,0.2,nan,0.4,0.5,0.6").unwrap();
    writeln!(f, "T1,1.0,2.0,3.0,4.0,5.0,6.0").unwrap();
    drop(f);

    let output = std::process::Command::new(&bin)
        .arg("--artifact").arg(&stem)
        .arg("--input").arg(&bad_csv)
        .arg("--output").arg(&bad_out)
        .output()
        .expect("running score-panel");

    assert!(!output.status.success(), "score-panel must reject NaN input");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("non-finite") || stderr.contains("NaN"),
        "stderr should explain why: got {}",
        stderr
    );

    let _ = std::fs::remove_file(&bad_csv);
    let _ = std::fs::remove_file(&bad_out);
}

#[test]
fn cli_binary_rejects_inf_csv_input() {
    use std::io::Write;
    let dir = fixtures();
    let stem = dir.join("poc_panel");

    let mut bin = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    bin.pop();
    bin.push("target");
    bin.push("release");
    bin.push("score-panel");
    if !bin.exists() {
        eprintln!("score-panel binary not found; skipping CLI inf test");
        return;
    }

    let tmp_dir = std::env::temp_dir();
    let bad_csv = tmp_dir.join("rust_test_inf_input.csv");
    let bad_out = tmp_dir.join("rust_test_inf_out.csv");

    let mut f = std::fs::File::create(&bad_csv).expect("create tmp csv");
    writeln!(f, "ticker,f0,f1,f2,f3,f4,f5").unwrap();
    writeln!(f, "T0,0.1,0.2,inf,0.4,0.5,0.6").unwrap();
    drop(f);

    let output = std::process::Command::new(&bin)
        .arg("--artifact").arg(&stem)
        .arg("--input").arg(&bad_csv)
        .arg("--output").arg(&bad_out)
        .output()
        .expect("running score-panel");

    assert!(!output.status.success(), "score-panel must reject inf input");

    let _ = std::fs::remove_file(&bad_csv);
    let _ = std::fs::remove_file(&bad_out);
}
