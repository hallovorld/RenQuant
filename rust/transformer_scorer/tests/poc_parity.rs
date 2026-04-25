//! End-to-end parity test: load the POC artifact, score the POC features,
//! confirm the scores match the Python reference at FP32 tolerance.
//!
//! The fixtures under `tests/fixtures/` are the exact files produced by
//! `scripts/poc_rust_transformer.py` — committed so this test runs hermetically
//! in CI without needing the Python toolchain.
//!
//! Reference scores come from running the Python forward pass at the same
//! deterministic seed (manual_seed(0) + manual_seed(7) for the uniform init).
//! See `scripts/poc_rust_transformer.py` for the producer.

use approx::assert_abs_diff_eq;
use candle_core::Device;
use std::path::PathBuf;
use transformer_scorer::PanelScorer;

fn fixture_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p
}

/// Python-side scores from the POC run, in T0..T3 order.
/// Reproduce: `python scripts/poc_rust_transformer.py`.
const PY_SCORES: [f32; 4] = [0.118015, 0.115734, 0.119074, 0.126078];

fn read_features(path: &std::path::Path) -> (Vec<String>, Vec<String>, ndarray::Array2<f32>) {
    let mut rdr = csv::Reader::from_path(path).expect("opening features fixture");
    let headers = rdr.headers().expect("headers").clone();
    let feature_names: Vec<String> = headers.iter().skip(1).map(|s| s.to_string()).collect();

    let mut tickers = Vec::new();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for r in rdr.records() {
        let r = r.expect("record");
        tickers.push(r.get(0).unwrap_or("").to_string());
        let row: Vec<f32> = r
            .iter()
            .skip(1)
            .map(|s| s.parse().expect("parse f32"))
            .collect();
        rows.push(row);
    }
    let t = tickers.len();
    let f = feature_names.len();
    let mut m = ndarray::Array2::<f32>::zeros((t, f));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            m[[i, j]] = v;
        }
    }
    (tickers, feature_names, m)
}

#[test]
fn poc_artifact_loads_and_matches_python_scores() {
    let dir = fixture_dir();
    let stem = dir.join("poc_panel");
    let scorer = PanelScorer::load(&stem, Device::Cpu).expect("load fixture");

    let (tickers, feature_names, matrix) = read_features(&dir.join("poc_features.csv"));
    assert_eq!(tickers, vec!["T0", "T1", "T2", "T3"]);
    assert_eq!(feature_names.len(), scorer.feature_cols().len());

    let scores = scorer.score(&matrix, &feature_names).expect("score");
    assert_eq!(scores.len(), 4);

    // Tolerance budget: Python reference is rounded to 6 decimals; FP32
    // matmul on different platforms can differ by ~1e-6. 5e-5 is a
    // generous-but-meaningful bar — the empirical max diff on Apple
    // Silicon was 4e-7 at scaffold time.
    for (i, (got, expected)) in scores.iter().zip(PY_SCORES.iter()).enumerate() {
        assert_abs_diff_eq!(*got, *expected, epsilon = 5e-5);
        let abs_diff = (*got - *expected).abs();
        assert!(
            abs_diff < 1e-4,
            "ticker {} (T{}): got {} expected {} diff {:.2e}",
            i, i, got, expected, abs_diff
        );
    }
}

#[test]
fn missing_feature_column_returns_error() {
    let dir = fixture_dir();
    let stem = dir.join("poc_panel");
    let scorer = PanelScorer::load(&stem, Device::Cpu).expect("load fixture");

    // Build a feature matrix with WRONG column names — should fail loud.
    let bogus_names: Vec<String> = (0..6).map(|i| format!("not_a_real_feature_{}", i)).collect();
    let m = ndarray::Array2::<f32>::zeros((4, 6));
    let result = scorer.score(&m, &bogus_names);
    assert!(result.is_err(), "should reject unknown feature names");
    let err = format!("{}", result.unwrap_err());
    assert!(
        err.contains("missing feature column"),
        "error message should name the missing column: {}",
        err
    );
}

#[test]
fn extra_columns_are_tolerated_via_reorder() {
    // Caller passes the feature matrix in some arbitrary column order
    // (possibly with extras the model doesn't care about). The reorder
    // path inside `score()` should pick the right cols and ignore the rest.
    let dir = fixture_dir();
    let stem = dir.join("poc_panel");
    let scorer = PanelScorer::load(&stem, Device::Cpu).expect("load fixture");

    let (_tickers, feature_names, matrix) = read_features(&dir.join("poc_features.csv"));

    // Add an extra column at the front. Reordering the feature_names
    // to put the model's expected features in different positions is
    // the meaningful test — the scorer must reorder by NAME, not index.
    let mut shuffled_names = vec!["unrelated_extra".to_string()];
    shuffled_names.extend(feature_names.iter().rev().cloned());

    let mut shuffled = ndarray::Array2::<f32>::zeros((matrix.shape()[0], shuffled_names.len()));
    // Column 0 is the dummy extra; copy original cols in reversed order.
    for (i, name) in feature_names.iter().rev().enumerate() {
        let orig_col = feature_names.iter().position(|n| n == name).unwrap();
        for row in 0..matrix.shape()[0] {
            shuffled[[row, i + 1]] = matrix[[row, orig_col]];
        }
    }

    let scores = scorer.score(&shuffled, &shuffled_names).expect("score reordered");
    for (got, expected) in scores.iter().zip(PY_SCORES.iter()) {
        assert!(
            (*got - *expected).abs() < 1e-4,
            "reorder broke determinism: got {} expected {}",
            got,
            expected
        );
    }
}

#[test]
fn empty_matrix_produces_empty_output() {
    let dir = fixture_dir();
    let stem = dir.join("poc_panel");
    let scorer = PanelScorer::load(&stem, Device::Cpu).expect("load fixture");

    let (_, feature_names, _) = read_features(&dir.join("poc_features.csv"));
    let empty = ndarray::Array2::<f32>::zeros((0, feature_names.len()));
    let scores = scorer.score(&empty, &feature_names).expect("score empty");
    assert!(scores.is_empty(), "empty input → empty output");
}

#[test]
fn deterministic_repeat_invocations() {
    // Run the same scoring twice. Outputs must be bitwise identical —
    // our forward pass must not introduce any randomness in eval mode.
    let dir = fixture_dir();
    let stem = dir.join("poc_panel");
    let scorer = PanelScorer::load(&stem, Device::Cpu).expect("load fixture");

    let (_, feature_names, matrix) = read_features(&dir.join("poc_features.csv"));
    let a = scorer.score(&matrix, &feature_names).expect("score #1");
    let b = scorer.score(&matrix, &feature_names).expect("score #2");
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "repeated scoring drifted: {} != {}",
            x,
            y
        );
    }
}
