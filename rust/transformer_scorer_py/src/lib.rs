//! PyO3 bindings — call the Rust scorer from Python in-process, no spawn.
//!
//! Built via `maturin develop` (or `maturin build`). Once installed:
//!
//!     >>> import transformer_scorer_py as tspy
//!     >>> scorer = tspy.PanelScorer.load("/path/to/poc_panel")
//!     >>> import numpy as np
//!     >>> X = np.random.randn(4, 6).astype(np.float32)
//!     >>> cols = ["f0","f1","f2","f3","f4","f5"]
//!     >>> scores = scorer.score(X, cols)
//!     >>> print(scores)
//!
//! Single-call latency is now Python-native (no fork+exec each call).
//! The artifact is loaded once on `load(...)`; every `score(...)` is a
//! pure function call into Rust — same as a C extension. Combined with
//! the bench_parallel.rs result (16.59× rayon speedup), this gives both
//! single-call AND batch parallelism wins.

use ::transformer_scorer::PanelScorer as RustPanelScorer;
use candle_core::Device;
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Python-facing handle around a loaded artifact.
#[pyclass(name = "PanelScorer")]
pub struct PyPanelScorer {
    inner: RustPanelScorer,
}

#[pymethods]
impl PyPanelScorer {
    /// Load a scorer from `<stem>.safetensors` + `<stem>.json`.
    ///
    /// `device` is "cpu" (default) or "metal" (Apple GPU; requires the
    /// `metal` feature flag at compile time).
    #[staticmethod]
    #[pyo3(signature = (stem, device = "cpu"))]
    fn load(stem: &str, device: &str) -> PyResult<Self> {
        let dev = match device {
            "cpu" => Device::Cpu,
            #[cfg(feature = "metal")]
            "metal" => Device::new_metal(0)
                .map_err(|e| PyRuntimeError::new_err(format!("metal init: {e}")))?,
            #[cfg(not(feature = "metal"))]
            "metal" => return Err(PyRuntimeError::new_err(
                "metal feature not enabled — rebuild with `maturin develop --features metal`"
            )),
            other => return Err(PyRuntimeError::new_err(
                format!("unknown device '{}', use 'cpu' or 'metal'", other)
            )),
        };
        let inner = RustPanelScorer::load(stem, dev)
            .map_err(|e| PyRuntimeError::new_err(format!("load: {e}")))?;
        Ok(Self { inner })
    }

    /// Read-only feature_cols list.
    #[getter]
    fn feature_cols<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        let cols: Vec<String> = self.inner.feature_cols().to_vec();
        PyList::new_bound(py, cols)
    }

    /// Score a (T × F) NumPy float32 matrix. Returns a list of T floats.
    /// `feature_names` is the column order in `matrix`.
    fn score<'py>(
        &self,
        py: Python<'py>,
        matrix: PyReadonlyArray2<f32>,
        feature_names: Vec<String>,
    ) -> PyResult<Bound<'py, PyList>> {
        let arr = matrix.as_array().to_owned();
        // Sanity: caller must provide finite inputs (matches BRIDGE-6 in CLI).
        for &v in arr.iter() {
            if !v.is_finite() {
                return Err(PyRuntimeError::new_err(
                    "non-finite feature value — refusing to score (sanitize upstream)",
                ));
            }
        }
        // Release the GIL so concurrent scorers can run.
        let scores = py.allow_threads(|| -> Result<Vec<f32>, anyhow::Error> {
            let m: Array2<f32> = arr;
            self.inner.score(&m, &feature_names)
        });
        let v = scores.map_err(|e| {
            let msg = format!("{e}");
            if msg.contains("missing feature column") {
                PyKeyError::new_err(msg)
            } else {
                PyRuntimeError::new_err(msg)
            }
        })?;
        Ok(PyList::new_bound(py, v))
    }

    fn __repr__(&self) -> String {
        format!("<PanelScorer features={}>", self.inner.feature_cols().len())
    }
}

#[pymodule]
fn transformer_scorer_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPanelScorer>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
