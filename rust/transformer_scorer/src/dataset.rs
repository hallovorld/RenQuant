//! Panel-data loader — reads a flat CSV `(date, ticker, f0..fN, label)`
//! and groups rows by `date` into per-group tensors ready to feed
//! `Trainer::train_epoch`.
//!
//! Format mirrors the Python panel after BuildPanelTask:
//!   first column:  date (YYYY-MM-DD or any monotonic int as string)
//!   second column: ticker
//!   middle cols:   feature columns (f32)
//!   last column:   label (f32; NaN allowed at boundary)
//!
//! For now this is the primary data path; a parquet reader can land
//! next iteration without changing the public API.
//!
//! Usage:
//!   let panel = Panel::load_csv("/path/to/panel.csv")?;
//!   let groups = panel.to_grouped_tensors(&device)?;
//!   for (x, y) in groups { ... }

use anyhow::{anyhow, Context, Result};
use candle_core::{Device, Tensor};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug)]
pub struct Panel {
    /// Header (excluding `date` and `ticker`) — last entry is `label`.
    pub feature_cols: Vec<String>,
    /// Sorted unique dates encountered (preserves load order).
    pub dates:        Vec<String>,
    /// Per-date rows. Each row: (ticker, [f0..fN], label_f32).
    /// Inner Vec<f32> length = feature_cols.len() (label NOT included
    /// in features). Label comes back via `labels` parallel structure.
    pub features_by_date: BTreeMap<String, Vec<(String, Vec<f32>, f32)>>,
}

impl Panel {
    /// Load a flat CSV with header row.
    /// Header layout: `date,ticker,<feature_cols...>,label`.
    pub fn load_csv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let mut rdr = csv::Reader::from_path(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let header = rdr.headers()?.clone();
        // Audit fix DAT-RUST-BOM (Round 2 deep audit, 2026-04-25):
        // Excel / Windows CSV exports often include a UTF-8 BOM (\u{FEFF})
        // at the start of the first cell. csv crate doesn't strip it.
        // Strip per-cell so all comparisons see clean values; same
        // applies to whitespace from Excel quoting habits.
        let strip = |s: &str| -> String {
            s.trim_start_matches('\u{FEFF}').trim().to_string()
        };
        let h0 = header.get(0).map(strip).unwrap_or_default();
        let h1 = header.get(1).map(strip).unwrap_or_default();
        if header.len() < 4 || h0 != "date" || h1 != "ticker" {
            return Err(anyhow!(
                "expected header `date,ticker,<features>,label`, got {:?}",
                header.iter().take(4).collect::<Vec<_>>(),
            ));
        }
        let last = header.len() - 1;
        let h_last = header.get(last).map(strip).unwrap_or_default();
        if h_last != "label" {
            return Err(anyhow!(
                "last header column must be 'label', got '{}'",
                header.get(last).unwrap_or(""),
            ));
        }
        // feature_cols = everything between ticker and label.
        let feature_cols: Vec<String> = header
            .iter()
            .skip(2)
            .take(last - 2)
            .map(|s| strip(s))
            .collect();
        let n_feat = feature_cols.len();

        let mut by_date: BTreeMap<String, Vec<(String, Vec<f32>, f32)>> = BTreeMap::new();
        let mut row_idx = 0usize;
        for rec in rdr.records() {
            row_idx += 1;
            let rec = rec?;
            if rec.len() != header.len() {
                return Err(anyhow!(
                    "row {}: width {} != header width {}",
                    row_idx, rec.len(), header.len(),
                ));
            }
            // Audit fix DAT-RUST-WS (Round 2 deep audit, 2026-04-25):
            // strip leading/trailing whitespace + BOM (the BOM is on the
            // first cell of the first row only, but defense in depth).
            // Production CSVs from Excel often have stray spaces around
            // quoted fields; without trim, the BTreeMap groups
            // " 2024-01-02" and "2024-01-02" as different dates.
            let date = rec.get(0)
                .unwrap_or("")
                .trim_start_matches('\u{FEFF}')
                .trim()
                .to_string();
            if date.is_empty() {
                return Err(anyhow!("row {}: empty date", row_idx));
            }
            let ticker = rec.get(1).unwrap_or("").trim().to_string();
            if ticker.is_empty() {
                return Err(anyhow!("row {}: empty ticker", row_idx));
            }
            // Parse features.
            let mut feats = Vec::with_capacity(n_feat);
            for j in 0..n_feat {
                let s_raw = rec.get(2 + j).unwrap_or("");
                let s = s_raw.trim();   // tolerate ' 0.5' and '0.5 '
                let v: f32 = s.parse().with_context(|| {
                    format!("row {} col {} ('{}'): not a float", row_idx, j + 3, s_raw)
                })?;
                // Reject non-finite features. Labels CAN be NaN
                // (boundary lookahead); features cannot — sanitize
                // upstream via scripts/sanitize_bridge_csv.py.
                if !v.is_finite() {
                    return Err(anyhow!(
                        "row {} col {}: feature '{}' is non-finite",
                        row_idx, j + 3, feature_cols[j],
                    ));
                }
                feats.push(v);
            }
            // Parse label — NaN allowed. Trim WS from the cell so a
            // CRLF leftover or Excel-quoted '  1.0  ' parses cleanly.
            let label_raw = rec.get(last).unwrap_or("");
            let label_str = label_raw.trim();
            let label: f32 = label_str.parse().unwrap_or(f32::NAN);
            // Treat empty label string as NaN (boundary rows).
            let label = if label_str.is_empty() { f32::NAN } else { label };

            by_date.entry(date).or_default().push((ticker, feats, label));
        }

        let dates: Vec<String> = by_date.keys().cloned().collect();
        Ok(Panel { feature_cols, dates, features_by_date: by_date })
    }

    pub fn n_features(&self) -> usize {
        self.feature_cols.len()
    }

    pub fn n_dates(&self) -> usize {
        self.dates.len()
    }

    pub fn n_rows(&self) -> usize {
        self.features_by_date.values().map(|v| v.len()).sum()
    }

    /// Convert to `Vec<(x: (T,F), y: (T,))>` suitable for
    /// `Trainer::train_epoch`. One entry per date.
    pub fn to_grouped_tensors(&self, device: &Device) -> Result<Vec<(Tensor, Tensor)>> {
        let f = self.n_features();
        let mut out = Vec::with_capacity(self.dates.len());
        for date in &self.dates {
            let rows = self.features_by_date.get(date)
                .ok_or_else(|| anyhow!("missing date '{}' in features_by_date", date))?;
            let t = rows.len();
            if t == 0 {
                continue;   // skip empty groups
            }
            let mut x_buf = Vec::with_capacity(t * f);
            let mut y_buf = Vec::with_capacity(t);
            for (_, feats, label) in rows {
                if feats.len() != f {
                    return Err(anyhow!(
                        "date '{}': row has {} features, expected {}",
                        date, feats.len(), f,
                    ));
                }
                x_buf.extend_from_slice(feats);
                y_buf.push(*label);
            }
            let x = Tensor::from_vec(x_buf, (t, f), device)?;
            let y = Tensor::from_vec(y_buf, t,      device)?;
            out.push((x, y));
        }
        Ok(out)
    }

    /// Days-since-2000 encoding for each unique date (ASCII string parse).
    /// Matches the encoding the CV splitter expects (`&[i64]`). Falls
    /// back to row-index when a date doesn't parse as YYYY-MM-DD.
    pub fn dates_as_days_since_2000(&self) -> Vec<i64> {
        let mut out: Vec<i64> = Vec::with_capacity(self.n_rows());
        for date_str in &self.dates {
            let day = parse_date_days_since_2000(date_str)
                .unwrap_or_else(|| out.len() as i64);
            // Replicate one entry per row in this date group.
            let n = self.features_by_date
                .get(date_str)
                .map(Vec::len)
                .unwrap_or(0);
            for _ in 0..n {
                out.push(day);
            }
        }
        out
    }
}

/// Best-effort YYYY-MM-DD → days-since-2000-01-01 parse. Returns None
/// if the string isn't ISO-format. The CV splitter only needs
/// monotonic integers — month/year-only encoding works too if needed.
fn parse_date_days_since_2000(s: &str) -> Option<i64> {
    let mut parts = s.split('-');
    let y: i32 = parts.next()?.parse().ok()?;
    let m: u32 = parts.next()?.parse().ok()?;
    let d: u32 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || y < 1900 {
        return None;
    }
    let days = civil_to_days(y, m, d)?;
    // Subtract days to 2000-01-01 to land on a 0-anchored sequence.
    let epoch_2000 = civil_to_days(2000, 1, 1)?;
    Some(days - epoch_2000)
}

/// Howard Hinnant's `civil_from_days` ↔ `days_from_civil` algorithm.
/// http://howardhinnant.github.io/date_algorithms.html
/// Returns days since some absolute epoch — only differences matter.
fn civil_to_days(y_in: i32, m: u32, d: u32) -> Option<i64> {
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let y = (y_in - if m <= 2 { 1 } else { 0 }) as i64;
    // era = floor(y / 400), defined for negative years too.
    let era = if y >= 0 { y / 400 } else { (y - 399) / 400 };
    let yoe = y - era * 400;                                // [0, 399]
    let mp  = if m > 2 { m as i64 - 3 } else { m as i64 + 9 };   // [0, 11]
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146097 + doe)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_csv(path: &Path, content: &str) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn loads_well_formed_panel() {
        let tmp = std::env::temp_dir().join("rust_panel_test.csv");
        write_csv(&tmp, "\
date,ticker,f0,f1,label
2024-01-02,AAA,0.1,0.2,1.0
2024-01-02,BBB,0.3,0.4,-1.0
2024-01-03,AAA,0.5,0.6,0.5
2024-01-03,BBB,0.7,0.8,
");
        let p = Panel::load_csv(&tmp).expect("load");
        assert_eq!(p.feature_cols, vec!["f0", "f1"]);
        assert_eq!(p.n_dates(), 2);
        assert_eq!(p.n_rows(), 4);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn empty_label_becomes_nan() {
        let tmp = std::env::temp_dir().join("rust_panel_empty_label.csv");
        write_csv(&tmp, "\
date,ticker,f0,label
2024-01-02,AAA,0.5,
");
        let p = Panel::load_csv(&tmp).unwrap();
        let rows = &p.features_by_date["2024-01-02"];
        assert!(rows[0].2.is_nan(), "empty label should parse as NaN");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn rejects_nan_feature() {
        let tmp = std::env::temp_dir().join("rust_panel_nan_feature.csv");
        write_csv(&tmp, "\
date,ticker,f0,label
2024-01-02,AAA,nan,1.0
");
        let r = Panel::load_csv(&tmp);
        assert!(r.is_err(), "nan in feature column should fail loud");
        let err_msg = format!("{}", r.unwrap_err());
        assert!(err_msg.contains("non-finite"), "error should name non-finite: {}", err_msg);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn to_grouped_tensors_shapes() {
        let tmp = std::env::temp_dir().join("rust_panel_shapes.csv");
        write_csv(&tmp, "\
date,ticker,f0,f1,label
2024-01-02,AAA,0.1,0.2,1.0
2024-01-02,BBB,0.3,0.4,-1.0
2024-01-02,CCC,0.5,0.6,0.0
2024-01-03,AAA,0.7,0.8,0.5
");
        let p = Panel::load_csv(&tmp).unwrap();
        let groups = p.to_grouped_tensors(&Device::Cpu).unwrap();
        assert_eq!(groups.len(), 2);
        let (x0, y0) = &groups[0];
        assert_eq!(x0.dims(), &[3, 2]);
        assert_eq!(y0.dims(), &[3]);
        let (x1, y1) = &groups[1];
        assert_eq!(x1.dims(), &[1, 2]);
        assert_eq!(y1.dims(), &[1]);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn date_parsing_2024() {
        // 2024-01-01 is days-since-2000-01-01 = ?  (2000→2024 = 24 years
        // including 6 leap years: 2000,2004,2008,2012,2016,2020 → 24*365+6 = 8766).
        let days = parse_date_days_since_2000("2024-01-01").unwrap();
        assert_eq!(days, 8766);
        // Same algorithm: 2000-01-01 = 0
        assert_eq!(parse_date_days_since_2000("2000-01-01").unwrap(), 0);
        // monotonic: later date is larger
        let d1 = parse_date_days_since_2000("2024-06-15").unwrap();
        let d2 = parse_date_days_since_2000("2024-12-31").unwrap();
        assert!(d2 > d1);
    }

    #[test]
    fn rejects_bad_header() {
        let tmp = std::env::temp_dir().join("rust_panel_bad_header.csv");
        write_csv(&tmp, "ticker,date,f0,label\nAAA,2024-01-02,0.1,1.0\n");
        let r = Panel::load_csv(&tmp);
        assert!(r.is_err());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn rejects_missing_label_col() {
        let tmp = std::env::temp_dir().join("rust_panel_no_label.csv");
        write_csv(&tmp, "date,ticker,f0,f1\n2024-01-02,AAA,0.1,0.2\n");
        let r = Panel::load_csv(&tmp);
        assert!(r.is_err());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn dat_rust_bom_excel_export_loads_cleanly() {
        // Audit fix DAT-RUST-BOM: production CSVs from Excel/Windows
        // include a UTF-8 BOM. csv crate doesn't strip it, so the
        // first header field reads as "\u{FEFF}date" not "date".
        // Without the strip, header validation fails.
        let tmp = std::env::temp_dir().join("rust_panel_bom.csv");
        let content = "\u{FEFF}date,ticker,f0,label\n2024-01-02,AAA,0.5,1.0\n";
        write_csv(&tmp, content);
        let p = Panel::load_csv(&tmp).expect("BOM-prefixed CSV must load");
        assert_eq!(p.feature_cols, vec!["f0"]);
        assert_eq!(p.n_dates(), 1);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn dat_rust_ws_dates_grouped_correctly() {
        // Audit fix DAT-RUST-WS: stray whitespace around dates from
        // Excel quoting was creating phantom date groups.
        let tmp = std::env::temp_dir().join("rust_panel_ws.csv");
        // Note the trailing space after "2024-01-02" — would be a
        // separate BTreeMap key without the trim.
        write_csv(&tmp, "\
date,ticker,f0,label
2024-01-02,AAA,0.5,1.0
 2024-01-02 ,BBB,0.6,1.5
2024-01-02,CCC,0.7,2.0
");
        let p = Panel::load_csv(&tmp).expect("ws-padded dates must group");
        assert_eq!(p.n_dates(), 1, "all 3 rows on the same trimmed date");
        assert_eq!(p.n_rows(), 3);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn dat_rust_ws_features_parse() {
        // Padded numeric cells should still parse as f32.
        let tmp = std::env::temp_dir().join("rust_panel_ws_feat.csv");
        write_csv(&tmp, "\
date,ticker,f0,f1,label
2024-01-02,AAA, 0.5 , -0.3 , 1.0
");
        let p = Panel::load_csv(&tmp).expect("ws-padded features parse");
        let rows = &p.features_by_date["2024-01-02"];
        assert_eq!(rows[0].1, vec![0.5_f32, -0.3_f32]);
        assert_eq!(rows[0].2, 1.0_f32);
        let _ = std::fs::remove_file(&tmp);
    }
}
