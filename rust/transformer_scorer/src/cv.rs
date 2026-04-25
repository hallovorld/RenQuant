//! Purged K-fold cross-validation with embargo. Direct port of the
//! Python `PurgedKFold` in training_panel/purged_cv.py, post-CV-1 fix
//! (purge window equals full lookahead_days, not L-1).
//!
//! Reference: López de Prado, *Advances in Financial Machine Learning*,
//! ch.7. The standard K-fold is biased on time-series panels because
//! train and test labels overlap; purge + embargo restore independence.
//!
//! API:
//!   let cv = PurgedKFold { n_splits, embargo_days, lookahead_days };
//!   for (train_idx, test_idx) in cv.split(&dates_days_since_epoch)? {
//!       train(&panel.select(train_idx), ...);
//!       test (&panel.select(test_idx),  ...);
//!   }
//!
//! `dates` is `&[i64]` of days-since-epoch, one entry per row of the
//! panel. The caller is responsible for converting their date column
//! to that format (chrono::NaiveDate::num_days_from_ce, or
//! `(date - UNIX_EPOCH).num_days()`).
//!
//! Same purge/embargo arithmetic as the Python version — verified
//! by the parity test below on a synthetic 30-row, 6-fold split.

use anyhow::{anyhow, Result};

#[derive(Debug, Clone, Copy)]
pub struct PurgedKFold {
    pub n_splits:       usize,
    pub embargo_days:   i64,
    pub lookahead_days: i64,
}

impl Default for PurgedKFold {
    fn default() -> Self {
        Self { n_splits: 5, embargo_days: 5, lookahead_days: 5 }
    }
}

impl PurgedKFold {
    /// Yield `(train_idx, test_idx)` pairs of positional row indices.
    ///
    /// `dates`: one date per row, in days-since-epoch (or any
    /// monotonic integer; only relative differences matter for purge
    /// + embargo arithmetic).
    pub fn split(&self, dates: &[i64]) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if self.n_splits < 2 {
            return Err(anyhow!("n_splits must be >= 2"));
        }
        let n_rows = dates.len();
        if n_rows == 0 {
            return Err(anyhow!("dates slice is empty"));
        }

        // Unique sorted dates.
        let mut unique: Vec<i64> = dates.to_vec();
        unique.sort_unstable();
        unique.dedup();
        let n_dates = unique.len();
        if n_dates < self.n_splits {
            return Err(anyhow!(
                "not enough unique dates ({}) for {}-fold CV",
                n_dates, self.n_splits,
            ));
        }

        // Contiguous date-fold edges via linspace.
        let fold_edges: Vec<usize> = (0..=self.n_splits)
            .map(|k| {
                ((k as f64) * (n_dates as f64) / (self.n_splits as f64)).round() as usize
            })
            .collect();

        let mut out = Vec::with_capacity(self.n_splits);
        for k in 0..self.n_splits {
            let lo = fold_edges[k];
            let hi = fold_edges[k + 1];
            if hi <= lo {
                continue;   // empty fold edge case
            }
            let test_dates = &unique[lo..hi];
            let test_start = *test_dates.first().unwrap();
            let test_end   = *test_dates.last().unwrap();

            let purge_start = test_start - self.lookahead_days;
            let embargo_end = test_end + self.embargo_days;

            let mut test_idx  = Vec::new();
            let mut train_idx = Vec::new();
            for (i, &d) in dates.iter().enumerate() {
                let is_test = d >= test_start && d <= test_end;
                if is_test {
                    test_idx.push(i);
                    continue;
                }
                // Purge: rows dated in [test_start - L, test_start)
                if d >= purge_start && d < test_start {
                    continue;
                }
                // Embargo: rows dated in (test_end, test_end + emb]
                if d > test_end && d <= embargo_end {
                    continue;
                }
                train_idx.push(i);
            }
            out.push((train_idx, test_idx));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a panel of `(date, ticker)` rows where date increments by
    /// 1 per row block; mimics the Python panel layout.
    fn synthetic_dates(n_dates: usize, tickers_per_date: usize) -> Vec<i64> {
        let mut out = Vec::with_capacity(n_dates * tickers_per_date);
        for d in 0..n_dates as i64 {
            for _ in 0..tickers_per_date {
                out.push(d);
            }
        }
        out
    }

    #[test]
    fn split_yields_n_folds() {
        let dates = synthetic_dates(30, 4);   // 30 dates × 4 tickers
        let cv = PurgedKFold { n_splits: 6, embargo_days: 0, lookahead_days: 0 };
        let folds = cv.split(&dates).unwrap();
        assert_eq!(folds.len(), 6);
    }

    #[test]
    fn test_indices_disjoint_per_fold() {
        let dates = synthetic_dates(30, 1);
        let cv = PurgedKFold { n_splits: 5, embargo_days: 0, lookahead_days: 0 };
        let folds = cv.split(&dates).unwrap();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (_, test_idx) in &folds {
            for &i in test_idx {
                assert!(seen.insert(i), "test index {} duplicated across folds", i);
            }
        }
        assert_eq!(seen.len(), 30, "every row should appear in exactly one test fold");
    }

    #[test]
    fn purge_drops_train_rows_in_window() {
        // 20 dates, 1 ticker each. 5 folds → fold k=2 covers dates [8..12).
        // With L=3, purge window is [8-3, 8) = [5, 8). Those rows should
        // NOT be in the train set for fold 2.
        let dates = synthetic_dates(20, 1);
        let cv = PurgedKFold { n_splits: 5, embargo_days: 0, lookahead_days: 3 };
        let folds = cv.split(&dates).unwrap();
        let (train_idx, test_idx) = &folds[2];
        // Test fold should be the rows dated 8..12 (linspace(0,20,6) = [0,4,8,12,16,20])
        let test_dates: Vec<i64> = test_idx.iter().map(|&i| dates[i]).collect();
        assert_eq!(test_dates, vec![8, 9, 10, 11]);
        // Train must NOT contain rows dated 5,6,7 (purged).
        for &i in train_idx {
            assert!(
                !(5..=7).contains(&dates[i]),
                "row {} (date={}) should be purged but is in train",
                i, dates[i],
            );
        }
        // Train MUST contain rows dated 0..4 (before purge window).
        let train_dates: Vec<i64> = train_idx.iter().map(|&i| dates[i]).collect();
        for d in 0..=4 {
            assert!(train_dates.contains(&(d as i64)),
                    "row dated {} should be in train (before purge window)", d);
        }
    }

    #[test]
    fn embargo_drops_train_rows_after_test() {
        let dates = synthetic_dates(20, 1);
        let cv = PurgedKFold { n_splits: 5, embargo_days: 2, lookahead_days: 0 };
        let folds = cv.split(&dates).unwrap();
        // Fold 1 covers dates [4..8). Embargo window: (7, 9].
        let (train_idx, _) = &folds[1];
        let train_dates: Vec<i64> = train_idx.iter().map(|&i| dates[i]).collect();
        // Embargoed dates 8 + 9 must NOT be in train.
        for d in [8, 9] {
            assert!(!train_dates.contains(&d),
                    "row dated {} should be embargoed but is in train", d);
        }
        // Date 10 (just outside embargo) MUST be in train.
        assert!(train_dates.contains(&10),
                "row dated 10 should be in train (just outside embargo)");
    }

    #[test]
    fn cv1_full_lookahead_purge() {
        // Regression for CV-1 (Python 2026-04-25 fix): with lookahead=5,
        // train must NOT contain test_start-5. Pre-fix purged only L-1=4
        // days, leaving test_start-5 as a leaky training row.
        let dates = synthetic_dates(30, 1);
        let cv = PurgedKFold { n_splits: 6, embargo_days: 0, lookahead_days: 5 };
        let folds = cv.split(&dates).unwrap();
        let (train_idx, test_idx) = &folds[1];   // dates 5..10
        let test_start = dates[test_idx[0]];
        // train must NOT include test_start - L = 0 here (with k=1, test_start=5, L=5).
        // Actually with linspace(0,30,7) = [0,5,10,15,20,25,30] → k=1 dates 5..10.
        // test_start=5, L=5, purge_start=0. So [0..5) is purged.
        let train_dates: Vec<i64> = train_idx.iter().map(|&i| dates[i]).collect();
        for d in 0..5 {
            assert!(
                !train_dates.contains(&(d as i64)),
                "CV-1 regression: date {} (test_start - L = {}) leaked into train",
                d, test_start - 5,
            );
        }
    }

    #[test]
    fn rejects_n_splits_too_few() {
        let dates = synthetic_dates(3, 1);
        let cv = PurgedKFold { n_splits: 5, embargo_days: 0, lookahead_days: 0 };
        let result = cv.split(&dates);
        assert!(result.is_err(), "should reject when fewer dates than n_splits");
    }

    #[test]
    fn rejects_n_splits_under_2() {
        let dates = synthetic_dates(20, 1);
        let cv = PurgedKFold { n_splits: 1, embargo_days: 0, lookahead_days: 0 };
        assert!(cv.split(&dates).is_err());
    }
}
