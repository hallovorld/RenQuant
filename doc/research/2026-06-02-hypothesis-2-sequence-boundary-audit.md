# Hypothesis #2 audit — sequence boundary leak (PatchTST seq_len=24 windows cross train/val/embargo)

**Date**: 2026-06-02
**Author**: Claude
**Hypothesis (per 2026-06-02 experiment validity audit §4 #2)**:
> Sequence boundary (seq_len=24 PatchTST window 跨 train/val) — 最物理的 timeshift_placebo > real 解释. 下一步重点.

**Verdict**: **ruled_in** as a real split-purity violation under the project's PatchTST tensor contract. **Inconclusive** as the sole / dominant cause of the 2026-06-01 placebo > real verdict (E1 is still required to size the impact; other hypotheses still need separate audits before attribution).

R1/R4 compliance: this memo is the prerequisite for any new B_tuned re-run. No experiments launched.

---

## 1 · Code path examined

Repository: `renquant-model`. All file:line refs are from the audit snapshot pinned by umbrella commit `0620a77`: `renquant-model` commit `f9c598f` (per `subrepos.lock.json`).

| File | Lines | Role |
|---|---|---|
| `src/renquant_model_patchtst/hf_trainer.py` | 599-617 | `PerDayDataset.__init__` — window construction (the suspect path) |
| `src/renquant_model_patchtst/hf_trainer.py` | 363-407 | split assignment + 2026-05-31 timeshift-cross-split-leak fix (commit `7245d84`) |
| `src/renquant_model_patchtst/sequence_training.py` | 159-162 | `BuildDatasetsTask` — instantiates train + val datasets from same panel |
| `src/renquant_model_patchtst/splits.py` | 45-72 | `assign_patchtst_split` — emits `train`/`embargo`/`val`/`test` labels |

Relevant commit: `7245d84` `fix(hf_trainer): timeshift placebo MUST NOT cross split boundary (Tier-3 root cause)` — the 2026-05-31 fix that this audit re-examines for sufficiency.

## 2 · The window construction loop

`PerDayDataset.__init__` (hf_trainer.py:599-617):

```python
for ticker, idxs in panel.groupby("ticker", sort=False).indices.items():
    idxs = np.asarray(sorted(idxs))
    for i in range(seq_len, len(idxs)):
        end_pos = idxs[i]
        if panel.iloc[end_pos]["split_label"] != split:    # ← filter is end_pos only
            continue
        window = feat_arr[idxs[i - seq_len: i]]            # ← seq_len previous panel rows
        ...
```

The split filter is applied to `end_pos` only. The window `idxs[i - seq_len: i]` then collects the **previous `seq_len` panel positions for that ticker** without any split check.

For `split="val"`, an end_pos at the start of the val period builds a window from the most recent 24 panel rows preceding it — which are typically in the embargo zone (default `embargo_days=60`, hf_trainer.py:332) and can reach trailing train rows when a ticker has too few eligible embargo rows after NaN/drop filtering.

## 3 · Two leak surfaces

### 3.1 · Embargo zone bleeding into val windows

`assign_patchtst_split` returns labels `train` / `embargo` / `val` / `test`. The `embargo` zone exists specifically to gap-separate train from val so that train-period labels (with their forward-looking horizon) don't extend into val features. The dataset construction at hf_trainer.py:610 leaves `embargo` rows in `panel` (only rows whose `end_pos` matches the requested split are emitted as samples), but the window lookback at line 612 freely indexes into pre-val rows when building the first val sample windows.

**Concrete consequence**: for `split="val"`, a sample's `(window, label)` pair has:
- `window`: `seq_len` panel rows ending the day BEFORE end_pos for that ticker. With `seq_len=24`, the first 24 emitted val samples per ticker can have windows that span the embargo zone. This is a `seq_len`-bounded window issue, not a "first 60 val days" issue: `embargo_days=60` describes the pre-val gap length, while the number of emitted val samples affected is bounded by the lookback length until the ticker has accumulated `seq_len` clean val rows. After that, regularly covered tickers draw from earlier val rows (clean within-split). Gappy tickers can still cross farther back in calendar time because the code walks panel positions, not calendar days.

The embargo is supposed to be the explicit invariant per CLAUDE.md §7.1: `max(train_date) + label_lookahead_days < min(val_date)`. The dataset construction does NOT enforce this invariant on a per-window basis.

### 3.2 · Train→val feature contamination

Even after a ticker has enough clean val history, the window construction can still pull in earlier non-val rows if `idxs[i - seq_len: i]` spans them because val rows are gapped (e.g. ticker missing data in val) and the position-based lookback jumps back across the split — less common but not impossible.

The deeper structural concern: `idxs[i - seq_len: i]` walks `seq_len` POSITIONS in the NaN-dropped panel, NOT `seq_len` CALENDAR DAYS. This is the same defect class flagged by the 2026-05-31 placebo cross-split-leak fix (hf_trainer.py:383-391 comment), but that fix only addressed the LABEL shift path. The WINDOW path inherits the same NaN-positions-not-days issue without any split-boundary guard.

## 4 · Interaction with the 2026-05-31 cross-split-leak fix

Commit `7245d84` added at hf_trainer.py:395-402:

```python
n_shift = int(label_shift_days)
ticker_groups = panel.groupby("ticker", sort=False)
shifted_label = ticker_groups[label_col].shift(-n_shift)
shifted_split = ticker_groups["split_label"].shift(-n_shift)
panel.loc[train_mask, label_col] = shifted_label.loc[train_mask]
before = len(panel)
cross_split_leak = train_mask & shifted_split.ne("train")
nan_after_shift = train_mask & panel[label_col].isna()
panel = panel.loc[~(cross_split_leak | nan_after_shift)].copy()
```

This fixes ONE leak path: under `--label-shift-days N` placebo, train rows whose shifted SOURCE LABEL came from val/embargo are dropped. The fix is correct for the LABEL side.

But the fix does **NOT** address the WINDOW side. Specifically:
- Train labels: post-fix, clean within train (no cross-split shifted labels).
- Train windows: built from training rows; can still pull in earlier train rows that are themselves shifted-relabeled (not a leak — same split).
- Val labels: never shifted (placebo only modifies `train_mask`).
- **Val windows: still built from prev `seq_len` panel rows of the ticker, with no split check on the window members.** This is the surface §3.1 above identifies.

Result: the 2026-05-31 fix closes the label-side leak but the window-side split-purity violation persists. A timeshift placebo run after this fix trains on shifted train labels with train-end_pos windows, then evaluates on real val labels where the earliest val windows can include embargo/pre-val rows. Cross-period feature contamination on the val side is unchanged.

## 5 · Mechanism for placebo IC > real IC under this bug

For shuffle / timeshift placebo to produce val IC > 0 (let alone > real), there must be some path from feature periodicity to a "spurious" model output. The relevant path under the window bug:

1. **Embargo rows preserve feature continuity** with the trailing train period (no feature shuffling, just label-side embargo).
2. **Early val windows include those embargo rows**, so the val sample's feature tensor can contain up to `seq_len` days of pre-val context.
3. **Both real and placebo evaluation** see the same early-val feature distribution. The model learns weights from train; placebo's training labels are corrupted (shuffled / shifted), real's are intact.
4. **During val IC computation**, the model's output on an early val window is a function of features that can include embargo/pre-val period info. Slow factor persistence (momentum, low-frequency regime) creates a baseline correlation between recent features and the val target — even for a model with noisy weights.

The structural reason placebo can come within range of (or exceed) real IC: the val-window contamination can provide an **early-val IC component** that both real and placebo enjoy. Real adds the learned signal on top; if the learned signal is small (which the PatchTST B_tuned trial reports of ~+0.04 suggest), even a localized contamination component can be material. E1 is required to measure whether this component is large enough to dominate the full validation IC.

This is a plausible mechanism, but mechanism plausibility is not proof. The audit cannot rule out the OTHER 5 hypotheses without their own memos. In particular, hypothesis #3 (PerRegimeICCallback regime injection) could provide its own contamination floor; hypothesis #4 (CSRankNorm NaN-fill cross-day) is its own path.

## 6 · Falsifying experiments (planned, not run)

Per R4, list the experiments that would CONFIRM ruled-in:

**E1 — split-aware window construction:**
Modify `PerDayDataset.__init__` to add `if panel.iloc[idxs[i - seq_len:i]]["split_label"].ne(split).any(): continue` and re-run the B_tuned Tier-3 harness. If placebo IC materially drops while real IC holds, hypothesis #2 is a dominant contributor. If both drop equally or placebo stays high, it is one of several contributors.

**E2 — pure-val windows (no embargo bleed):**
Manually slice the panel so val sample windows can ONLY draw from val rows of the ticker (require `min_val_history >= seq_len` before emitting any val sample). Re-run; same comparison as E1. This is a stricter form of E1 — confirms the sufficient version of the fix.

**E3 — feature-only contamination floor:**
Replace the model with a constant (e.g. predict the mean of window feature column 0). Compute val IC against shifted-by-N labels for N ∈ {0, 5, 20, 60, 90}. If IC at N=60 is non-trivial, the contamination floor exists regardless of model learning. This isolates the feature-side leak from the label-side.

E1 is the cheapest (one line of code, one Tier-3 run). E3 is the cleanest mechanistic separator.

## 7 · Smallest fix (if E1 confirms)

```python
# hf_trainer.py:608-617, add split check on the entire window
for i in range(seq_len, len(idxs)):
    end_pos = idxs[i]
    if panel.iloc[end_pos]["split_label"] != split:
        continue
    window_idxs = idxs[i - seq_len: i]
    if panel.iloc[window_idxs]["split_label"].ne(split).any():
        continue                              # ← new: drop windows that cross split
    window = feat_arr[window_idxs]
    ...
```

This loses the first `seq_len` emitted val samples per ticker (no qualifying clean window). For seq_len=24 and val ~12-18 months, that's typically <5% of val samples for regularly covered tickers. Acceptable tradeoff for the cross-split invariant.

A stricter alternative (E2): require the ticker to accumulate `embargo_days + seq_len` clean val history before emitting val samples, so the window is both split-pure and far past the train/val gap. This loses more val samples but stress-tests the splitter's explicit invariant.

## 8 · Verdict

**Ruled_in**: the dataset construction at hf_trainer.py:610-612 admits early val sample windows that include embargo/pre-val rows and can include train rows for gappy tickers, in violation of the project's split-pure PatchTST tensor contract. This is a real cross-split feature contamination path that the 2026-05-31 label-side fix does not address.

**Inconclusive** on attribution: whether this surface is THE cause of the 2026-06-01 placebo > real verdict requires E1 (one-line fix + Tier-3 re-run) to confirm. Other hypotheses (#3 PerRegimeICCallback, #4 CSRankNorm, #5 Winsor, #6 dataloader/seed) need their own memos before any one can be declared "the" cause.

## 9 · References

- 2026-06-02 experiment validity audit: `doc/research/2026-06-02-experiment-validity-audit.md`
- 2026-06-01 process post-mortem: `doc/research/2026-06-01-leakage-reflection.md`
- 2026-06-01 leakage architecture (PR #43 v12): `doc/research/2026-06-01-leakage-architecture.md`
- 2026-05-31 cross-split-leak fix: `renquant-model` commit `7245d84`
- CLAUDE.md §7.2 sanity discipline, §7.1 splitter embargo invariant
- CLAUDE.md §7.2.1 R1 (verdict-from-file) + R4 (hypothesis-by-hypothesis audit memo)
