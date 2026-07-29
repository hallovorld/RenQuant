# Progress: the production trainer could not be pointed at a rebuilt panel

STATUS:   delivered (one flag, threaded as a parameter). No production file
          touched, no training run performed by this PR.

WHAT:     `scripts/train_production_model.py` gains `--panel`. The input panel
          path was hardcoded at what is now line ~240
          (`pd.read_parquet("data/alpha158_291_fundamental_dataset.parquet")`).

WHY/DIR:  Found while trying to retrain on a rebuilt 300-ticker panel for the
          9-ticker watchlist batch, WITHOUT overwriting production.

          `--output-path`, `--watchlist-file` and `--fingerprint-config` were
          all already redirectable. The INPUT PANEL was the one thing that was
          not — so the only way to train on a rebuilt panel was to overwrite
          `data/alpha158_291_fundamental_dataset.parquet` in place. That is the
          same move that gutted 82 calibrators on 2026-06-17.

          This is the second instance of the same gap today: the umbrella's
          `scripts/build_alpha158_fund_panel.py` also hardcodes its input and
          output (its base-data counterpart,
          `renquant_base_data.alpha158_fund_panel`, does NOT — it takes
          `data_dir` + `output_path`, which is what made the isolated panel
          rebuild possible at all). The pattern: the base-data modules are
          parameterised, the umbrella wrappers are not, and the wrappers are
          what the retrain pipeline calls.

EVIDENCE: artifact: `scripts/train_production_model.py`.
  prod or exp:      PROD script, one new optional flag. Default behaviour is
                    byte-identical: `panel_path or
                    "data/alpha158_291_fundamental_dataset.parquet"`.
  existing data:    Yes, established this session by reading the code:
                    `train_production_model.py` read the panel from a hardcoded
                    literal `[VERIFIED]`; `--output-path`, `--watchlist-file`,
                    `--fingerprint-config` already existed `[VERIFIED]`;
                    `TrainPanelLTRTask` already passes `--output-path` to a
                    staged candidate and its docstring states the boundary
                    ("training may create candidate artifacts, but active
                    production files are touched only by promote()")
                    `[VERIFIED]`. So the output side was already isolated and
                    only the input side blocked an isolated run.
  best-known?:      Yes for the gap. NOT claimed: that any retrain has been
                    run, or that the rebuilt panel trains a better model.
  scope:            One umbrella script. No pin advanced, no config edited, no
                    data file written, no artifact produced.

A BUG IN MY OWN FIRST PATCH, caught before committing:
          The first version read `getattr(args, "panel", None)` inside
          `load_and_slice_panel(...)`, which has NO `args` in scope — a
          NameError on every call, including the default path. It is now an
          explicit `panel_path` parameter threaded from the single call site,
          verified by walking the AST of that function and asserting `args` is
          not referenced in it.

NEXT:     With this flag, Phase 4 of the 9-ticker batch can train against the
          isolated 300-ticker panel and write to a staged candidate artifact,
          touching no production file. The candidate still does NOT go live —
          `promote()` after the acceptance/WF gates is the only path to that,
          and it is not part of this work.
