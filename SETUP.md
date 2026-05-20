# RenQuant Setup Guide ⚙️

> **📅 Superseded — see [`doc/ops/setup.md`](doc/ops/setup.md) for current setup.**
> The instructions previously in this file directed users to Miniconda; the project
> now uses a project-local `.venv` per `feedback_python_env` memory. CLAUDE.md
> Environment section + `doc/ops/setup.md` are the canonical references.

## Quick reference

```bash
# Current setup (2026-05-20)
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
# Includes transformers >= 5.8.1 + accelerate >= 1.1.0 for HF Trainer-based
# PatchTST shadow path
lean login
```

For HF PatchTST training context: `doc/research/2026-05-19-patchtst-improvement-plan.md`.
For broker mode (LIVE vs PAPER): CLAUDE.md Environment §"e2e".

Full setup guide: [`doc/ops/setup.md`](doc/ops/setup.md).
