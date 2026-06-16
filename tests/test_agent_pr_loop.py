from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "agent_pr_loop.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("agent_pr_loop_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_agent_prompt_review_mentions_visible_review_marker() -> None:
    mod = _load_module()

    prompt = mod.build_agent_prompt("codex", "review")

    assert "reviewed by codex" in prompt
    assert "repos agent --as codex --workflow review --repo all" in prompt
    assert "Do not touch PRs outside that queue." in prompt


def test_build_agent_prompt_fix_mentions_visible_fix_marker() -> None:
    mod = _load_module()

    prompt = mod.build_agent_prompt("claude", "fix")

    assert "fixed by claude" in prompt
    assert "repos agent --as claude --workflow fix --repo all" in prompt


def test_queue_total_sums_cross_repo_plan_rows() -> None:
    mod = _load_module()

    total = mod._queue_total(
        {
            "repos": [
                {"repo": "RenQuant", "plan": {"queue": [{"number": 1}, {"number": 2}]}},
                {"repo": "renquant-pipeline", "plan": {"queue": [{"number": 3}]}},
                {"repo": "renquant-model", "plan": {"queue": []}},
            ]
        }
    )

    assert total == 3
