"""Regression guards for agent workflow startup behavior."""
from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parent.parent


def _workflow(name: str) -> str:
    return (REPO / ".github" / "workflows" / name).read_text()


def test_agent_review_template_does_not_require_repo_secrets() -> None:
    src = _workflow("_agent-review-template.yml")
    assert "api_key:\n        required: false" in src
    assert "Check agent API key availability" in src
    assert "skipping ${AGENT} review without failing workflow startup" in src
    assert "steps.secret_gate.outputs.has_api_key == 'true'" in src
    assert "always() && steps.secret_gate.outputs.has_api_key == 'true'" in src
    assert "issues: write" in src


def test_agent_fix_template_does_not_require_repo_secrets() -> None:
    src = _workflow("_agent-fix-template.yml")
    assert "api_key: { required: false }" in src
    assert "required: false" in src
    assert "Check agent secrets availability" in src
    assert "skipping ${AGENT} auto-fix without failing workflow startup" in src
    assert "secrets.git_push_token || github.token" in src
    assert "contents: write" in src
    assert "issues: write" in src
    assert "always() && steps.secret_gate.outputs.has_api_key == 'true'" in src


def test_renquant_wrappers_use_branch_local_templates() -> None:
    wrappers = {
        "agent-attribution-check.yml": [
            "./.github/workflows/_agent-attribution-check-template.yml",
        ],
        "agent-review.yml": [
            "./.github/workflows/_agent-review-template.yml",
        ],
        "agent-autofix.yml": [
            "./.github/workflows/_agent-fix-template.yml",
        ],
    }
    for workflow_name, expected_uses in wrappers.items():
        src = _workflow(workflow_name)
        assert "uses: hallovorld/RenQuant/.github/workflows/" not in src
        assert "permissions:" in src
        for expected in expected_uses:
            assert f"uses: {expected}" in src


def test_attribution_template_creates_a_job_for_non_agent_prs() -> None:
    src = _workflow("_agent-attribution-check-template.yml")
    assert "Decide whether attribution enforcement applies" in src
    assert "no agent authorship label present; attribution check not applicable" in src
    assert "if: steps.enforce.outputs.run == 'true'" in src
    assert "jobs:\n  check:\n    # Only enforce" not in src
