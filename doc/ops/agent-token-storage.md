# SOP — agent GitHub token storage (Claude + Codex)

**Status:** canonical operating procedure. **Scope:** all renquant repos +
the orchestrator's multi-agent PR workflows.

Each agent (Claude, Codex) authenticates to GitHub with its **own**
fine-grained PAT, so tokens are independently revocable, rate-limit-isolated,
and individually auditable. This SOP defines how those tokens are **stored,
loaded, and rotated** — safely. It complements
[`agent-automation.md`](agent-automation.md) (§3.7 identity/attribution canon)
and the orchestrator's `doc/agent-pr-workflows.md` (which already reads
`RENQUANT_<AGENT>_GH_TOKEN`).

## 0 · The three rules (non-negotiable)

1. **A token never appears in plaintext** in chat, a transcript, a commit, a
   file in any repo, or shell history/argv.
2. **Storage is the OS Keychain**, encrypted at rest. Not `.env`, not a dotfile.
3. **Each agent's token is independently revocable** — revoking one never
   affects the other.

A pre-push hook (`scripts/install_pr_hook.sh`) blocks any push whose diff
contains a real token shape — defense-in-depth for rule 1.

## 1 · Provision the PATs (GitHub → Settings → Developer settings → Fine-grained tokens)

Two tokens, **Resource owner = `hallovorld`**, **Repository access = only the
`renquant-*` repositories**, **Repository permissions**:

| Permission | Access |
|---|---|
| Contents | Read & write |
| Pull requests | Read & write |
| Issues | Read & write |
| Workflows | Read & write |
| (everything else) | No access |

Name them `renquant-gh-claude` and `renquant-gh-codex`. Set an expiry (90 days)
and calendar the rotation.

> Attribution stays via the §3.7 convention (`Co-Authored-By` trailers +
> `agent:claude` / `agent:codex` labels); both tokens act as `hallovorld`. If
> true distinct authorship is ever needed, upgrade to machine-user accounts —
> the storage/loading below is unchanged.

## 2 · Store them in the Keychain (run YOURSELF, in a terminal)

The `-w` with **no value** prompts (hidden), so the token never reaches argv,
history, or any agent/transcript. **Do not paste a token into an agent chat.**

```bash
security add-generic-password -U -a "$USER" -s renquant-gh-claude -w
security add-generic-password -U -a "$USER" -s renquant-gh-codex  -w
# paste the matching token at each hidden prompt
```

Verify presence without revealing the value:

```bash
security find-generic-password -s renquant-gh-claude >/dev/null && echo "claude: stored"
security find-generic-password -s renquant-gh-codex  >/dev/null && echo "codex: stored"
```

## 3 · Load them at runtime (`scripts/agent_gh_env.sh`, never prints the token)

```bash
# A single agent's interactive session → GH_TOKEN + RENQUANT_<AGENT>_GH_TOKEN:
source scripts/agent_gh_env.sh claude
source scripts/agent_gh_env.sh codex

# The orchestrator (dispatches BOTH agents) → RENQUANT_{CLAUDE,CODEX}_GH_TOKEN:
source scripts/agent_gh_env.sh --orchestrator
python -m renquant_orchestrator ...   # picks up RENQUANT_<AGENT>_GH_TOKEN per its --as <agent>
```

The shim reads the Keychain into the env vars the orchestrator + `gh` already
expect (`RENQUANT_CLAUDE_GH_TOKEN`, `RENQUANT_CODEX_GH_TOKEN`, `GH_TOKEN`). The
token lives only in that shell's environment — never on disk.

## 4 · Leak-prevention hook (install once per clone)

```bash
bash scripts/install_pr_hook.sh --all
```

The pre-push hook blocks a push whose diff contains a real token shape
(`gh{p,o,u,s}_…`, `github_pat_…`). Bare prefix mentions in docs do not match,
so this SOP pushes cleanly. Emergency override: `PR_HOOK_BYPASS=1 git push …`.

## 5 · Rotation runbook

1. Revoke the old PAT on GitHub (Fine-grained tokens → Revoke).
2. Mint a replacement with the same scope (§1).
3. Re-run the **same** `add-generic-password -U` from §2 — it overwrites.
4. Done. No code or file changes; the next `source` picks up the new value.

Rotate immediately (not on schedule) if a token is ever exposed — including
being pasted into any chat/transcript.

## 6 · What NOT to do

- ❌ Paste a token into an agent chat or any transcript (rotate it if you do).
- ❌ Put tokens in `.env`, a dotfile, `git config`, or a CI variable file in-repo.
- ❌ `echo $GH_TOKEN`, or pass a token as a command-line argument.
- ❌ Reuse one token for both agents (breaks independent revocation + audit).
