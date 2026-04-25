#!/usr/bin/env bash
# rust/setup.sh — one-shot installer for the RenQuant Rust workspace.
#
# Mirrors the conda env bootstrap pattern: idempotent, prints what it
# does, exits non-zero on any failure. Safe to re-run.
#
# What this does:
#   1. Installs rustup (toolchain manager) if missing — official script.
#   2. The rust-toolchain.toml pin auto-installs the matching toolchain
#      on the first cargo invocation; we trigger that explicitly here.
#   3. Adds rust-analyzer + clippy + rustfmt components (already pinned
#      in rust-toolchain.toml; this is the verification step).
#   4. Pre-fetches dependencies into the workspace cache (`cargo fetch`)
#      so the first build is offline-clean.
#   5. Verifies the build compiles (`cargo check`).
#
# Usage:
#   bash rust/setup.sh                # install + verify
#   bash rust/setup.sh --skip-fetch   # skip the network prefetch
#
# Output: tells you the build command for downstream use.

set -euo pipefail

# ── Resolve script directory + project root ──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colour helpers ───────────────────────────────────────────────────────
say()  { printf "\033[1;36m[setup]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m[ok]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

SKIP_FETCH=false
for arg in "$@"; do
    case "$arg" in
        --skip-fetch) SKIP_FETCH=true ;;
        --help|-h)
            sed -n '1,/^set -euo/p' "$0" | grep '^#' | sed 's/^# //'
            exit 0
            ;;
        *) die "unknown arg: $arg" ;;
    esac
done

# ── 1. Install rustup if missing ─────────────────────────────────────────
if ! command -v rustup >/dev/null 2>&1; then
    say "rustup not found — installing via official rustup-init.sh"
    if ! command -v curl >/dev/null 2>&1; then
        die "curl is required to install rustup; install curl first"
    fi
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- --default-toolchain none --no-modify-path -y
    export PATH="$HOME/.cargo/bin:$PATH"
    ok  "rustup installed at $HOME/.cargo/bin"
    say "Add this line to ~/.zshrc or ~/.bashrc:  source \"\$HOME/.cargo/env\""
else
    ok "rustup present: $(rustup --version)"
fi

# Make `cargo` reachable in this script regardless of shell init order.
export PATH="$HOME/.cargo/bin:$PATH"

# ── 2. Install toolchain pinned in rust-toolchain.toml ───────────────────
# The pinned channel is auto-installed lazily by cargo, but a controlled
# explicit install gives a clean log.
say "Installing pinned toolchain (rust-toolchain.toml drives the channel)"
rustup show active-toolchain
rustup component add rustfmt clippy rust-src 2>/dev/null || true

ok  "Toolchain ready: $(rustc --version)"

# ── 3. Pre-fetch crates ─────────────────────────────────────────────────
if [ "$SKIP_FETCH" = false ]; then
    say "Fetching workspace dependencies"
    cargo fetch --locked 2>/dev/null || cargo fetch
    ok  "Crates fetched"
else
    warn "Skipping cargo fetch (--skip-fetch)"
fi

# ── 4. Verify build ─────────────────────────────────────────────────────
say "Compiling (cargo check) to verify environment"
cargo check --workspace
ok  "cargo check passed"

# ── 5. Final smoke test: build the score-panel binary ───────────────────
say "Building release binary"
cargo build --workspace --release
BIN="target/release/score-panel"
if [ -x "$BIN" ]; then
    ok "Built $BIN"
    say "Try: ./$BIN --help"
else
    die "binary missing after build: $BIN"
fi

cat <<'EOF'

──────────────────────────────────────────────────────────────────────
Rust workspace ready.

Common commands (run from rust/):
  cargo dev          # debug build of all crates
  cargo rls          # release build (default — fastest binary)
  cargo metal        # release build with Apple Metal backend
  cargo test-all     # run all tests, all features
  cargo lint         # clippy with -D warnings
  cargo fmt-check    # rustfmt CI gate

Run the scorer:
  ./target/release/score-panel \
      --artifact ../backtesting/renquant_104/artifacts/panel-transformer \
      --input    /tmp/panel_features.csv \
      --output   /tmp/panel_scores.csv

──────────────────────────────────────────────────────────────────────
EOF
