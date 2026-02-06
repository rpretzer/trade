#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Stock Arbitrage Model – Quick Install
#
# Downloads dependencies, sets up a virtual environment, and drops you
# into the interactive CLI where you can install optional feature packs
# (LSTM training, live trading, advanced analysis) on demand.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── colour helpers (honour NO_COLOR) ─────────────────────────────────
if [ -z "${NO_COLOR:-}" ] && [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' RESET=''
fi
info()    { printf "${CYAN}[info]${RESET}    %s\n" "$*"; }
ok()      { printf "${GREEN}[ok]${RESET}      %s\n" "$*"; }
warn()    { printf "${YELLOW}[warn]${RESET}    %s\n" "$*"; }
err()     { printf "${RED}[error]${RESET}   %s\n" "$*" >&2; }

# ── 1. Python version check ─────────────────────────────────────────
# Prefer 3.12 (TensorFlow compat); 3.11 is fine; 3.10+ required.
pick_python() {
    for candidate in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$candidate" &>/dev/null; then
            local ver
            ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            local major minor
            major="${ver%%.*}"
            minor="${ver#*.}"
            if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

PYTHON="$(pick_python)" || {
    err "Python 3.10+ is required but none was found on PATH."
    err "Install Python 3.12 (recommended) and try again."
    exit 1
}
PYVER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Using $PYTHON ($PYVER)"

if [ "$PYVER" != "3.12" ] && [ "$PYVER" != "3.11" ]; then
    warn "Python $PYVER detected.  LSTM training requires 3.11 or 3.12."
    warn "You can still run backtests and the dashboard on this version."
fi

# ── 2. Virtual environment ───────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/venv"
if [ -d "$VENV_DIR" ]; then
    info "Virtual environment already exists at $VENV_DIR"
else
    info "Creating virtual environment …"
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# Activate (source so pip/python resolve inside venv)
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ── 3. Upgrade pip ───────────────────────────────────────────────────
info "Upgrading pip …"
pip install --quiet --upgrade pip

# ── 4. Core dependencies ─────────────────────────────────────────────
info "Installing core dependencies …"
pip install --quiet -r requirements.txt
ok "Core dependencies installed"

# ── 5. Interactive optional-dependency installer ────────────────────
info "Checking optional feature packs …"
"$PYTHON" main.py deps

# ── 6. Done ──────────────────────────────────────────────────────────
printf "\n${BOLD}${GREEN}────────────────────────────────────────────────────────────────────${RESET}\n"
printf "${BOLD}  You're all set.${RESET}\n\n"
printf "  Start the interactive CLI:\n"
printf "      ${BOLD}source $VENV_DIR/bin/activate && python main.py${RESET}\n\n"
printf "  Or run a single command directly:\n"
printf "      ${BOLD}python main.py backtest${RESET}\n"
printf "      ${BOLD}python main.py dashboard${RESET}\n"
printf "${BOLD}${GREEN}────────────────────────────────────────────────────────────────────${RESET}\n\n"
