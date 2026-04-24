#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_BIN="$SCRIPT_DIR/ldpc_gui/build/ldpc_gui"

cd "$SCRIPT_DIR"

if [[ ! -x "$GUI_BIN" ]]; then
  printf 'LDPC GUI binary not found or not executable:\n%s\n\n' "$GUI_BIN" >&2
  printf 'Build it first with:\ncd "%s/ldpc_gui" && mkdir -p build && cd build && cmake .. && make\n' "$SCRIPT_DIR" >&2
  exit 1
fi

exec "$GUI_BIN" "$@"
