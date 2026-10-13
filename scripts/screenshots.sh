#!/usr/bin/env bash
# Capture dashboard screenshots (server must be running on :7860). Requires Google Chrome.
set -euo pipefail
CHROME="${CHROME:-google-chrome}"
[ -x "/c/Program Files/Google/Chrome/Application/chrome.exe" ] && CHROME="/c/Program Files/Google/Chrome/Application/chrome.exe"
OUT="$(cd "$(dirname "$0")/.." && pwd)/docs/screenshots"; mkdir -p "$OUT"
command -v cygpath >/dev/null && OUTP="$(cygpath -w "$OUT")" || OUTP="$OUT"
URL="${URL:-http://127.0.0.1:7860}"
shot() { "$CHROME" --headless=new --disable-gpu --hide-scrollbars --virtual-time-budget=15000 --window-size="$2" --screenshot="$OUTP/$1.png" "$URL/$3" >/dev/null 2>&1; echo "saved $1.png"; }
shot home        1440,1500 "?theme=dark#/"
shot analyze     1440,1500 "?theme=dark#/analyze?preset=vague&run=1"
shot performance 1440,1700 "?theme=dark#/performance"
shot explain     1440,1500 "?theme=dark#/explain"
shot errors      1440,1300 "?theme=dark#/errors"
shot dataset     1440,1500 "?theme=dark#/dataset"
shot experiments 1440,1300 "?theme=dark#/experiments"
shot batch       1440,1100 "?theme=light#/batch"
