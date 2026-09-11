#!/usr/bin/env bash
#
# Cron entry point for the weekly benchmark. Runs `check` and acts on what it
# reports: `dispatch` when a run is due, `finish` when a dispatched run has
# landed. Meant for a host that holds a persistent, already-authenticated ssh
# ControlMaster to Vulcan (so it never hits the MFA prompt), not a laptop that
# sleeps.
#
# crontab -e:
#   0 * * * * PATH="$HOME/.local/bin:$PATH" /path/to/online-gsp/benchmarks/core/cron.sh
#
# Notifies by email (WEEKLY_BENCHMARK_NOTIFY, default below) via the local
# `mail` command - "finish" every time (success or failure), and a Vulcan
# connection failure once when it starts and once when it clears (not every
# tick of a multi-hour outage). A dispatch blocked by a lapsed MFA session is
# that same connection failure: reopen it with `ssh vulcan true` and the next
# tick dispatches by itself, since the run stays due until one succeeds. A
# dispatch that fails for any other reason reports once too, and again once a
# later dispatch has gone through.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

NOTIFY="${WEEKLY_BENCHMARK_NOTIFY:-parham1@ualberta.ca}"
LOG="$REPO/.cluster/weekly-cron.log"
UNREACHABLE_MARKER="$REPO/.cluster/weekly-cron.unreachable"
FAILED_MARKER="$REPO/.cluster/weekly-cron.dispatch-failed"
mkdir -p "$(dirname "$LOG")"

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"; }

notify_unreachable() {
  if [ ! -f "$UNREACHABLE_MARKER" ]; then
    echo "$1" | mail -s "[online-gsp] weekly benchmark: can't reach Vulcan" "$NOTIFY"
    touch "$UNREACHABLE_MARKER"
  fi
}

check_out=$(uv run benchmarks/core/weekly.py check 2>&1)
check_status=$?
log "check ($check_status): $check_out"

unreachable=0

case "$check_status" in
  0)
    ;;  # idle, or a dispatch is still queued - nothing to do
  10)
    dispatch_out=$(uv run benchmarks/core/weekly.py dispatch 2>&1)
    dispatch_status=$?
    log "dispatch ($dispatch_status): $dispatch_out"
    if [ "$dispatch_status" = "30" ]; then
      unreachable=1
      notify_unreachable "$dispatch_out"
    elif [ "$dispatch_status" != "0" ]; then
      if [ ! -f "$FAILED_MARKER" ]; then
        echo "$dispatch_out" \
          | mail -s "[online-gsp] weekly benchmark dispatch FAILED" "$NOTIFY"
        touch "$FAILED_MARKER"
      fi
    else
      rm -f "$FAILED_MARKER"
    fi
    ;;
  20)
    if finish_out=$(uv run benchmarks/core/weekly.py finish 2>&1); then
      log "finish: $finish_out"
      echo "$finish_out" | mail -s "[online-gsp] weekly benchmark PR opened" "$NOTIFY"
    else
      log "finish FAILED: $finish_out"
      echo "$finish_out" | mail -s "[online-gsp] weekly benchmark finish FAILED" "$NOTIFY"
    fi
    ;;
  30)
    unreachable=1
    notify_unreachable "$check_out"
    ;;
  *)
    log "check exited unexpectedly"
    echo "$check_out" | mail -s "[online-gsp] weekly benchmark check FAILED ($check_status)" "$NOTIFY"
    ;;
esac

if [ "$unreachable" = "0" ] && [ -f "$UNREACHABLE_MARKER" ]; then
  rm -f "$UNREACHABLE_MARKER"
  echo "Vulcan is reachable again as of this tick." \
    | mail -s "[online-gsp] weekly benchmark: Vulcan connection recovered" "$NOTIFY"
fi
