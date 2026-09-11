#!/usr/bin/env bash
# Turn the automated weekly benchmark on or off, on a remote host.
#
#   ./scripts/weekly-benchmarks.sh --on     # provision the host and install the cron
#   ./scripts/weekly-benchmarks.sh --off    # stop the cron, leaving the clone in place
#   ./scripts/weekly-benchmarks.sh --off --purge   # ... and delete the clone too
#
# --on is idempotent: run it again after a change to re-point the host at this
# repo's origin/main, re-sync the venv, and replace the crontab entry in place.
# --off only removes the crontab entry, so the clone and its
# .cluster/weekly-cron.log survive for diagnosis and --on is cheap afterwards.
# --purge deletes the clone as well, log included.
#
# The host must hold a persistent, already-authenticated ssh ControlMaster to
# Vulcan, because that is how cron.sh reaches the cluster without an MFA
# prompt. This script cannot open it - MFA needs a terminal - so it reports
# whether one is live and names the command to run if not.
#
# See the repo README's "Weekly Automated Benchmarks" section for what the
# cron then does on each tick.
set -euo pipefail

# --- what to provision, and where -------------------------------------------
AUTOMATION_HOST="${AUTOMATION_HOST:-innisfree}"
INSTALL_PATH="${INSTALL_PATH:-/cshome/parham1}"
# ----------------------------------------------------------------------------

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGIN="$(git -C "$PROJECT" remote get-url origin)"
REPO_DIR="$INSTALL_PATH/$(basename "$ORIGIN" .git)"
CRON_MARKER="benchmarks/core/cron.sh"

# ssh re-parses its command through the remote shell rather than preserving
# argv, so quote each value for that shell instead of passing it bare.
run_remote() {
  ssh "$AUTOMATION_HOST" "bash -s -- $(printf '%q ' "$@")"
}

usage() {
  echo "usage: $(basename "$0") --on | --off [--purge]" >&2
  exit 2
}

turn_on() {
run_remote "$ORIGIN" "$REPO_DIR" "$CRON_MARKER" "$AUTOMATION_HOST" <<'REMOTE'
set -euo pipefail
origin="$1"; repo_dir="$2"; cron_marker="$3"; host="$4"
export PATH="$HOME/.local/bin:$PATH"
cron_line="0 * * * * PATH=\"\$HOME/.local/bin:\$PATH\" $repo_dir/$cron_marker"

if ! ssh-keygen -F github.com >/dev/null 2>&1; then
  echo ">> trusting github.com's host keys"
  scanned="$(ssh-keyscan -t rsa,ecdsa,ed25519 github.com 2>/dev/null)"
  [ -n "$scanned" ] || { echo "error: ssh-keyscan github.com returned nothing" >&2; exit 1; }
  published="$(curl -fsSL https://api.github.com/meta \
    | python3 -c 'import json,sys; print("\n".join(json.load(sys.stdin)["ssh_keys"]))')"
  [ -n "$published" ] || { echo "error: could not read api.github.com/meta" >&2; exit 1; }
  while read -r _ type key; do
    grep -qxF "$type $key" <<<"$published" \
      || { echo "error: $type key from github.com is not one GitHub publishes" >&2; exit 1; }
  done <<<"$scanned"
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  printf '%s\n' "$scanned" >> "$HOME/.ssh/known_hosts"
  echo "   verified against api.github.com/meta and added"
fi

if [ -d "$repo_dir/.git" ]; then
  echo ">> updating $repo_dir"
  git -C "$repo_dir" pull --ff-only --quiet
else
  echo ">> cloning $origin into $repo_dir"
  mkdir -p "$(dirname "$repo_dir")"
  git clone --quiet "$origin" "$repo_dir"
fi
echo "   at $(git -C "$repo_dir" log --oneline -1)"

command -v uv >/dev/null \
  || { echo "error: uv not on PATH on $host (expected ~/.local/bin/uv)" >&2; exit 1; }
echo ">> uv sync"
( cd "$repo_dir" && uv sync --quiet )

echo ">> installing the crontab entry"
kept="$(crontab -l 2>/dev/null | grep -vF "$cron_marker" || true)"
printf '%s\n%s\n' "$kept" "$cron_line" | grep -v '^[[:space:]]*$' | crontab -
crontab -l | sed -n "\|$cron_marker|s|^|   |p"

if ssh -O check vulcan >/dev/null 2>&1; then
  echo ">> vulcan ControlMaster is live"
else
  echo ">> WARNING: no live vulcan ControlMaster on $host, so every dispatch"
  echo "   will fail until you open one there yourself (MFA needs a terminal):"
  echo "     ssh $host"
  echo "     ssh vulcan true"
fi

echo ">> weekly.py check:"
set +e
out="$(cd "$repo_dir" && uv run benchmarks/core/weekly.py check 2>&1)"
code=$?
set -e
echo "   $out"
echo "   (exit $code)"
REMOTE
}

turn_off() {
run_remote "$REPO_DIR" "$CRON_MARKER" "$PURGE" <<'REMOTE'
set -euo pipefail
repo_dir="$1"; cron_marker="$2"; purge="$3"

kept="$(crontab -l 2>/dev/null | grep -vF "$cron_marker" || true)"
if [ -n "$kept" ]; then
  printf '%s\n' "$kept" | crontab -
else
  crontab -r 2>/dev/null || true
fi
echo ">> removed the crontab entry"

if ls "$repo_dir"/.cluster/*.json >/dev/null 2>&1; then
  echo ">> WARNING: a dispatch is still in flight, and nothing will finish it"
  echo "   now. Its jobs are listed by:  ssh vulcan squeue -u \$USER"
fi

if [ "$purge" = 1 ]; then
  rm -rf "$repo_dir"
  echo ">> deleted $repo_dir"
else
  echo ">> clone left at $repo_dir (--off --purge also deletes it)"
fi
REMOTE
}

[ $# -ge 1 ] || usage
case "$1" in
  --on) MODE=on ;;
  --off) MODE=off ;;
  *) usage ;;
esac

PURGE=0
if [ $# -ge 2 ]; then
  [ "$2" = "--purge" ] && [ "$MODE" = off ] || usage
  PURGE=1
fi

ssh -o BatchMode=yes -o ConnectTimeout=15 "$AUTOMATION_HOST" true 2>/dev/null \
  || { echo "error: cannot ssh to $AUTOMATION_HOST" >&2; exit 1; }
echo ">> $AUTOMATION_HOST reachable, turning the weekly benchmark $MODE"

if [ "$MODE" = on ]; then
  turn_on
  echo ">> the weekly benchmark is on; the next hourly tick acts on it"
else
  turn_off
  echo ">> the weekly benchmark is off"
fi
