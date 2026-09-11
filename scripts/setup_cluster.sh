#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: setup_cluster.sh [--config PATH]

Bootstrap the SLURM cluster for running experiments.

options:
  --config PATH  cluster/slurm configuration file (default <repo>/cluster.toml)
  -h, --help     show this message
EOF
}

die() { printf '%s\n' "$*" >&2; exit 1; }

config=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config)
      config="${2:-}"
      [ -n "$config" ] || { usage >&2; exit 2; }
      shift 2
      ;;
    --config=*) config="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

repo="$PWD"
while [ ! -f "$repo/cluster.toml" ]; do
  if [ "$repo" = "/" ]; then
    die "could not locate the repo root (no cluster.toml above $PWD)"
  fi
  repo="$(dirname "$repo")"
done
cd "$repo"
config="${config:-$repo/cluster.toml}"
if [ ! -f "$config" ]; then
  die "no cluster config at $config (see cluster.toml in the repo root)"
fi

toml_raw() {
  awk -v sec="$1" -v key="$2" '
    /^[[:space:]]*#/ { next }
    /^[[:space:]]*\[/ {
      cur = $0
      sub(/^[[:space:]]*\[/, "", cur)
      sub(/\][[:space:]]*$/, "", cur)
      next
    }
    index($0, "=") == 0 { next }
    {
      k = $0
      sub(/=.*/, "", k)
      gsub(/[[:space:]]/, "", k)
      if (cur == sec && k == key) {
        v = $0
        sub(/^[^=]*=[[:space:]]*/, "", v)
        sub(/[[:space:]]+$/, "", v)
        print v
        exit
      }
    }
  ' "$config"
}

toml_str() {
  local v
  v="$(toml_raw "$1" "$2")"
  v="${v#\"}"
  printf '%s' "${v%\"}"
}

toml_arr() {
  toml_raw "$1" "$2" | tr -d '[]"' | tr ',' '\n' | awk 'NF { $1 = $1; print }'
}

field() {
  awk -v k="$1" '
    index($0, k "=") == 1 { v = substr($0, length(k) + 2) }
    END {
      sub(/^[[:space:]]+/, "", v)
      sub(/[[:space:]]+$/, "", v)
      print v
    }
  '
}

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum
  else
    shasum -a 256
  fi | cut -d' ' -f1
}

host="$(toml_str cluster host)"
root_cfg="$(toml_str cluster root)"
account="$(toml_str cluster account)"
post_sync="$(toml_str project post_sync)"
project="$(toml_str project name)"
[ -n "$project" ] || project="$(basename "$root_cfg")"

remote_dir=""
for candidate in \
  "${EXPERIMENT_REMOTE_DIR:-}" \
  "${VIRTUAL_ENV:-}"/lib/python*/site-packages/experiment/remote \
  "$repo"/.venv/lib/python*/site-packages/experiment/remote; do
  if [ -d "$candidate" ]; then
    remote_dir="$candidate"
    break
  fi
done
if [ -z "$remote_dir" ]; then
  die "could not find experiment/remote (set EXPERIMENT_REMOTE_DIR)"
fi

local_mode() { [ "${EXPERIMENT_LOCAL_MODE:-}" = "1" ]; }

errfile="$(mktemp)"
snapshot=""
root=""
cleanup() {
  if [ -n "$snapshot" ]; then
    rsh "rm -rf $(quote "$snapshot") $(quote "$root/results/.setup")" nocheck \
      >/dev/null || true
  fi
  rm -f "$errfile"
}
trap cleanup EXIT

quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

check_auth() {
  local err
  err="$(cat "$errfile")"
  case "$err" in
    *keyboard-interactive*|*"Permission denied"*) ;;
    *) return 0 ;;
  esac
  printf 'ssh to %s was refused:\n  %s\n' "$host" \
    "$(printf '%s' "$err" | tail -n 1)" >&2
  printf 'MFA needs a terminal, so open the connection once yourself:\n' >&2
  printf '    ssh %s true\n' "$host" >&2
  printf 'ControlMaster keeps it alive for the commands that follow. ' >&2
  printf 'Then retry.\n' >&2
  exit 1
}

rsh() {
  local out rc=0
  if local_mode; then
    out="$(bash -c "$1" 2>"$errfile")" || rc=$?
  else
    out="$(ssh "$host" "$1" 2>"$errfile")" || rc=$?
  fi
  check_auth
  if [ "${2:-check}" = check ] && [ "$rc" -ne 0 ]; then
    die "remote command failed: $1
$(cat "$errfile")"
  fi
  printf '%s' "$out"
}

rscript() {
  local script="$1" out rc=0 quoted
  shift
  if local_mode; then
    out="$(bash -s -- "$@" <"$script" 2>"$errfile")" || rc=$?
  else
    quoted="$(printf '%q ' "$@")"
    out="$(ssh "$host" "bash -s -- $quoted" <"$script" 2>"$errfile")" || rc=$?
  fi
  check_auth
  cat "$errfile" >&2
  [ "$rc" -eq 0 ] || die "$(basename "$script") failed on $host"
  printf '%s' "$out"
}

lock_hash() {
  local extras first=1 extra
  extras="$(printf '%s\n' "$@" | sort)"
  {
    git show "$sha:uv.lock"
    printf '\0'
    for extra in $extras; do
      [ "$first" -eq 1 ] || printf '\0'
      printf '%s' "$extra"
      first=0
    done
    if [ -n "$post_sync" ]; then
      printf '\0'
      git show "$sha:$post_sync"
    fi
  } | sha256 | cut -c1-12
}

dirty="$(git status --porcelain)"
if [ -n "$dirty" ]; then
  die "working tree is dirty - commit or stash before dispatching:
$dirty"
fi
sha="$(git rev-parse HEAD)"
if [ -n "$post_sync" ] && ! git cat-file -e "$sha:$post_sync" 2>/dev/null; then
  die "[project] post_sync is '$post_sync', which the commit being dispatched
does not carry"
fi

printf 'setting up %s at %s\n' "$host" "$root_cfg" >&2

remote_root="$(rsh "printf \"%s\" \"$root_cfg\"")"
bootstrap="$(rscript "$remote_dir/bootstrap.sh" "$remote_root" "$project")"
root="$(printf '%s' "$bootstrap" | field ROOT)"
bare="$(printf '%s' "$bootstrap" | field BARE)"
if [ -z "$root" ] || [ -z "$bare" ]; then
  die "bootstrap.sh did not report a root and bare repo"
fi

remote_name="cluster-$host"
if local_mode; then url="$bare"; else url="$host:$bare"; fi
if git remote get-url "$remote_name" >/dev/null 2>&1; then
  git remote set-url "$remote_name" "$url"
else
  git remote add "$remote_name" "$url"
fi
if ! git push --quiet "$remote_name" "+$sha:refs/gsp/setup" 2>"$errfile"; then
  check_auth
  die "push to $remote_name failed - run setup_cluster.sh if the bare repo is gone
$(cat "$errfile")"
fi

prepared="$(
  rscript "$remote_dir/prepare.sh" "$root" .setup "$sha" .setup . "$project"
)"
snapshot="$(printf '%s' "$prepared" | field RUNDIR)"
if [ -z "$snapshot" ]; then
  die "prepare.sh did not report a snapshot for the venv build"
fi

cpu_venv=""
gpu_venv=""
for name in cpu gpu; do
  extras="$(toml_arr venvs "$name" | tr '\n' ' ')"
  built="$(
    rscript "$remote_dir/build_env.sh" "$root" "$name" "$(lock_hash $extras)" \
      "$snapshot" "$post_sync" $extras
  )"
  eval "${name}_venv=\$(printf '%s' \"\$built\" | field VENV)"
done

printf '\nready on %s\n' "$host" >&2
printf '  root       %s\n' "$root" >&2
printf '  bare repo  %s\n' "$bare" >&2
printf '  cpu venv   %s\n' "$cpu_venv" >&2
printf '  gpu venv   %s\n' "$gpu_venv" >&2
printf '  remote     %s\n' "$remote_name" >&2
if [ -z "$account" ]; then
  found="$(printf '%s' "$bootstrap" | field ACCOUNTS)"
  [ -n "$found" ] || found="none found under ~/projects"
  printf '\nset account in %s before dispatching (candidates: %s)\n' \
    "$config" "$found" >&2
fi
