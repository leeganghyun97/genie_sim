#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
branch="codex/g2-preloss-v2-repro"
remote_name="g2-preloss-fork"
fork_url=""
dry_run=false

usage() {
  cat <<'EOF'
Usage:
  bash scripts/publish_g2_preloss_v2_branch.sh --fork-url URL [options]

Options:
  --fork-url URL      User-owned GitHub/GitLab fork URL (required)
  --remote-name NAME  Local remote name (default: g2-preloss-fork)
  --dry-run           Validate the branch and LFS objects without pushing
  -h, --help          Show this help

The script deliberately refuses to publish to the repository's origin URL.
It pushes codex/g2-preloss-v2-repro and its Git LFS objects only.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fork-url)
      [[ $# -ge 2 ]] || { echo "--fork-url requires a value" >&2; exit 2; }
      fork_url="$2"
      shift 2
      ;;
    --fork-url=*)
      fork_url="${1#*=}"
      shift
      ;;
    --remote-name)
      [[ $# -ge 2 ]] || { echo "--remote-name requires a value" >&2; exit 2; }
      remote_name="$2"
      shift 2
      ;;
    --remote-name=*)
      remote_name="${1#*=}"
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "$fork_url" ]] || { echo "--fork-url is required" >&2; exit 2; }
[[ "$remote_name" != "origin" ]] || {
  echo "refusing remote name 'origin'; use a dedicated fork remote" >&2
  exit 2
}

cd "$repository_root"
git show-ref --verify --quiet "refs/heads/$branch" || {
  echo "missing local reproducibility branch: $branch" >&2
  exit 2
}

origin_url="$(git remote get-url origin 2>/dev/null || true)"
if [[ -n "$origin_url" && "$fork_url" == "$origin_url" ]]; then
  echo "refusing to push the reproducibility branch to origin" >&2
  exit 2
fi

if git remote get-url "$remote_name" >/dev/null 2>&1; then
  configured_url="$(git remote get-url "$remote_name")"
  [[ "$configured_url" == "$fork_url" ]] || {
    echo "remote $remote_name already points to a different URL" >&2
    exit 2
  }
else
  git remote add "$remote_name" "$fork_url"
fi

git lfs env >/dev/null
missing_lfs="$(git lfs fsck "$branch" 2>&1 || true)"
if [[ -n "$missing_lfs" && "$missing_lfs" != *"Git LFS fsck OK"* ]]; then
  echo "$missing_lfs" >&2
  echo "local LFS verification failed" >&2
  exit 2
fi

printf 'branch=%s\ncommit=%s\nremote=%s\nurl=%s\n' \
  "$branch" "$(git rev-parse "$branch")" "$remote_name" "$fork_url"

if [[ "$dry_run" == true ]]; then
  echo "PUBLISH_DRY_RUN_PASS"
  exit 0
fi

# The installed Git LFS pre-push hook uploads every LFS object referenced by
# this branch before its Git ref is accepted by the fork.
git push --set-upstream "$remote_name" "$branch:$branch"
echo "PUBLISH_PASS"
