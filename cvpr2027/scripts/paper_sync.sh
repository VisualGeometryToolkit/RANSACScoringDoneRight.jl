#!/usr/bin/env bash
# =============================================================================
# paper_sync.sh — sync cvpr2027/paper with its Overleaf project
# =============================================================================
#
# The manuscript lives INSIDE this repository at cvpr2027/paper; its Overleaf
# project (Overleaf's git integration, branch main) is where co-authors
# edit it. `push` puts the prefix's tree on Overleaf's tip as one commit;
# `pull` merges the co-authors' edits back under the prefix. The protocol is
# radiometric-camera's cvpr2027/scripts/paper_sync.sh (the Blens subtree),
# adapted to a remote that starts empty: `init` makes the blank Overleaf
# project hold this paper once, and `pull` is a subtree-strategy merge, which
# needs no earlier `git subtree add`.
#
#   cvpr2027/scripts/paper_sync.sh setup <url>   # bind the remote 'overleaf'
#   cvpr2027/scripts/paper_sync.sh init          # ONCE: the blank project -> this paper
#   cvpr2027/scripts/paper_sync.sh pull          # Overleaf edits -> this repo (a merge commit)
#   cvpr2027/scripts/paper_sync.sh push          # local edits -> Overleaf (fast-forward only)
#   cvpr2027/scripts/paper_sync.sh status
#
# Rules that keep the two sides mergeable:
#   * Only TRACKED files travel. Nothing of the data may sit under
#     cvpr2027/paper (test/runtests.jl checks: no data/, results/, *.jld2).
#   * The generated files (paper/figures/: the tables and synthetic_numbers.tex)
#     are the drivers' (REPRODUCE.md): co-authors edit the prose, never them.
#   * Pull BEFORE you push after anyone edited on Overleaf. A push that is not
#     a fast-forward is refused; never `--force`, the remote's history is the
#     co-authors' history.
#   * Overleaf asks for a git authentication token (Account Settings -> Git
#     integration) as the password; a credential helper keeps it.
set -euo pipefail

PREFIX="cvpr2027/paper"
REMOTE="overleaf"
REMOTE_BRANCH="main"
BRANCH="split/cvpr2027-paper"     # the commit the last push made (local)
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

usage() { sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
need_remote() {
  git remote get-url "$REMOTE" >/dev/null 2>&1 || {
    echo "no git remote '$REMOTE' — run: $0 setup <overleaf git url>" >&2
    exit 1
  }
}
clean_tree() {
  git diff --quiet && git diff --cached --quiet || {
    echo "working tree has uncommitted changes — commit or stash first" >&2; exit 1
  }
}
# The remote tip. A failed fetch (no network, no credentials) stops the
# script: a command substitution does not inherit `set -e`.
fetch_tip() {
  git fetch -q "$REMOTE" "$REMOTE_BRANCH" >&2 || {
    echo "could not fetch $REMOTE $REMOTE_BRANCH — check the network and your Overleaf git token" >&2
    return 1
  }
  git rev-parse FETCH_HEAD
}
# The prefix's tree as one commit on the remote tip, pushed as a fast-forward.
push_tree() {
  local tip=$1 msg=$2 tree commit
  tree=$(git rev-parse "HEAD:$PREFIX")
  if [ "$(git rev-parse "$tip^{tree}")" = "$tree" ]; then
    echo "nothing to push: $REMOTE $REMOTE_BRANCH already holds $PREFIX"; return 0
  fi
  commit=$(git commit-tree "$tree" -p "$tip" -m "$msg")
  git branch -f "$BRANCH" "$commit"
  git push "$REMOTE" "$BRANCH:$REMOTE_BRANCH"     # no --force, ever: a fast-forward by construction
  echo "pushed $PREFIX -> $REMOTE $REMOTE_BRANCH (${commit:0:8})"
}

case "${1:-}" in
  setup)
    url="${2:?usage: $0 setup <overleaf git url>}"
    if git remote get-url "$REMOTE" >/dev/null 2>&1; then git remote set-url "$REMOTE" "$url"; else git remote add "$REMOTE" "$url"; fi
    echo "remote '$REMOTE' -> $url"
    ;;
  init)
    need_remote; clean_tree
    git rev-parse -q --verify "$BRANCH" >/dev/null && { echo "already initialised ($BRANCH exists) — use push/pull" >&2; exit 1; }
    tip=$(fetch_tip) || exit 1
    # The blank project's files are replaced by this paper (a fast-forward on
    # its tip), and the result is merged here with this tree kept, so later
    # pulls have a merge base.
    push_tree "$tip" "cvpr2027/paper from RANSACScoringDoneRight.jl $(git rev-parse --short HEAD)"
    git merge -q -s ours --allow-unrelated-histories --no-edit "$BRANCH" \
      -m "cvpr2027: bind $PREFIX to its Overleaf project"
    ;;
  push)
    need_remote; clean_tree
    tip=$(fetch_tip) || exit 1
    pushed=$(git rev-parse -q --verify "$BRANCH" || true)
    [ "$tip" = "$pushed" ] || git merge-base --is-ancestor "$tip" HEAD || {
      echo "Overleaf $REMOTE_BRANCH is at ${tip:0:8}, which is not merged here — run: $0 pull" >&2; exit 1
    }
    push_tree "$tip" "Sync from RANSACScoringDoneRight.jl $(git rev-parse --short HEAD) ($PREFIX)"
    ;;
  pull)
    need_remote; clean_tree
    tip=$(fetch_tip) || exit 1
    git merge --no-ff -X subtree="$PREFIX" "$tip" -m "cvpr2027: merge Overleaf edits into $PREFIX"
    ;;
  status)
    # Offline-safe: the remote may be unreachable or unbound; status still reports what is local.
    if git remote get-url "$REMOTE" >/dev/null 2>&1; then
      echo "remote : $REMOTE -> $(git remote get-url "$REMOTE") ($REMOTE_BRANCH)"
      git fetch "$REMOTE" "$REMOTE_BRANCH" >/dev/null 2>&1 \
        && echo "fetched: $REMOTE/$REMOTE_BRANCH @ $(git rev-parse --short FETCH_HEAD)" \
        || echo "fetched: remote unreachable (offline?)"
    else
      echo "remote : none bound (run: $0 setup <overleaf git url>)"
    fi
    echo "prefix : $PREFIX ($(git ls-files "$PREFIX" | wc -l) tracked files)"
    if git rev-parse --verify -q "$BRANCH" >/dev/null; then
      echo "split  : $BRANCH @ $(git rev-parse --short "$BRANCH")"
    else
      echo "split  : none yet (created by init)"
    fi
    ;;
  *) usage ;;
esac
