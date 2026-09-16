#!/usr/bin/env bash
# Assemble the Morty image build context. The persona/skills are GENERATED from agents/
# via `make prompt` (single source of truth, gitignored) — never hand-maintained here.
# Usage: prepare-context.sh [context-dir]   (default: containers/morty/.build-context)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CTX="${1:-$HERE/.build-context}"

# Generate k8s/base/morty-* (persona + skills) from agents/.
( cd "$REPO" && make prompt )

rm -rf "$CTX"
mkdir -p "$CTX/skills"
# Map make-prompt output -> the layout the Dockerfile COPYs.
cp    "$REPO/k8s/base/morty-prompt.md"            "$CTX/morty.md"       # orchestrator identity+workflow
cp    "$REPO/k8s/base/morty-sdg-workflow.md"      "$CTX/sdg.md"
cp    "$REPO/k8s/base/morty-training-workflow.md" "$CTX/training.md"
cp -R "$REPO/k8s/base/morty-skills/."             "$CTX/skills/"        # sdg/ + training/ skill trees
cp    "$HERE/opencode.json"                       "$CTX/opencode.json"
cp    "$HERE/Dockerfile"                          "$CTX/Dockerfile"

echo "$CTX"
