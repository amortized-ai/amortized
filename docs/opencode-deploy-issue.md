# OpenCode Deployment Issue

## Problem

Deploying `main` to a user namespace does not include OpenCode. The chat
agent is unreachable — Studio connects but gets no responses.

## Root Cause

Commit `fe34ee4` ("remove Claude Code agent deployment") in
`amortized-deploy` removed OpenCode resources from
`k8s/base-internal/kustomization.yaml`:

```diff
 resources:
   - ../../../amortized/k8s/base
-  - opencode-configmap.yaml
-  - opencode-deployment.yaml
-  - opencode-service.yaml
+  - opencode-secret.yaml
```

The OpenCode deployment, configmap, and service YAML files still exist in
`k8s/base-internal/` but are no longer referenced by kustomize.

## Additional Issue: ConfigMaps

On `fix/prompt-revamp`, the amortized repo's `k8s/base/kustomization.yaml`
includes a `configMapGenerator` that creates `morty-config` and
`morty-skills` ConfigMaps. On `main`, these generators don't exist.

The OpenCode deployment mounts two ConfigMaps that must exist:

- **`morty-config`** — agent prompt file (`morty.md`). On `main` this is
  the single combined `morty-prompt.md`. On `fix/prompt-revamp` it
  includes separate identity, workflow, and subagent prompt files.
- **`morty-skills`** — skill guide files with `__`-delimited flat keys
  (e.g. `sdg__classification__guide.md`). An init container reconstructs
  the directory tree from these flat keys.

If either ConfigMap is missing, OpenCode pods stay in `Pending` with
`FailedMount` errors.

## Workaround

After deploying main, manually create the ConfigMaps and restart:

```bash
KUBECTL="kubectl --context kind-amortized"
NS=amortized-mathale
AMORTIZED=/home/shiv/mathale/amortized
SKILLS=$AMORTIZED/k8s/base/morty-skills

# Re-add OpenCode to kustomization
# (edit k8s/base-internal/kustomization.yaml to include opencode-*.yaml)

# Create morty-config
$KUBECTL -n $NS create configmap morty-config \
  --from-file=morty.md=$AMORTIZED/k8s/base/morty-prompt.md

# Create morty-skills with flat keys
$KUBECTL -n $NS create configmap morty-skills \
  --from-file=sdg__classification__guide.md=$SKILLS/sdg/classification/guide.md \
  --from-file=sdg__knowledge-ingestion__guide.md=$SKILLS/sdg/knowledge-ingestion/guide.md \
  --from-file=sdg__guidance.md=$SKILLS/sdg/guidance.md \
  --from-file=training__guidance.md=$SKILLS/training/guidance.md \
  --from-file=training__knowledge-ingestion__osft__guide.md=$SKILLS/training/knowledge-ingestion/osft/guide.md \
  --from-file=training__knowledge-ingestion__osft__training-config-template.json=$SKILLS/training/knowledge-ingestion/osft/training-config-template.json

# Restart OpenCode
$KUBECTL -n $NS delete pod -l component=opencode
```

## Fix

When `fix/prompt-revamp` merges, the amortized repo's kustomization will
include the `configMapGenerator` for both ConfigMaps, and the deploy
repo's `base-internal/kustomization.yaml` should re-add the OpenCode
resources. Until then, manual workaround above is needed for main.
