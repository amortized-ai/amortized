---
description: Deploy a branch or PR to the kind cluster on the GPU node. Use when asked to deploy, redeploy, refresh, or push to the node.
---

# Deploy to Kind Cluster

Deploy the current or specified branch to a developer namespace on the GPU node.

## Arguments

Parse the user's input for:
- **user** (required): The developer namespace to deploy to. If not provided, ask the user before proceeding. Valid users: `mathale`, `meyceoz`, `ssudalai`, `nmalepat`, `esivaram`.
- **branch**: A branch name (e.g., `refactor/structured-agent-tools`). Defaults to the current branch.
- **pr**: A PR number (e.g., `276`). Resolves to the PR's head branch.

## Remote Host

- **Host:** `shiv@169.62.17.147`
- **Workspace:** `/home/shiv/{user}/amortized`
- **Cluster context:** `kind-amortized`

## Steps

### 1. Resolve Branch

If a PR number was given, resolve it to a branch:
```bash
gh pr view <pr> --json headRefName --jq '.headRefName'
```

If no branch or PR specified, use the current local branch:
```bash
git branch --show-current
```

### 2. Push Local Changes

Check if there are uncommitted changes or unpushed commits. If so, warn the user — don't push automatically.

### 3. Deploy to Remote

Run the following as a single SSH command:
```bash
ssh shiv@169.62.17.147 'cd /home/shiv/{user}/amortized && git fetch origin && git reset --hard origin/{branch} && make refresh-{user}'
```

Note: `git reset --hard` is expected here — the remote workspace is a deployment target, not a development environment.

### 4. Set Up Tunnel

Check if an SSH tunnel is already running for the user's ports. If not, start one.

Port mapping per user:
| User | Studio | Server |
|------|--------|--------|
| mathale | 31120 | 31121 |
| meyceoz | 31100 | 31101 |
| ssudalai | 31110 | 31111 |
| nmalepat | 31130 | 31131 |
| esivaram | 31140 | 31141 |

```bash
ssh -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L {studio_port}:localhost:{studio_port} \
  -L {server_port}:localhost:{server_port} \
  shiv@169.62.17.147
```

### 5. Verify

Confirm the deployment by checking the pod status:
```bash
ssh shiv@169.62.17.147 'kubectl --context kind-amortized -n amortized-{user} get pods -o custom-columns="NAME:.metadata.name,STATUS:.status.phase,STARTED:.status.startTime"'
```

Report the Studio URL: `http://localhost:{studio_port}`

## Error Handling

- If `make refresh-{user}` fails with a build error, show the error and suggest fixes.
- If the tunnel port is already in use, kill the existing process first.
- If SSH fails, suggest the user check their SSH key and connectivity.
