# Multi-Tenancy — Deep Research Summary

Adversarially verified research across 105 agents, 24+ sources. Focus on open-source self-hosted platforms.

## Key Findings

### 1. Determined AI — The Reference Model

Determined AI has the most comprehensive open-source multi-tenancy implementation:

**Hierarchy:** Workspace → Project → Experiment (PostgreSQL, FK-enforced)
- `workspaces(id, name, archived, user_id)`
- `projects(workspace_id FK, UNIQUE(name, workspace_id))`
- `experiments(project_id FK)` — NOT NULL, hierarchy is mandatory

**Compute isolation** (opt-in):
- `rp_workspace_bindings` table — many-to-many between workspaces and resource pools
- Pools can be unbound (globally available) or bound to specific workspaces
- Default: all pools are global. Binding makes isolation explicit.
- Per-workspace `default_compute_pool` and `default_aux_pool`

**K8s namespace isolation:**
- `workspace_namespace_bindings` table — maps workspaces to K8s namespaces per cluster
- Auto-create namespace support via CLI flags
- Supersedes older per-resource-pool namespace field

**RBAC** (Enterprise Edition only):
- 5 tables: `roles`, `permissions`, `role_permissions`, `role_assignments`, `permission_assignments`
- Roles: ClusterAdmin, WorkspaceAdmin, WorkspaceCreator, Editor, Viewer (5 default)
- Scopes: global OR workspace-level role assignments
- Open-source lacks RBAC — workspace isolation is not enforced without it

**Checkpoint storage:** Per-workspace checkpoint storage config (S3, GCS, Azure, shared_fs)

**GPU quotas:** `resource_quotas` table — GPU-only, per-workspace limit (integer, GPU-weighted by slot count)

### 2. MLflow — Lightweight Workspace Isolation

> **Note:** RHOAI uses Kubeflow Model Registry (Kubeflow Hub) for model versioning, not MLflow Model Registry. MLflow on RHOAI is for experiment tracking only. The workspace isolation described here applies to MLflow's experiment tracking features.

MLflow v3.10+ added workspace-level isolation:

- Experiments, models, prompts, gateway resources scoped to workspaces
- "Upgrade-only" permission floor — workspace default permission can only be raised, not lowered
- Permission types: READ, EDIT, MANAGE, NO_PERMISSIONS
- Simpler than Determined — no compute isolation, no K8s namespace mapping
- Workspaces are logical grouping, not hard isolation boundaries

### 3. Kubernetes Namespaces — The Foundation

Every platform uses K8s namespaces as the basic isolation primitive:
- Object name isolation and API scoping
- ResourceQuota enforcement per namespace
- NetworkPolicy for network isolation
- **BUT:** Namespace isolation alone is insufficient for hard multi-tenancy — needs network policies, storage isolation, container sandboxing

### 4. Secret Management Patterns

| Approach | Used By | Pros | Cons |
|---|---|---|---|
| K8s Secrets per namespace | Determined AI | Native, namespace-scoped | Plain base64, etcd access = all secrets |
| Platform-native encrypted store | Amortized (current) | Simple, self-contained | Single-tenant only |
| HashiCorp Vault | Enterprise platforms | Strong isolation, audit trail | Ops complexity |
| Per-workspace secret scope | Determined AI (Enterprise) | Clean scoping | Requires RBAC |

### 5. Artifact Isolation Patterns

| Approach | Used By | Pros | Cons |
|---|---|---|---|
| Path-based (`/workspace-id/artifacts/`) | Most platforms | Simple, works with any storage | Relies on access control |
| Bucket-per-tenant | Enterprise/cloud | Hard isolation | Operational overhead |
| Per-workspace checkpoint storage config | Determined AI | Flexible, tenant-controlled | Complex config |

## What Amortized Should Implement

### Phase 1: Project Scoping (minimum viable multi-tenancy)

Add a `projects` table that scopes all resources:

```sql
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    settings TEXT DEFAULT '{}'  -- JSON: default_backend, default_model, etc.
);

-- Add project_id FK to existing tables:
ALTER TABLE jobs ADD COLUMN project_id TEXT REFERENCES projects(id);
ALTER TABLE artifacts ADD COLUMN project_id TEXT REFERENCES projects(id);
ALTER TABLE evaluators ADD COLUMN project_id TEXT REFERENCES projects(id);
ALTER TABLE evaluations ADD COLUMN project_id TEXT REFERENCES projects(id);
ALTER TABLE conversations ADD COLUMN project_id TEXT REFERENCES projects(id);
```

All API endpoints gain a `?project=<name>` query param or `X-Project` header. Studio gets a project switcher in the header (like Oumi's "Default Project" dropdown).

### Phase 2: Per-Project Settings

- Default compute backend per project
- Default model per project (for SDG and eval)
- API keys scoped to project (or inherited from global)
- Artifact storage path prefix: `{storage_root}/{project_id}/`

### Phase 3: RBAC (when multi-user)

Only needed when amortized supports multiple users:
- Project-level roles: Admin, Member, Viewer
- Admin: full access, manage project settings
- Member: submit jobs, create datasets, run evals
- Viewer: read-only access to artifacts and results

### Phase 4: K8s Namespace Mapping (when on K8s)

Follow Determined AI's pattern:
- `project_namespace_bindings` table
- Each project maps to a K8s namespace
- Jobs for that project run in that namespace
- ResourceQuota per namespace for GPU limits

## What NOT to Build Yet

- **Org → Team → Project hierarchy** — premature. Start with flat project list.
- **Hard multi-tenancy** (network policies, container sandboxing) — only needed for untrusted tenants.
- **Billing/usage metering** — not needed for self-hosted.
- **Cross-project sharing** — keep it simple, no shared artifacts between projects.

## Sources

- Determined AI workspaces: https://docs.determined.ai/latest/manage/workspaces-rpools.html
- Determined AI RBAC: https://docs.determined.ai/latest/manage/security/rbac.html
- Determined AI K8s namespaces: https://docs.determined.ai/latest/setup-cluster/k8s/resource-caps.html
- MLflow workspaces: MLflow v3.10 release notes
- K8s multi-tenancy: https://kubernetes.io/docs/concepts/security/multi-tenancy/
