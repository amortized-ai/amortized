-- Amortized v1 database schema — single jobs table

CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    config          TEXT NOT NULL DEFAULT '{}',
    recipe          TEXT DEFAULT '',
    user_id         TEXT DEFAULT '',
    k8s_job_name    TEXT DEFAULT '',
    k8s_namespace   TEXT DEFAULT '',
    mlflow_run_id   TEXT DEFAULT '',
    mlflow_experiment TEXT DEFAULT '',
    parent_job_id   TEXT DEFAULT '',
    error           TEXT DEFAULT '',
    created_at      TEXT NOT NULL,
    started_at      TEXT DEFAULT '',
    completed_at    TEXT DEFAULT '',
    backend_handle  TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);

-- Prompt store (#541, migration 0004): reviewed prompts persisted once and cited
-- from SDG configs by an unguessable id (<prompt:prm_...>), so configs stay small at
-- emission. Full text is re-materialized only at the container boundary (jobs/sdg.py).
CREATE TABLE IF NOT EXISTS prompts (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT '',
    conversation_id TEXT NOT NULL DEFAULT '',
    role            TEXT NOT NULL DEFAULT '',
    label           TEXT NOT NULL DEFAULT '',
    purpose         TEXT NOT NULL DEFAULT '',
    content         TEXT NOT NULL,
    content_sha256  TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS prompts_content_sha256_key ON prompts (content_sha256);
CREATE INDEX IF NOT EXISTS prompts_user_id_idx ON prompts (user_id);

