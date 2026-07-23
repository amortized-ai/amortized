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

CREATE TABLE IF NOT EXISTS documents (
    document_id     TEXT PRIMARY KEY,
    mlflow_run_id   TEXT DEFAULT '',
    filename        TEXT NOT NULL,
    format          TEXT NOT NULL DEFAULT 'md',
    content         TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
