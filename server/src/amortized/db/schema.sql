-- Amortized runtime database schema

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    config TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error TEXT,
    pid INTEGER,
    output_dir TEXT,
    backend_handle TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    job_id TEXT,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    size INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    location TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    producer_job TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    data TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_job_id ON artifacts(job_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE TABLE IF NOT EXISTS evaluators (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT 'llm',
    prompt TEXT NOT NULL,
    judgment_type TEXT NOT NULL DEFAULT 'bool',
    response_format TEXT NOT NULL DEFAULT 'json',
    variables TEXT NOT NULL DEFAULT '[]',
    model TEXT,
    inference_params TEXT DEFAULT '{}',
    rule_config TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    evaluator_id TEXT NOT NULL,
    dataset_artifact_id TEXT,
    job_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    results TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (evaluator_id) REFERENCES evaluators(id)
);

CREATE INDEX IF NOT EXISTS idx_events_job_id ON events(job_id);
CREATE INDEX IF NOT EXISTS idx_events_job_timestamp ON events(job_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_evaluations_evaluator ON evaluations(evaluator_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_status ON evaluations(status);
