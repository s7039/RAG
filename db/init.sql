-- 健康档案 RAG：PostgreSQL 初始化脚本（容器首次启动时自动执行）
-- 内容：知识库 chunk 元数据 + Agent 的待办 / 会话 / 消息 / 用户偏好

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT        NOT NULL,
    page        INT,
    section     TEXT,
    patient     TEXT,
    sha256      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,            -- 复用 Chroma 的 chunk_id，保证两边对得上
    doc_id      BIGINT      REFERENCES documents(id) ON DELETE CASCADE,
    ord         INT         NOT NULL,
    content     TEXT        NOT NULL,
    patient     TEXT,                        -- 反范式，方便按患者过滤
    tsvec       TSVECTOR,                    -- 用 tokenize.py 的分词结果填
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx     ON chunks USING gin (tsvec);
CREATE INDEX IF NOT EXISTS chunks_patient_idx ON chunks (patient);

CREATE TABLE IF NOT EXISTS tasks (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT        NOT NULL,
    due_at      TIMESTAMPTZ,
    status      TEXT        NOT NULL DEFAULT 'open',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    thread_id   TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    thread_id   TEXT NOT NULL REFERENCES sessions(thread_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT,
    tool_calls  JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_thread_idx ON messages (thread_id, created_at);

CREATE TABLE IF NOT EXISTS user_prefs (
    user_id  TEXT NOT NULL,
    key      TEXT NOT NULL,
    value    JSONB,
    PRIMARY KEY (user_id, key)
);
