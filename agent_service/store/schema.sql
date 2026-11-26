CREATE TABLE IF NOT EXISTS conversations (
    id              text PRIMARY KEY,
    state           text NOT NULL DEFAULT 'bot' CHECK (state IN ('bot', 'human')),
    language        text,
    handoff_reason  text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- One row per inbound message id. The primary key IS the dedup:
-- WhatsApp providers redeliver, and a redelivery must not trigger a second reply.
CREATE TABLE IF NOT EXISTS inbound_messages (
    message_id       text PRIMARY KEY,
    conversation_id  text NOT NULL,
    reply            text,
    completed_at     timestamptz,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id               bigserial PRIMARY KEY,
    conversation_id  text NOT NULL REFERENCES conversations(id),
    role             text NOT NULL CHECK (role IN ('customer', 'assistant')),
    content          text NOT NULL,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation_id, id DESC);

-- Durable customer memory: small structured facts, separate from the transcript,
-- so the prompt carries "what we know" without replaying every message.
CREATE TABLE IF NOT EXISTS customer_facts (
    conversation_id  text NOT NULL REFERENCES conversations(id),
    key              text NOT NULL,
    value            text NOT NULL,
    updated_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (conversation_id, key)
);

-- Stand-in for the CRM. One lead per conversation; the agent can only upsert it.
CREATE TABLE IF NOT EXISTS leads (
    conversation_id  text PRIMARY KEY REFERENCES conversations(id),
    name             text,
    interest         text,
    preferred_date   text,
    language         text,
    status           text NOT NULL DEFAULT 'new',
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);

-- One row per handled message: what it cost, how long it took, what the guards caught.
CREATE TABLE IF NOT EXISTS turns (
    id               bigserial PRIMARY KEY,
    conversation_id  text NOT NULL,
    message_id       text NOT NULL,
    outcome          text NOT NULL,
    models           jsonb NOT NULL DEFAULT '[]',
    input_tokens     integer NOT NULL DEFAULT 0,
    output_tokens    integer NOT NULL DEFAULT 0,
    cost_usd         numeric(12, 6) NOT NULL DEFAULT 0,
    latency_ms       integer NOT NULL DEFAULT 0,
    attempts         integer NOT NULL DEFAULT 0,
    tool_calls       jsonb NOT NULL DEFAULT '[]',
    guard_reasons    jsonb NOT NULL DEFAULT '[]',
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS turns_conversation_idx ON turns (conversation_id, id);
