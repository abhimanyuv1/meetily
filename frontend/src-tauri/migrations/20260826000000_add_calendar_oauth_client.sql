-- Bring-your-own Google OAuth client credentials (Desktop-app type).
-- The user creates their own (free) Google Cloud project, downloads the
-- Desktop-app client JSON, and pastes it into Settings -> Calendar. Keeping
-- credentials here instead of compiling them into the binary means every
-- build ships working calendar integration and each user's API usage counts
-- against their own quota, not ours.
CREATE TABLE IF NOT EXISTS calendar_oauth_clients (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    client_id TEXT NOT NULL,
    client_secret TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
