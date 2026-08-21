-- Online transcription: base URL for the OpenAI speech-to-text provider.
-- Kept configurable so users can point Meetily at any OpenAI-compatible STT
-- service (Groq, LiteLLM, a local faster-whisper-server, ...) instead of only
-- api.openai.com. NULL/empty means "use the official OpenAI endpoint".
ALTER TABLE transcript_settings ADD COLUMN openaiBaseUrl TEXT;
