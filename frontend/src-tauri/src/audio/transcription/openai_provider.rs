// audio/transcription/openai_provider.rs
//
// OpenAI speech-to-text provider — an *online* transcription backend.
//
// Wired through the generic `TranscriptionProvider` trait alongside Sarvam, this
// lets any user bring their own OpenAI API key instead of running Whisper or
// Parakeet locally. Because the request shape is the de-facto standard for
// hosted STT, the base URL is user-configurable: pointing it at Groq, together.ai,
// a local LiteLLM/vLLM/faster-whisper-server, or any other OpenAI-compatible
// endpoint works without code changes.
//
// Each audio chunk from the transcription worker is encoded as 16 kHz mono WAV
// and POSTed as multipart/form-data.
//
// Docs: https://platform.openai.com/docs/api-reference/audio/createTranscription

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use super::wav::encode_wav_16k_mono;
use async_trait::async_trait;
use log::{info, warn};
use serde::Deserialize;
use std::time::Duration;

/// Default OpenAI API root. Users may override this to target any
/// OpenAI-compatible service.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
/// Recommended default: cheapest/fastest of the hosted Whisper-family models
/// that is universally available on the OpenAI platform.
pub const DEFAULT_MODEL: &str = "whisper-1";

/// Model ids OpenAI's audio endpoints accept. Anything else (e.g. a local
/// Whisper model id like "large-v3" leaking in from shared settings) is coerced
/// to DEFAULT_MODEL so we never send an invalid model and get a 400.
///
/// Note: only `whisper-1` supports the `/audio/translations` endpoint and
/// verbose response formats; the gpt-4o transcribe models are transcription-only.
const VALID_MODELS: &[&str] = &[
    "whisper-1",
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
];

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Below this many samples a request is wasteful/noisy, so we skip it (mirrors
/// the other providers' "audio too short" guard). 100ms at 16kHz.
const MIN_SAMPLES: usize = 1600;

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    #[serde(default)]
    text: String,
}

/// Error envelope returned by OpenAI (and most compatible services) on failure.
#[derive(Debug, Deserialize)]
struct OpenAiErrorEnvelope {
    error: Option<OpenAiErrorBody>,
}

#[derive(Debug, Deserialize)]
struct OpenAiErrorBody {
    #[serde(default)]
    message: String,
}

/// Online transcription via the OpenAI (or OpenAI-compatible) audio REST API.
pub struct OpenAiProvider {
    api_key: String,
    model: String,
    base_url: String,
    client: reqwest::Client,
}

impl OpenAiProvider {
    /// `model` is the OpenAI audio model id (e.g. "whisper-1"); falls back to
    /// the recommended default when empty or unrecognized. `base_url` may be
    /// empty to use the official OpenAI endpoint.
    pub fn new(api_key: String, model: String, base_url: Option<String>) -> Self {
        Self {
            api_key,
            model: Self::sanitize_model(&model),
            base_url: Self::sanitize_base_url(base_url.as_deref()),
            client: reqwest::Client::new(),
        }
    }

    /// Coerce a requested model id to one the audio endpoint actually accepts.
    /// Empty or unrecognized values fall back to DEFAULT_MODEL, preventing a 400.
    ///
    /// Custom/self-hosted OpenAI-compatible servers often expose their own model
    /// names (e.g. "Systran/faster-whisper-large-v3"), so an id containing '/'
    /// is treated as an explicit custom model and passed through untouched.
    fn sanitize_model(model: &str) -> String {
        let trimmed = model.trim();
        if trimmed.is_empty() {
            return DEFAULT_MODEL.to_string();
        }
        if VALID_MODELS.contains(&trimmed) || trimmed.contains('/') {
            return trimmed.to_string();
        }
        warn!(
            "OpenAI: model '{}' is not a known OpenAI audio model; falling back to '{}'",
            trimmed, DEFAULT_MODEL
        );
        DEFAULT_MODEL.to_string()
    }

    /// Normalizes a user-supplied base URL: trims whitespace and trailing
    /// slashes, and falls back to the official endpoint when unset. A URL that
    /// already ends in an audio route is reduced back to its API root so users
    /// pasting a full endpoint from docs still get a working configuration.
    fn sanitize_base_url(base_url: Option<&str>) -> String {
        let raw = base_url.unwrap_or("").trim().trim_end_matches('/');
        if raw.is_empty() {
            return DEFAULT_BASE_URL.to_string();
        }
        for suffix in ["/audio/transcriptions", "/audio/translations"] {
            if let Some(root) = raw.strip_suffix(suffix) {
                return root.trim_end_matches('/').to_string();
            }
        }
        raw.to_string()
    }

    /// Full URL of the endpoint to call. Meetily's "auto-translate" language
    /// hint means "translate into English", which OpenAI exposes as a separate
    /// `/audio/translations` route rather than a parameter.
    fn endpoint_for(&self, translate: bool) -> String {
        if translate {
            format!("{}/audio/translations", self.base_url)
        } else {
            format!("{}/audio/transcriptions", self.base_url)
        }
    }

    /// True when the language hint asks for English translation rather than
    /// verbatim transcription.
    fn wants_translation(language: &Option<String>) -> bool {
        matches!(
            language.as_ref().map(|l| l.trim().to_lowercase()),
            Some(ref l) if l == "auto-translate"
        )
    }

    /// Maps Meetily's language hint to OpenAI's optional ISO-639-1 `language`
    /// parameter. Auto-detect hints and unknown values return None, letting
    /// OpenAI detect the language itself.
    fn openai_language_code(language: &Option<String>) -> Option<String> {
        let lang = match language {
            Some(l) if !l.trim().is_empty() => l.trim().to_lowercase(),
            _ => return None,
        };

        // Meetily's synthetic hints: let OpenAI auto-detect.
        if lang == "auto" || lang == "auto-translate" || lang == "unknown" {
            return None;
        }

        // OpenAI wants a bare ISO-639-1 code, so reduce BCP-47 ("en-IN" -> "en").
        let base = lang.split('-').next().unwrap_or(&lang).to_string();

        // ISO-639-1 codes are two letters; anything else (e.g. "sat", "kok")
        // is not representable, so fall back to auto-detection.
        if base.len() == 2 && base.chars().all(|c| c.is_ascii_alphabetic()) {
            Some(base)
        } else {
            None
        }
    }

    /// Extracts the most useful message from an error body, falling back to the
    /// raw text for compatible servers that don't use OpenAI's envelope.
    fn extract_error_message(body: &str) -> String {
        serde_json::from_str::<OpenAiErrorEnvelope>(body)
            .ok()
            .and_then(|e| e.error)
            .map(|e| e.message)
            .filter(|m| !m.trim().is_empty())
            .unwrap_or_else(|| body.trim().to_string())
    }
}

#[async_trait]
impl TranscriptionProvider for OpenAiProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        if audio.len() < MIN_SAMPLES {
            return Err(TranscriptionError::AudioTooShort {
                samples: audio.len(),
                minimum: MIN_SAMPLES,
            });
        }

        let translate = Self::wants_translation(&language);
        let wav = encode_wav_16k_mono(&audio);

        let file_part = reqwest::multipart::Part::bytes(wav)
            .file_name("chunk.wav")
            .mime_str("audio/wav")
            .map_err(|e| {
                TranscriptionError::EngineFailed(format!("multipart build failed: {}", e))
            })?;

        let mut form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .text("model", self.model.clone())
            .text("response_format", "json");

        // The translations endpoint always outputs English and rejects a
        // `language` parameter, so only send it when transcribing.
        if !translate {
            if let Some(code) = Self::openai_language_code(&language) {
                form = form.text("language", code);
            }
        }

        let endpoint = self.endpoint_for(translate);
        info!(
            "OpenAI: {} {} samples with model '{}' via {}",
            if translate { "translating" } else { "transcribing" },
            audio.len(),
            self.model,
            endpoint
        );

        let response = self
            .client
            .post(&endpoint)
            .bearer_auth(&self.api_key)
            .multipart(form)
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .map_err(|e| {
                TranscriptionError::EngineFailed(format!("request to OpenAI failed: {}", e))
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            let detail = Self::extract_error_message(&body);
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(TranscriptionError::EngineFailed(format!(
                    "OpenAI rejected the API key (HTTP {}). Check the key in Transcript settings. {}",
                    status, detail
                )));
            }
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                return Err(TranscriptionError::EngineFailed(format!(
                    "OpenAI rate limit or quota exceeded (HTTP {}): {}",
                    status, detail
                )));
            }
            return Err(TranscriptionError::EngineFailed(format!(
                "OpenAI API error (HTTP {}): {}",
                status, detail
            )));
        }

        let parsed: OpenAiResponse = response.json().await.map_err(|e| {
            TranscriptionError::EngineFailed(format!("failed to parse OpenAI response: {}", e))
        })?;

        Ok(TranscriptResult {
            text: parsed.text.trim().to_string(),
            // OpenAI's json response format doesn't include a confidence score.
            confidence: None,
            is_partial: false,
        })
    }

    async fn is_model_loaded(&self) -> bool {
        // No local model to load — the provider is "ready" as long as a key is set.
        if self.api_key.trim().is_empty() {
            warn!("OpenAI provider has no API key configured");
            return false;
        }
        true
    }

    async fn get_current_model(&self) -> Option<String> {
        Some(self.model.clone())
    }

    fn provider_name(&self) -> &'static str {
        "OpenAI"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_model_falls_back_to_default() {
        let p = OpenAiProvider::new("k".to_string(), String::new(), None);
        assert_eq!(p.model, DEFAULT_MODEL);
        let p2 = OpenAiProvider::new("k".to_string(), "  ".to_string(), None);
        assert_eq!(p2.model, DEFAULT_MODEL);
    }

    #[test]
    fn valid_models_pass_through() {
        for m in VALID_MODELS {
            let p = OpenAiProvider::new("k".to_string(), m.to_string(), None);
            assert_eq!(&p.model, m);
        }
        // Whitespace tolerated.
        let p = OpenAiProvider::new("k".to_string(), "  whisper-1  ".to_string(), None);
        assert_eq!(p.model, "whisper-1");
    }

    #[test]
    fn invalid_model_is_coerced_to_default() {
        // A local Whisper model id leaking in from shared settings must never be
        // sent to OpenAI (it returns HTTP 400). It is coerced to the default.
        let p = OpenAiProvider::new("k".to_string(), "large-v3".to_string(), None);
        assert_eq!(p.model, DEFAULT_MODEL);
        // A Sarvam id likewise.
        let p2 = OpenAiProvider::new("k".to_string(), "saaras:v3".to_string(), None);
        assert_eq!(p2.model, DEFAULT_MODEL);
    }

    #[test]
    fn namespaced_custom_model_passes_through() {
        // Self-hosted OpenAI-compatible servers expose their own model names.
        let p = OpenAiProvider::new(
            "k".to_string(),
            "Systran/faster-whisper-large-v3".to_string(),
            Some("http://localhost:8000/v1".to_string()),
        );
        assert_eq!(p.model, "Systran/faster-whisper-large-v3");
    }

    #[test]
    fn base_url_defaults_and_normalizes() {
        let p = OpenAiProvider::new("k".to_string(), String::new(), None);
        assert_eq!(p.base_url, DEFAULT_BASE_URL);
        let p2 = OpenAiProvider::new("k".to_string(), String::new(), Some("  ".to_string()));
        assert_eq!(p2.base_url, DEFAULT_BASE_URL);
        // Trailing slashes trimmed.
        let p3 = OpenAiProvider::new(
            "k".to_string(),
            String::new(),
            Some("http://localhost:8000/v1/".to_string()),
        );
        assert_eq!(p3.base_url, "http://localhost:8000/v1");
    }

    #[test]
    fn base_url_with_full_endpoint_is_reduced_to_root() {
        // Users often paste the full endpoint from the docs.
        let p = OpenAiProvider::new(
            "k".to_string(),
            String::new(),
            Some("https://api.openai.com/v1/audio/transcriptions".to_string()),
        );
        assert_eq!(p.base_url, "https://api.openai.com/v1");
        assert_eq!(
            p.endpoint_for(false),
            "https://api.openai.com/v1/audio/transcriptions"
        );
        let p2 = OpenAiProvider::new(
            "k".to_string(),
            String::new(),
            Some("https://api.groq.com/openai/v1/audio/translations/".to_string()),
        );
        assert_eq!(p2.base_url, "https://api.groq.com/openai/v1");
    }

    #[test]
    fn endpoint_switches_for_translation() {
        let p = OpenAiProvider::new("k".to_string(), String::new(), None);
        assert_eq!(
            p.endpoint_for(false),
            "https://api.openai.com/v1/audio/transcriptions"
        );
        assert_eq!(
            p.endpoint_for(true),
            "https://api.openai.com/v1/audio/translations"
        );
    }

    #[test]
    fn auto_translate_hint_selects_translation() {
        assert!(OpenAiProvider::wants_translation(&Some(
            "auto-translate".to_string()
        )));
        assert!(OpenAiProvider::wants_translation(&Some(
            "AUTO-TRANSLATE".to_string()
        )));
        assert!(!OpenAiProvider::wants_translation(&Some("auto".to_string())));
        assert!(!OpenAiProvider::wants_translation(&Some("en".to_string())));
        assert!(!OpenAiProvider::wants_translation(&None));
    }

    #[test]
    fn language_none_or_auto_maps_to_autodetect() {
        assert_eq!(OpenAiProvider::openai_language_code(&None), None);
        assert_eq!(
            OpenAiProvider::openai_language_code(&Some("   ".to_string())),
            None
        );
        assert_eq!(
            OpenAiProvider::openai_language_code(&Some("auto".to_string())),
            None
        );
        assert_eq!(
            OpenAiProvider::openai_language_code(&Some("auto-translate".to_string())),
            None
        );
        assert_eq!(
            OpenAiProvider::openai_language_code(&Some("unknown".to_string())),
            None
        );
    }

    #[test]
    fn language_is_reduced_to_iso_639_1() {
        assert_eq!(
            OpenAiProvider::openai_language_code(&Some("en".to_string())),
            Some("en".to_string())
        );
        assert_eq!(
            OpenAiProvider::openai_language_code(&Some("HI".to_string())),
            Some("hi".to_string())
        );
        // BCP-47 gets reduced to the base code.
        assert_eq!(
            OpenAiProvider::openai_language_code(&Some("en-IN".to_string())),
            Some("en".to_string())
        );
    }

    #[test]
    fn non_iso6391_language_falls_back_to_autodetect() {
        // Codes with no two-letter ISO-639-1 form must not be sent verbatim.
        assert_eq!(
            OpenAiProvider::openai_language_code(&Some("sat".to_string())),
            None
        );
        assert_eq!(
            OpenAiProvider::openai_language_code(&Some("kok".to_string())),
            None
        );
        assert_eq!(
            OpenAiProvider::openai_language_code(&Some("1a".to_string())),
            None
        );
    }

    #[test]
    fn error_message_extraction_handles_envelope_and_raw() {
        let enveloped = r#"{"error":{"message":"Incorrect API key provided","type":"invalid_request_error"}}"#;
        assert_eq!(
            OpenAiProvider::extract_error_message(enveloped),
            "Incorrect API key provided"
        );
        // Compatible servers that don't use the envelope fall back to raw text.
        assert_eq!(
            OpenAiProvider::extract_error_message("plain failure"),
            "plain failure"
        );
        // Empty message in envelope falls back to raw body.
        let empty = r#"{"error":{"message":""}}"#;
        assert_eq!(OpenAiProvider::extract_error_message(empty), empty);
    }

    #[tokio::test]
    async fn short_audio_is_rejected() {
        let p = OpenAiProvider::new("k".to_string(), "whisper-1".to_string(), None);
        let res = p.transcribe(vec![0.0f32; 10], None).await;
        assert!(matches!(res, Err(TranscriptionError::AudioTooShort { .. })));
    }

    #[tokio::test]
    async fn is_model_loaded_requires_api_key() {
        let with_key = OpenAiProvider::new("k".to_string(), String::new(), None);
        assert!(with_key.is_model_loaded().await);
        let no_key = OpenAiProvider::new("   ".to_string(), String::new(), None);
        assert!(!no_key.is_model_loaded().await);
    }

    #[tokio::test]
    async fn provider_metadata_is_reported() {
        let p = OpenAiProvider::new("k".to_string(), "gpt-4o-transcribe".to_string(), None);
        assert_eq!(p.provider_name(), "OpenAI");
        assert_eq!(p.get_current_model().await, Some("gpt-4o-transcribe".to_string()));
    }

    // --- Integration-boundary tests against a local mock HTTP server ---------
    //
    // These exercise the real reqwest multipart POST + response parsing without
    // needing an OpenAI key or network. A tiny one-shot TCP server reads the
    // request, captures it for assertions, and replies with a canned response.

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc as std_mpsc;

    /// Spawns a one-shot HTTP/1.1 server on an ephemeral port. Returns the base
    /// URL and a receiver that yields the raw request bytes once received.
    fn spawn_once(response: &'static str) -> (String, std_mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = std_mpsc::channel();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                loop {
                    let n = stream.read(&mut tmp).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    let text = String::from_utf8_lossy(&buf);
                    if let Some(hdr_end) = text.find("\r\n\r\n") {
                        let headers = &text[..hdr_end];
                        let content_len = headers
                            .lines()
                            .find_map(|l| {
                                let l = l.to_ascii_lowercase();
                                l.strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                            })
                            .unwrap_or(0);
                        let body_start = hdr_end + 4;
                        if buf.len() >= body_start + content_len {
                            break;
                        }
                    }
                }
                let _ = tx.send(String::from_utf8_lossy(&buf).to_string());
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        (format!("http://{}/v1", addr), rx)
    }

    /// Builds a canned HTTP response with the correct Content-Length, leaked to
    /// obtain the 'static lifetime the server thread requires.
    fn canned(status_line: &str, body: &str) -> &'static str {
        let owned = format!(
            "{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            status_line,
            body.len(),
            body
        );
        Box::leak(owned.into_boxed_str())
    }

    #[tokio::test]
    async fn transcribe_sends_expected_request_and_parses_response() {
        let (url, rx) = spawn_once(canned("HTTP/1.1 200 OK", r#"{"text":"hello world"}"#));
        let provider =
            OpenAiProvider::new("secret-key-xyz".to_string(), "whisper-1".to_string(), Some(url));

        let audio = vec![0.1f32; MIN_SAMPLES + 100];
        let result = provider
            .transcribe(audio, Some("en".to_string()))
            .await
            .expect("transcribe should succeed against mock");

        // Response parsing
        assert_eq!(result.text, "hello world");
        assert!(!result.is_partial);
        assert_eq!(result.confidence, None);

        // Request assertions
        let req = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        let head = req.splitn(2, "\r\n\r\n").next().unwrap().to_ascii_lowercase();
        assert!(
            head.contains("post /v1/audio/transcriptions"),
            "method/path: {}",
            head
        );
        assert!(
            head.contains("authorization: bearer secret-key-xyz"),
            "auth header missing: {}",
            head
        );
        assert!(
            head.contains("content-type: multipart/form-data"),
            "not multipart: {}",
            head
        );
        // Multipart field values appear in the body
        assert!(req.contains("name=\"model\""), "model field missing");
        assert!(req.contains("whisper-1"), "model value missing");
        assert!(req.contains("name=\"language\""), "language field missing");
        assert!(req.contains("name=\"response_format\""), "response_format missing");
        assert!(req.contains("name=\"file\""), "file field missing");
        assert!(req.contains("filename=\"chunk.wav\""), "wav filename missing");
        assert!(req.contains("RIFF"), "wav payload missing");
    }

    #[tokio::test]
    async fn auto_translate_posts_to_translations_without_language_field() {
        let (url, rx) = spawn_once(canned("HTTP/1.1 200 OK", r#"{"text":"translated"}"#));
        let provider =
            OpenAiProvider::new("k".to_string(), "whisper-1".to_string(), Some(url));

        let result = provider
            .transcribe(
                vec![0.1f32; MIN_SAMPLES + 10],
                Some("auto-translate".to_string()),
            )
            .await
            .expect("translate should succeed against mock");
        assert_eq!(result.text, "translated");

        let req = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        let head = req.splitn(2, "\r\n\r\n").next().unwrap().to_ascii_lowercase();
        assert!(
            head.contains("post /v1/audio/translations"),
            "should hit translations route: {}",
            head
        );
        // The translations endpoint rejects a language parameter.
        assert!(
            !req.contains("name=\"language\""),
            "language must not be sent to translations endpoint"
        );
    }

    #[tokio::test]
    async fn auto_language_omits_language_field() {
        let (url, rx) = spawn_once(canned("HTTP/1.1 200 OK", r#"{"text":"detected"}"#));
        let provider = OpenAiProvider::new("k".to_string(), String::new(), Some(url));

        provider
            .transcribe(vec![0.1f32; MIN_SAMPLES + 10], Some("auto".to_string()))
            .await
            .expect("auto-detect should succeed");

        let req = rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        assert!(
            !req.contains("name=\"language\""),
            "auto-detect must not pin a language"
        );
        // Default model is used when none configured.
        assert!(req.contains("whisper-1"), "default model missing");
    }

    #[tokio::test]
    async fn transcribe_maps_401_to_key_error() {
        let (url, _rx) = spawn_once(canned(
            "HTTP/1.1 401 Unauthorized",
            r#"{"error":{"message":"Incorrect API key provided"}}"#,
        ));
        let provider = OpenAiProvider::new("bad".to_string(), "whisper-1".to_string(), Some(url));

        let err = provider
            .transcribe(vec![0.0f32; MIN_SAMPLES + 1], None)
            .await
            .expect_err("401 should be an error");
        match err {
            TranscriptionError::EngineFailed(msg) => {
                assert!(msg.contains("API key"), "expected key-hint message, got: {}", msg);
                assert!(
                    msg.contains("Incorrect API key provided"),
                    "server detail should be surfaced, got: {}",
                    msg
                );
            }
            other => panic!("expected EngineFailed, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn transcribe_maps_429_to_quota_error() {
        let (url, _rx) = spawn_once(canned(
            "HTTP/1.1 429 Too Many Requests",
            r#"{"error":{"message":"You exceeded your current quota"}}"#,
        ));
        let provider = OpenAiProvider::new("k".to_string(), "whisper-1".to_string(), Some(url));

        let err = provider
            .transcribe(vec![0.0f32; MIN_SAMPLES + 1], None)
            .await
            .expect_err("429 should be an error");
        match err {
            TranscriptionError::EngineFailed(msg) => {
                assert!(
                    msg.contains("rate limit") || msg.contains("quota"),
                    "expected quota hint, got: {}",
                    msg
                );
            }
            other => panic!("expected EngineFailed, got {:?}", other),
        }
    }
    /// Live wire-compatibility check against the real api.openai.com.
    ///
    /// Without a paid key we cannot assert on transcript text, but we CAN prove
    /// the request we build is well-formed enough for OpenAI to parse and
    /// authenticate: a malformed multipart body or wrong route would yield 400
    /// or 404, whereas a correctly-shaped request with a bad key yields 401.
    /// This is the closest we get to the real acceptance path without billing.
    ///
    /// Ignored by default so CI/offline runs are unaffected; run with
    /// `cargo test -- --ignored openai_live`.
    #[tokio::test]
    #[ignore]
    async fn openai_live_endpoint_rejects_bad_key_not_bad_request() {
        let provider =
            OpenAiProvider::new("sk-invalid-key-for-testing".to_string(), "whisper-1".to_string(), None);
        let audio = vec![0.05f32; 16_000]; // 1 second of quiet audio
        let err = provider
            .transcribe(audio, Some("en".to_string()))
            .await
            .expect_err("an invalid key must fail");

        let msg = err.to_string();
        println!("LIVE OPENAI RESPONSE: {}", msg);
        assert!(
            msg.contains("API key"),
            "expected an auth rejection (proving the request shape was accepted \
             and it failed only on credentials), got: {}",
            msg
        );
    }

    /// End-to-end success path against a real OpenAI-compatible HTTP server
    /// running out-of-process (see /tmp/mock_oai_server.py). Unlike the in-test
    /// mock, that server genuinely parses the multipart body and decodes the
    /// WAV with a standard library, so a malformed container fails loudly.
    ///
    /// Ignored by default (needs the external server); run with
    /// `cargo test -- --ignored openai_e2e`.
    #[tokio::test]
    #[ignore]
    async fn openai_e2e_against_external_compatible_server() {
        let provider = OpenAiProvider::new(
            "sk-local".to_string(),
            "Systran/faster-whisper-large-v3".to_string(),
            Some("http://127.0.0.1:8731/v1".to_string()),
        );
        let audio = vec![0.25f32; 16_000]; // 1s
        let result = provider
            .transcribe(audio, Some("en".to_string()))
            .await
            .expect("self-hosted OpenAI-compatible transcription should succeed");
        println!("E2E TRANSCRIPT: {:?}", result.text);
        assert_eq!(result.text, "the quick brown fox");
        assert!(!result.is_partial);
    }

}
