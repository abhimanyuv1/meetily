// audio/transcription/sarvam_provider.rs
//
// Sarvam AI speech-to-text provider — an *online* transcription backend.
//
// This is the first cloud provider wired through the generic
// `TranscriptionProvider` trait, giving Meetily an alternative to the local
// Whisper/Parakeet engines when the machine is too weak to transcribe in
// real time without stutter (or when higher accuracy for Indic languages is
// wanted). Each audio chunk handed to us by the transcription worker is
// encoded as a 16 kHz mono WAV and POSTed to Sarvam's REST endpoint.
//
// Docs: https://docs.sarvam.ai/api-reference/speech-to-text/transcribe

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use super::wav::encode_wav_16k_mono;
use async_trait::async_trait;
use log::{info, warn};
use serde::Deserialize;
use std::time::Duration;

const SARVAM_ENDPOINT: &str = "https://api.sarvam.ai/speech-to-text";
const DEFAULT_MODEL: &str = "saaras:v3";
/// Model ids the Sarvam /speech-to-text endpoint accepts. Anything else (e.g. a
/// leaked local Whisper model like "large-v3") is coerced to DEFAULT_MODEL so we
/// never send an invalid model and get a 400.
const VALID_MODELS: &[&str] = &[
    "saarika:v2.5",
    "saaras:v3",
    "saaras:v3-realtime",
    "saaras:v4",
    "saaras:v4-multispk",
    "saarika:v1",
    "saarika:v2",
    "saarika:flash",
];
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Sarvam's REST endpoint targets clips under ~30s; below this many samples a
/// request is wasteful/noisy, so we skip it (mirrors the other providers'
/// "audio too short" guard). 100ms at 16kHz.
const MIN_SAMPLES: usize = 1600;

#[derive(Debug, Deserialize)]
struct SarvamResponse {
    #[serde(default)]
    transcript: String,
    #[allow(dead_code)]
    #[serde(default)]
    language_code: Option<String>,
}

/// Online transcription via the Sarvam AI REST API.
pub struct SarvamProvider {
    api_key: String,
    model: String,
    endpoint: String,
    client: reqwest::Client,
}

impl SarvamProvider {
    /// `model` is the Sarvam model id (e.g. "saaras:v3" / "saaras:v4"); falls
    /// back to the recommended default when empty.
    pub fn new(api_key: String, model: String) -> Self {
        Self::with_endpoint(api_key, model, SARVAM_ENDPOINT.to_string())
    }

    /// Same as `new` but with a custom endpoint URL. Primarily used to point at
    /// a local mock server in integration tests; production code uses `new`.
    pub fn with_endpoint(api_key: String, model: String, endpoint: String) -> Self {
        Self {
            api_key,
            model: Self::sanitize_model(&model),
            endpoint,
            client: reqwest::Client::new(),
        }
    }

    /// Coerce a requested model id to one Sarvam actually accepts. Empty or
    /// unrecognized values (e.g. a local Whisper model id like "large-v3" that
    /// leaked in from shared settings) fall back to DEFAULT_MODEL, preventing a
    /// 400 from the API.
    fn sanitize_model(model: &str) -> String {
        let trimmed = model.trim();
        if VALID_MODELS.contains(&trimmed) {
            trimmed.to_string()
        } else {
            if !trimmed.is_empty() {
                warn!(
                    "Sarvam: model '{}' is not a valid Sarvam model; falling back to '{}'",
                    trimmed, DEFAULT_MODEL
                );
            }
            DEFAULT_MODEL.to_string()
        }
    }

    /// Maps Meetily's short language hint (e.g. "en", "hi") to Sarvam's BCP-47
    /// `language_code` (e.g. "en-IN", "hi-IN"). Anything unknown / unset maps to
    /// "unknown", letting Sarvam auto-detect.
    fn sarvam_language_code(language: &Option<String>) -> String {
        let lang = match language {
            Some(l) if !l.trim().is_empty() => l.trim().to_lowercase(),
            _ => return "unknown".to_string(),
        };

        // Meetily's synthetic hints for auto-detection map to Sarvam's "unknown"
        // (it auto-detects). "auto-translate" is Meetily-specific and not a real
        // BCP-47 code, so it must be caught before the hyphen passthrough below.
        if lang == "auto" || lang == "auto-translate" || lang == "unknown" {
            return "unknown".to_string();
        }

        // Already a BCP-47 code like "en-IN"? pass through.
        if lang.contains('-') {
            return lang
                .split_once('-')
                .map(|(a, b)| format!("{}-{}", a, b.to_uppercase()))
                .unwrap_or(lang);
        }

        match lang.as_str() {
            "en" => "en-IN",
            "hi" => "hi-IN",
            "bn" => "bn-IN",
            "kn" => "kn-IN",
            "ml" => "ml-IN",
            "mr" => "mr-IN",
            "od" | "or" => "od-IN",
            "pa" => "pa-IN",
            "ta" => "ta-IN",
            "te" => "te-IN",
            "gu" => "gu-IN",
            "as" => "as-IN",
            "ur" => "ur-IN",
            "ne" => "ne-IN",
            "kok" => "kok-IN",
            "ks" => "ks-IN",
            "sd" => "sd-IN",
            "sa" => "sa-IN",
            "sat" => "sat-IN",
            "mni" => "mni-IN",
            "brx" => "brx-IN",
            "mai" => "mai-IN",
            "doi" => "doi-IN",
            _ => "unknown",
        }
        .to_string()
    }
}

#[async_trait]
impl TranscriptionProvider for SarvamProvider {
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

        let wav = encode_wav_16k_mono(&audio);
        let language_code = Self::sarvam_language_code(&language);

        let file_part = reqwest::multipart::Part::bytes(wav)
            .file_name("chunk.wav")
            .mime_str("audio/wav")
            .map_err(|e| TranscriptionError::EngineFailed(format!("multipart build failed: {}", e)))?;

        let form = reqwest::multipart::Form::new()
            .part("file", file_part)
            .text("model", self.model.clone())
            .text("language_code", language_code);

        info!(
            "Sarvam: transcribing {} samples with model '{}'",
            audio.len(),
            self.model
        );

        let response = self
            .client
            .post(&self.endpoint)
            .header("api-subscription-key", &self.api_key)
            .multipart(form)
            .timeout(REQUEST_TIMEOUT)
            .send()
            .await
            .map_err(|e| TranscriptionError::EngineFailed(format!("request to Sarvam failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            if status == reqwest::StatusCode::UNAUTHORIZED
                || status == reqwest::StatusCode::FORBIDDEN
            {
                return Err(TranscriptionError::EngineFailed(format!(
                    "Sarvam rejected the API key (HTTP {}). Check the key in Transcript settings.",
                    status
                )));
            }
            return Err(TranscriptionError::EngineFailed(format!(
                "Sarvam API error (HTTP {}): {}",
                status, body
            )));
        }

        let parsed: SarvamResponse = response
            .json()
            .await
            .map_err(|e| TranscriptionError::EngineFailed(format!("failed to parse Sarvam response: {}", e)))?;

        Ok(TranscriptResult {
            text: parsed.transcript.trim().to_string(),
            // Sarvam's REST response doesn't include a per-transcript confidence.
            confidence: None,
            is_partial: false,
        })
    }

    async fn is_model_loaded(&self) -> bool {
        // No local model to load — the provider is "ready" as long as a key is set.
        if self.api_key.trim().is_empty() {
            warn!("Sarvam provider has no API key configured");
            return false;
        }
        true
    }

    async fn get_current_model(&self) -> Option<String> {
        Some(self.model.clone())
    }

    fn provider_name(&self) -> &'static str {
        "Sarvam"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_none_or_empty_maps_to_unknown() {
        assert_eq!(SarvamProvider::sarvam_language_code(&None), "unknown");
        assert_eq!(
            SarvamProvider::sarvam_language_code(&Some("   ".to_string())),
            "unknown"
        );
        assert_eq!(
            SarvamProvider::sarvam_language_code(&Some("zz".to_string())),
            "unknown"
        );
    }

    #[test]
    fn language_auto_hints_map_to_unknown() {
        // Meetily's auto-detect codes must not be sent to Sarvam verbatim.
        // "auto-translate" has a hyphen and would otherwise pass through the
        // BCP-47 branch as an invalid code.
        assert_eq!(
            SarvamProvider::sarvam_language_code(&Some("auto".to_string())),
            "unknown"
        );
        assert_eq!(
            SarvamProvider::sarvam_language_code(&Some("auto-translate".to_string())),
            "unknown"
        );
        assert_eq!(
            SarvamProvider::sarvam_language_code(&Some("AUTO-TRANSLATE".to_string())),
            "unknown"
        );
    }

    #[test]
    fn language_short_hints_map_to_bcp47() {
        assert_eq!(
            SarvamProvider::sarvam_language_code(&Some("en".to_string())),
            "en-IN"
        );
        assert_eq!(
            SarvamProvider::sarvam_language_code(&Some("HI".to_string())),
            "hi-IN"
        );
        assert_eq!(
            SarvamProvider::sarvam_language_code(&Some("ta".to_string())),
            "ta-IN"
        );
    }

    #[test]
    fn language_existing_bcp47_is_normalized_passthrough() {
        assert_eq!(
            SarvamProvider::sarvam_language_code(&Some("en-in".to_string())),
            "en-IN"
        );
        assert_eq!(
            SarvamProvider::sarvam_language_code(&Some("hi-IN".to_string())),
            "hi-IN"
        );
    }

    #[test]
    fn empty_model_falls_back_to_default() {
        let p = SarvamProvider::new("k".to_string(), "".to_string());
        assert_eq!(p.model, DEFAULT_MODEL);
        let p2 = SarvamProvider::new("k".to_string(), "saaras:v4".to_string());
        assert_eq!(p2.model, "saaras:v4");
    }

    #[test]
    fn invalid_model_is_coerced_to_default() {
        // A local Whisper model id leaking in from shared settings must never be
        // sent to Sarvam (it returns HTTP 400). It is coerced to the default.
        let p = SarvamProvider::new("k".to_string(), "large-v3".to_string());
        assert_eq!(p.model, DEFAULT_MODEL);
        // Whitespace around a valid id is tolerated.
        let p2 = SarvamProvider::new("k".to_string(), "  saaras:v4  ".to_string());
        assert_eq!(p2.model, "saaras:v4");
        // Other valid ids pass through.
        let p3 = SarvamProvider::new("k".to_string(), "saarika:v2.5".to_string());
        assert_eq!(p3.model, "saarika:v2.5");
    }

    #[tokio::test]
    async fn short_audio_is_rejected() {
        let p = SarvamProvider::new("k".to_string(), "saaras:v3".to_string());
        let res = p.transcribe(vec![0.0f32; 10], None).await;
        assert!(matches!(
            res,
            Err(TranscriptionError::AudioTooShort { .. })
        ));
    }

    #[tokio::test]
    async fn is_model_loaded_requires_api_key() {
        let with_key = SarvamProvider::new("k".to_string(), String::new());
        assert!(with_key.is_model_loaded().await);
        let no_key = SarvamProvider::new("   ".to_string(), String::new());
        assert!(!no_key.is_model_loaded().await);
    }

    // --- Integration-boundary tests against a local mock HTTP server ---------
    //
    // These exercise the real reqwest multipart POST + response parsing without
    // needing a Sarvam key or network. A tiny one-shot TCP server reads the
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
                // Read until we have headers + body. reqwest sends Content-Length,
                // so read that many body bytes after the header terminator.
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
                        // Determine expected body length from Content-Length.
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
        (format!("http://{}/speech-to-text", addr), rx)
    }

    #[tokio::test]
    async fn transcribe_sends_expected_request_and_parses_response() {
        let body = "{\"request_id\":\"r1\",\"transcript\":\"hello world\",\"language_code\":\"en-IN\"}";
        let response_owned = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        // Leak to obtain the 'static lifetime required by the server thread.
        let response_static: &'static str = Box::leak(response_owned.into_boxed_str());

        let (url, rx) = spawn_once(response_static);
        let provider = SarvamProvider::with_endpoint(
            "secret-key-xyz".to_string(),
            "saaras:v4".to_string(),
            url,
        );

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
        assert!(head.contains("post /speech-to-text"), "method/path: {}", head);
        assert!(
            head.contains("api-subscription-key: secret-key-xyz"),
            "auth header missing: {}",
            head
        );
        assert!(head.contains("content-type: multipart/form-data"), "not multipart: {}", head);
        // Multipart field values appear in the body
        assert!(req.contains("name=\"model\""), "model field missing");
        assert!(req.contains("saaras:v4"), "model value missing");
        assert!(req.contains("name=\"language_code\""), "language_code field missing");
        assert!(req.contains("en-IN"), "mapped language missing");
        assert!(req.contains("name=\"file\""), "file field missing");
        assert!(req.contains("filename=\"chunk.wav\""), "wav filename missing");
        assert!(req.contains("RIFF"), "wav payload missing");
    }

    #[tokio::test]
    async fn transcribe_maps_401_to_key_error() {
        let body = "{\"error\":\"unauthorized\"}";
        let response_owned = format!(
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let response_static: &'static str = Box::leak(response_owned.into_boxed_str());
        let (url, _rx) = spawn_once(response_static);
        let provider =
            SarvamProvider::with_endpoint("bad".to_string(), "saaras:v3".to_string(), url);

        let err = provider
            .transcribe(vec![0.0f32; MIN_SAMPLES + 1], None)
            .await
            .expect_err("401 should be an error");
        match err {
            TranscriptionError::EngineFailed(msg) => {
                assert!(msg.contains("API key"), "expected key-hint message, got: {}", msg);
            }
            other => panic!("expected EngineFailed, got {:?}", other),
        }
    }
}
