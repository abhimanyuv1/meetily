use crate::database::models::{Setting, TranscriptSetting};
use crate::summary::CustomOpenAIConfig;
use sqlx::SqlitePool;

#[derive(serde::Deserialize, Debug)]
pub struct SaveModelConfigRequest {
    pub provider: String,
    pub model: String,
    #[serde(rename = "whisperModel")]
    pub whisper_model: String,
    #[serde(rename = "apiKey")]
    pub api_key: Option<String>,
    #[serde(rename = "ollamaEndpoint")]
    pub ollama_endpoint: Option<String>,
}

#[derive(serde::Deserialize, Debug)]
pub struct SaveTranscriptConfigRequest {
    pub provider: String,
    pub model: String,
    #[serde(rename = "apiKey")]
    pub api_key: Option<String>,
}

pub struct SettingsRepository;

// Transcript providers: localWhisper, deepgram, elevenLabs, groq, openai
// Summary providers: openai, claude, ollama, groq, added openrouter
// NOTE: Handle data exclusion in the higher layer as this is database abstraction layer(using SELECT *)

impl SettingsRepository {
    pub async fn get_model_config(
        pool: &SqlitePool,
    ) -> std::result::Result<Option<Setting>, sqlx::Error> {
        let setting = sqlx::query_as::<_, Setting>("SELECT * FROM settings LIMIT 1")
            .fetch_optional(pool)
            .await?;
        Ok(setting)
    }

    pub async fn save_model_config(
        pool: &SqlitePool,
        provider: &str,
        model: &str,
        whisper_model: &str,
        ollama_endpoint: Option<&str>,
    ) -> std::result::Result<(), sqlx::Error> {
        // Using id '1' for backward compatibility
        sqlx::query(
            r#"
            INSERT INTO settings (id, provider, model, whisperModel, ollamaEndpoint)
            VALUES ('1', $1, $2, $3, $4)
            ON CONFLICT(id) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model,
                whisperModel = excluded.whisperModel,
                ollamaEndpoint = excluded.ollamaEndpoint
            "#,
        )
        .bind(provider)
        .bind(model)
        .bind(whisper_model)
        .bind(ollama_endpoint)
        .execute(pool)
        .await?;

        Ok(())
    }

    pub async fn save_api_key(
        pool: &SqlitePool,
        provider: &str,
        api_key: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        // Custom OpenAI uses JSON config (customOpenAIConfig) instead of a separate API key column
        if provider == "custom-openai" {
            return Err(sqlx::Error::Protocol(
                "custom-openai provider should use save_custom_openai_config() instead of save_api_key()".into(),
            ));
        }

        let api_key_column = match provider {
            "openai" => "openaiApiKey",
            "claude" => "anthropicApiKey",
            "ollama" => "ollamaApiKey",
            "groq" => "groqApiKey",
            "openrouter" => "openRouterApiKey",
            "builtin-ai" => return Ok(()), // No API key needed
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };

        let query = format!(
            r#"
            INSERT INTO settings (id, provider, model, whisperModel, "{}")
            VALUES ('1', 'openai', 'gpt-4o-2024-11-20', 'large-v3', $1)
            ON CONFLICT(id) DO UPDATE SET
                "{}" = $1
            "#,
            api_key_column, api_key_column
        );
        sqlx::query(&query).bind(api_key).execute(pool).await?;

        Ok(())
    }

    pub async fn get_api_key(
        pool: &SqlitePool,
        provider: &str,
    ) -> std::result::Result<Option<String>, sqlx::Error> {
        // Custom OpenAI uses JSON config - extract API key from there
        if provider == "custom-openai" {
            let config = Self::get_custom_openai_config(pool).await?;
            return Ok(config.and_then(|c| c.api_key));
        }

        let api_key_column = match provider {
            "openai" => "openaiApiKey",
            "ollama" => "ollamaApiKey",
            "groq" => "groqApiKey",
            "claude" => "anthropicApiKey",
            "openrouter" => "openRouterApiKey",
            "builtin-ai" => return Ok(None), // No API key needed
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };

        let query = format!(
            "SELECT {} FROM settings WHERE id = '1' LIMIT 1",
            api_key_column
        );
        let api_key = sqlx::query_scalar(&query).fetch_optional(pool).await?;
        Ok(api_key)
    }

    pub async fn get_transcript_config(
        pool: &SqlitePool,
    ) -> std::result::Result<Option<TranscriptSetting>, sqlx::Error> {
        let setting =
            sqlx::query_as::<_, TranscriptSetting>("SELECT * FROM transcript_settings LIMIT 1")
                .fetch_optional(pool)
                .await?;
        Ok(setting)

    }

    pub async fn save_transcript_config(
        pool: &SqlitePool,
        provider: &str,
        model: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            INSERT INTO transcript_settings (id, provider, model)
            VALUES ('1', $1, $2)
            ON CONFLICT(id) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model
            "#,
        )
        .bind(provider)
        .bind(model)
        .execute(pool)
        .await?;

        Ok(())
    }

    pub async fn save_transcript_api_key(
        pool: &SqlitePool,
        provider: &str,
        api_key: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        let api_key_column = match provider {
            "localWhisper" => "whisperApiKey",
            "parakeet" => return Ok(()), // Parakeet doesn't need an API key, return early
            "deepgram" => "deepgramApiKey",
            "elevenLabs" => "elevenLabsApiKey",
            "groq" => "groqApiKey",
            "openai" => "openaiApiKey",
            "sarvam" => "sarvamApiKey",
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };

        let query = format!(
            r#"
            INSERT INTO transcript_settings (id, provider, model, "{}")
            VALUES ('1', 'parakeet', '{}', $1)
            ON CONFLICT(id) DO UPDATE SET
                "{}" = $1
            "#,
            api_key_column, crate::config::DEFAULT_PARAKEET_MODEL, api_key_column
        );
        sqlx::query(&query).bind(api_key).execute(pool).await?;

        Ok(())
    }

    /// Reads the optional OpenAI base-URL override. Empty string means "unset",
    /// i.e. use the official OpenAI endpoint. Exposed so users can point the
    /// OpenAI provider at any OpenAI-compatible STT service.
    pub async fn get_openai_base_url(
        pool: &SqlitePool,
    ) -> std::result::Result<Option<String>, sqlx::Error> {
        let url: Option<Option<String>> =
            sqlx::query_scalar("SELECT openaiBaseUrl FROM transcript_settings WHERE id = '1' LIMIT 1")
                .fetch_optional(pool)
                .await?;
        Ok(url
            .flatten()
            .map(|u| u.trim().to_string())
            .filter(|u| !u.is_empty()))
    }

    /// Persists the OpenAI base-URL override. Passing an empty/whitespace value
    /// clears it, restoring the official OpenAI endpoint.
    pub async fn save_openai_base_url(
        pool: &SqlitePool,
        base_url: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        let trimmed = base_url.trim();
        let value: Option<&str> = if trimmed.is_empty() { None } else { Some(trimmed) };
        sqlx::query(
            r#"
            INSERT INTO transcript_settings (id, provider, model, openaiBaseUrl)
            VALUES ('1', 'openai', '', $1)
            ON CONFLICT(id) DO UPDATE SET
                openaiBaseUrl = $1
            "#,
        )
        .bind(value)
        .execute(pool)
        .await?;
        Ok(())
    }

    pub async fn get_transcript_api_key(
        pool: &SqlitePool,
        provider: &str,
    ) -> std::result::Result<Option<String>, sqlx::Error> {
        let api_key_column = match provider {
            "localWhisper" => "whisperApiKey",
            "parakeet" => return Ok(None), // Parakeet doesn't need an API key
            "deepgram" => "deepgramApiKey",
            "elevenLabs" => "elevenLabsApiKey",
            "groq" => "groqApiKey",
            "openai" => "openaiApiKey",
            "sarvam" => "sarvamApiKey",
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };

        let query = format!(
            "SELECT {} FROM transcript_settings WHERE id = '1' LIMIT 1",
            api_key_column
        );
        let api_key = sqlx::query_scalar(&query).fetch_optional(pool).await?;
        Ok(api_key)
    }

    pub async fn delete_api_key(
        pool: &SqlitePool,
        provider: &str,
    ) -> std::result::Result<(), sqlx::Error> {
        // Custom OpenAI uses JSON config - clear the entire config
        if provider == "custom-openai" {
            sqlx::query("UPDATE settings SET customOpenAIConfig = NULL WHERE id = '1'")
                .execute(pool)
                .await?;
            return Ok(());
        }

        let api_key_column = match provider {
            "openai" => "openaiApiKey",
            "ollama" => "ollamaApiKey",
            "groq" => "groqApiKey",
            "claude" => "anthropicApiKey",
            "openrouter" => "openRouterApiKey",
            "builtin-ai" => return Ok(()), // No API key needed
            _ => {
                return Err(sqlx::Error::Protocol(
                    format!("Invalid provider: {}", provider).into(),
                ))
            }
        };

        let query = format!(
            "UPDATE settings SET {} = NULL WHERE id = '1'",
            api_key_column
        );
        sqlx::query(&query).execute(pool).await?;

        Ok(())
    }

    // ===== CUSTOM OPENAI CONFIG METHODS =====

    /// Gets the custom OpenAI configuration from JSON
    ///
    /// # Returns
    /// * `Ok(Some(CustomOpenAIConfig))` - Config exists and is valid JSON
    /// * `Ok(None)` - No config stored
    /// * `Err(sqlx::Error)` - Database error
    pub async fn get_custom_openai_config(
        pool: &SqlitePool,
    ) -> std::result::Result<Option<CustomOpenAIConfig>, sqlx::Error> {
        use sqlx::Row;

        let row = sqlx::query(
            r#"
            SELECT customOpenAIConfig
            FROM settings
            WHERE id = '1'
            LIMIT 1
            "#
        )
        .fetch_optional(pool)
        .await?;

        match row {
            Some(record) => {
                let config_json: Option<String> = record.get("customOpenAIConfig");

                if let Some(json) = config_json {
                    // Parse JSON into CustomOpenAIConfig
                    let config: CustomOpenAIConfig = serde_json::from_str(&json)
                        .map_err(|e| sqlx::Error::Protocol(
                            format!("Invalid JSON in customOpenAIConfig: {}", e).into()
                        ))?;

                    Ok(Some(config))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    /// Saves the custom OpenAI configuration as JSON
    ///
    /// # Arguments
    /// * `pool` - Database connection pool
    /// * `config` - CustomOpenAIConfig to save (includes endpoint, apiKey, model, maxTokens, temperature, topP)
    ///
    /// # Returns
    /// * `Ok(())` - Config saved successfully
    /// * `Err(sqlx::Error)` - Database or JSON serialization error
    pub async fn save_custom_openai_config(
        pool: &SqlitePool,
        config: &CustomOpenAIConfig,
    ) -> std::result::Result<(), sqlx::Error> {
        // Serialize config to JSON
        let config_json = serde_json::to_string(config)
            .map_err(|e| sqlx::Error::Protocol(
                format!("Failed to serialize config to JSON: {}", e).into()
            ))?;

        // Upsert into settings table
        sqlx::query(
            r#"
            INSERT INTO settings (id, provider, model, whisperModel, customOpenAIConfig)
            VALUES ('1', 'custom-openai', $1, 'large-v3', $2)
            ON CONFLICT(id) DO UPDATE SET
                customOpenAIConfig = excluded.customOpenAIConfig
            "#,
        )
        .bind(&config.model)
        .bind(config_json)
        .execute(pool)
        .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::manager::DatabaseManager;

    /// Builds a real database the same way the app does: an on-disk SQLite file
    /// created and migrated by `DatabaseManager::new`, which runs the actual
    /// `migrations/` directory. This exercises the real migration chain rather
    /// than a hand-written schema, so a broken/missing migration fails here.
    async fn real_migrated_db() -> (DatabaseManager, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("test.sqlite");
        let db_str = db_path.to_str().unwrap().to_string();
        let mgr = DatabaseManager::new(&db_str, "/nonexistent-legacy.db")
            .await
            .expect("DatabaseManager::new should create and migrate the DB");
        (mgr, dir)
    }

    #[tokio::test]
    async fn openai_base_url_roundtrips_through_real_migrated_db() {
        let (mgr, _dir) = real_migrated_db().await;
        let pool = mgr.pool();

        // Unset by default (the column exists thanks to the new migration).
        let initial = SettingsRepository::get_openai_base_url(pool)
            .await
            .expect("get should succeed on a freshly migrated DB");
        assert_eq!(initial, None, "base URL should start unset");

        // Save a custom endpoint and read it back.
        SettingsRepository::save_openai_base_url(pool, "http://localhost:8000/v1")
            .await
            .expect("save should succeed");
        assert_eq!(
            SettingsRepository::get_openai_base_url(pool).await.unwrap(),
            Some("http://localhost:8000/v1".to_string())
        );

        // Whitespace is trimmed on the way in.
        SettingsRepository::save_openai_base_url(pool, "  http://example.com/v1  ")
            .await
            .unwrap();
        assert_eq!(
            SettingsRepository::get_openai_base_url(pool).await.unwrap(),
            Some("http://example.com/v1".to_string())
        );

        // An empty value clears it, restoring the official OpenAI endpoint.
        SettingsRepository::save_openai_base_url(pool, "   ")
            .await
            .unwrap();
        assert_eq!(
            SettingsRepository::get_openai_base_url(pool).await.unwrap(),
            None,
            "empty value should clear the override"
        );
    }

    #[tokio::test]
    async fn saving_transcript_config_preserves_openai_base_url() {
        // Regression guard: the base URL lives in the same row as provider/model,
        // so a later config save must not clobber it.
        let (mgr, _dir) = real_migrated_db().await;
        let pool = mgr.pool();

        SettingsRepository::save_openai_base_url(pool, "https://api.groq.com/openai/v1")
            .await
            .unwrap();
        SettingsRepository::save_transcript_config(pool, "openai", "whisper-1")
            .await
            .unwrap();

        assert_eq!(
            SettingsRepository::get_openai_base_url(pool).await.unwrap(),
            Some("https://api.groq.com/openai/v1".to_string()),
            "base URL must survive a transcript config save"
        );

        let config = SettingsRepository::get_transcript_config(pool)
            .await
            .unwrap()
            .expect("config should exist");
        assert_eq!(config.provider, "openai");
        assert_eq!(config.model, "whisper-1");
    }

    #[tokio::test]
    async fn openai_api_key_roundtrips_for_transcript_provider() {
        // The end-user path: settings UI saves provider+model+key, and the
        // recording engine reads the key back for the 'openai' provider.
        let (mgr, _dir) = real_migrated_db().await;
        let pool = mgr.pool();

        SettingsRepository::save_transcript_config(pool, "openai", "whisper-1")
            .await
            .unwrap();
        SettingsRepository::save_transcript_api_key(pool, "openai", "sk-test-123")
            .await
            .unwrap();

        let key = SettingsRepository::get_transcript_api_key(pool, "openai")
            .await
            .unwrap();
        assert_eq!(key, Some("sk-test-123".to_string()));

        // Saving the OpenAI key must not disturb the Sarvam key (separate columns).
        SettingsRepository::save_transcript_api_key(pool, "sarvam", "sarvam-key")
            .await
            .unwrap();
        assert_eq!(
            SettingsRepository::get_transcript_api_key(pool, "openai")
                .await
                .unwrap(),
            Some("sk-test-123".to_string())
        );
        assert_eq!(
            SettingsRepository::get_transcript_api_key(pool, "sarvam")
                .await
                .unwrap(),
            Some("sarvam-key".to_string())
        );
    }
}

#[cfg(test)]
mod upgrade_tests {
    use super::*;
    use crate::database::manager::DatabaseManager;

    /// Upgrade path for an EXISTING install: a database created before the
    /// openaiBaseUrl migration, already holding user settings, must gain the
    /// column without losing data when the app next starts. Creating the DB
    /// fresh (as the other tests do) would not catch a migration that only
    /// works on an empty schema.
    #[tokio::test]
    async fn existing_database_upgrades_and_preserves_user_data() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("existing.sqlite");
        let db_str = db_path.to_str().unwrap().to_string();

        // Build the pre-migration state using sqlx's own migrator restricted to
        // the migrations that existed before mine. Using the real migrator (not
        // raw SQL) means the _sqlx_migrations ledger is written correctly, so
        // the later upgrade applies only the new migration, exactly as it will
        // on a real user's machine.
        {
            let url = format!("sqlite://{}?mode=rwc", db_str);
            let pool = sqlx::SqlitePool::connect(&url).await.unwrap();
            let mut migrator = sqlx::migrate!("./migrations");
            migrator
                .migrations
                .to_mut()
                .retain(|m| !m.description.contains("openai transcript base url"));
            assert!(
                !migrator.migrations.is_empty(),
                "sanity: pre-migration set must not be empty"
            );
            migrator.run(&pool).await.expect("baseline migrations apply");

            // The column must genuinely be absent in this baseline, otherwise
            // the test proves nothing about the upgrade.
            let cols: Vec<(i64, String, String, i64, Option<String>, i64)> =
                sqlx::query_as("PRAGMA table_info(transcript_settings)")
                    .fetch_all(&pool)
                    .await
                    .unwrap();
            assert!(
                !cols.iter().any(|c| c.1 == "openaiBaseUrl"),
                "baseline DB must predate the openaiBaseUrl column"
            );

            sqlx::query(
                "INSERT INTO transcript_settings (id, provider, model, sarvamApiKey)
                 VALUES ('1','sarvam','saaras:v3','existing-sarvam-key')",
            )
            .execute(&pool)
            .await
            .unwrap();
            pool.close().await;
        }

        // Now start the app's real DatabaseManager against that existing file,
        // which runs the full migration chain including the new column.
        let mgr = DatabaseManager::new(&db_str, "/nonexistent-legacy.db")
            .await
            .expect("migrating an existing database must succeed");
        let pool = mgr.pool();

        // Pre-existing settings survived the upgrade.
        let config = SettingsRepository::get_transcript_config(pool)
            .await
            .unwrap()
            .expect("existing config must survive migration");
        assert_eq!(config.provider, "sarvam");
        assert_eq!(config.model, "saaras:v3");
        assert_eq!(
            SettingsRepository::get_transcript_api_key(pool, "sarvam")
                .await
                .unwrap(),
            Some("existing-sarvam-key".to_string()),
            "an existing user's Sarvam key must not be lost"
        );

        // The new column exists and is usable on the upgraded database.
        assert_eq!(
            SettingsRepository::get_openai_base_url(pool).await.unwrap(),
            None
        );
        SettingsRepository::save_openai_base_url(pool, "https://api.openai.com/v1")
            .await
            .expect("new column must be writable after upgrade");
        assert_eq!(
            SettingsRepository::get_openai_base_url(pool).await.unwrap(),
            Some("https://api.openai.com/v1".to_string())
        );
    }
}
