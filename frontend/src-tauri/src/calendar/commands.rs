use super::models::{
    CalendarAccountStatusDto, CalendarAutoStartSettingsDto, CalendarEventDto,
    UpdateAutoStartSettingsRequest,
};
use super::oauth;
use crate::database::repositories::calendar::CalendarRepository;
use crate::state::AppState;
use chrono::{Duration as ChronoDuration, Utc};

/// Builds the full status DTO, including bring-your-own credential presence.
async fn build_status_dto(
    pool: &sqlx::SqlitePool,
) -> Result<CalendarAccountStatusDto, String> {
    let account = CalendarRepository::get_account(pool)
        .await
        .map_err(|e| e.to_string())?;
    let creds = CalendarRepository::get_oauth_client(pool)
        .await
        .map_err(|e| e.to_string())?;

    let mut dto = match account {
        Some(acc) => CalendarAccountStatusDto {
            connected: acc.status == "connected",
            email: Some(acc.email),
            status: acc.status,
            credentials_configured: false,
            client_id_hint: None,
        },
        None => CalendarAccountStatusDto {
            connected: false,
            email: None,
            status: "disconnected".to_string(),
            credentials_configured: false,
            client_id_hint: None,
        },
    };
    dto.credentials_configured = creds.is_some();
    dto.client_id_hint = creds.as_ref().map(|c| shorten_client_id(&c.client_id));
    Ok(dto)
}

/// Client ids are not secret (they appear in browser URLs during sign-in), but
/// they're long — trim the middle so users can recognize which Google project
/// a stored credential belongs to.
fn shorten_client_id(id: &str) -> String {
    let chars: Vec<char> = id.chars().collect();
    if chars.len() <= 48 {
        return id.to_string();
    }
    let head: String = chars[..12].iter().collect();
    let tail: String = chars[chars.len() - 24..].iter().collect();
    format!("{}…{}", head, tail)
}

/// Runs the full OAuth loopback flow and persists the resulting tokens. Blocks (async)
/// until the user finishes (or abandons) the browser sign-in.
#[tauri::command]
pub async fn calendar_connect(
    state: tauri::State<'_, AppState>,
) -> Result<CalendarAccountStatusDto, String> {
    let pool = state.db_manager.pool();
    let creds = oauth::load_client_credentials(pool).await?;

    let pkce = oauth::generate_pkce();
    let state_token = oauth::generate_state();
    let (listener, port) = oauth::bind_loopback_listener().await?;
    let auth_url = oauth::build_auth_url(port, &state_token, &pkce.challenge, &creds)?;

    log::info!("Opening browser for Google Calendar sign-in");
    crate::api::open_external_url(auth_url).await?;

    let code = oauth::await_callback(listener, &state_token).await?;
    let tokens = oauth::exchange_code(&code, &pkce.verifier, port, &creds).await?;
    let email = oauth::fetch_connected_email(&tokens.access_token).await?;

    let pool = state.db_manager.pool();
    let expires_at = Utc::now() + ChronoDuration::seconds(tokens.expires_in);
    CalendarRepository::upsert_account(
        pool,
        &email,
        &tokens.access_token,
        &tokens.refresh_token,
        expires_at,
        oauth::SCOPE,
    )
    .await
    .map_err(|e| e.to_string())?;

    log::info!("Google Calendar connected: {}", email);

    build_status_dto(pool).await
}

#[tauri::command]
pub async fn calendar_disconnect(state: tauri::State<'_, AppState>) -> Result<(), String> {
    CalendarRepository::disconnect(state.db_manager.pool())
        .await
        .map_err(|e| e.to_string())
}

/// Stores user-supplied Google OAuth Desktop-app client credentials (parsed
/// from the JSON downloaded out of Google Cloud Console).
#[tauri::command]
pub async fn calendar_set_credentials(
    state: tauri::State<'_, AppState>,
    raw_json: String,
) -> Result<(), String> {
    let creds = oauth::parse_client_json(&raw_json)?;
    CalendarRepository::save_oauth_client(
        state.db_manager.pool(),
        &creds.client_id,
        &creds.client_secret,
    )
    .await
    .map_err(|e| e.to_string())?;

    log::info!("Google OAuth client credentials saved (bring-your-own)");
    Ok(())
}

/// Removes the stored client credentials. Only allowed while no account is
/// actively connected: an account's refresh tokens only work with the client that
/// made them, so clearing mid-connection would leave tokens unusable on refresh.
/// If the account has expired tokens (status "needs_reauth"), clearing is allowed
/// since the user will need to reconnect anyway, and the stale account is removed.
#[tauri::command]
pub async fn calendar_clear_credentials(state: tauri::State<'_, AppState>) -> Result<(), String> {
    let pool = state.db_manager.pool();
    if let Some(acc) = CalendarRepository::get_account(pool)
        .await
        .map_err(|e| e.to_string())?
    {
        // Only block if the account is actively connected (has valid tokens)
        if acc.status == "connected" {
            return Err("Disconnect your Google account before removing its sign-in credentials".to_string());
        }
        // If status is "needs_reauth" or other non-connected state, also disconnect
        // the account since it has invalid/stale tokens. This gives a clean slate
        // for the user to reconnect with fresh credentials if they wish.
        CalendarRepository::disconnect(pool)
            .await
            .map_err(|e| e.to_string())?;
        log::info!("Cleared stale account (status: {}) along with OAuth credentials", acc.status);
    }
    CalendarRepository::clear_oauth_client(pool)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn calendar_get_status(
    state: tauri::State<'_, AppState>,
) -> Result<CalendarAccountStatusDto, String> {
    build_status_dto(state.db_manager.pool()).await
}

/// Meetings synced from the connected calendar in the next 24h (plus a 1h lookback so
/// an in-progress meeting doesn't disappear from the list).
#[tauri::command]
pub async fn calendar_get_upcoming_events(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<CalendarEventDto>, String> {
    let pool = state.db_manager.pool();
    let now = Utc::now();
    let events = CalendarRepository::get_events_in_range(
        pool,
        now - ChronoDuration::hours(1),
        now + ChronoDuration::hours(24),
    )
    .await
    .map_err(|e| e.to_string())?;

    Ok(events
        .into_iter()
        .map(|e| CalendarEventDto {
            id: e.id,
            title: e.title.unwrap_or_else(|| "Untitled meeting".to_string()),
            start_time: e.start_time.to_rfc3339(),
            end_time: e.end_time.to_rfc3339(),
            is_meeting: e.meeting_url.is_some(),
            meeting_url: e.meeting_url,
            meeting_provider: e.meeting_provider,
        })
        .collect())
}

#[tauri::command]
pub async fn calendar_get_auto_start_settings(
    state: tauri::State<'_, AppState>,
) -> Result<CalendarAutoStartSettingsDto, String> {
    let account = CalendarRepository::get_account(state.db_manager.pool())
        .await
        .map_err(|e| e.to_string())?;

    Ok(match account {
        Some(acc) => CalendarAutoStartSettingsDto {
            enabled: acc.auto_start_enabled,
            mode: acc.auto_start_mode,
            grace_minutes: acc.auto_stop_grace_minutes,
        },
        // No account connected yet — report the same defaults the migration seeds.
        None => CalendarAutoStartSettingsDto {
            enabled: false,
            mode: "ask".to_string(),
            grace_minutes: 5,
        },
    })
}

#[tauri::command]
pub async fn calendar_update_auto_start_settings(
    state: tauri::State<'_, AppState>,
    settings: UpdateAutoStartSettingsRequest,
) -> Result<(), String> {
    if settings.mode != "ask" && settings.mode != "silent" {
        return Err(format!("Invalid auto-start mode '{}'", settings.mode));
    }

    CalendarRepository::update_auto_start_settings(
        state.db_manager.pool(),
        settings.enabled,
        &settings.mode,
        settings.grace_minutes,
    )
    .await
    .map_err(|e| e.to_string())
}

/// Called by the frontend right after it actually starts a calendar-triggered recording,
/// so the poller knows which event the active recording belongs to (for auto-stop) and
/// doesn't mistake a later, unrelated manual recording for this event's auto-stop target.
#[tauri::command]
pub async fn calendar_confirm_auto_start(event_id: String) -> Result<(), String> {
    super::poller::set_active_auto_event(Some(event_id));
    Ok(())
}
