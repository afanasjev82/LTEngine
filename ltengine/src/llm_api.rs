use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(thiserror::Error, Debug)]
pub enum ApiError {
    #[error("LLM API request timed out")]
    Timeout,
    #[error("LLM API unreachable: {0}")]
    Unreachable(String),
    #[error("LLM API returned status {status}: {body}")]
    Remote { status: u16, body: String },
    #[error("LLM API response parse error: {0}")]
    Parse(String),
    #[error("LLM API returned no choices")]
    NoChoices,
}

impl ApiError {
    /// Map an API failure to an HTTP status. Transient/retryable (timeout,
    /// unreachable, gateway/unavailable/timeout/too-many-requests) -> 503;
    /// everything else (other remote status, parse, empty) -> 500.
    pub fn http_status(&self) -> u16 {
        match self {
            ApiError::Timeout | ApiError::Unreachable(_) => 503,
            ApiError::Remote { status, .. } if matches!(*status, 502 | 503 | 504 | 429) => 503,
            _ => 500,
        }
    }
}

pub struct LLM {
    url: String,
    api_key: String,
    model: String,
    max_tokens: Option<u32>,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatTemplateKwargs {
    enable_thinking: bool,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    chat_template_kwargs: ChatTemplateKwargs,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct Usage {
    #[serde(default)]
    completion_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

impl LLM {
    pub fn new(url: String, api_key: String, model: String, max_tokens: Option<u32>, timeout_secs: u64) -> Result<Self> {
        // timeout_secs == 0 means "no timeout": reqwest has no timeout by default,
        // so we only apply one when a positive value is configured. Passing
        // Duration::from_secs(0) would instead make every request time out instantly.
        let mut builder = reqwest::Client::builder();
        if timeout_secs > 0 {
            builder = builder.timeout(Duration::from_secs(timeout_secs));
        }
        let client = builder
            .build()
            .with_context(|| "Failed to build HTTP client")?;
        Ok(LLM { url, api_key, model, max_tokens, client })
    }

    pub async fn resolve_model(&mut self) -> Result<String> {
        let endpoint = format!("{}/v1/models", self.url.trim_end_matches('/'));

        let mut req = self.client.get(&endpoint);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let response = req.send().await
            .with_context(|| format!("Failed to fetch models from {}", endpoint))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM API returned status {} when fetching models: {}", status, body);
        }

        let models: ModelsResponse = response.json().await
            .with_context(|| "Failed to parse /v1/models response")?;

        let model_id = models.data.into_iter()
            .next()
            .map(|m| m.id)
            .ok_or_else(|| anyhow::anyhow!("No models available on the server"))?;

        self.model = model_id.clone();
        Ok(model_id)
    }

    pub async fn run_prompt(
        &self,
        system: String,
        user: String,
        max_output_tokens: Option<u32>,
    ) -> Result<String> {
        let endpoint = format!("{}/v1/chat/completions", self.url.trim_end_matches('/'));

        // Dynamic per-request cap when provided, otherwise the static ceiling.
        let max_tokens = max_output_tokens.or(self.max_tokens);

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage { role: "system".to_string(), content: system },
                ChatMessage { role: "user".to_string(), content: user },
            ],
            // Translation pipeline invariants — enforced here, not relied on from
            // the vLLM serve flags or model defaults.
            temperature: 0.0,
            max_tokens,
            chat_template_kwargs: ChatTemplateKwargs { enable_thinking: false },
        };

        let mut req = self.client.post(&endpoint)
            .json(&request);

        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let response = req.send().await.map_err(|e| {
            if e.is_timeout() { ApiError::Timeout } else { ApiError::Unreachable(e.to_string()) }
        })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(ApiError::Remote { status: status.as_u16(), body }.into());
        }

        let completion: ChatCompletionResponse = response.json().await
            .map_err(|e| ApiError::Parse(e.to_string()))?;

        let completion_tokens = completion.usage.and_then(|u| u.completion_tokens);
        let choice = completion.choices.into_iter()
            .next()
            .ok_or(ApiError::NoChoices)?;

        // Truncation safety signal: the cap may have cut a real translation.
        if choice.finish_reason.as_deref() == Some("length") {
            eprintln!(
                "⚠️ LLM stopped at max_tokens cap (max_tokens={:?}, completion_tokens={:?}) — output may be truncated",
                max_tokens, completion_tokens
            );
        }

        Ok(choice.message.content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_enforces_determinism_invariants() {
        let req = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![],
            temperature: 0.0,
            max_tokens: Some(123),
            chat_template_kwargs: ChatTemplateKwargs { enable_thinking: false },
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["temperature"], serde_json::json!(0.0));
        assert_eq!(v["chat_template_kwargs"]["enable_thinking"], serde_json::json!(false));
        assert_eq!(v["max_tokens"], serde_json::json!(123));
    }

    #[test]
    fn max_tokens_omitted_when_none() {
        let req = ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![],
            temperature: 0.0,
            max_tokens: None,
            chat_template_kwargs: ChatTemplateKwargs { enable_thinking: false },
        };
        let v = serde_json::to_value(&req).unwrap();
        assert!(v.get("max_tokens").is_none());
    }

    #[test]
    fn api_error_status_maps_transient_to_503() {
        assert_eq!(ApiError::Timeout.http_status(), 503);
        assert_eq!(ApiError::Unreachable("x".into()).http_status(), 503);
        assert_eq!(ApiError::Remote { status: 502, body: String::new() }.http_status(), 503);
        assert_eq!(ApiError::Remote { status: 503, body: String::new() }.http_status(), 503);
        assert_eq!(ApiError::Remote { status: 504, body: String::new() }.http_status(), 503);
        assert_eq!(ApiError::Remote { status: 429, body: String::new() }.http_status(), 503);
    }

    #[test]
    fn api_error_status_maps_permanent_to_500() {
        assert_eq!(ApiError::Remote { status: 400, body: String::new() }.http_status(), 500);
        assert_eq!(ApiError::Remote { status: 401, body: String::new() }.http_status(), 500);
        assert_eq!(ApiError::Remote { status: 500, body: String::new() }.http_status(), 500);
        assert_eq!(ApiError::Parse("x".into()).http_status(), 500);
        assert_eq!(ApiError::NoChoices.http_status(), 500);
    }
}
