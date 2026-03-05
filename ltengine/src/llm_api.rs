use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};

pub struct LLM {
    url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
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
    pub fn new(url: String, api_key: String, model: String) -> Result<Self> {
        let client = reqwest::Client::new();
        Ok(LLM { url, api_key, model, client })
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

    pub async fn run_prompt(&self, system: String, user: String) -> Result<String> {
        let endpoint = format!("{}/v1/chat/completions", self.url.trim_end_matches('/'));

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage { role: "system".to_string(), content: system },
                ChatMessage { role: "user".to_string(), content: user },
            ],
            temperature: 0.0,
        };

        let mut req = self.client.post(&endpoint)
            .json(&request);

        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let response = req.send().await
            .with_context(|| format!("Failed to send request to {}", endpoint))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("LLM API returned status {}: {}", status, body);
        }

        let completion: ChatCompletionResponse = response.json().await
            .with_context(|| "Failed to parse LLM API response")?;

        completion.choices.into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("LLM API returned no choices"))
    }
}
