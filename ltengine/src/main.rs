use actix_web::{
    get, post, web, App, HttpRequest, HttpResponse, 
    HttpServer, Responder, http::header, FromRequest
};
use actix_multipart::form::{MultipartForm, text::Text as MPText};
use actix_web_prom::PrometheusMetricsBuilder;
use actix_web_static_files::ResourceFiles;
use std::sync::Arc;
use clap::Parser;
use serde::{Deserialize, Serialize};

mod error_response;
mod languages;
#[cfg(feature = "local")]
mod models;
#[cfg(feature = "local")]
mod llm;
#[cfg(feature = "api")]
#[path = "llm_api.rs"]
mod llm;
mod banner;
mod prompt;
mod startup_info;
#[cfg(feature = "api")]
mod token_budget;

use languages::{detect_lang, get_language_from_code, LANGUAGES};
use error_response::ErrorResponse;
#[cfg(feature = "local")]
use models::{MODELS, load_model};
use banner::print_banner;
use prompt::PromptBuilder;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    /// Hostname to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind to
    #[arg(short, long, default_value_t = 5050)]
    port: u16,

    /// Character limit for translation requests
    #[arg(long, default_value_t = 5000)]
    char_limit: usize,

    /// Model to use (local inference)
    #[cfg(feature = "local")]
    #[arg(short='m', long, value_parser = MODELS.keys().collect::<Vec<_>>(), default_value = "gemma3-4b")]
    model: String,

    /// Path to .gguf model file (local inference)
    #[cfg(feature = "local")]
    #[arg(long, default_value = "")]
    model_file: String,

    /// Set an API key
    #[arg(long, default_value = "")]
    api_key: String,  

    /// Use CPU only (local inference)
    #[cfg(feature = "local")]
    #[arg(long)]
    cpu: bool,

    /// OpenAI-compatible API URL (e.g. http://localhost:8080)
    #[cfg(feature = "api")]
    #[arg(long)]
    llm_url: String,

    /// API key for the LLM API
    #[cfg(feature = "api")]
    #[arg(long, default_value = "")]
    llm_api_key: String,

    /// Model name for the LLM API (auto-resolved if not provided)
    #[cfg(feature = "api")]
    #[arg(long, default_value = "")]
    llm_model: String,

    /// Maximum output tokens per LLM request (0 = no limit)
    #[cfg(feature = "api")]
    #[arg(long, default_value_t = 0)]
    llm_max_tokens: u32,

    /// Dynamic cap: conservative characters-per-token divisor
    #[cfg(feature = "api")]
    #[arg(long, default_value_t = 2.0)]
    llm_chars_per_token: f32,

    /// Dynamic cap: output safety multiple (0 disables the dynamic cap)
    #[cfg(feature = "api")]
    #[arg(long, default_value_t = 3.0)]
    llm_max_tokens_mult: f32,

    /// Dynamic cap: minimum output tokens for tiny inputs
    #[cfg(feature = "api")]
    #[arg(long, default_value_t = 64)]
    llm_max_tokens_floor: u32,

    /// HTTP timeout in seconds for LLM API requests (0 = no timeout)
    #[cfg(feature = "api")]
    #[arg(long, default_value_t = 120)]
    llm_timeout: u64,

    /// Enable verbose logging
    #[arg(short = 'v', long)]
    verbose: bool
}

#[derive(Debug, Deserialize, Serialize)]
struct TranslateRequest {
    q: Option<String>,
    source: Option<String>,
    target: Option<String>,
    format: Option<String>,
    api_key: Option<String>,
    alternatives: Option<u32>
}

#[derive(MultipartForm)]
struct MPTranslateRequest {
    q: Option<MPText<String>>,
    source: Option<MPText<String>>,
    target: Option<MPText<String>>,
    format: Option<MPText<String>>,
    api_key: Option<MPText<String>>,
    alternatives: Option<MPText<u32>>
}
impl MPTranslateRequest {
    fn into_translate_request(self) -> TranslateRequest {
        TranslateRequest {
            q: self.q.map(|v| v.into_inner()),
            source: self.source.map(|v| v.into_inner()),
            target: self.target.map(|v| v.into_inner()),
            format: self.format.map(|v| v.into_inner()),
            api_key: self.api_key.map(|v| v.into_inner()),
            alternatives: self.alternatives.map(|v| v.into_inner()),
        }
    }
}

async fn parse_payload(req: HttpRequest, payload: web::Payload) -> Result<TranslateRequest, ErrorResponse>{
    let content_type = req.headers().get(header::CONTENT_TYPE).map(|h| h.to_str().unwrap_or("")).unwrap_or("");
    let body: TranslateRequest;

    if content_type.starts_with("application/json") {
        let json = actix_web::web::Json::<TranslateRequest>::from_request(&req, &mut payload.into_inner()).await?;
        body = json.into_inner()
    } else if content_type.starts_with("application/x-www-form-urlencoded") {
        let form = actix_web::web::Form::<TranslateRequest>::from_request(&req, &mut payload.into_inner()).await?;
        body = form.into_inner()
    } else if content_type.starts_with("multipart/form-data") {
        let form = MultipartForm::<MPTranslateRequest>::from_request(&req, &mut payload.into_inner()).await?;
        body = form.into_inner().into_translate_request();
    } else {
        return Err(ErrorResponse{ error: "Unsupported content-type".to_string(), status: 400 });
    }

    return Ok(body);
}

fn check_params(body: &TranslateRequest, args: &Args, required_params: &[(&str, &Option<String>)]) -> Result<bool, ErrorResponse> {
    // Validate required params
    for (key, value) in required_params {
        if value.as_ref().is_none_or(|v| v.trim().is_empty()) {
            return Err(ErrorResponse {
                error: format!("Invalid request: missing {} parameter", key),
                status: 400,
            });
        }
    }
    
    // Check key
    if !args.api_key.is_empty() && body.api_key.as_ref().is_none_or(|key| *key != args.api_key) {
        return Err(ErrorResponse {
            error: format!("Invalid API key"),
            status: 403,
        });
    }

    let q = body.q.as_ref().unwrap();
    if q.len() > args.char_limit {
        return Err(ErrorResponse {
            error: format!("Invalid request: request ({}) exceeds text limit ({})", q.len(), args.char_limit),
            status: 400,
        });
    }

    Ok(true)
}

fn improve_formatting(q: &String, translation: &String) -> String {
    let t = translation.trim().to_string();

    if q.len() == 0 {
        return String::new();
    }

    if t.len() == 0 {
        return q.clone();
    }

    let q_last_char = q.chars().rev().next().unwrap();
    let translation_last_char = t.chars().rev().next().unwrap();
    let mut result = t.clone();

    const PUNCTUATION_CHARS: [char; 6] = ['!', '?', '.', ',', ';', '。'];
    if PUNCTUATION_CHARS.contains(&q_last_char){
        if q_last_char != translation_last_char{
            if PUNCTUATION_CHARS.contains(&translation_last_char){
                result.pop();
            }

            result.push(q_last_char);
        }
    }else if PUNCTUATION_CHARS.contains(&translation_last_char) {
        result.pop();   
    }

    if q.chars().all(|c| c.is_lowercase()) {
        result = result.to_lowercase();
    }

    if q.chars().all(|c| c.is_uppercase()) {
        result = result.to_uppercase();
    }

    if let (Some(q0), Some(r0)) = (q.chars().next(), result.chars().next()) {
        if q0.is_lowercase() && r0.is_uppercase() {
            result.replace_range(0..r0.len_utf8(), &r0.to_lowercase().to_string());
        }else if q0.is_uppercase() && r0.is_lowercase() {
            result.replace_range(0..r0.len_utf8(), &r0.to_uppercase().to_string());
        }
    }

    result.trim().to_string()
}

#[post("/detect")]
async fn detect(req: HttpRequest, payload: web::Payload, args: web::Data<Arc<Args>>) -> Result<HttpResponse, ErrorResponse> {
    let body = parse_payload(req, payload).await?;
    check_params(&body, &args, &[
        ("q", &body.q)
    ])?;

    let q = body.q.unwrap();
    let d = detect_lang(&q);

    Ok(HttpResponse::Ok().json(serde_json::json!([{
        "language": d.language.code,
        "confidence": d.confidence
    }])))
}

fn check_format(format: &str) -> Result<bool, ErrorResponse> {
    match format {
        "text" | "html" => Ok(true),
        _ => Err(ErrorResponse {
            error: "Invalid format. Supported formats: text, html".to_string(),
            status: 400,
        })
    }
}

/// Per-request output-token cap. The dynamic cap is fully opt-in: it applies
/// only when all three of `--llm-max-tokens`, `--llm-chars-per-token`, and
/// `--llm-max-tokens-mult` are > 0. With `--llm-max-tokens 0` nothing is capped
/// and `max_tokens` is omitted from the request.
#[cfg(feature = "api")]
fn output_cap(input_chars: usize, args: &Args) -> Option<u32> {
    if args.llm_max_tokens == 0 {
        return None; // no ceiling => no cap at all
    }
    let cfg = token_budget::TokenBudgetConfig {
        chars_per_token: args.llm_chars_per_token,
        output_mult: args.llm_max_tokens_mult,
        floor: args.llm_max_tokens_floor,
        ceiling: Some(args.llm_max_tokens), // always Some here (> 0)
    };
    // Returns None unless chars_per_token > 0 && output_mult > 0.
    token_budget::dynamic_output_cap(input_chars, &cfg)
}

#[post("/translate")]
async fn translate(req: HttpRequest, payload: web::Payload, args: web::Data<Arc<Args>>, llm: actix_web::web::Data<Arc<llm::LLM>>) -> Result<HttpResponse, ErrorResponse> {
    let body = parse_payload(req, payload).await?;
    check_params(&body, &args, &[
        ("q", &body.q),
        ("source", &body.source),
        ("target", &body.target),
    ])?;

    let q = body.q.unwrap();
    let source = body.source.unwrap();
    let target = body.target.unwrap();
    let format = body.format.unwrap_or("text".to_string());
    check_format(&format)?;
    
    let mut pb = PromptBuilder::new();
    pb.set_format(&format);

    // TODO: add HTML support
    
    if source == "auto"{
        pb.set_source_language("auto");
    }else{
        let src_lang = get_language_from_code(&source).ok_or_else(|| ErrorResponse {
            error: format!("{} is not supported", source),
            status: 400,
        })?;
        pb.set_source_language(src_lang.name);
    }

    let tgt_lang = get_language_from_code(&target).ok_or_else(|| ErrorResponse {
        error: format!("{} is not supported", target),
        status: 400,
    })?;
    pb.set_target_language(tgt_lang.name);

    let llm = llm.get_ref();
    let prompt = pb.build(&q);
    
    let translated_text = if source != target {
        #[cfg(feature = "api")]
        {
            let cap = output_cap(q.chars().count(), &args);
            llm.run_prompt(prompt.system, prompt.user, cap).await.unwrap_or(q.clone())
        }
        #[cfg(not(feature = "api"))]
        { llm.run_prompt(prompt.system, prompt.user).unwrap_or(q.clone()) }
    }else{
        q.clone()
    };
    
    let mut response = serde_json::json!({"translatedText": improve_formatting(&q, &translated_text)});

    // TODO: we just add this for compatibility for now
    // we should allow multiple alternatives to be generated
    if body.alternatives.is_some_and(|v| v > 0) {
        response["alternatives"] = serde_json::json!([]);
    }

    if source == "auto" {
        let d = detect_lang(&q);
        response["detectedLanguage"] = serde_json::json!({
            "language": d.language.code,
            "confidence": d.confidence
        });
    }

    Ok(HttpResponse::Ok().json(response))
}

#[post("/translate_file")]
async fn translate_file() -> Result<HttpResponse, ErrorResponse> {
    Err(ErrorResponse{
        error: "Not implemented".to_string(),
        status: 501
    })
}

#[post("/suggest")]
async fn suggest() -> Result<HttpResponse, ErrorResponse> {
    Err(ErrorResponse{
        error: "Not implemented".to_string(),
        status: 501
    })
}

#[get("/languages")]
async fn get_languages() -> impl Responder {
    HttpResponse::Ok().json(&*LANGUAGES)
}

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().finish()
}

#[get("/frontend/settings")]
async fn get_frontend_settings(args: web::Data<Arc<Args>>) -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "apiKeys": false,
        "charLimit": args.char_limit,
        "filesTranslation": false,
        "frontendTimeout": 1000,
        "keyRequired": false,
        "language": {
            "source": {
                "code": "auto",
                "name": "Auto Detect"
            },
            "target": {
                "code": "en",
                "name": "English"
            }
        },
        "suggestions": false,
        "supportedFilesFormat": []
    }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Arc::new(Args::parse());

    let host = args.host.clone();
    let port = args.port;

    #[cfg(feature = "local")]
    let (llm, facts) = {
        let model_path = load_model(&args.model, &args.model_file).unwrap_or_else(|err| {
            eprintln!("Failed to load model: {}", err);
            std::process::exit(1);
        });
        let facts = startup_info::RuntimeFacts { model_path: model_path.clone() };
        let llm = Arc::new(llm::LLM::new(model_path, args.cpu, args.verbose).unwrap_or_else(|err| {
            eprintln!("Failed to initialize LLM: {}", err);
            std::process::exit(1);
        }));
        (llm, facts)
    };

    #[cfg(feature = "api")]
    let (llm, facts) = {
        let max_tokens = if args.llm_max_tokens > 0 { Some(args.llm_max_tokens) } else { None };
        let timeout_secs = if args.llm_timeout > 0 { args.llm_timeout } else { 0 };
        let mut llm_instance = llm::LLM::new(args.llm_url.clone(), args.llm_api_key.clone(), args.llm_model.clone(), max_tokens, timeout_secs).unwrap_or_else(|err| {
            eprintln!("Failed to initialize LLM API client: {}", err);
            std::process::exit(1);
        });
        let (resolved_model, model_resolved) = if args.llm_model.is_empty() {
            match llm_instance.resolve_model().await {
                Ok(model_name) => (model_name, true),
                Err(err) => {
                    eprintln!("Failed to resolve model from server: {}", err);
                    std::process::exit(1);
                }
            }
        } else {
            (args.llm_model.clone(), false)
        };
        let facts = startup_info::RuntimeFacts { resolved_model, model_resolved };
        (Arc::new(llm_instance), facts)
    };

    print_banner();
    startup_info::print(&args, &facts);

    let prometheus = PrometheusMetricsBuilder::new("ltengine")
        .endpoint("/metrics")
        .build()
        .unwrap();

    let server = HttpServer::new(move || {
        let generated = generate();

        App::new()
            .wrap(prometheus.clone())
            // .service(index)
            .app_data(web::Data::new(llm.clone()))
            .app_data(web::Data::new(args.clone()))
            .service(health)
            .service(get_languages)
            .service(get_frontend_settings)
            .service(translate)
            .service(translate_file)
            .service(detect)
            .service(suggest)
            .service(ResourceFiles::new("/", generated))
    })
    .bind((host.clone(), port))?
    .run();

    println!("Listening on http://{}:{}", host, port);

    return server.await;
}

#[cfg(all(test, feature = "api"))]
mod tests {
    use super::*;
    use clap::Parser;

    /// Build an `Args` from CLI tokens; `--llm-url` is required.
    fn args_with(extra: &[&str]) -> Args {
        let mut argv = vec!["ltengine", "--llm-url", "http://localhost:8080"];
        argv.extend_from_slice(extra);
        Args::parse_from(argv)
    }

    #[test]
    fn output_cap_none_without_ceiling() {
        // --llm-max-tokens 0 => no cap, even with positive estimator knobs.
        let args = args_with(&[
            "--llm-max-tokens", "0",
            "--llm-chars-per-token", "2",
            "--llm-max-tokens-mult", "3",
        ]);
        assert_eq!(output_cap(1000, &args), None);
    }

    #[test]
    fn output_cap_none_when_mult_zero() {
        // Ceiling set but mult 0 => dynamic disabled; handler falls back to static ceiling.
        let args = args_with(&[
            "--llm-max-tokens", "256",
            "--llm-chars-per-token", "2",
            "--llm-max-tokens-mult", "0",
        ]);
        assert_eq!(output_cap(1000, &args), None);
    }

    #[test]
    fn output_cap_clamped_to_ceiling_when_all_positive() {
        let args = args_with(&[
            "--llm-max-tokens", "256",
            "--llm-chars-per-token", "2",
            "--llm-max-tokens-mult", "3",
        ]);
        // 1000/2 = 500, *3 = 1500, clamped to ceiling 256.
        assert_eq!(output_cap(1000, &args), Some(256));
    }

    #[test]
    fn output_cap_uses_floor_for_tiny_input() {
        let args = args_with(&[
            "--llm-max-tokens", "4096",
            "--llm-chars-per-token", "2",
            "--llm-max-tokens-mult", "3",
            "--llm-max-tokens-floor", "64",
        ]);
        // 10/2 = 5, *3 = 15, raised to floor 64.
        assert_eq!(output_cap(10, &args), Some(64));
    }
}