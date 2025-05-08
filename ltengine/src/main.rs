use actix_web::{
    get, post, web, App, HttpRequest, HttpResponse, 
    HttpServer, Responder, http::header, FromRequest
};
use actix_multipart::form::{MultipartForm, text::Text as MPText};
use actix_web_static_files::ResourceFiles;
use std::sync::Arc;
use clap::Parser;
use serde::{Deserialize, Serialize};

mod error_response;
mod languages;
mod models;
mod llm;
mod banner;

use languages::LANGUAGES;
use error_response::ErrorResponse;
use models::{MODELS, load_model};
use banner::print_banner;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    /// Hostname to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind to
    #[arg(short, long, default_value_t = 5000)]
    port: u16,

    /// Character limit for translation requests
    #[arg(long, default_value_t = 5000)]
    char_limit: usize,

    /// Model to use
    #[arg(short='m', long, value_parser = MODELS.keys().collect::<Vec<_>>(), default_value = "gemma3-1b")]
    model: String,

    /// Path to .gguf model file
    #[arg(long, default_value = "")]
    model_file: String,

    /// Set an API key
    #[arg(long, default_value = "")]
    api_key: String,  

    /// Use CPU only
    #[arg(long)]
    cpu: bool,

    /// Enable verbose logging
    #[arg(short = 'v', long)]
    verbose: bool,
}

#[get("/test")]
async fn test_func(data: actix_web::web::Data<Arc<llm::LLM>>) -> impl Responder{
    let llm = data.get_ref();

    let system = "You are an expert linguist, specializing in translation. You are able to capture the nuances of the languages you translate. You pay attention to masculine/feminine/plural and proper use of articles and grammar. You always provide natural sounding translations that fully preserve the meaning of the original text. You never provide explanations for your work. You always answer with the translated text and nothing else.".to_string();
    let user: String = "Translate the text below from English to Italian.\n\nEnglish: Ovechkin’s first assist of the night was on the game-winning goal by rookie Nicklas Backstrom;\n\nItalian:\n".to_string();

    let result = llm.run_prompt(system, user).unwrap_or_else(|err| {
        eprintln!("Failed prompt: {}", err);
        std::process::exit(1);
    });

    HttpResponse::Ok().body(result)
}

#[derive(Debug, Deserialize, Serialize)]
struct TranslateRequest {
    q: Option<String>,
    source: Option<String>,
    target: Option<String>,
    api_key: Option<String>,
    alternatives: Option<u32>
}

#[derive(MultipartForm)]
struct MPTranslateRequest {
    q: Option<MPText<String>>,
    source: Option<MPText<String>>,
    target: Option<MPText<String>>,
    api_key: Option<MPText<String>>,
    alternatives: Option<MPText<u32>>
}
impl MPTranslateRequest {
    fn into_translate_request(self) -> TranslateRequest {
        TranslateRequest {
            q: self.q.map(|v| v.into_inner()),
            source: self.source.map(|v| v.into_inner()),
            target: self.target.map(|v| v.into_inner()),
            api_key: self.api_key.map(|v| v.into_inner()),
            alternatives: self.alternatives.map(|v| v.into_inner()),
        }
    }
}

#[post("/translate")]
async fn translate(req: HttpRequest, payload: web::Payload, args: web::Data<Arc<Args>>, llm: actix_web::web::Data<Arc<llm::LLM>>) -> Result<HttpResponse, ErrorResponse> {
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

    // Check key
    if !args.api_key.is_empty() && body.api_key.is_none_or(|key| key != args.api_key){
        return Err(ErrorResponse{
            error: format!("Invalid API key"),
            status: 403
        });
    }

    // Validate required params
    for (key, value) in [
        ("q", &body.q),
        ("source", &body.source),
        ("target", &body.target),
    ] {
        if value.as_ref().is_none_or(|v| v.trim().is_empty()) {
            return Err(ErrorResponse {
                error: format!("Invalid request: missing {} parameter", key),
                status: 400,
            });
        }
    }

    // Check limits
    let q = body.q.unwrap();
    let source = body.source.unwrap();
    let target = body.target.unwrap();

    if q.len() > args.char_limit {
        return Err(ErrorResponse{
            error: format!("Invalid request: request ({}) exceeds text limit ({})", q.len(), args.char_limit),
            status: 400
        });
    }

    let llm = llm.get_ref();
    let system = "You are an expert linguist, specializing in translation. You are able to capture the nuances of the languages you translate. You pay attention to masculine/feminine/plural and proper use of articles and grammar. You always provide natural sounding translations that fully preserve the meaning of the original text. You never provide explanations for your work. You always answer with the translated text and nothing else.".to_string();
    let user: String = format!("Translate the text below from English to Italian.\n\nEnglish: {}\n\nItalian:\n", q).to_string();

    let translated_text = llm.run_prompt(system, user).unwrap_or(q);
    let mut response = serde_json::json!({"translatedText": translated_text});

    // TODO: we just add this for compatibility for now
    // we should allow multiple alternatives to be generated
    if body.alternatives.is_some_and(|v| v > 0) {
        response["alternatives"] = serde_json::json!([]);
    }
    Ok(HttpResponse::Ok().json(response))
}

#[get("/languages")]
async fn get_languages() -> impl Responder {
    HttpResponse::Ok().json(&*LANGUAGES)
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

    let model_path = load_model(&args.model, &args.model_file).unwrap_or_else(|err| {
        eprintln!("Failed to load model: {}", err);
        std::process::exit(1);
    });
    
    println!("Loading model: {}", model_path.display());

    let llm = Arc::new(llm::LLM::new(model_path, args.cpu, args.verbose).unwrap_or_else(|err| {
        eprintln!("Failed to initialize LLM: {}", err);
        std::process::exit(1);
    }));

    print_banner();

    let server = HttpServer::new(move || {
        let generated = generate();

        App::new()
            // .service(index)
            .app_data(web::Data::new(llm.clone()))
            .app_data(web::Data::new(args.clone()))
            .service(get_languages)
            .service(get_frontend_settings)
            .service(test_func)
            .service(translate)
            .service(ResourceFiles::new("/", generated))
    })
    .bind((host.clone(), port))?
    .run();

    println!("Running on: http://{}:{}", host, port);

    return server.await;
}