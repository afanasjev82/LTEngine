use actix_web::{get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder, Error, FromRequest, error};
use actix_multipart::form::{MultipartForm, text::Text as MPText};
use actix_web_static_files::ResourceFiles;
use std::sync::Arc;
use clap::Parser;
use serde::Deserialize;

mod languages;
mod models;
mod llm;
mod banner;

use languages::LANGUAGES;
use models::{MODELS, load_model};
use banner::print_banner;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

#[derive(Parser, Debug)]
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

    let prompt: String = "Translate this sentence from English to Italian, output just the translation, nothing else: the world is on fire.".to_string();
    let result = llm.run_prompt(prompt).unwrap_or_else(|err| {
        eprintln!("Failed prompt: {}", err);
        std::process::exit(1);
    });

    HttpResponse::Ok().body(result)
}

#[derive(Debug, Deserialize)]
struct TranslateRequest {
    q: String,
    source: String,
    target: String,
    api_key: Option<String>,
}

#[derive(MultipartForm)]
struct MPTranslateRequest {
    q: MPText<String>,
    source: MPText<String>,
    target: MPText<String>,
    api_key: Option<MPText<String>>,
}
impl MPTranslateRequest {
    fn into_translate_request(self) -> TranslateRequest {
        TranslateRequest {
            q: self.q.into_inner(),
            source: self.source.into_inner(),
            target: self.target.into_inner(),
            api_key: self.api_key.map(|key| key.into_inner()),
        }
    }
}

#[post("/translate")]
async fn translate(req: HttpRequest, mut payload: web::Payload) -> Result<HttpResponse, Error> {
    let content_type = req.headers().get("content-type").map(|h| h.to_str().unwrap_or("")).unwrap_or("");

    if content_type.starts_with("application/json") {
        let json = actix_web::web::Json::<TranslateRequest>::from_request(&req, &mut payload.into_inner()).await?;
        println!("JSON: {:?}", json);
    } else if content_type.starts_with("application/x-www-form-urlencoded") {
        let form = actix_web::web::Form::<TranslateRequest>::from_request(&req, &mut payload.into_inner()).await?;
        println!("Form: {:?}", form);
    } else if content_type.starts_with("multipart/form-data") {
        // let multipart = Multipart::from_request
    } else {
        return Ok(HttpResponse::BadRequest().body("Unsupported content-type"));
    }

    let args = Args::parse();

    // println!("{}", req.q);

    // if req.q.len() > args.char_limit {
    //     return HttpResponse::BadRequest().json(serde_json::json!({
    //         "error": format!("Invalid request: request ({}) exceeds text limit ({})", req.q.len(), args.char_limit)
    //     }));
    // }

    Ok(HttpResponse::Ok().json(serde_json::json!({})))
}

#[get("/languages")]
async fn get_languages() -> impl Responder {
    HttpResponse::Ok().json(&*LANGUAGES)
}

#[get("/frontend/settings")]
async fn get_frontend_settings() -> impl Responder {
    let args = Args::parse();

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
    let args = Args::parse();
    let model_path = load_model(args.model, args.model_file).unwrap_or_else(|err| {
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

        let form_error = |err: error::UrlencodedError, _req: &HttpRequest| {
            let msg = err.to_string();
            error::InternalError::from_response(
                err,
                HttpResponse::BadRequest().json(serde_json::json!({"error": msg})),
            )
            .into()
        };
        let json_error = |err: error::JsonPayloadError, _req: &HttpRequest| {
            let msg = err.to_string();
            error::InternalError::from_response(
                err,
                HttpResponse::BadRequest().json(serde_json::json!({"error": msg})),
            )
            .into()
        };

        App::new()
            // .service(index)
            .app_data(web::Data::new(llm.clone()))
            .app_data(web::FormConfig::default().error_handler(form_error))
            .app_data(web::JsonConfig::default().error_handler(json_error))
            
            .service(get_languages)
            .service(get_frontend_settings)
            .service(test_func)
            .service(translate)
            .service(ResourceFiles::new("/", generated))
    })
    .bind((args.host.clone(), args.port))?
    .run();

    println!("Running on: http://{}:{}", args.host, args.port);

    return server.await;
}