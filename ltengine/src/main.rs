use actix_web::{get, App, HttpResponse, HttpServer, Responder};
use actix_web_static_files::ResourceFiles;
use clap::Parser;

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
    char_limit: u32,

    /// Model to use
    #[arg(long, value_parser = MODELS.keys().collect::<Vec<_>>(), default_value = "gemma3-1b")]
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
    let model_path = match load_model(args.model, args.model_file) {
        Ok(path) => path,
        Err(err) => {
            eprintln!("Failed to load model: {}", err);
            std::process::exit(1);
        }
    };
    
    println!("Loading model: {}", model_path.display());

    let llm = match llm::LLM::new(model_path, args.cpu) {
        Ok(llm) => llm,
        Err(err) => {
            eprintln!("Failed to initialize LLM: {}", err);
            std::process::exit(1);
        }
    };

    print_banner();

    let server = HttpServer::new(|| {
        let generated = generate();

        App::new()
            // .service(index)
            .service(get_languages)
            .service(get_frontend_settings)
            .service(ResourceFiles::new("/", generated))
    })
    .bind((args.host.clone(), args.port))?
    .run();

    println!("Running on: http://{}:{}", args.host, args.port);

    let prompt: String = "Translate this sentence from English to Italian, output just the translation, nothing else: the world is on fire.".to_string();
    
    match llm.run_prompt(prompt){
        Ok(result) => println!("{}", result),
        Err(err) => {
            eprintln!("Failed prompt: {}", err);
            std::process::exit(1);
        }
    }

    return server.await;
}