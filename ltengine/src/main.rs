use actix_web::{get, App, HttpResponse, HttpServer, Responder};
use clap::Parser;

mod languages;
use languages::LANGUAGES;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Hostname to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to bind to
    #[arg(short, long, default_value_t = 5000)]
    port: u16,
}

#[get("/")]
async fn index() -> impl Responder {
    HttpResponse::Ok().body("LT Engine is running")
}

#[get("/languages")]
async fn get_languages() -> impl Responder {
    HttpResponse::Ok().json(&*LANGUAGES)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

    HttpServer::new(|| {
        App::new()
            .service(index)
            .service(get_languages)
    })
    .bind((args.host, args.port))?
    .run()
    .await
}