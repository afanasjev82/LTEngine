use static_files::resource_dir;
use std::process::Command;

fn main() -> std::io::Result<()> {
    emit_build_metadata();
    resource_dir("./resources").build()
}

/// Emit build-time metadata as `LTE_*` compile-time env vars consumed by
/// `startup_info.rs` via `env!`. Every lookup is best-effort: a missing `git`,
/// absent `.git`, or any command failure degrades to `"unknown"` and never
/// fails the build.
fn emit_build_metadata() {
    // Git short SHA, with a `-dirty` suffix when the working tree is not clean.
    let git_sha = match git_output(&["rev-parse", "--short", "HEAD"]) {
        Some(sha) => {
            let dirty = git_output(&["status", "--porcelain"])
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false);
            if dirty { format!("{sha}-dirty") } else { sha }
        }
        None => "unknown".to_string(),
    };
    println!("cargo:rustc-env=LTE_GIT_SHA={git_sha}");

    let branch = git_output(&["rev-parse", "--abbrev-ref", "HEAD"])
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=LTE_GIT_BRANCH={branch}");

    // Wall-clock UTC build time (RFC3339) via the `time` build-dependency.
    let build_time = time::OffsetDateTime::now_utc()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=LTE_BUILD_TIME={build_time}");

    // rustc version (cargo provides the rustc path in the RUSTC env var).
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let rustc_ver = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=LTE_RUSTC={rustc_ver}");

    // Refresh git-derived values when the checked-out commit or tree changes.
    if let Some(head) = git_output(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head}");
    }
    if let Some(index) = git_output(&["rev-parse", "--git-path", "index"]) {
        println!("cargo:rerun-if-changed={index}");
    }
}

/// Run `git <args>` and return trimmed stdout, or `None` on any failure or
/// empty output.
fn git_output(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}
