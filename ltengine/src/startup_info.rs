//! Human-readable startup banner: build and runtime info rendered as compact,
//! aligned key/value sections.
//!
//! Build-time values (`LTE_*`) are injected by `build.rs`. The module only
//! formats already-validated data, so it introduces no fallible paths.

use crate::Args;

/// Facts that `main` has already computed, passed in so this module stays
/// decoupled from clap/LLM internals.
pub struct RuntimeFacts {
    /// Resolved path to the local `.gguf` model file.
    pub model_path: std::path::PathBuf,
}

/// Print the full startup summary (call after `print_banner`).
pub fn print(args: &Args, facts: &RuntimeFacts) {
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    let features = feature_list();
    let features = if features.is_empty() {
        "cpu".to_string()
    } else {
        features.join(", ")
    };
    let title = format!("LTEngine {} · {features} · {profile}", env!("CARGO_PKG_VERSION"));
    println!("{title}");
    println!("{}", "─".repeat(title.chars().count()));

    println!();
    print!("{}", render_section("Build", &build_rows()));
    println!();
    print!("{}", render_section("Runtime", &runtime_rows(args, facts)));
    println!();
}

/// Render a titled section: the title on its own line, then each row indented
/// two spaces with the label column padded so all values line up. The returned
/// string ends in a newline.
fn render_section(title: &str, rows: &[(&str, String)]) -> String {
    let width = rows.iter().map(|(k, _)| k.chars().count()).max().unwrap_or(0);
    let mut out = format!("{title}\n");
    for (k, v) in rows {
        out.push_str(&format!("  {k:<width$}  {v}\n"));
    }
    out
}

/// Compile-time enabled GPU Cargo features, in display order.
fn feature_list() -> Vec<&'static str> {
    let mut v = Vec::new();
    if cfg!(feature = "cuda") {
        v.push("cuda");
    }
    if cfg!(feature = "metal") {
        v.push("metal");
    }
    if cfg!(feature = "vulkan") {
        v.push("vulkan");
    }
    v
}

fn build_rows() -> Vec<(&'static str, String)> {
    vec![
        ("git", format!("{} ({})", env!("LTE_GIT_SHA"), env!("LTE_GIT_BRANCH"))),
        ("built", env!("LTE_BUILD_TIME").to_string()),
        // `LTE_RUSTC` is the raw `rustc --version` line; drop the leading word
        // so it doesn't read "rustc  rustc 1.96.0".
        ("rustc", env!("LTE_RUSTC").trim_start_matches("rustc ").to_string()),
    ]
}

/// `"set"`/`"none"` — never echoes secret values.
fn secret_state(s: &str) -> &'static str {
    if s.trim().is_empty() { "none" } else { "set" }
}

fn runtime_rows(args: &Args, facts: &RuntimeFacts) -> Vec<(&'static str, String)> {
    vec![
        ("mode", "local".to_string()),
        ("model", args.model.clone()),
        ("model path", facts.model_path.display().to_string()),
        ("device", if args.cpu { "cpu" } else { "gpu" }.to_string()),
        ("server auth", secret_state(&args.api_key).to_string()),
        ("bind", format!("{}:{}", args.host, args.port)),
        ("char limit", args.char_limit.to_string()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_state_masks_values() {
        assert_eq!(secret_state(""), "none");
        assert_eq!(secret_state("   "), "none");
        assert_eq!(secret_state("hunter2"), "set");
    }

    #[test]
    fn render_section_aligns_values() {
        let rows = vec![("a", "1".to_string()), ("long", "2".to_string())];
        // Labels padded to the widest ("long" = 4), two-space indent + gap.
        assert_eq!(render_section("S", &rows), "S\n  a     1\n  long  2\n");
    }

    #[test]
    fn feature_list_reflects_enabled_gpu_features() {
        let features = feature_list();
        assert_eq!(cfg!(feature = "cuda"), features.contains(&"cuda"));
        assert_eq!(cfg!(feature = "metal"), features.contains(&"metal"));
        assert_eq!(cfg!(feature = "vulkan"), features.contains(&"vulkan"));
    }
}
