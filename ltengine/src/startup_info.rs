//! Human-readable startup banner: build, runtime and token-budget info rendered
//! as compact, aligned key/value sections.
//!
//! Build-time values (`LTE_*`) are injected by `build.rs`. The module only
//! formats already-validated data, so it introduces no fallible paths.

use crate::Args;
#[cfg(feature = "api")]
use crate::token_budget::TokenBudgetConfig;

/// Mode-specific facts that `main` has already computed, passed in so this
/// module stays decoupled from clap/LLM internals.
pub struct RuntimeFacts {
    /// Final model name (explicit flag or auto-resolved from the server).
    #[cfg(feature = "api")]
    pub resolved_model: String,
    /// Whether `resolved_model` was auto-resolved (vs. given via `--llm-model`).
    #[cfg(feature = "api")]
    pub model_resolved: bool,
    /// Resolved path to the local `.gguf` model file.
    #[cfg(feature = "local")]
    pub model_path: std::path::PathBuf,
}

/// Print the full startup summary (call after `print_banner`).
pub fn print(args: &Args, facts: &RuntimeFacts) {
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    let title = format!(
        "LTEngine {} · {} · {profile}",
        env!("CARGO_PKG_VERSION"),
        feature_list().join(", "),
    );
    println!("{title}");
    println!("{}", "─".repeat(title.chars().count()));

    println!();
    print!("{}", render_section("Build", &build_rows()));
    println!();
    print!("{}", render_section("Runtime", &runtime_rows(args, facts)));

    #[cfg(feature = "api")]
    {
        let (enabled, rows) = token_budget_rows(&token_budget_config(args));
        let state = if enabled { "ENABLED" } else { "DISABLED" };
        println!();
        print!("{}", render_section(&format!("Token budget  {state}"), &rows));
    }
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

/// Compile-time enabled Cargo features, in display order.
fn feature_list() -> Vec<&'static str> {
    let mut v = Vec::new();
    if cfg!(feature = "api") {
        v.push("api");
    }
    if cfg!(feature = "local") {
        v.push("local");
    }
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
        (
            "git",
            format!("{} ({})", env!("LTE_GIT_SHA"), env!("LTE_GIT_BRANCH")),
        ),
        ("built", env!("LTE_BUILD_TIME").to_string()),
        // `LTE_RUSTC` is the raw `rustc --version` line ("rustc 1.96.0 (…)");
        // drop the leading word so it doesn't read "rustc  rustc 1.96.0".
        (
            "rustc",
            env!("LTE_RUSTC").trim_start_matches("rustc ").to_string(),
        ),
    ]
}

/// `"set"`/`"none"` — never echoes secret values.
fn secret_state(s: &str) -> &'static str {
    if s.trim().is_empty() { "none" } else { "set" }
}

#[cfg(feature = "api")]
fn runtime_rows(args: &Args, facts: &RuntimeFacts) -> Vec<(&'static str, String)> {
    let model = if facts.model_resolved {
        format!("{} (resolved)", facts.resolved_model)
    } else {
        facts.resolved_model.clone()
    };
    let timeout = if args.llm_timeout > 0 {
        format!("{}s", args.llm_timeout)
    } else {
        "none".to_string()
    };
    vec![
        ("mode", "api".to_string()),
        ("llm url", args.llm_url.clone()),
        ("llm model", model),
        ("llm api key", secret_state(&args.llm_api_key).to_string()),
        ("server auth", secret_state(&args.api_key).to_string()),
        ("bind", format!("{}:{}", args.host, args.port)),
        ("char limit", args.char_limit.to_string()),
        ("llm timeout", timeout),
    ]
}

#[cfg(feature = "local")]
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

#[cfg(feature = "api")]
fn token_budget_config(args: &Args) -> TokenBudgetConfig {
    TokenBudgetConfig {
        chars_per_token: args.llm_chars_per_token,
        output_mult: args.llm_max_tokens_mult,
        floor: args.llm_max_tokens_floor,
        ceiling: if args.llm_max_tokens > 0 {
            Some(args.llm_max_tokens)
        } else {
            None
        },
    }
}

/// `(enabled, rows)` for the token-budget section. Pure — unit-tested.
/// "Enabled" mirrors `dynamic_output_cap`'s own gate.
#[cfg(feature = "api")]
fn token_budget_rows(cfg: &TokenBudgetConfig) -> (bool, Vec<(&'static str, String)>) {
    let enabled = cfg.output_mult > 0.0 && cfg.chars_per_token > 0.0;
    let rows = vec![
        ("chars/token", format!("{}", cfg.chars_per_token)),
        ("mult", format!("×{}", cfg.output_mult)),
        ("floor", cfg.floor.to_string()),
        (
            "ceiling",
            cfg.ceiling
                .map(|c| c.to_string())
                .unwrap_or_else(|| "none".to_string()),
        ),
    ];
    (enabled, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_list_includes_active_features() {
        let features = feature_list();
        // Exactly one of api/local is active in any valid build.
        assert_eq!(
            cfg!(feature = "api") || cfg!(feature = "local"),
            !features.is_empty()
        );
        #[cfg(feature = "api")]
        assert!(features.contains(&"api"));
        #[cfg(feature = "local")]
        assert!(features.contains(&"local"));
    }

    #[test]
    fn secret_state_masks_values() {
        assert_eq!(secret_state(""), "none");
        assert_eq!(secret_state("   "), "none");
        assert_eq!(secret_state("hunter2"), "set");
    }

    #[test]
    fn render_section_aligns_values() {
        let rows = vec![
            ("a", "1".to_string()),
            ("long", "2".to_string()),
        ];
        // Labels padded to the widest ("long" = 4), two-space indent + gap.
        assert_eq!(
            render_section("S", &rows),
            "S\n  a     1\n  long  2\n",
        );
    }

    #[cfg(feature = "api")]
    #[test]
    fn token_budget_enabled_with_positive_params() {
        let cfg = TokenBudgetConfig {
            chars_per_token: 2.0,
            output_mult: 3.0,
            floor: 64,
            ceiling: Some(16384),
        };
        let (enabled, rows) = token_budget_rows(&cfg);
        assert!(enabled);
        assert_eq!(rows[3], ("ceiling", "16384".to_string()));
    }

    #[cfg(feature = "api")]
    #[test]
    fn token_budget_disabled_by_mult() {
        let cfg = TokenBudgetConfig {
            chars_per_token: 2.0,
            output_mult: 0.0,
            floor: 64,
            ceiling: None,
        };
        assert!(!token_budget_rows(&cfg).0);
    }

    #[cfg(feature = "api")]
    #[test]
    fn token_budget_disabled_by_chars_per_token() {
        let cfg = TokenBudgetConfig {
            chars_per_token: 0.0,
            output_mult: 3.0,
            floor: 64,
            ceiling: None,
        };
        assert!(!token_budget_rows(&cfg).0);
    }

    #[cfg(feature = "api")]
    #[test]
    fn token_budget_no_ceiling_reads_none() {
        let cfg = TokenBudgetConfig {
            chars_per_token: 2.0,
            output_mult: 3.0,
            floor: 64,
            ceiling: None,
        };
        let (_, rows) = token_budget_rows(&cfg);
        assert_eq!(rows[3], ("ceiling", "none".to_string()));
    }
}
