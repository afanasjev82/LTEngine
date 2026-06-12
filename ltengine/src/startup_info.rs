//! Human-readable startup banner: build, runtime and token-budget info rendered
//! as bordered tables.
//!
//! Build-time values (`LTE_*`) are injected by `build.rs`. The module only
//! formats already-validated data, so it introduces no fallible paths.

use comfy_table::{Cell, Table};

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
    println!("{}", build_table());
    println!("{}", runtime_table(args, facts));
    #[cfg(feature = "api")]
    println!("{}", token_budget_table(args));
}

/// A table with the box-drawing preset forced on, so output is consistent in
/// Docker logs regardless of TTY detection.
fn new_table() -> Table {
    let mut t = Table::new();
    t.load_preset(comfy_table::presets::UTF8_FULL);
    t
}

/// Two-column table with a section header in the first cell.
fn section(title: &str) -> Table {
    let mut t = new_table();
    t.set_header(vec![Cell::new(title), Cell::new("")]);
    t
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

fn build_table() -> Table {
    let mut t = section("Build");
    t.add_row(vec![Cell::new("Version"), Cell::new(env!("CARGO_PKG_VERSION"))]);
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    t.add_row(vec![Cell::new("Profile"), Cell::new(profile)]);
    t.add_row(vec![Cell::new("Features"), Cell::new(feature_list().join(", "))]);
    t.add_row(vec![
        Cell::new("Git"),
        Cell::new(format!("{} ({})", env!("LTE_GIT_SHA"), env!("LTE_GIT_BRANCH"))),
    ]);
    t.add_row(vec![Cell::new("Built"), Cell::new(env!("LTE_BUILD_TIME"))]);
    t.add_row(vec![Cell::new("Rustc"), Cell::new(env!("LTE_RUSTC"))]);
    t
}

/// `"set"`/`"none"` — never echoes secret values.
fn secret_state(s: &str) -> &'static str {
    if s.trim().is_empty() { "none" } else { "set" }
}

#[cfg(feature = "api")]
fn runtime_table(args: &Args, facts: &RuntimeFacts) -> Table {
    let mut t = section("Runtime");
    t.add_row(vec![Cell::new("Mode"), Cell::new("api")]);
    t.add_row(vec![Cell::new("LLM URL"), Cell::new(&args.llm_url)]);
    let model = if facts.model_resolved {
        format!("{} (resolved)", facts.resolved_model)
    } else {
        facts.resolved_model.clone()
    };
    t.add_row(vec![Cell::new("LLM model"), Cell::new(model)]);
    t.add_row(vec![Cell::new("LLM API key"), Cell::new(secret_state(&args.llm_api_key))]);
    t.add_row(vec![Cell::new("Server auth"), Cell::new(secret_state(&args.api_key))]);
    t.add_row(vec![Cell::new("Bind"), Cell::new(format!("{}:{}", args.host, args.port))]);
    t.add_row(vec![Cell::new("Char limit"), Cell::new(args.char_limit)]);
    let timeout = if args.llm_timeout > 0 {
        format!("{}s", args.llm_timeout)
    } else {
        "none".to_string()
    };
    t.add_row(vec![Cell::new("LLM timeout"), Cell::new(timeout)]);
    t
}

#[cfg(feature = "local")]
fn runtime_table(args: &Args, facts: &RuntimeFacts) -> Table {
    let mut t = section("Runtime");
    t.add_row(vec![Cell::new("Mode"), Cell::new("local")]);
    t.add_row(vec![Cell::new("Model"), Cell::new(&args.model)]);
    t.add_row(vec![Cell::new("Model path"), Cell::new(facts.model_path.display())]);
    let device = if args.cpu { "cpu" } else { "gpu" };
    t.add_row(vec![Cell::new("Device"), Cell::new(device)]);
    t.add_row(vec![Cell::new("Server auth"), Cell::new(secret_state(&args.api_key))]);
    t.add_row(vec![Cell::new("Bind"), Cell::new(format!("{}:{}", args.host, args.port))]);
    t.add_row(vec![Cell::new("Char limit"), Cell::new(args.char_limit)]);
    t
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

/// `(enabled, rows)` for the token-budget table. Pure — unit-tested.
/// "Enabled" mirrors `dynamic_output_cap`'s own gate.
#[cfg(feature = "api")]
fn token_budget_rows(cfg: &TokenBudgetConfig) -> (bool, Vec<(&'static str, String)>) {
    let enabled = cfg.output_mult > 0.0 && cfg.chars_per_token > 0.0;
    let rows = vec![
        ("chars/token", format!("{}", cfg.chars_per_token)),
        ("mult", format!("x{}", cfg.output_mult)),
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

#[cfg(feature = "api")]
fn token_budget_table(args: &Args) -> Table {
    let cfg = token_budget_config(args);
    let (enabled, rows) = token_budget_rows(&cfg);
    let mut t = new_table();
    t.set_header(vec![
        Cell::new("Token budget (dynamic cap)"),
        Cell::new(if enabled { "ENABLED" } else { "DISABLED" }),
    ]);
    for (k, v) in rows {
        t.add_row(vec![Cell::new(k), Cell::new(v)]);
    }
    t
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
