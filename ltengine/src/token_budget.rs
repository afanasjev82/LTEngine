//! Dynamic per-request output-token budget for api/vLLM translation requests.
//!
//! Turns the input text size into a `max_tokens` cap so a hallucinating
//! generation is cut early instead of running to the static ceiling.

/// Configuration for the dynamic output-token cap.
#[derive(Debug, Clone, Copy)]
pub struct TokenBudgetConfig {
    /// Conservative characters-per-token divisor used to estimate input tokens.
    pub chars_per_token: f32,
    /// Output safety multiple applied to estimated input tokens (the "x3").
    pub output_mult: f32,
    /// Minimum cap (tokens) so tiny inputs are never starved.
    pub floor: u32,
    /// Hard ceiling (tokens); `None` means no ceiling.
    pub ceiling: Option<u32>,
}

/// Compute the dynamic output-token cap for an input of `input_chars` characters.
///
/// Returns `None` when the dynamic policy is disabled (non-positive
/// `output_mult` or `chars_per_token`), signalling the caller to fall back to
/// the static cap.
#[must_use]
pub fn dynamic_output_cap(input_chars: usize, cfg: &TokenBudgetConfig) -> Option<u32> {
    if cfg.output_mult <= 0.0 || cfg.chars_per_token <= 0.0 {
        return None;
    }
    let est_input_tokens = (input_chars as f32 / cfg.chars_per_token).ceil();
    let budget = (est_input_tokens * cfg.output_mult).ceil();
    // Guard NaN/negative before the saturating f32 -> u32 cast.
    let budget = if budget.is_finite() && budget >= 0.0 { budget } else { 0.0 };
    let mut cap = budget as u32; // saturates to u32::MAX on overflow
    cap = cap.max(cfg.floor);
    if let Some(c) = cfg.ceiling {
        // Ceiling is a hard limit (model/server max) and takes priority over the
        // floor if misconfigured with ceiling < floor.
        cap = cap.min(c);
    }
    Some(cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> TokenBudgetConfig {
        TokenBudgetConfig { chars_per_token: 2.0, output_mult: 3.0, floor: 64, ceiling: Some(16384) }
    }

    #[test]
    fn disabled_when_mult_non_positive() {
        let mut c = cfg();
        c.output_mult = 0.0;
        assert_eq!(dynamic_output_cap(1000, &c), None);
    }

    #[test]
    fn disabled_when_chars_per_token_non_positive() {
        let mut c = cfg();
        c.chars_per_token = 0.0;
        assert_eq!(dynamic_output_cap(1000, &c), None);
    }

    #[test]
    fn tiny_input_uses_floor() {
        // 12 chars / 2.0 = 6 -> *3 = 18 -> raised to floor 64
        assert_eq!(dynamic_output_cap(12, &cfg()), Some(64));
    }

    #[test]
    fn mid_input_uses_formula() {
        // 1000 / 2.0 = 500 -> *3 = 1500 (above floor, below ceiling)
        assert_eq!(dynamic_output_cap(1000, &cfg()), Some(1500));
    }

    #[test]
    fn huge_input_clamped_to_ceiling() {
        // 100000 / 2.0 = 50000 -> *3 = 150000 -> clamped to 16384
        assert_eq!(dynamic_output_cap(100_000, &cfg()), Some(16384));
    }

    #[test]
    fn no_ceiling_allows_large_cap() {
        let mut c = cfg();
        c.ceiling = None;
        assert_eq!(dynamic_output_cap(100_000, &c), Some(150_000));
    }
}
