use std::collections::HashMap;

pub struct HuggingFace {
    pub repo: &'static str,
    pub model: &'static str,
}

pub static MODELS: once_cell::sync::Lazy<HashMap<&'static str, HuggingFace>> = once_cell::sync::Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("gemma3-1b", HuggingFace { repo: "libretranslate/gemma3", model: "gemma-3-1b-it-q4_0.gguf" });
    m.insert("gemma3-4b", HuggingFace { repo: "libretranslate/gemma3", model: "gemma-3-4b-it-q4_0.gguf" });
    m
});
