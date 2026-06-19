use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::model::{LlamaModel, LlamaChatMessage};
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::{send_logs_to_log, LogOptions};
use llama_cpp_bindings::{list_llama_ggml_backend_devices, LlamaBackendDeviceType};
use llama_cpp_bindings::{DecodeError, LlamaModelLoadError};
use std::num::NonZeroU32;
use std::path::PathBuf;
use parking_lot::Mutex;
use anyhow::{Result, Context};

#[derive(thiserror::Error, Debug)]
pub enum LLMError {
    #[error("LLM busy")]
    Busy,
    #[error("model produced empty output")]
    EmptyOutput,
}

/// Sentinel "offload everything" layer count. llama.cpp clamps any value >= the
/// real layer count to "all layers", so 9999 means "all" for every model we ship.
const GPU_LAYERS_ALL: i32 = 9999;

/// Next layer count after a *decode* probe failure: shed ~10%, floor of 1.
fn next_layers_on_decode_fail(n: i32) -> i32 {
    let step = (n / 10).max(1);
    (n - step).max(1)
}

/// Next layer count after a *load* failure: first failure (still at "all") jumps to
/// 64, then halve. Can reach 0 -> caller falls back to CPU.
fn next_layers_on_load_fail(n: i32) -> i32 {
    if n >= GPU_LAYERS_ALL { 64 } else { n / 2 }
}

/// Pick `n_ubatch` from the primary GPU's total VRAM. <6 GiB -> 128 (avoids OOM on
/// compute buffers); otherwise the binding default (512).
fn pick_n_ubatch(total_vram_mib: u64) -> u32 {
    const DEFAULT_N_UBATCH: u32 = 512;
    if total_vram_mib < 6 * 1024 { 128 } else { DEFAULT_N_UBATCH }
}

/// Total VRAM (MiB) of the first GPU via the binding's safe device enumeration.
fn primary_gpu_total_vram_mib() -> Option<u64> {
    list_llama_ggml_backend_devices()
        .into_iter()
        .find(|d| matches!(
            d.device_type,
            LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu
        ))
        .map(|d| d.memory_total as u64 / (1024 * 1024))
}

pub struct LLM {
    backend: LlamaBackend,
    model: LlamaModel,
    prompt_lock: Mutex<()>,
    n_ubatch: u32,
}

pub struct LLMContext<'a>{
    llm: &'a LLM,
    ctx: LlamaContext<'a>,
    ctx_size: i32
}

impl LLM {
    pub fn new(model_path: PathBuf, cpu: bool, verbose: bool) -> Result<Self> {
        if !verbose{
            send_logs_to_log(LogOptions::default().with_logs_enabled(false));
        }
        
        let backend = LlamaBackend::init()?;

        let use_gpu = !cpu && cfg!(any(feature = "cuda", feature = "vulkan"));

        // Optional forced starting cap to exercise the probe path on a big GPU.
        // Unset in production. e.g. LTENGINE_GPU_LAYERS_START=8.
        let start_layers = std::env::var("LTENGINE_GPU_LAYERS_START")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(GPU_LAYERS_ALL);

        let model = if use_gpu {
            Self::load_model_probing_gpu(&backend, &model_path, start_layers)?
        } else {
            let m = LlamaModel::load_from_file(
                &backend,
                &model_path,
                &LlamaModelParams::default().with_n_gpu_layers(0),
            )
            .with_context(|| "Unable to load model")?;
            eprintln!("ltengine: CPU only ({} layers)", m.n_layer().unwrap_or(0));
            m
        };

        let n_ubatch = if use_gpu {
            match primary_gpu_total_vram_mib() {
                Some(total_mib) => {
                    let n = pick_n_ubatch(total_mib);
                    eprintln!("ltengine: {total_mib} MiB total VRAM, n_ubatch={n}");
                    n
                }
                None => pick_n_ubatch(u64::MAX),
            }
        } else {
            pick_n_ubatch(u64::MAX)
        };
        
        Ok(LLM { backend, model, prompt_lock: Mutex::new(()), n_ubatch })
    }

    /// Load on GPU, shedding layers until both load AND a minimal trial decode
    /// succeed. Adapts upstream 3cdef35 + cdc2ba2 + 9b6c7ee to intentee typed errors.
    fn load_model_probing_gpu(
        backend: &LlamaBackend,
        model_path: &PathBuf,
        start_layers: i32,
    ) -> Result<LlamaModel> {
        let mut n_gpu = start_layers;
        loop {
            let model = match LlamaModel::load_from_file(
                backend,
                model_path,
                &LlamaModelParams::default().with_n_gpu_layers(n_gpu),
            ) {
                Ok(m) => m,
                Err(e @ LlamaModelLoadError::FileNotFound(_)) => {
                    return Err(anyhow::Error::new(e).context("Unable to load model"));
                }
                Err(e) => {
                    let next = next_layers_on_load_fail(n_gpu);
                    eprintln!("ltengine: model load failed at {n_gpu} GPU layers ({e}), retrying with {next}");
                    if next == 0 {
                        return Err(anyhow::Error::new(e).context("Unable to load model even with 0 GPU layers"));
                    }
                    n_gpu = next;
                    continue;
                }
            };

            match Self::probe_decode(&model, backend) {
                Ok(()) => {
                    let total = model.n_layer().unwrap_or(0);
                    let on_gpu = if n_gpu < 0 { total } else { (n_gpu as u32).min(total) };
                    if on_gpu >= total {
                        eprintln!("ltengine: {total}/{total} layers offloaded to GPU");
                    } else {
                        eprintln!("ltengine: {on_gpu}/{total} layers offloaded to GPU, rest on CPU");
                    }
                    return Ok(model);
                }
                Err(e) => {
                    let total = model.n_layer().unwrap_or(0);
                    let current = if n_gpu < 0 { total as i32 } else { n_gpu.min(total as i32) };
                    let next = next_layers_on_decode_fail(current);
                    eprintln!("ltengine: GPU probe decode failed at {current} layers ({e}), retrying with {next}");
                    drop(model);
                    if next <= 0 {
                        return Err(anyhow::anyhow!("GPU inference failed even with minimal layers"));
                    }
                    n_gpu = next;
                }
            }
        }
    }

    /// Minimal one-token decode to confirm the GPU can run compute at the current
    /// offload. Maps context/batch allocation failures to a decode-style error so
    /// the caller sheds layers.
    fn probe_decode(model: &LlamaModel, backend: &LlamaBackend) -> std::result::Result<(), DecodeError> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(NonZeroU32::new(8).expect("8 is non-zero")))
            .with_n_ubatch(1);

        let mut ctx = match LlamaContext::from_model(model, backend, ctx_params) {
            Ok(c) => c,
            Err(_) => return Err(DecodeError::DecodeOutOfMemory),
        };

        let mut batch = match LlamaBatch::new(8, 1) {
            Ok(b) => b,
            Err(_) => return Err(DecodeError::DecodeOutOfMemory),
        };
        if batch.add(&SampledToken::Content(LlamaToken(0)), 0, &[0], true).is_err() {
            return Err(DecodeError::DecodeOutOfMemory);
        }

        ctx.decode(&mut batch)
    }

    pub fn create_context(&self, ctx_size: i32) -> Result<LLMContext<'_>>{
        let ctx_params =
            LlamaContextParams::default()
                .with_n_ctx(Some(NonZeroU32::new(ctx_size as u32).unwrap()))
                .with_n_batch(ctx_size as u32)  // Set n_batch to match n_ctx to avoid assertion failure with large texts
                .with_n_ubatch(self.n_ubatch);  // Bundle B: cap micro-batch on low-VRAM GPUs

        // Use all threads
        // ctx_params = ctx_params.with_n_threads(threads);
        // ctx_params = ctx_params.with_n_threads_batch(threads_batch);

        let ctx = LlamaContext::from_model(&self.model, &self.backend, ctx_params)
            .with_context(|| "Unable to create the llama context")?;
        Ok(LLMContext{ llm: self, ctx, ctx_size })
    }

    pub fn run_prompt(&self, system: String, user: String) -> Result<String>{
        let tmpl = self.model.chat_template(None)?;
        let llm_input = self.model.apply_chat_template(&tmpl, &[
            LlamaChatMessage::new("system".to_string(), system)?,
            LlamaChatMessage::new("user".to_string(), user)?
        ], true)?;

        let tokens_list = self.model
            .str_to_token(&llm_input
            , AddBos::Always)
            .with_context(|| format!("Failed to tokenize {llm_input}"))?;
        // for token in &tokens_list {
        //     eprint!("{} {} | ", self.model.token_to_str(*token, Special::Tokenize)?, token);
        // }
        let ctx_size: i32 = tokens_list.len() as i32 * 3;
        // Bundle B (upstream f2faec9): take the lock BEFORE create_context — GPU
        // context allocation must be serialized too, not just inference. 120s -> 503.
        let _lock = self.prompt_lock.try_lock_for(std::time::Duration::from_secs(120))
            .ok_or(LLMError::Busy)?;
        let mut ctx = self.create_context(ctx_size)?;
        ctx.process(tokens_list)
    }
}

impl LLMContext<'_>{
    pub fn process(&mut self, tokens_list: Vec<LlamaToken>) -> Result<String>{
        // let ctx_size: i32 = tokens_list.len() as i32 * 3;
        
        // We use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(self.ctx_size.try_into()?, 1)
            .with_context(|| "Failed to allocate llama batch")?;

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(&SampledToken::Content(token), i, &[0], is_last)?;
        }

        self.ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed")?;

        let mut n_cur = batch.n_tokens();

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let seq_breakers = vec![b"\n", b":", b"\"", b"*"];

        let dry = LlamaSampler::dry(&self.llm.model, 0.0, 1.75, 2, -1, seq_breakers)
            .with_context(|| "Failed to build dry sampler")?;
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(64, 1.0, 0.0, 0.0),
            dry,
            LlamaSampler::top_k(40),
            LlamaSampler::typical(1.0, 0),
            LlamaSampler::top_p(0.95, 0),
            LlamaSampler::min_p(0.05, 0),
            LlamaSampler::xtc(0.0, 0.1, 0, 42),
            LlamaSampler::temp_ext(0.0, 0.0, 1.0),
            LlamaSampler::dist(42)
        ]);

        let mut output = String::new();

        while n_cur <= self.ctx_size {

            // sample the next token
            {
                let raw_token = sampler.sample(&self.ctx, batch.n_tokens() - 1)
                    .with_context(|| "Sampler failed")?;

                sampler.accept(raw_token)
                    .with_context(|| "Sampler accept failed")?;

                let sampled = SampledToken::Content(raw_token);

                // is it an end of stream?
                if self.llm.model.is_eog_token(&sampled) {
                    break;
                }
                    
                let output_string = self.llm.model.token_to_piece(&sampled, &mut decoder, true, None)?;
                output.push_str(&output_string);

                batch.clear();
                batch.add(&sampled, n_cur, &[0], true)?;
            }

            n_cur += 1;

            self.ctx.decode(&mut batch).with_context(|| "Failed to eval")?;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::{next_layers_on_decode_fail, next_layers_on_load_fail, pick_n_ubatch, GPU_LAYERS_ALL};

    #[test]
    fn decode_fail_sheds_about_ten_percent() {
        assert_eq!(next_layers_on_decode_fail(100), 90);
        assert_eq!(next_layers_on_decode_fail(48), 44);
        assert_eq!(next_layers_on_decode_fail(9999), 9000);
    }

    #[test]
    fn decode_fail_floors_at_one_and_always_progresses() {
        assert_eq!(next_layers_on_decode_fail(2), 1);
        assert_eq!(next_layers_on_decode_fail(1), 1);
        for n in 2..=200 {
            assert!(next_layers_on_decode_fail(n) < n, "n={n} did not decrease");
        }
    }

    #[test]
    fn load_fail_first_jumps_to_64_then_halves() {
        assert_eq!(next_layers_on_load_fail(GPU_LAYERS_ALL), 64);
        assert_eq!(next_layers_on_load_fail(10_000), 64);
        assert_eq!(next_layers_on_load_fail(64), 32);
        assert_eq!(next_layers_on_load_fail(1), 0);
    }

    #[test]
    fn ubatch_drops_on_small_cards_keeps_default_on_big() {
        assert_eq!(pick_n_ubatch(0), 128);
        assert_eq!(pick_n_ubatch(3500), 128);
        assert_eq!(pick_n_ubatch(6 * 1024 - 1), 128);
        assert_eq!(pick_n_ubatch(6 * 1024), 512);
        assert_eq!(pick_n_ubatch(24 * 1024), 512);
    }
}
