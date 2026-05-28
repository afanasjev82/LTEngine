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
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Mutex;
use anyhow::{Result, Context};

pub struct LLM {
    backend: LlamaBackend,
    model: LlamaModel,
    prompt_lock: Mutex<bool>
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

        let model_params = {
            if !cpu && cfg!(any(feature = "cuda", feature = "vulkan")) {
                LlamaModelParams::default().with_n_gpu_layers(9999)
            } else {
                LlamaModelParams::default()
            }
        };

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| "Unable to load model")?;
        
        Ok(LLM { backend, model, prompt_lock: Mutex::new(true) })
    }

    pub fn create_context(&self, ctx_size: i32) -> Result<LLMContext<'_>>{
        let ctx_params =
            LlamaContextParams::default()
                .with_n_ctx(Some(NonZeroU32::new(ctx_size as u32).unwrap()))
                .with_n_batch(ctx_size as u32);  // Set n_batch to match n_ctx to avoid assertion failure with large texts

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
        let mut ctx = self.create_context(ctx_size)?;
        {
            // TODO: The llama bindings (or llama itself?) do not appear to be totally thread-safe
            // as garbage starts to come out when we run inference in parallel
            // this might need to be investigated and fixed. For now we lock and process requests
            // one at a time.
            // TODO: consider locking with a timeout: https://docs.rs/parking_lot/latest/parking_lot/type.Mutex.html#method.try_lock_for
            let _lock = self.prompt_lock.lock();
            ctx.process(tokens_list)
        }
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
