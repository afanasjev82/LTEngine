use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
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
            send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
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

    pub fn create_context(&self, ctx_size: i32) -> Result<LLMContext>{
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(ctx_size as u32).unwrap()));

        // Use all threads
        // ctx_params = ctx_params.with_n_threads(threads);
        // ctx_params = ctx_params.with_n_threads_batch(threads_batch);

        let ctx = self.model
            .new_context(&self.backend, ctx_params)
            .with_context(|| "Unable to create the llama context")?;
        Ok(LLMContext{ llm: self, ctx, ctx_size })
    }

    pub fn run_prompt(&self, prompt: String) -> Result<String>{
        let tokens_list = self.model
            .str_to_token(&prompt, AddBos::Always)
            .with_context(|| format!("Failed to tokenize {prompt}"))?;
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
        let mut batch = LlamaBatch::new(512, 1);

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        self.ctx.decode(&mut batch)
            .with_context(|| "llama_decode() failed")?;

        let mut n_cur = batch.n_tokens();

        let mut decoder = encoding_rs::UTF_8.new_decoder();

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 0),
            LlamaSampler::temp(0.8),
            LlamaSampler::dist(42),
        ]);

        let mut output = String::new();

        while n_cur <= self.ctx_size {
            // sample the next token
            {
                let token = sampler.sample(&self.ctx, batch.n_tokens() - 1);

                sampler.accept(token);

                // is it an end of stream?
                if self.llm.model.is_eog_token(token) {
                    break;
                }
                    
                let output_bytes = self.llm.model.token_to_bytes(token, Special::Tokenize)?;
                // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                let mut output_string = String::with_capacity(32);
                let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);
                output.push_str(&output_string);

                batch.clear();
                batch.add(token, n_cur, &[0], true)?;
            }

            n_cur += 1;

            self.ctx.decode(&mut batch).with_context(|| "Failed to eval")?;
        }

        Ok(output)
    }
}
