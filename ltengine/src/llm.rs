use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::context::LlamaContext;
use std::num::NonZeroU32;
use std::path::PathBuf;
use anyhow::{Result, Context};

pub struct LLM {
    backend: LlamaBackend,
    model: LlamaModel,
}

impl LLM {
    pub fn new(model_path: PathBuf, cpu: bool) -> Result<Self> {
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
        
        Ok(LLM { backend, model })
    }

    pub fn create_context(&self) -> Result<LlamaContext<'_>>{
        let ctx_params =
            LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));

        // Use all threads
        // ctx_params = ctx_params.with_n_threads(threads);
        // ctx_params = ctx_params.with_n_threads_batch(threads_batch);

        let ctx = self.model
            .new_context(&self.backend, ctx_params)
            .with_context(|| "Unable to create the llama context")?;
        Ok(ctx)
    }
}
