use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::path::PathBuf;
use anyhow::{Result, Context};

pub fn init_llm(model_path: PathBuf, cpu: bool) -> Result<LlamaModel>{
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

    return Ok(model);
}