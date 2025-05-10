# LTEngine

Free and Open Source Local AI Machine Translation API, written in Rust, entirely self-hosted and compatible with [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate). Its translation capabilities are powered by large language models (LLMs) that run locally on your machine via [llama.cpp](https://github.com/ggml-org/llama.cpp). 

![Translation](https://github.com/user-attachments/assets/457696b5-dbff-40ab-a18e-7bfb152c5121)

The LLMs in LTEngine are much larger than the lightweight transformer models in [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate). Thus memory usage and speed are traded off for quality of outputs. 

It is possible to run LTEngine entirely on the CPU, but an accelerator will greatly improve performance. Supported accelerators currently include:

 * CUDA
 * Vulkan
 * Metal (macOS)

## Requirements

 * [Rust](https://www.rust-lang.org/)
 * [clang](https://clang.llvm.org/)
 * [CMake](https://cmake.org/)
 * A C++ compiler (g++, MSVC) for building the llama.cpp bindings

## Build

```bash
git clone https://github.com/LibreTranslate/LTEngine --recursive
cd LTEngine
cargo build [--features cuda,vulkan,metal] --bin ltengine --release
```

## Run

```bash
./target/release/ltengine [--host 0.0.0.0] [--port 5050] [-m gemma3-4b | --model-file path-to-model.gguf]
```

## Models

LTEngine supports any GGUF language model supported by [llama.cpp](https://github.com/ggml-org/llama.cpp). You can pass a path to load a custom .gguf model using the `--model-file` parameter. Otherwise LTEngine will download one of the Gemma3 models based on the `-m` parameter: 

| Model      | RAM Usage | GPU Usage | Notes                     | Default            |
| ---------- | --------- | --------- | ------------------------- | ------------------ |
| gemma3-1b  |           |           | Only good for development |                    |
| gemma3-4b  |           |           |                           | :heavy_check_mark: |
| gemma3-12b |           |           |                           |                    |
| gemma4-27b |           |           | Best translation quality  |                    |

### Simple

Request:

```javascript
const res = await fetch("http://0.0.0.0:5050/translate", {
  method: "POST",
  body: JSON.stringify({
    q: "Hello!",
    source: "en",
    target: "es",
  }),
  headers: { "Content-Type": "application/json" },
});

console.log(await res.json());
```

Response:

```javascript
{
    "translatedText": "¡Hola!"
}
```

List of language codes: https://0.0.0.0:5000/languages

### Auto Detect Language

Request:

```javascript
const res = await fetch("http://0.0.0.0:5000/translate", {
  method: "POST",
  body: JSON.stringify({
    q: "Ciao!",
    source: "auto",
    target: "en",
  }),
  headers: { "Content-Type": "application/json" },
});

console.log(await res.json());
```

Response:

```javascript
{
    "detectedLanguage": {
        "confidence": 83,
        "language": "it"
    },
    "translatedText": "Bye!"
}
```

## Language Bindings

You can use the LTEngine API using the following bindings:

- Rust: <https://github.com/DefunctLizard/libretranslate-rs>
- Node.js: <https://github.com/franciscop/translate>
- TypeScript: <https://github.com/tderflinger/libretranslate-ts>
- .Net: <https://github.com/sigaloid/LibreTranslate.Net>
- Go: <https://github.com/SnakeSel/libretranslate>
- Python: <https://github.com/argosopentech/LibreTranslate-py>
- PHP: <https://github.com/jefs42/libretranslate>
- C++: <https://github.com/argosopentech/LibreTranslate-cpp>
- Swift: <https://github.com/wacumov/libretranslate>
- Unix: <https://github.com/argosopentech/LibreTranslate-sh>
- Shell: <https://github.com/Hayao0819/Hayao-Tools/tree/master/libretranslate-sh>
- Java: <https://github.com/suuft/libretranslate-java>
- Ruby: <https://github.com/noesya/libretranslate>
- R: <https://github.com/myanesp/libretranslateR>


## Contributing

We welcome contributions! Just open a pull request.

## Credits

This work is largely possible thanks [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) which provide the Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[GNU Affero General Public License v3](https://www.gnu.org/licenses/agpl-3.0.en.html)

## Trademark

See [Trademark Guidelines](https://github.com/LibreTranslate/LibreTranslate/blob/main/TRADEMARK.md)
