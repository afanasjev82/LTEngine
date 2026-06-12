# LTEngine

Free and Open Source Local AI Machine Translation API, written in Rust, entirely self-hosted and compatible with [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate). Its translation capabilities are powered by large language models (LLMs) that run locally on your machine via [llama.cpp](https://github.com/ggml-org/llama.cpp). Alternatively, LTEngine can delegate inference to a remote OpenAI-compatible server (e.g. vLLM or llama.cpp's `llama-server`) via the optional [API mode](#api-mode-remote-openai-compatible-backend) build.

![Translation](https://github.com/user-attachments/assets/37dd4e20-382b-459d-bcc1-5de3ed4b4c18)

The LLMs in LTEngine are much larger than the lightweight transformer models in [LibreTranslate](https://github.com/LibreTranslate/LibreTranslate). Thus memory usage and speed are traded off for quality of outputs, which for some languages has been reported as being [on par or better than DeepL](https://community.libretranslate.com/t/ltengine-llm-powered-local-machine-translation/1862/5).

It is possible to run LTEngine entirely on the CPU, but an accelerator will greatly improve performance. Supported accelerators currently include:

 * CUDA
 * Metal (macOS)
 * Vulkan

 The largest model (`gemma3-27b`) can fit on a single consumer RTX 3090 with 24G of VRAM.

> ⚠️ LTEngine is in active development. Check the [Roadmap](#roadmap) for current limitations.


## Requirements

 * [Rust](https://www.rust-lang.org/)
 * [clang](https://clang.llvm.org/)
 * [CMake](https://cmake.org/)
 * A C++ compiler (g++, MSVC) for building the llama.cpp bindings

clang, CMake and the C++ compiler are only needed for the default local-inference build, which compiles the bundled llama.cpp. [API mode](#api-mode-remote-openai-compatible-backend) builds skip llama.cpp entirely and need only Rust and a C compiler (plus OpenSSL development headers and `pkg-config` on Linux).

## Build

```bash
git clone https://github.com/LibreTranslate/LTEngine
cd LTEngine
cargo build [--features cuda,vulkan,metal] --release
```

The llama.cpp bindings are fetched automatically by Cargo as a pinned git dependency ([llama-cpp-bindings](https://github.com/intentee/llama-cpp-bindings)) — there are no git submodules, so `--recursive` is not needed.

Available build features:

| Feature | Description |
| ------- | ----------- |
| `local` (default) | In-process inference via llama.cpp |
| `cuda` / `metal` / `vulkan` | GPU acceleration for local inference (each implies `local`) |
| `api` | Remote inference via an OpenAI-compatible server. Build with `cargo build --no-default-features --features api --release` — `api` and `local` are mutually exclusive |

## Run

```bash
./target/release/ltengine
```

To run different LLM models:

```bash
./target/release/ltengine -m gemma3-12b [--model-file /path/to/model.gguf]
```

Common options (see `ltengine --help` for the full list):

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--host` | `0.0.0.0` | Address to bind |
| `-p, --port` | `5050` | Port to bind |
| `--char-limit` | `5000` | Maximum size of the `q` parameter (bytes); larger requests are rejected with HTTP 400 |
| `--api-key` | _(none)_ | When set, requests must include a matching `api_key` parameter or receive HTTP 403 |
| `--cpu` | off | Force CPU-only inference (local builds only) |
| `-v, --verbose` | off | Enable verbose llama.cpp logging |

`-m`, `--model-file` and `--cpu` exist only in local-inference builds; [API mode](#api-mode-remote-openai-compatible-backend) builds expose the `--llm-*` flags instead.

Once the server is running, open http://localhost:5050 in a browser to use the built-in LibreTranslate-style web translation UI (served on the same port as the API).

On startup LTEngine prints a summary of the build (version, enabled features, git commit/branch, build time, rustc version), the runtime configuration (model, bind address, character limit) and — in API mode — the token-budget settings. API keys are never echoed; they are shown only as `set`/`none`.

## API Mode (remote OpenAI-compatible backend)

Instead of running an LLM in-process, LTEngine can forward translation prompts to any OpenAI-compatible server (vLLM, llama.cpp's `llama-server`, etc.):

```bash
cargo build --no-default-features --features api --release
./target/release/ltengine --llm-url http://localhost:8080 [--llm-api-key KEY] [--llm-model NAME]
```

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--llm-url` | _(required)_ | Base URL of the server; LTEngine calls `{url}/v1/models` and `{url}/v1/chat/completions` |
| `--llm-api-key` | _(none)_ | Bearer token sent to the server |
| `--llm-model` | _(auto)_ | Model name; when omitted it is auto-resolved from `GET /v1/models` |
| `--llm-timeout` | `120` | HTTP timeout in seconds (`0` = no timeout) |
| `--llm-max-tokens` | `0` | Static output-token ceiling (`0` = no ceiling) |
| `--llm-chars-per-token` | `2.0` | Dynamic cap: conservative characters-per-token estimate |
| `--llm-max-tokens-mult` | `3.0` | Dynamic cap: output budget multiplier (`0` disables the dynamic cap) |
| `--llm-max-tokens-floor` | `64` | Dynamic cap: minimum output budget for tiny inputs |

Requests are deterministic: `temperature 0.0` and `chat_template_kwargs.enable_thinking = false` are always sent. Output tokens are capped dynamically per request: `ceil(ceil(input_chars / chars-per-token) × mult)`, never below the floor and never above `--llm-max-tokens` (when set). Set `--llm-max-tokens-mult 0` (or `--llm-chars-per-token 0`) to disable the dynamic cap and use only the static ceiling, which is omitted from requests when `0`. A warning is logged when a response is truncated at the cap (`finish_reason = "length"`).

## Models

LTEngine supports any GGUF language model supported by [llama.cpp](https://github.com/ggml-org/llama.cpp). You can pass a path to load a custom .gguf model using the `--model-file` parameter. Otherwise LTEngine will download one of the Gemma3 models based on the `-m` parameter: 

| Model      | RAM Usage | GPU Usage | Notes                               | Default            |
| ---------- | --------- | --------- | ----------------------------------- | ------------------ |
| gemma3-1b  | 1G        | 2G        | Good for testing, poor translations |                    |
| gemma3-4b  | 4G        | 4G        |                                     | :heavy_check_mark: |
| gemma3-12b | 8G        | 10G       |                                     |                    |
| gemma3-27b | 16G       | 18G       | Best translation quality, slowest   |                    |

Memory usage numbers are approximate.

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

List of language codes: http://0.0.0.0:5050/languages

### Auto Detect Language

Request:

```javascript
const res = await fetch("http://0.0.0.0:5050/translate", {
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

### Detect Language

Request:

```javascript
const res = await fetch("http://0.0.0.0:5050/detect", {
  method: "POST",
  body: JSON.stringify({
    q: "Ciao!",
  }),
  headers: { "Content-Type": "application/json" },
});

console.log(await res.json());
```

Response:

```javascript
[
    {
        "confidence": 83,
        "language": "it"
    }
]
```

### HTML Translation

`/translate` accepts an optional `format` parameter: `"text"` (default) or `"html"`. With `"html"`, the model is instructed to preserve all HTML tags and elements in the translation (best-effort, prompt-based — there is no tag-aware parsing yet). Any other value returns HTTP 400.

### Other Endpoints

| Endpoint | Description |
| -------- | ----------- |
| `GET /languages` | List of supported languages |
| `POST /detect` | Standalone language detection |
| `GET /health` | Liveness check; returns `200 OK` once the server is up |
| `GET /metrics` | Prometheus metrics (`ltengine` namespace) |

`/translate` and `/detect` accept JSON, `application/x-www-form-urlencoded` and `multipart/form-data` request bodies. `/translate_file` and `/suggest` are not implemented yet (HTTP 501).

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

## Roadmap

 - [ ] Remove mutex block that currently limits the software to process one single translation request at a time due to a possible bug in llama.cpp. 
 - [ ] Cancel inference (stop generating tokens) when HTTP connections are aborted by clients. I'm unsure how this could done with actix-web.
 - [ ] Add support for `/translate_file` (ability to translate files).
 - [ ] Add support for sentence splitting. Currently text is sent to the LLM as-is, but longer texts (like documents) should be split into chunks, translated and merged back.
 - [ ] Better language detection for short texts (port [LexiLang](https://github.com/LibreTranslate/LexiLang) to Rust)
 - [ ] Test/add more LLM models aside from Gemma3
 - [ ] Create comparative benchmarks between LTEngine and proprietary software.
 - [ ] Add support for command line inference (run `./ltengine translate` as a command line app separate from `./ltengine server`)
 - [ ] Make ltengine available as a library, possibly creating bindings for other languages like Python.
 - [x] Automated builds / CI (GitHub Actions: build + test on Ubuntu, on every push)
 - [ ] Multi-platform / GPU-feature CI and release builds (Windows/macOS, CUDA/Vulkan matrix, tagged release artifacts)
 - [ ] Your ideas? We welcome contributions.

## Contributing

We welcome contributions! Just open a pull request.

## Credits

This work is largely possible thanks to [llama-cpp-bindings](https://github.com/intentee/llama-cpp-bindings) which provides the Rust bindings to [llama.cpp](https://github.com/ggml-org/llama.cpp). Earlier versions were built on [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs).

## License

[GNU Affero General Public License v3](https://www.gnu.org/licenses/agpl-3.0.en.html)

## Trademark

See [Trademark Guidelines](https://github.com/LibreTranslate/LibreTranslate/blob/main/TRADEMARK.md)
