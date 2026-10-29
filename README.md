# tiny-llm-infer

A small LLM inference engine written from scratch in C++ and CUDA, built as a self-learning project. It runs **Llama 3.2 1B Instruct** directly from its `model.safetensors` file, with every operation implemented as a CUDA kernel or a cuBLAS call.

## Inspiration

This project follows **[tiny-vllm](https://github.com/jmaczan/tiny-vllm)** by [Jędrzej Maczan](https://github.com/jmaczan) (Apache-2.0): a course that walks through building a high-performance LLM inference engine in C++ and CUDA, a smaller sibling of [vLLM](https://github.com/vllm-project/vllm). The course covers loading safetensors, the full Llama forward pass, KV cache, batching, online softmax and PagedAttention.

This repo is my own implementation while working through that course, not a copy of it. It adds a few things along the way:

- **A PyTorch golden reference** (`python/reference.py`) that runs the forward pass op by op, dumps every intermediate tensor, and checks itself against Hugging Face `transformers`. Each CUDA kernel is verified against these dumps with `python/compare.py` before moving on.
- **Hugging Face RoPE layout and Llama 3 frequency scaling.** The Hugging Face Llama weights pair dimension `i` with `i + 32` ("rotate half"), and Llama 3.x applies `rope_scaling: llama3` to the rotation speeds. Both are reproduced here.

## Setup

| | |
|---|---|
| GPU | NVIDIA RTX 5060 (8 GB, compute capability 12.0) |
| OS | Windows 11 + WSL2 Ubuntu 24.04 |
| CUDA | 13.3 |
| Model | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (gated; accept Meta's license on Hugging Face) |

Model weights are not included in this repo. Download them with:

```bash
hf download meta-llama/Llama-3.2-1B-Instruct \
  model.safetensors config.json tokenizer.json tokenizer_config.json \
  --local-dir models/llama-3.2-1b-instruct
```

Python tools (reference and checks) need `torch transformers safetensors numpy`.

## Build and run

```bash
python python/reference.py                     # tokenize the prompt, write golden dumps to dumps/
cmake -B build -G Ninja && cmake --build build
./build/tiny_infer                             # run from the repo root
```

Set `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` to match your GPU (120 = RTX 50 series).

## Layout

```
src/model.h                    model constants, Weights struct, CUDA_CHECK
src/loader.cu                  safetensors → one GPU allocation + per-tensor pointers
src/main.cu                    entry point
python/inspect_safetensors.py  parse the safetensors header by hand
python/reference.py            PyTorch golden reference + Hugging Face check
python/compare.py              compare an engine output against a reference dump
include/json.hpp               nlohmann/json 3.12.0
```

## Roadmap

Milestone 1 is the course's "single token inference": produce the same first token as Hugging Face for the prompt *"What is the capital of France?"* (expected: `The`).

- [x] **Step 0 — Environment.** Driver, WSL2, CUDA toolkit, a first test kernel, Python venv.
- [x] **Step 1 — Download the model.** Llama 3.2 1B Instruct, about 2.47 GB in bf16.
- [x] **Step 2 — Parse safetensors by hand.** 8-byte header size, JSON header, 146 tensors, 1,235,814,400 parameters. Weights are stored `[out, in]`; there is no `lm_head` (tied to the embedding table).
- [x] **Step 3 — Golden reference.** Hand-written forward pass in PyTorch. Predicts token 791 `The`, same top-5 as Hugging Face; relative error vs HF about 1e-2.
- [x] **Step 4 — C++ loader.** One `cudaMalloc` for all weights, streamed through a pinned buffer, each tensor's pointer bound and shape-checked. Values read back from the GPU match the Python parser exactly.
- [x] **Step 5 — Token ids to the GPU.** Ids come from the Python tokenizer; prompts are capped at 512 tokens.
- [ ] **Step 6 — Embedding gather kernel.** One block per token; exact match with the reference.
- [ ] **Step 7 — RMSNorm kernel.** fp32 tree reduction, eps 1e-5.
- [ ] **Step 8 — Q/K/V projections** with `cublasGemmEx` (row-major/column-major transposition trick).
- [ ] **Step 9 — RoPE kernel.** Rotate-half pairing, Llama 3 scaled frequencies precomputed on the CPU.
- [ ] **Step 10 — Attention (prefill).** Per-head scores, GQA (8 KV heads shared by 32 query heads), causal mask, softmax, weighted sum of values, output projection, residual.
- [ ] **Step 11 — MLP (SwiGLU).** gate and up (2048 → 8192), SiLU gating, down (8192 → 2048), residual.
- [ ] **Step 12 — Full forward pass → first token.** 16 layers, final norm, logits via the tied embedding table, argmax.

After the first token:

- [ ] Decode loop with a KV cache
- [ ] Static batching
- [ ] Continuous batching
- [ ] Online softmax (FlashAttention-style)
- [ ] PagedAttention

## License

The code in this repo is my own. The course it follows, [tiny-vllm](https://github.com/jmaczan/tiny-vllm), is Apache-2.0. Llama 3.2 weights are subject to the [Llama 3.2 Community License](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).