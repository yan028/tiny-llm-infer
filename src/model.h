// Model constants and weight pointers for Llama 3.2 1B Instruct.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>

// ---- Architecture (from config.json). Hardcoded on purpose: we build for one model first.
constexpr int HIDDEN = 2048;                   // hidden_size
constexpr int N_LAYERS = 16;                   // num_hidden_layers
constexpr int N_Q_HEADS = 32;                  // num_attention_heads
constexpr int N_KV_HEADS = 8;                  // num_key_value_heads
constexpr int HEAD_DIM = 64;                   // HIDDEN / N_Q_HEADS
constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM;  // 512
constexpr int FFN = 8192;                      // intermediate_size
constexpr int VOCAB = 128256;                  // vocab_size
constexpr float RMS_EPS = 1e-5f;               // rms_norm_eps
// RoPE (config.json: rope_theta and rope_scaling, rope_type "llama3")
constexpr float ROPE_THETA = 500000.0f;
constexpr float ROPE_FACTOR = 32.0f;           // slow pairs rotate this many times slower
constexpr float ROPE_LOW_FREQ_FACTOR = 1.0f;   // < 1 full turn in 8192 positions: fully slowed
constexpr float ROPE_HIGH_FREQ_FACTOR = 4.0f;  // > 4 full turns: unchanged; between: blended
constexpr float ROPE_ORIGINAL_CTX = 8192.0f;   // original_max_position_embeddings
constexpr int ROPE_PAIRS = HEAD_DIM / 2;       // 32 pairs per head: dims (i, i + 32)
constexpr int MAX_PROMPT_LEN = 512;  // our own limit; buffers are sized for this many tokens

using bf16 = __nv_bfloat16;

// Stop with file:line if a CUDA call fails. Wrap every CUDA runtime call in this.
#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                          \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                         cudaGetErrorString(err_));                                         \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

// One transformer layer's 9 tensors. Every pointer points into the single GPU block.
struct LayerWeights {
    bf16* attn_norm;  // input_layernorm.weight            [2048]
    bf16 *q, *k, *v;  // self_attn.{q,k,v}_proj.weight     [2048,2048] [512,2048] [512,2048]
    bf16* o;          // self_attn.o_proj.weight           [2048,2048]
    bf16* mlp_norm;   // post_attention_layernorm.weight   [2048]
    bf16 *gate, *up;  // mlp.{gate,up}_proj.weight         [8192,2048]
    bf16* down;       // mlp.down_proj.weight              [2048,8192]
};

struct Weights {
    void* gpu_block = nullptr;  // the one cudaMalloc holding all 2.47 GB
    size_t bytes = 0;
    // model.embed_tokens.weight [128256, 2048], also the output layer (tied weights)
    bf16* embed = nullptr;
    bf16* final_norm = nullptr;  // model.norm.weight         [2048]
    LayerWeights layers[N_LAYERS];
};

// Parse the safetensors file, copy the whole data section to the GPU, fill in every pointer.
Weights loadWeights(const std::string& path);
void freeWeights(Weights& w);
