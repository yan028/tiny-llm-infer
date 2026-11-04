// Step 10: causal self-attention for the whole prompt (prefill), one layer.
#pragma once
#include "linear.h"
#include "model.h"

// Scratch buffers attention needs, allocated once for MAX_PROMPT_LEN tokens.
struct AttentionBuffers {
    float* scores = nullptr;  // [32 heads, T, T] fp32: Q·Kᵀ/8 for every head
    bf16* probs = nullptr;    // [32 heads, T, T] bf16: softmax of the masked scores
    bf16* out = nullptr;      // [T, 2048]: the 32 heads' outputs side by side (concatenated)
};
AttentionBuffers allocAttention();
void freeAttention(AttentionBuffers& a);

// q: [T, 2048] (32 heads, after RoPE), k and v: [T, 512] (8 heads, k after RoPE).
// Fills a.scores, a.probs and a.out. GQA: query head h reads key/value head h / 4.
void attention(cublasHandle_t handle, const bf16* q, const bf16* k, const bf16* v,
               AttentionBuffers& a, int T);
