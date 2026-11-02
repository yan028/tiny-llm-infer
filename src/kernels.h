// Declarations for the CUDA kernels' launch functions. main.cu calls these.
#pragma once
#include "model.h"

// Step 6: hidden[t] = embed[tokens[t]] for t in 0..T-1. Each row is HIDDEN bf16 values.
void embeddingGather(const int* d_tokens, const bf16* embed, bf16* d_hidden, int T);

// Step 7: out[t] = in[t] / rms(in[t]) * weight, for each of the T rows (RMSNorm, eps = RMS_EPS).
void rmsNorm(const bf16* in, const bf16* weight, bf16* out, int T);

// Step 9: RoPE.
// computeRopeInvFreq fills the 32 rotation speeds (radians per position), with Llama 3 scaling.
// Runs on the CPU once; the result is copied to the GPU.
void computeRopeInvFreq(float inv_freq[ROPE_PAIRS]);
// Rotate every head of x in place. x is [T, n_heads * 64] (Q: 32 heads, K: 8 heads).
void rope(bf16* x, const float* d_inv_freq, int T, int n_heads);
