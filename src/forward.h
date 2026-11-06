// Step 12: the full forward pass — embedding, 16 layers, final norm, logits, next token.
#pragma once
#include "attention.h"
#include "linear.h"
#include "model.h"

// Every buffer the forward pass uses, allocated once for MAX_PROMPT_LEN tokens and reused by
// all 16 layers (each layer overwrites the previous layer's scratch values).
struct Activations {
    bf16* hidden = nullptr;     // [T, 2048] the residual stream x, carried from layer to layer
    bf16* norm = nullptr;       // [T, 2048] RMSNorm output (attention half, then MLP half)
    bf16* q = nullptr;          // [T, 2048]
    bf16* k = nullptr;          // [T, 512]
    bf16* v = nullptr;          // [T, 512]
    bf16* o = nullptr;          // [T, 2048] O projection, then reused for the down projection
    bf16* gate = nullptr;       // [T, 8192] gate, then silu(gate) * up in place
    bf16* up = nullptr;         // [T, 8192]
    float* inv_freq = nullptr;  // [32] RoPE speeds, computed once
    float* logits = nullptr;    // [VOCAB] fp32 scores for the last token only
    AttentionBuffers att;
};
Activations allocActivations();
void freeActivations(Activations& a);

struct Prediction {
    int top_ids[5];
    float top_logits[5];
};

// Runs the whole prompt through the model and returns the 5 most likely next tokens.
// dumps = true writes layer 0's intermediates and every layer's output to out/ (slow; for checks).
Prediction forward(cublasHandle_t handle, const Weights& w, const int* d_tokens, int T,
                   Activations& a, bool dumps);
