// Declarations for the CUDA kernels' launch functions. main.cu calls these.
#pragma once
#include "model.h"

// Step 6: hidden[t] = embed[tokens[t]] for t in 0..T-1. Each row is HIDDEN bf16 values.
void embeddingGather(const int* d_tokens, const bf16* embed, bf16* d_hidden, int T);
