// CUDA kernels. Each kernel has a small launch function that sets the grid and checks errors.
#include "kernels.h"

// ---- Step 6: embedding gather
// Launched with one block per token and 1024 threads per block.
//   blockIdx.x  = which token (0..T-1) this block handles
//   threadIdx.x = 0..1023; each thread copies 2 of the token's 2048 values
__global__ void embeddingGatherKernel(const int* tokens, const bf16* table, bf16* out) {
    const int t = blockIdx.x;                       // Q1: my block = token position t (0..T-1)
    const int id = tokens[t];                       // Q2: that token's id (e.g. 128000)
    const bf16* src = table + (size_t)id * HIDDEN;  // Q3: start of row `id` in the table
    bf16* dst = out + (size_t)t * HIDDEN;           // Q4: start of row t in the output
    const int c = threadIdx.x;                      // Q5: my two columns are c and c + 1024
    dst[c] = src[c];
    dst[c + HIDDEN / 2] = src[c + HIDDEN / 2];
}

void embeddingGather(const int* d_tokens, const bf16* embed, bf16* d_hidden, int T) {
    embeddingGatherKernel<<<T, 1024>>>(d_tokens, embed, d_hidden);
    CUDA_CHECK(cudaGetLastError());  // catches bad launch configs right away
}
