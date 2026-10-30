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

// ---- Step 7: RMSNorm
// One block per token, 1024 threads, 2 values per thread (same layout as the gather).
// The block needs one number from all 2048 values (the row's RMS), so threads cooperate:
//   1. each thread squares its 2 values and adds them             -> 1024 partial sums
//   2. tree reduction in shared memory halves the count each round -> 1 total in sums[0]
//   3. every thread reads the total and normalizes its own 2 values
// All math is done in fp32; only the stored result is bf16.
constexpr int RMS_THREADS = 1024;

__global__ void rmsNormKernel(const bf16* in, const bf16* weight, bf16* out) {
    __shared__ float sums[RMS_THREADS];  // one slot per thread, visible to the whole block

    const int t = blockIdx.x;
    const int c = threadIdx.x;
    const bf16* x = in + (size_t)t * HIDDEN;
    bf16* y = out + (size_t)t * HIDDEN;

    // 1. Read my two values once, keep them in registers (fp32).
    const float a = __bfloat162float(x[c]);
    const float b = __bfloat162float(x[c + HIDDEN / 2]);
    sums[c] = a * a + b * b;
    __syncthreads();  // every slot must be written before anyone reads a neighbour's

    // 2. Tree reduction: 1024 -> 512 -> 256 -> ... -> 1 (10 rounds).
    //    In each round, the first `stride` threads add the slot `stride` places to their right.
    for (int stride = RMS_THREADS / 2; stride > 0; stride /= 2) {
        if (c < stride) {
            sums[c] += sums[c + stride];
        }
        __syncthreads();  // finish this round before the next one reads the results
    }

    // 3. sums[0] now holds the sum of all 2048 squares. Every thread computes 1/RMS from it.
    const float inv_rms = 1.0f / sqrtf(sums[0] / HIDDEN + RMS_EPS);
    y[c] = __float2bfloat16(a * inv_rms * __bfloat162float(weight[c]));
    y[c + HIDDEN / 2] = __float2bfloat16(b * inv_rms * __bfloat162float(weight[c + HIDDEN / 2]));
}

void rmsNorm(const bf16* in, const bf16* weight, bf16* out, int T) {
    rmsNormKernel<<<T, RMS_THREADS>>>(in, weight, out);
    CUDA_CHECK(cudaGetLastError());
}
