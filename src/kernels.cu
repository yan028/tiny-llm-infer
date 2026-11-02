// CUDA kernels. Each kernel has a small launch function that sets the grid and checks errors.
#include <cmath>

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

// ---- Step 9: RoPE
// Same formulas as rope_inv_freq() in python/reference.py (Hugging Face's llama3 rope type).
void computeRopeInvFreq(float inv_freq[ROPE_PAIRS]) {
    const float low_wavelen = ROPE_ORIGINAL_CTX / ROPE_LOW_FREQ_FACTOR;    // 8192
    const float high_wavelen = ROPE_ORIGINAL_CTX / ROPE_HIGH_FREQ_FACTOR;  // 2048
    for (int i = 0; i < ROPE_PAIRS; ++i) {
        // Standard RoPE speed for pair i: 1 / theta^(2i / 64).
        const float freq = 1.0f / std::pow(ROPE_THETA, (2.0f * i) / HEAD_DIM);
        const float wavelen = 2.0f * (float)M_PI / freq;  // positions per full turn
        if (wavelen < high_wavelen) {
            inv_freq[i] = freq;  // fast pair: unchanged
        } else if (wavelen > low_wavelen) {
            inv_freq[i] = freq / ROPE_FACTOR;  // slow pair: 32x slower
        } else {
            // In between: blend from "unchanged" (smooth = 1) to "32x slower" (smooth = 0).
            const float smooth = (ROPE_ORIGINAL_CTX / wavelen - ROPE_LOW_FREQ_FACTOR) /
                                 (ROPE_HIGH_FREQ_FACTOR - ROPE_LOW_FREQ_FACTOR);
            inv_freq[i] = (1.0f - smooth) * freq / ROPE_FACTOR + smooth * freq;
        }
    }
}

// One block per token position; one thread per (head, pair): n_heads * 32 threads.
//   blockIdx.x  = position p (0..T-1): how far to rotate
//   threadIdx.x = head * 32 + i: which head, and which pair (dims i and i + 32) inside it
__global__ void ropeKernel(bf16* x, const float* inv_freq, int n_heads) {
    const int p = blockIdx.x;
    const int head = threadIdx.x / ROPE_PAIRS;
    const int i = threadIdx.x % ROPE_PAIRS;

    // Start of this head's 64 values in row p.
    bf16* h = x + (size_t)p * n_heads * HEAD_DIM + (size_t)head * HEAD_DIM;

    float s, c;
    sincosf((float)p * inv_freq[i], &s, &c);  // angle = position x speed

    // Read both values before writing either: the second line needs the original a.
    const float a = __bfloat162float(h[i]);
    const float b = __bfloat162float(h[i + ROPE_PAIRS]);
    h[i] = __float2bfloat16(a * c - b * s);
    h[i + ROPE_PAIRS] = __float2bfloat16(b * c + a * s);
}

void rope(bf16* x, const float* d_inv_freq, int T, int n_heads) {
    ropeKernel<<<T, n_heads * ROPE_PAIRS>>>(x, d_inv_freq, n_heads);  // Q: 1024 threads, K: 256
    CUDA_CHECK(cudaGetLastError());
}
