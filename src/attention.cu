#include <cfloat>

#include "attention.h"

constexpr int GQA_GROUP = N_Q_HEADS / N_KV_HEADS;  // 4 query heads share one key/value head
constexpr int SOFTMAX_THREADS = 256;

AttentionBuffers allocAttention() {
    AttentionBuffers a;
    const size_t tt = (size_t)N_Q_HEADS * MAX_PROMPT_LEN * MAX_PROMPT_LEN;  // 8.4M per buffer
    CUDA_CHECK(cudaMalloc(&a.scores, tt * sizeof(float)));                  // 32 MB
    CUDA_CHECK(cudaMalloc(&a.probs, tt * sizeof(bf16)));                    // 16 MB
    CUDA_CHECK(cudaMalloc(&a.out, (size_t)MAX_PROMPT_LEN * HIDDEN * sizeof(bf16)));
    return a;
}

void freeAttention(AttentionBuffers& a) {
    CUDA_CHECK(cudaFree(a.scores));
    CUDA_CHECK(cudaFree(a.probs));
    CUDA_CHECK(cudaFree(a.out));
    a = AttentionBuffers{};
}

// ---- Causal mask + softmax, one block per (head, query row).
// Row r of a head may only look at columns 0..r (itself and earlier tokens); later columns get
// probability exactly 0. Two block-wide reductions, like RMSNorm: first the row's max (subtracted
// before exp so nothing overflows), then the sum of exps (to divide by).
__device__ float blockReduce(float v, float* scratch, bool is_max) {
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other = scratch[threadIdx.x + stride];
            scratch[threadIdx.x] =
                is_max ? fmaxf(scratch[threadIdx.x], other) : scratch[threadIdx.x] + other;
        }
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();  // everyone has read scratch[0] before it is reused
    return result;
}

__global__ void causalSoftmaxKernel(const float* scores, bf16* probs, int T) {
    __shared__ float scratch[SOFTMAX_THREADS];
    const int row = blockIdx.x % T;              // query position
    const size_t base = (size_t)blockIdx.x * T;  // = (head * T + row) * T: start of this row
    const float* s = scores + base;
    bf16* p = probs + base;
    const int allowed = row + 1;  // columns 0..row are visible

    // A row (up to 512 values) can be longer than the block, so each thread strides over it.
    float m = -FLT_MAX;
    for (int j = threadIdx.x; j < allowed; j += blockDim.x) m = fmaxf(m, s[j]);
    m = blockReduce(m, scratch, true);

    float sum = 0.0f;
    for (int j = threadIdx.x; j < allowed; j += blockDim.x) sum += expf(s[j] - m);
    sum = blockReduce(sum, scratch, false);

    const float inv_sum = 1.0f / sum;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        p[j] = __float2bfloat16(j < allowed ? expf(s[j] - m) * inv_sum : 0.0f);
    }
}

void attention(cublasHandle_t handle, const bf16* q, const bf16* k, const bf16* v,
               AttentionBuffers& a, int T) {
    const float scale = 1.0f / sqrtf((float)HEAD_DIM);  // 1/8
    const float one = 1.0f, zero = 0.0f;

    for (int h = 0; h < N_Q_HEADS; ++h) {
        const int g = h / GQA_GROUP;         // GQA: which key/value head this query head uses
        const bf16* q_h = q + h * HEAD_DIM;  // [T, 64] slice, rows HIDDEN apart
        const bf16* k_g = k + g * HEAD_DIM;  // [T, 64] slice, rows KV_DIM apart
        float* s_h = a.scores + (size_t)h * T * T;  // [T, T]

        // 1. scores_h = q_h · k_gᵀ / 8   → [T queries, T keys], fp32 output.
        //    Column-major view: computes k_g · q_hᵀ = [T keys, T queries], which read back
        //    row-major is [T queries, T keys]. Leading dims pick the 64-column head slices.
        // clang-format off
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  T, T, HEAD_DIM,          // m = keys, n = queries, k = 64
                                  &scale,
                                  k_g, CUDA_R_16BF, KV_DIM,
                                  q_h, CUDA_R_16BF, HIDDEN,
                                  &zero,
                                  s_h, CUDA_R_32F, T,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        // clang-format on
    }

    // 2. Causal mask + softmax for all 32 × T rows at once.
    causalSoftmaxKernel<<<N_Q_HEADS * T, SOFTMAX_THREADS>>>(a.scores, a.probs, T);
    CUDA_CHECK(cudaGetLastError());

    for (int h = 0; h < N_Q_HEADS; ++h) {
        const int g = h / GQA_GROUP;
        const bf16* v_g = v + g * HEAD_DIM;
        const bf16* p_h = a.probs + (size_t)h * T * T;
        bf16* o_h = a.out + h * HEAD_DIM;  // this head's 64 columns of the output

        // 3. out_h = probs_h · v_g   → [T, 64], written straight into columns h*64..h*64+63
        //    of a.out (leading dim HIDDEN), so the heads end up concatenated for free.
        //    Column-major view: computes v_gᵀ · probs_hᵀ = [64, T].
        // clang-format off
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  HEAD_DIM, T, T,          // m = 64, n = queries, k = keys
                                  &one,
                                  v_g, CUDA_R_16BF, KV_DIM,
                                  p_h, CUDA_R_16BF, T,
                                  &zero,
                                  o_h, CUDA_R_16BF, HIDDEN,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        // clang-format on
    }
}
