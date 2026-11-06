#include "linear.h"

// cuBLAS assumes COLUMN-major matrices; our buffers are ROW-major. Reading a row-major [r, c]
// buffer as column-major gives its transpose [c, r] for free, so we never move any data:
//
//   cuBLAS sees W (row-major [out, in]) as Wᵀ  [in, out]  → ask for OP_T to get W  [out, in]
//   cuBLAS sees x (row-major [T, in])   as xᵀ  [in, T]    → OP_N keeps it          [in, T]
//   it computes  W · xᵀ = [out, T] in column-major,
//   which, read back row-major, is exactly y = x · Wᵀ  [T, out].
void linear(cublasHandle_t handle, const bf16* x, const bf16* W, bf16* y, int T, int in_dim,
            int out_dim) {
    const float alpha = 1.0f;  // y = alpha * (W · xᵀ) + beta * y
    const float beta = 0.0f;   // beta = 0: overwrite y instead of adding to it
    // clang-format off
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_T, CUBLAS_OP_N,  // transpose W, keep x
                              out_dim, T, in_dim,        // m, n, k: result is m×n, inner dim k
                              &alpha,
                              W, CUDA_R_16BF, in_dim,    // A = W; leading dim = its row length
                              x, CUDA_R_16BF, in_dim,    // B = x
                              &beta,
                              y, CUDA_R_16BF, out_dim,   // C = y
                              CUBLAS_COMPUTE_32F,        // multiply-adds accumulate in fp32
                              CUBLAS_GEMM_DEFAULT));
    // clang-format on
}

void linearF32(cublasHandle_t handle, const bf16* x, const bf16* W, float* y, int T, int in_dim,
               int out_dim) {
    const float alpha = 1.0f, beta = 0.0f;
    // clang-format off
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, T, in_dim, &alpha,
                              W, CUDA_R_16BF, in_dim,
                              x, CUDA_R_16BF, in_dim,
                              &beta,
                              y, CUDA_R_32F, out_dim,    // only difference: fp32 output
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    // clang-format on
}
