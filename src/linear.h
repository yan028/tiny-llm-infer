// Linear layers (matrix multiplies) through cuBLAS.
#pragma once
#include <cublas_v2.h>

#include "model.h"

// Stop with file:line if a cuBLAS call fails (cuBLAS has its own status codes, not cudaError_t).
#define CUBLAS_CHECK(call)                                                                       \
    do {                                                                                         \
        cublasStatus_t st_ = (call);                                                             \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                      \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d: %s\n", (int)st_, __FILE__, __LINE__, \
                         #call);                                                                 \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

// y = x · Wᵀ, all row-major bf16:
//   x: [T, in_dim]      (T token rows)
//   W: [out_dim, in_dim] (as stored in safetensors: one row per output feature)
//   y: [T, out_dim]
// Same as `h @ W.T` in python/reference.py. Accumulates in fp32.
void linear(cublasHandle_t handle, const bf16* x, const bf16* W, bf16* y, int T, int in_dim,
            int out_dim);
