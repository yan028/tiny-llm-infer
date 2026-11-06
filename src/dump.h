// Debug dumps: copy a GPU buffer to the CPU and write it raw, for python/compare.py.
#pragma once
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "model.h"

template <typename T>
inline void dumpRaw(const std::string& path, const T* d_ptr, size_t n) {
    CUDA_CHECK(cudaDeviceSynchronize());  // make sure the GPU has finished writing it
    std::vector<T> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(host.data()), n * sizeof(T));
}
// bf16 buffers: compare with --dtype bf16. fp32 buffers: compare with --dtype f32.
inline void dumpBuffer(const std::string& path, const bf16* d, size_t n) {
    dumpRaw(path, d, n);
}
inline void dumpFloats(const std::string& path, const float* d, size_t n) {
    dumpRaw(path, d, n);
}
