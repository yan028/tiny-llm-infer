// Step 4: load model.safetensors onto the GPU.
//
// File layout:  [8 bytes: header size N][N bytes: JSON header][tensor data...]
// Plan: one cudaMalloc for the whole data section, stream the file into it in chunks,
// then every tensor pointer is just gpu_block + its start offset from the header.
#include <chrono>
#include <fstream>
#include <unordered_map>
#include <vector>

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "json.hpp"
#include "model.h"

using json = nlohmann::json;

namespace {

struct TensorInfo {
    size_t start, end;  // byte offsets relative to the data section
    std::vector<int64_t> shape;
};

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "loader: %s\n", msg.c_str());
    std::exit(1);
}

// Look up a tensor by name, check it is bf16 with the shape we expect, return its GPU pointer.
bf16* bind(const std::unordered_map<std::string, TensorInfo>& tensors, char* gpu_block,
           const std::string& name, const std::vector<int64_t>& expected_shape) {
    auto it = tensors.find(name);
    if (it == tensors.end()) die("missing tensor " + name);
    const TensorInfo& t = it->second;
    if (t.shape != expected_shape) die("unexpected shape for " + name);
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    if ((int64_t)(t.end - t.start) != n * (int64_t)sizeof(bf16))
        die("byte size mismatch for " + name);
    return reinterpret_cast<bf16*>(gpu_block + t.start);
}

}  // namespace

Weights loadWeights(const std::string& path) {
    auto t0 = std::chrono::steady_clock::now();

    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);

    // 1. Header size: 8 bytes, little-endian uint64 (x86 is little-endian, so read it directly).
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char*>(&header_size), 8);

    // 2. JSON header.
    std::string header_str(header_size, '\0');
    f.read(header_str.data(), header_size);
    json header = json::parse(header_str);

    f.seekg(0, std::ios::end);
    const size_t file_size = f.tellg();
    const size_t data_start = 8 + header_size;
    const size_t data_size = file_size - data_start;

    std::unordered_map<std::string, TensorInfo> tensors;
    for (auto& [name, info] : header.items()) {
        if (name == "__metadata__") continue;
        if (info["dtype"].get<std::string>() != "BF16") die(name + " is not BF16");
        TensorInfo t;
        t.start = info["data_offsets"][0].get<size_t>();
        t.end = info["data_offsets"][1].get<size_t>();
        t.shape = info["shape"].get<std::vector<int64_t>>();
        if (t.end > data_size) die(name + " runs past the end of the file");
        tensors[name] = t;
    }
    std::printf("header: %zu tensors, data section %.2f GB\n", tensors.size(), data_size / 1e9);

    // 3. One GPU allocation for everything.
    Weights w;
    w.bytes = data_size;
    CUDA_CHECK(cudaMalloc(&w.gpu_block, data_size));

    // 4. Stream the data section to the GPU in 256 MB chunks through a pinned host buffer.
    //    Pinned (page-locked) memory lets cudaMemcpy run at full PCIe speed and avoids
    //    needing 2.47 GB of host RAM at once.
    constexpr size_t CHUNK = 256ull << 20;
    char* staging = nullptr;
    CUDA_CHECK(cudaMallocHost(&staging, CHUNK));
    f.seekg(data_start);
    for (size_t done = 0; done < data_size;) {
        size_t n = std::min(CHUNK, data_size - done);
        f.read(staging, n);
        if ((size_t)f.gcount() != n) die("short read from " + path);
        CUDA_CHECK(
            cudaMemcpy(static_cast<char*>(w.gpu_block) + done, staging, n, cudaMemcpyHostToDevice));
        done += n;
    }
    CUDA_CHECK(cudaFreeHost(staging));

    // 5. Point every field at its tensor inside the block, checking shapes as we go.
    char* base = static_cast<char*>(w.gpu_block);
    w.embed = bind(tensors, base, "model.embed_tokens.weight", {VOCAB, HIDDEN});
    w.final_norm = bind(tensors, base, "model.norm.weight", {HIDDEN});
    for (int L = 0; L < N_LAYERS; ++L) {
        const std::string p = "model.layers." + std::to_string(L) + ".";
        LayerWeights& l = w.layers[L];
        l.attn_norm = bind(tensors, base, p + "input_layernorm.weight", {HIDDEN});
        l.q = bind(tensors, base, p + "self_attn.q_proj.weight", {HIDDEN, HIDDEN});
        l.k = bind(tensors, base, p + "self_attn.k_proj.weight", {KV_DIM, HIDDEN});
        l.v = bind(tensors, base, p + "self_attn.v_proj.weight", {KV_DIM, HIDDEN});
        l.o = bind(tensors, base, p + "self_attn.o_proj.weight", {HIDDEN, HIDDEN});
        l.mlp_norm = bind(tensors, base, p + "post_attention_layernorm.weight", {HIDDEN});
        l.gate = bind(tensors, base, p + "mlp.gate_proj.weight", {FFN, HIDDEN});
        l.up = bind(tensors, base, p + "mlp.up_proj.weight", {FFN, HIDDEN});
        l.down = bind(tensors, base, p + "mlp.down_proj.weight", {HIDDEN, FFN});
    }
    // 2 global + 16 layers x 9 = 146. Anything else in the file would mean a different model.
    if (tensors.size() != 2 + 9 * N_LAYERS) die("expected 146 tensors");

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("loaded %.2f GB to GPU in %.2f s\n", data_size / 1e9, secs);
    return w;
}

void freeWeights(Weights& w) {
    if (w.gpu_block) CUDA_CHECK(cudaFree(w.gpu_block));
    w = Weights{};
}
