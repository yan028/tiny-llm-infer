// tiny-llm-infer entry point. Step 4: load the weights and prove the pointers are right.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "model.h"

static void printGPU() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("GPU: %s | sm_%d%d | %d SMs | %.1f / %.1f GB free\n", p.name, p.major, p.minor,
                p.multiProcessorCount, free_b / 1e9, total_b / 1e9);
}

// Copy the first n bf16 values of a GPU tensor back to the host and print them as floats.
// Same format as python/inspect_safetensors.py, so the two outputs can be compared by eye.
static void printFirst(const char *name, const bf16 *gpu_ptr, int n = 8) {
    bf16 host[8];
    CUDA_CHECK(cudaMemcpy(host, gpu_ptr, n * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::printf("  %s: [", name);
    for (int i = 0; i < n; ++i) std::printf("'%.6g'%s", __bfloat162float(host[i]), i + 1 < n ? ", " : "");
    std::printf("]\n");
}

// Step 5: read the prompt's token ids (written by python/reference.py) from a text file.
static std::vector<int> readTokens(const std::string &path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "cannot open %s (run python/reference.py first)\n", path.c_str());
        std::exit(1);
    }
    std::vector<int> ids;
    for (int id; f >> id;) {  // read whitespace-separated integers until the file ends
        if (id < 0 || id >= VOCAB) {
            std::fprintf(stderr, "token id %d is outside the vocabulary\n", id);
            std::exit(1);
        }
        ids.push_back(id);
    }
    if (ids.empty() || (int)ids.size() > MAX_PROMPT_LEN) {
        std::fprintf(stderr, "prompt has %zu tokens; need 1..%d\n", ids.size(), MAX_PROMPT_LEN);
        std::exit(1);
    }
    return ids;
}

int main(int argc, char **argv) {
    std::string path = argc > 1 ? argv[1] : "models/llama-3.2-1b-instruct/model.safetensors";
    printGPU();

    Weights w = loadWeights(path);
    printGPU();

    std::printf("first 8 values (compare with python/inspect_safetensors.py):\n");
    printFirst("model.embed_tokens.weight", w.embed);
    printFirst("model.layers.15.mlp.down_proj.weight", w.layers[15].down);

    // ---- Step 5: token ids → GPU
    std::vector<int> tokens = readTokens("dumps/tokens.txt");
    const int T = (int)tokens.size();

    int *d_tokens = nullptr;  // d_ prefix = lives on the device (GPU)
    CUDA_CHECK(cudaMalloc(&d_tokens, MAX_PROMPT_LEN * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens.data(), T * sizeof(int), cudaMemcpyHostToDevice));

    // Round trip: copy back from the GPU and check nothing changed.
    std::vector<int> check(T);
    CUDA_CHECK(cudaMemcpy(check.data(), d_tokens, T * sizeof(int), cudaMemcpyDeviceToHost));
    std::printf("tokens: %d on GPU, round trip %s\n", T, check == tokens ? "OK" : "MISMATCH");
    std::printf("  first 5: %d %d %d %d %d ... last: %d\n", tokens[0], tokens[1], tokens[2],
                tokens[3], tokens[4], tokens[T - 1]);

    CUDA_CHECK(cudaFree(d_tokens));
    freeWeights(w);
    return 0;
}