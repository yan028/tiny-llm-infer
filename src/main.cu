// tiny-llm-infer entry point. Step 4: load the weights and prove the pointers are right.
#include <cstdio>
#include <string>

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

int main(int argc, char **argv) {
    std::string path = argc > 1 ? argv[1] : "models/llama-3.2-1b-instruct/model.safetensors";
    printGPU();

    Weights w = loadWeights(path);
    printGPU();

    std::printf("first 8 values (compare with python/inspect_safetensors.py):\n");
    printFirst("model.embed_tokens.weight", w.embed);
    printFirst("model.layers.15.mlp.down_proj.weight", w.layers[15].down);

    freeWeights(w);
    return 0;
}
