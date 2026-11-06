// tiny-llm-infer: run Llama 3.2 1B Instruct on a prompt and predict the next token.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "forward.h"
#include "linear.h"
#include "model.h"

static void printGPU() {
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("GPU: %s | sm_%d%d | %d SMs | %.1f / %.1f GB free\n", p.name, p.major, p.minor,
                p.multiProcessorCount, free_b / 1e9, total_b / 1e9);
}

// Read the prompt's token ids (written by python/reference.py) from a text file.
static std::vector<int> readTokens(const std::string& path) {
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

int main(int argc, char** argv) {
    std::string path = argc > 1 ? argv[1] : "models/llama-3.2-1b-instruct/model.safetensors";
    printGPU();
    Weights w = loadWeights(path);

    std::vector<int> tokens = readTokens("dumps/tokens.txt");
    const int T = (int)tokens.size();
    int* d_tokens = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tokens, MAX_PROMPT_LEN * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens.data(), T * sizeof(int), cudaMemcpyHostToDevice));

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    Activations act = allocActivations();
    printGPU();

    // Run 1: with dumps, for checking against the reference.
    Prediction p = forward(cublas, w, d_tokens, T, act, /*dumps=*/true);
    std::printf("prompt: %d tokens\nnext token: %d\ntop-5:", T, p.top_ids[0]);
    for (int i = 0; i < 5; ++i) std::printf("  %d (%.2f)", p.top_ids[i], p.top_logits[i]);
    std::printf("\n");

    // Run 2: no dumps, timed with CUDA events (GPU-side timestamps).
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    Prediction p2 = forward(cublas, w, d_tokens, T, act, /*dumps=*/false);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::printf("prefill of %d tokens: %.2f ms (same token: %s)\n", T, ms,
                p2.top_ids[0] == p.top_ids[0] ? "yes" : "NO");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    freeActivations(act);
    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaFree(d_tokens));
    freeWeights(w);
    return 0;
}
