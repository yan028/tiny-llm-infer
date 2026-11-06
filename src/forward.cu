#include <algorithm>
#include <string>
#include <vector>

#include "dump.h"
#include "forward.h"
#include "kernels.h"

Activations allocActivations() {
    Activations a;
    auto alloc = [](bf16** p, size_t width) {
        CUDA_CHECK(cudaMalloc(p, (size_t)MAX_PROMPT_LEN * width * sizeof(bf16)));
    };
    alloc(&a.hidden, HIDDEN);
    alloc(&a.norm, HIDDEN);
    alloc(&a.q, HIDDEN);
    alloc(&a.k, KV_DIM);
    alloc(&a.v, KV_DIM);
    alloc(&a.o, HIDDEN);
    alloc(&a.gate, FFN);
    alloc(&a.up, FFN);
    a.att = allocAttention();
    CUDA_CHECK(cudaMalloc(&a.logits, VOCAB * sizeof(float)));

    float inv_freq[ROPE_PAIRS];
    computeRopeInvFreq(inv_freq);
    CUDA_CHECK(cudaMalloc(&a.inv_freq, sizeof(inv_freq)));
    CUDA_CHECK(cudaMemcpy(a.inv_freq, inv_freq, sizeof(inv_freq), cudaMemcpyHostToDevice));
    return a;
}

void freeActivations(Activations& a) {
    for (bf16* p : {a.hidden, a.norm, a.q, a.k, a.v, a.o, a.gate, a.up}) CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(a.logits));
    CUDA_CHECK(cudaFree(a.inv_freq));
    freeAttention(a.att);
    a = Activations{};
}

// One transformer layer: x = x + attention(norm(x)), then x = x + mlp(norm(x)).
// Exactly the Step 7–11 code, with layer L's weights instead of layer 0's.
static void layerForward(cublasHandle_t cb, const LayerWeights& lw, Activations& a, int T, int L,
                         bool dumps) {
    const bool detail = dumps && L == 0;  // intermediates only for layer 0, like reference.py
    const size_t n = (size_t)T * HIDDEN;

    // ---- attention half (Steps 7–10)
    rmsNorm(a.hidden, lw.attn_norm, a.norm, T);
    linear(cb, a.norm, lw.q, a.q, T, HIDDEN, HIDDEN);
    linear(cb, a.norm, lw.k, a.k, T, HIDDEN, KV_DIM);
    linear(cb, a.norm, lw.v, a.v, T, HIDDEN, KV_DIM);
    if (detail) {
        dumpBuffer("out/l0_attn_norm.bin", a.norm, n);
        dumpBuffer("out/l0_q.bin", a.q, n);
        dumpBuffer("out/l0_k.bin", a.k, (size_t)T * KV_DIM);
        dumpBuffer("out/l0_v.bin", a.v, (size_t)T * KV_DIM);
    }
    rope(a.q, a.inv_freq, T, N_Q_HEADS);
    rope(a.k, a.inv_freq, T, N_KV_HEADS);
    if (detail) {
        dumpBuffer("out/l0_q_rope.bin", a.q, n);
        dumpBuffer("out/l0_k_rope.bin", a.k, (size_t)T * KV_DIM);
    }
    attention(cb, a.q, a.k, a.v, a.att, T);
    linear(cb, a.att.out, lw.o, a.o, T, HIDDEN, HIDDEN);
    if (detail) {
        dumpFloats("out/l0_scores.bin", a.att.scores, (size_t)N_Q_HEADS * T * T);
        dumpBuffer("out/l0_probs.bin", a.att.probs, (size_t)N_Q_HEADS * T * T);
        dumpBuffer("out/l0_attn_out.bin", a.att.out, n);
        dumpBuffer("out/l0_o_proj.bin", a.o, n);
    }
    residualAdd(a.hidden, a.o, n);
    if (detail) dumpBuffer("out/l0_resid1.bin", a.hidden, n);

    // ---- MLP half (Step 11)
    rmsNorm(a.hidden, lw.mlp_norm, a.norm, T);
    linear(cb, a.norm, lw.gate, a.gate, T, HIDDEN, FFN);
    linear(cb, a.norm, lw.up, a.up, T, HIDDEN, FFN);
    if (detail) {
        dumpBuffer("out/l0_mlp_norm.bin", a.norm, n);
        dumpBuffer("out/l0_gate.bin", a.gate, (size_t)T * FFN);
        dumpBuffer("out/l0_up.bin", a.up, (size_t)T * FFN);
    }
    siluMul(a.gate, a.up, (size_t)T * FFN);
    linear(cb, a.gate, lw.down, a.o, T, FFN, HIDDEN);
    if (detail) {
        dumpBuffer("out/l0_silu_mul.bin", a.gate, (size_t)T * FFN);
        dumpBuffer("out/l0_down.bin", a.o, n);
    }
    residualAdd(a.hidden, a.o, n);
    if (dumps) dumpBuffer("out/layer" + std::to_string(L) + "_out.bin", a.hidden, n);
}

Prediction forward(cublasHandle_t cb, const Weights& w, const int* d_tokens, int T, Activations& a,
                   bool dumps) {
    // 1. Token ids → vectors (Step 6).
    embeddingGather(d_tokens, w.embed, a.hidden, T);
    if (dumps) dumpBuffer("out/embed.bin", a.hidden, (size_t)T * HIDDEN);

    // 2. 16 layers, each reading and updating the residual stream a.hidden.
    for (int L = 0; L < N_LAYERS; ++L) layerForward(cb, w.layers[L], a, T, L, dumps);

    // 3. Final RMSNorm (model.norm), for all rows so the dump matches reference.py.
    rmsNorm(a.hidden, w.final_norm, a.norm, T);
    if (dumps) dumpBuffer("out/final_norm.bin", a.norm, (size_t)T * HIDDEN);

    // 4. Logits for the LAST token only: it is the one predicting what comes next.
    //    Tied weights: the output layer is the embedding table, [128256, 2048], used as W.
    //    One row in, 128,256 scores out (fp32, so close scores are not merged by bf16 rounding).
    const bf16* last = a.norm + (size_t)(T - 1) * HIDDEN;
    linearF32(cb, last, w.embed, a.logits, 1, HIDDEN, VOCAB);

    // 5. Argmax on the CPU: copy 128,256 floats (0.5 MB) back and find the 5 largest.
    std::vector<float> logits(VOCAB);
    CUDA_CHECK(cudaMemcpy(logits.data(), a.logits, VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
    if (dumps) dumpFloats("out/logits_last.bin", a.logits, VOCAB);

    std::vector<int> ids(VOCAB);
    for (int i = 0; i < VOCAB; ++i) ids[i] = i;
    std::partial_sort(ids.begin(), ids.begin() + 5, ids.end(),
                      [&](int x, int y) { return logits[x] > logits[y]; });
    Prediction p;
    for (int i = 0; i < 5; ++i) {
        p.top_ids[i] = ids[i];
        p.top_logits[i] = logits[ids[i]];
    }
    return p;
}
