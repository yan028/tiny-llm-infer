"""Golden reference: Llama 3.2 1B forward pass written out op by op in PyTorch.

Every intermediate tensor is dumped to dumps/<name>.bin (raw float32, row-major) and its shape
is recorded in dumps/index.json. Your CUDA kernels get compared against these files with
python/compare.py.

At the end the script runs Hugging Face's LlamaForCausalLM on the same tokens and checks that
this hand-written version agrees with it. If it does, the dumps are trustworthy.

Usage:
  python python/reference.py                                   # default prompt
  python python/reference.py --prompt "Explain KV cache in one sentence."
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaForCausalLM

MODEL_DIR = "models/llama-3.2-1b-instruct"
DUMP_DIR = "dumps"
# The Llama 3.2 chat template inserts today's date into the system prompt. Pin it so the
# token ids (and every dump) are the same no matter which day you run this.
DATE_STRING = "26 Jul 2024"

index = {}


def dump(name, t):
    """Save a tensor as raw float32 so C++/CUDA can compare against it byte for byte."""
    t = t.detach().float().contiguous().cpu()
    t.numpy().tofile(os.path.join(DUMP_DIR, f"{name}.bin"))
    index[name] = list(t.shape)


# ---------------------------------------------------------------- building blocks

def rms_norm(x, w, eps):
    # x: (T, 2048). Divide each row by its root-mean-square, then scale by the learned weight.
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * w


def rope_inv_freq(cfg, head_dim):
    """Rotation speed for each of the 32 dimension pairs, including Llama 3's frequency scaling."""
    base = cfg["rope_theta"]  # 500000
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    s = cfg.get("rope_scaling") or {}
    if s.get("rope_type", s.get("type")) == "llama3":
        factor, lo, hi = s["factor"], s["low_freq_factor"], s["high_freq_factor"]
        old_ctx = s["original_max_position_embeddings"]
        wavelen = 2 * math.pi / inv_freq
        lo_wavelen, hi_wavelen = old_ctx / lo, old_ctx / hi
        # Long wavelengths (slow rotations) get slowed down by `factor`; short ones are kept;
        # the band in between is blended smoothly.
        scaled = torch.where(wavelen > lo_wavelen, inv_freq / factor, inv_freq)
        smooth = (old_ctx / wavelen - lo) / (hi - lo)
        blended = (1 - smooth) * scaled / factor + smooth * scaled
        is_mid = (wavelen >= hi_wavelen) & (wavelen <= lo_wavelen)
        inv_freq = torch.where(is_mid, blended, scaled)
    return inv_freq


def apply_rope(x, cos, sin):
    """x: (T, n_heads, 64). Hugging Face layout: dimension i is paired with i + 32 ("rotate half"),
    NOT with i + 1 as in the course's kernel."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


# ---------------------------------------------------------------- forward pass

@torch.no_grad()
def forward(tokens, W, cfg, dump_layer=0):
    D = cfg["hidden_size"]                         # 2048
    n_q = cfg["num_attention_heads"]               # 32
    n_kv = cfg["num_key_value_heads"]              # 8
    hd = cfg.get("head_dim", D // n_q)             # 64
    eps = cfg["rms_norm_eps"]                      # 1e-5
    T = tokens.shape[0]

    def w(name):  # weights are stored in bf16; compute in fp32 for a clean reference
        return W[name].float()

    # Embedding lookup: one row of the 128256 x 2048 table per token.
    x = w("model.embed_tokens.weight")[tokens]                     # (T, 2048)
    dump("embed", x)

    # RoPE angles for positions 0..T-1, shared by every layer.
    inv_freq = rope_inv_freq(cfg, hd).to(x.device)
    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)                            # (T, 32)
    angles = torch.cat([angles, angles], dim=-1)                   # (T, 64)
    cos, sin = angles.cos(), angles.sin()

    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()

    for L in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{L}."
        d = (lambda name, t: dump(f"l{L}_{name}", t)) if L == dump_layer else (lambda name, t: None)

        # ---- attention branch
        h = rms_norm(x, w(p + "input_layernorm.weight"), eps);     d("attn_norm", h)
        q = h @ w(p + "self_attn.q_proj.weight").T                 # (T, 2048)
        k = h @ w(p + "self_attn.k_proj.weight").T                 # (T, 512)
        v = h @ w(p + "self_attn.v_proj.weight").T                 # (T, 512)
        d("q", q); d("k", k); d("v", v)

        q = apply_rope(q.view(T, n_q, hd), cos, sin)               # (T, 32, 64)
        k = apply_rope(k.view(T, n_kv, hd), cos, sin)              # (T, 8, 64)
        v = v.view(T, n_kv, hd)
        d("q_rope", q.reshape(T, -1)); d("k_rope", k.reshape(T, -1))

        # GQA: Q head i uses KV head i // 4.
        rep = n_q // n_kv
        k_full = k.repeat_interleave(rep, dim=1)                   # (T, 32, 64)
        v_full = v.repeat_interleave(rep, dim=1)

        qh, kh, vh = q.transpose(0, 1), k_full.transpose(0, 1), v_full.transpose(0, 1)  # (32, T, 64)
        scores = qh @ kh.transpose(1, 2) / math.sqrt(hd)           # (32, T, T)
        d("scores", scores)
        scores = scores.masked_fill(~causal, float("-inf"))        # causal mask
        probs = torch.softmax(scores, dim=-1);                     d("probs", probs)
        attn = (probs @ vh).transpose(0, 1).reshape(T, n_q * hd)   # (T, 2048), heads concatenated
        d("attn_out", attn)
        o = attn @ w(p + "self_attn.o_proj.weight").T;             d("o_proj", o)
        x = x + o;                                                 d("resid1", x)

        # ---- MLP branch (SwiGLU)
        h = rms_norm(x, w(p + "post_attention_layernorm.weight"), eps); d("mlp_norm", h)
        gate = h @ w(p + "mlp.gate_proj.weight").T                 # (T, 8192)
        up = h @ w(p + "mlp.up_proj.weight").T                     # (T, 8192)
        d("gate", gate); d("up", up)
        act = torch.nn.functional.silu(gate) * up;                 d("silu_mul", act)
        down = act @ w(p + "mlp.down_proj.weight").T;              d("down", down)
        x = x + down

        dump(f"layer{L}_out", x)

    x = rms_norm(x, w("model.norm.weight"), eps);                  dump("final_norm", x)
    # Tied weights: the output projection reuses the embedding table.
    logits = x @ w("model.embed_tokens.weight").T                  # (T, 128256)
    dump("logits", logits)
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    args = ap.parse_args()
    os.makedirs(DUMP_DIR, exist_ok=True)
    dev = "cuda"

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True, date_string=DATE_STRING,
    )
    if not isinstance(ids, list):        # some transformers versions return a dict-like object
        ids = list(ids["input_ids"])
    with open(os.path.join(DUMP_DIR, "tokens.txt"), "w") as f:
        f.write(" ".join(map(str, ids)) + "\n")
    print(f"prompt: {args.prompt!r}")
    print(f"{len(ids)} tokens: {ids}")

    # ---- our hand-written forward pass
    W = load_file(os.path.join(args.model_dir, "model.safetensors"), device=dev)  # bf16, ~2.5 GB
    tokens = torch.tensor(ids, device=dev)
    logits = forward(tokens, W, cfg)
    ours_next = int(logits[-1].argmax())
    ours_top5 = logits[-1].topk(5).indices.tolist()
    ours_l0 = torch.from_numpy(
        np.fromfile(os.path.join(DUMP_DIR, "layer0_out.bin"), dtype=np.float32)
    ).view(len(ids), -1)
    with open(os.path.join(DUMP_DIR, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(f"\nours: next token {ours_next} -> {tok.decode([ours_next])!r}")
    print(f"ours: top-5 {ours_top5} -> {[tok.decode([t]) for t in ours_top5]}")
    print(f"wrote {len(index)} tensors to {DUMP_DIR}/ (shapes in index.json)")

    del W
    torch.cuda.empty_cache()

    # ---- Hugging Face check (bf16, so expect small differences from our fp32 math)
    model = LlamaForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16).to(dev).eval()
    with torch.no_grad():
        out = model(tokens[None], output_hidden_states=True)
    hf_logits = out.logits[0].float()
    hf_next = int(hf_logits[-1].argmax())
    hf_top5 = hf_logits[-1].topk(5).indices.tolist()
    hf_l0 = out.hidden_states[1][0].float().cpu()

    def rel(a, b):
        return float((a - b).abs().max() / b.abs().max())

    print(f"\nHF:   next token {hf_next} -> {tok.decode([hf_next])!r}")
    print(f"HF:   top-5 {hf_top5}")
    print(f"layer0_out rel err vs HF: {rel(ours_l0, hf_l0):.2e}")
    print(f"last-row logits rel err vs HF: {rel(logits[-1].cpu(), hf_logits[-1].cpu()):.2e}")
    print("PASS: first token matches HF" if ours_next == hf_next else "FAIL: first token differs from HF")


if __name__ == "__main__":
    main()