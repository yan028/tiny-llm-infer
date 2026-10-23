"""Parse a .safetensors file by hand (no safetensors library) and print what's inside.

File layout:
  [ 8 bytes: header size N (little-endian uint64) ][ N bytes: JSON header ][ tensor data ... ]

Usage:
  python python/inspect_safetensors.py models/llama-3.2-1b-instruct/model.safetensors
"""
import json
import struct
import sys

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4, "U8": 1}


def main(path):
    with open(path, "rb") as f:
        # 1. First 8 bytes: header size. '<Q' = little-endian unsigned 64-bit int.
        (header_size,) = struct.unpack("<Q", f.read(8))
        # 2. Next header_size bytes: JSON describing every tensor.
        header = json.loads(f.read(header_size))
        f.seek(0, 2)
        file_size = f.tell()

    data_start = 8 + header_size
    data_size = file_size - data_start
    metadata = header.pop("__metadata__", None)

    print(f"file size:   {file_size:,} bytes")
    print(f"header size: {header_size:,} bytes")
    print(f"data size:   {data_size:,} bytes")
    print(f"metadata:    {metadata}")
    print(f"tensors:     {len(header)}\n")

    # 3. Sort by start offset so we can check the data section is packed with no gaps.
    tensors = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    expected_start = 0
    total_params = 0
    problems = 0
    for name, info in tensors:
        start, end = info["data_offsets"]
        shape, dtype = info["shape"], info["dtype"]
        n = 1
        for d in shape:
            n *= d
        total_params += n

        if start != expected_start:
            print(f"  !! gap/overlap before {name}: expected {expected_start}, got {start}")
            problems += 1
        if end - start != n * DTYPE_BYTES[dtype]:
            print(f"  !! size mismatch for {name}: {end - start} bytes vs {n} x {dtype}")
            problems += 1
        expected_start = end

        # Print the non-layer tensors and layer 0 in full; skip the other layers' repeats.
        if ".layers." not in name or ".layers.0." in name:
            print(f"{name:55s} {dtype:5s} {str(shape):15s} [{start:>13,} .. {end:>13,})")

    print(f"\n(layers 1-15 omitted; same 9 tensors each)")
    print(f"total parameters: {total_params:,}")
    print(f"data section fully covered: {expected_start == data_size}")
    print(f"layout problems: {problems}")
    print(f"has lm_head.weight: {'lm_head.weight' in header}")

    # 4. Decode a few raw bf16 values by hand. bf16 is just the top 16 bits of a float32,
    #    so shift the 16 bits left by 16 and reinterpret as float32.
    print("\nfirst 8 values (save these to check your C++ loader in Step 4):")
    with open(path, "rb") as f:
        for name in ["model.embed_tokens.weight", "model.layers.15.mlp.down_proj.weight"]:
            start = header[name]["data_offsets"][0]
            f.seek(data_start + start)
            raw = struct.unpack("<8H", f.read(16))  # 8 little-endian uint16s
            vals = [struct.unpack("<f", struct.pack("<I", b << 16))[0] for b in raw]
            print(f"  {name}: {[f'{v:.6g}' for v in vals]}")


if __name__ == "__main__":
    main(sys.argv[1])