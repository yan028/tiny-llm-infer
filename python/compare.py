"""Compare a tensor your engine produced against the golden reference dump.

Both files are raw arrays with no header. The reference is always float32 (written by
reference.py); your engine's file can be float32 or bf16 (dump the GPU buffer as-is).

Usage:
  python python/compare.py dumps/embed.bin out/embed.bin --dtype bf16
  python python/compare.py dumps/l0_q.bin out/l0_q.bin --dtype f32 --tol 2e-2
"""

import argparse
import json
import os
import sys

import numpy as np


def load(path, dtype):
    if dtype == "f32":
        return np.fromfile(path, dtype=np.float32)
    if dtype == "bf16":
        # bf16 = top 16 bits of a float32: widen to uint32, shift left 16, reinterpret as float32.
        raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
        return raw.view(np.float32)
    sys.exit(f"unknown dtype {dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="reference .bin from dumps/ (float32)")
    ap.add_argument("test", help="your engine's .bin")
    ap.add_argument(
        "--dtype", default="bf16", choices=["bf16", "f32"], help="dtype of the test file"
    )
    ap.add_argument("--tol", type=float, default=2e-2, help="max relative error to call it a PASS")
    args = ap.parse_args()

    ref = load(args.ref, "f32")
    test = load(args.test, args.dtype)
    if ref.size != test.size:
        sys.exit(f"FAIL: size mismatch: ref has {ref.size} values, test has {test.size}")

    # Masked attention scores contain -inf; treat matching infinities as equal.
    both_inf = np.isinf(ref) & np.isinf(test) & (np.sign(ref) == np.sign(test))
    if np.isnan(test).any():
        print(f"warning: {int(np.isnan(test).sum())} NaNs in test")
    with np.errstate(invalid="ignore"):
        diff = np.where(both_inf, 0.0, np.abs(ref - test))
    finite_ref = np.abs(ref[np.isfinite(ref)])
    scale = finite_ref.max() if finite_ref.size else 1.0

    worst = int(np.nanargmax(diff))
    rel = float(np.nanmax(diff) / scale) if scale > 0 else float(np.nanmax(diff))

    # Show the worst element as a (row, col) using the shape from dumps/index.json if available.
    where = f"flat index {worst}"
    idx_path = os.path.join(os.path.dirname(args.ref), "index.json")
    name = os.path.splitext(os.path.basename(args.ref))[0]
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            shape = json.load(f).get(name)
        if shape:
            idx = tuple(int(i) for i in np.unravel_index(worst, shape))
            where = f"index {idx} of shape {tuple(shape)}"

    print(f"{name}: {ref.size:,} values")
    print(f"  max abs err:  {np.nanmax(diff):.4e}   at {where}")
    print(f"                ref {ref[worst]:.6g}, test {test[worst]:.6g}")
    print(f"  mean abs err: {np.nanmean(diff):.4e}")
    print(f"  max rel err:  {rel:.4e}   (relative to max |ref| = {scale:.4g})")
    print("  PASS" if rel <= args.tol and not np.isnan(test).any() else f"  FAIL (tol {args.tol})")


if __name__ == "__main__":
    main()
