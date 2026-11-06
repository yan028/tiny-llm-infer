"""Turn token ids into text with the model's tokenizer.

Usage:
  python python/decode.py 791
  python python/decode.py 791 60704 50100
"""

import sys

from transformers import AutoTokenizer

MODEL_DIR = "models/llama-3.2-1b-instruct"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
for i in map(int, sys.argv[1:]):
    print(f"{i:>7} -> {tok.decode([i])!r}")
