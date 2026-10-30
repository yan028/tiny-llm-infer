#!/usr/bin/env bash
# Format and lint the whole project. Run from the repo root: ./lint.sh
#   ./lint.sh          format files in place, then lint
#   ./lint.sh --check  only report; change nothing (use before committing)
set -u
cd "$(dirname "$0")"
CHECK=0; [[ "${1:-}" == "--check" ]] && CHECK=1
status=0

echo "== Python: ruff"
if [[ $CHECK == 1 ]]; then
    ruff format --check python/ || status=1
    ruff check python/ || status=1
else
    ruff format python/
    ruff check --fix python/ || status=1
fi

echo "== C++/CUDA: clang-format"
if [[ $CHECK == 1 ]]; then
    clang-format --dry-run --Werror src/*.cu src/*.h || status=1
else
    clang-format -i src/*.cu src/*.h && echo "formatted src/"
fi

echo "== C++/CUDA: clang-tidy (host code only)"
# clang parses .cu files in CUDA mode. --cuda-host-only skips GPU code generation, and the
# warning flag is because clang may not know CUDA 13 yet; it still understands the code.
for f in src/*.cu; do
    clang-tidy --quiet "$f" -- -x cuda --cuda-host-only --cuda-path=/usr/local/cuda \
        -Wno-unknown-cuda-version -std=c++17 -Iinclude -Isrc 2>/dev/null \
        | grep -E "^.*src/.*(warning|error):" && status=1
done

[[ $status == 0 ]] && echo "All clean." || echo "Issues found (see above)."
exit $status
