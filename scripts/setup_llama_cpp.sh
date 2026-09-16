#!/usr/bin/env bash
# Setup llama.cpp for GGUF conversion and quantisation.
# Run once per machine: bash scripts/setup_llama_cpp.sh
# On Colab: !bash scripts/setup_llama_cpp.sh

set -euo pipefail

LLAMA_DIR="vendor/llama.cpp"
REPO="https://github.com/ggerganov/llama.cpp"

echo "=== llama.cpp setup ==="

# Clone or update.
if [ -d "$LLAMA_DIR/.git" ]; then
    echo "Updating existing llama.cpp..."
    git -C "$LLAMA_DIR" pull --ff-only
else
    echo "Cloning llama.cpp..."
    mkdir -p vendor
    git clone --depth 1 "$REPO" "$LLAMA_DIR"
fi

# Build.
echo "Building llama.cpp..."
cd "$LLAMA_DIR"

# Use CUDA if available, otherwise CPU.
if command -v nvcc &>/dev/null; then
    echo "CUDA detected — building with GGML_CUDA=1"
    cmake -B build -DGGML_CUDA=1 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF
else
    echo "No CUDA — building CPU-only"
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF
fi

cmake --build build --config Release -j"$(nproc 2>/dev/null || echo 4)"

cd -

# Verify.
QUANTIZE_BIN="$LLAMA_DIR/build/bin/llama-quantize"
if [ ! -f "$QUANTIZE_BIN" ]; then
    # Older llama.cpp used a different bin path.
    QUANTIZE_BIN="$LLAMA_DIR/build/bin/quantize"
fi

if [ -f "$QUANTIZE_BIN" ]; then
    echo "✓ llama-quantize found: $QUANTIZE_BIN"
else
    echo "✗ Build may have failed — llama-quantize not found."
    echo "  Check cmake output above."
    exit 1
fi

# Install Python conversion dependencies.
echo "Installing llama.cpp Python requirements..."
pip install -q -r "$LLAMA_DIR/requirements.txt"

echo ""
echo "=== llama.cpp setup complete ==="
echo "LLAMA_DIR: $(realpath $LLAMA_DIR)"
echo "Run export with: alignforge export gguf --run <dpo-run-id>"
