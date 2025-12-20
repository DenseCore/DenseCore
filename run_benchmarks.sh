#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/python
VENV_PYTHON=$(pwd)/.venv/bin/python3
TIMEOUT_CMD="timeout 60s"

run_bench() {
    MODEL_NAME=$1
    MODEL_PATH=$2
    REPO_ID=$3
    echo "----------------------------------------------------------------"
    echo "Benchmarking $MODEL_NAME..."
    $TIMEOUT_CMD $VENV_PYTHON benchmarks/benchmark_throughput.py --model "$MODEL_PATH" --repo_id "$REPO_ID" --threads 8 --n-predict 32 > "bench_${MODEL_NAME}.log" 2>&1
    RET=$?
    if [ $RET -eq 124 ]; then
        echo "Create Timeout (Hang)"
    elif [ $RET -eq 0 ]; then
        grep "Generation Throughput" "bench_${MODEL_NAME}.log" || echo "Failed (No output)"
    else
        echo "Failed (Exit Code: $RET)"
    fi
}

# 1. Qwen2.5-0.5B
run_bench "Qwen2.5-0.5B" "/home/jaewook/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf" "Qwen/Qwen2.5-0.5B-Instruct"

# 2. Qwen3-4B
run_bench "Qwen3-4B" "/home/jaewook/.cache/huggingface/hub/models--Qwen--Qwen3-4B-GGUF/snapshots/bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf" "Qwen/Qwen3-4B"

# 3. Qwen3-8B
run_bench "Qwen3-8B" "/home/jaewook/.cache/huggingface/hub/models--Qwen--Qwen3-8B-GGUF/snapshots/7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf" "Qwen/Qwen3-8B"

# 4. TinyLlama-1.1B
# Warning: Update path if needed, assuming standard location or just skipping if not sure.
# Using a placeholder path based on pattern if not known, or skipping.
# Let's try known Llama-3 instead first.

# 5. Llama-3-8B
# Assuming path from previous context or skipping if not locally cached/known.
# Based on BENCHMARKS.md, Llama-3-8B was tested. I'll assume I can find it or skip it.
# I'll stick to the Qwens and update document based on those for now as they are the primary targets of the request.
