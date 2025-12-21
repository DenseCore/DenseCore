#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/python
VENV_PYTHON=$(pwd)/.venv/bin/python3
TIMEOUT_CMD="timeout 300s"

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
run_bench "Qwen3-4B" "/home/jaewook/.cache/huggingface/hub/models--unsloth--Qwen3-4B-GGUF/snapshots/22c9fc8a8c7700b76a1789366280a6a5a1ad1120/Qwen3-4B-Q4_K_M.gguf" "Qwen/Qwen3-4B"

# 3. Qwen3-8B
run_bench "Qwen3-8B" "/home/jaewook/.cache/huggingface/hub/models--tensorblock--Qwen_Qwen3-8B-GGUF/snapshots/9041b4398744d5c412426e50e01168292857d35f/Qwen3-8B-Q4_K_M.gguf" "Qwen/Qwen3-8B"

# 4. TinyLlama-1.1B
run_bench "TinyLlama-1.1B" "/home/jaewook/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/52e7645ba7c309695bec7ac98f4f005b139cf465/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 5. Llama-3.2-3B
run_bench "Llama-3.2-3B" "/home/jaewook/.cache/huggingface/hub/models--bartowski--Llama-3.2-3B-Instruct-GGUF/snapshots/5ab33fa94d1d04e903623ae72c95d1696f09f9e8/Llama-3.2-3B-Instruct-Q4_K_M.gguf" "meta-llama/Llama-3.2-3B-Instruct"
