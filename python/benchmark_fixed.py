#!/usr/bin/env python3
"""
Benchmark script for DenseCore inference engine.
Measures tokens/second on i7-10870H (AVX2).
"""

import argparse
import time

import densecore


def run_benchmark(model_id: str, max_tokens: int = 128, num_runs: int = 3):
    print("=" * 60)
    print(f"DenseCore Benchmark - {model_id}")
    print("=" * 60)

    # Download and load model
    try:
        model_path = densecore.download_model(model_id)
        tokenizer_id = model_id.replace("-GGUF", "").replace("-Instruct-GGUF", "-Instruct")
        print(f"Model: {model_path}")
        print(f"Tokenizer: {tokenizer_id}")

        model = densecore.DenseCore(model_path, hf_repo_id=tokenizer_id, threads=16)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Warmup
    print("\nWarmup (1 run)...")
    try:
        warmup_result = model.generate("Hi", max_tokens=8)
        print(f"Warmup output: {warmup_result[:50]}...")
    except Exception as e:
        print(f"Warmup failed: {e}")
        return None

    # Benchmark runs
    print(f"\nBenchmarking ({num_runs} runs, {max_tokens} tokens each)...")
    prompt = "Write an explanation of how neural networks work step by step:"

    total_tokens = 0
    total_time = 0.0
    ttft_samples = []
    decode_tps_samples = []

    for run in range(num_runs):
        tokens_generated = 0
        first_token_time = None
        start_time = time.perf_counter()

        try:
            # Use stream for accurate timing
            for token_text in model.stream(prompt, max_tokens=max_tokens):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                tokens_generated += 1
        except Exception as e:
            print(f"Run {run+1} error: {e}")
            continue

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        if first_token_time:
            ttft = first_token_time - start_time
            ttft_samples.append(ttft)

        if tokens_generated > 1 and first_token_time:
            decode_time = end_time - first_token_time
            decode_tps = (tokens_generated - 1) / decode_time if decode_time > 0 else 0
            decode_tps_samples.append(decode_tps)

        total_tokens += tokens_generated
        total_time += elapsed

        tps = tokens_generated / elapsed if elapsed > 0 else 0
        print(f"  Run {run+1}: {tokens_generated} tokens in {elapsed:.2f}s = {tps:.2f} tok/s")

    # Summary
    if total_tokens > 0 and total_time > 0:
        avg_tps = total_tokens / total_time
        avg_ttft = sum(ttft_samples) / len(ttft_samples) if ttft_samples else 0
        avg_decode_tps = (
            sum(decode_tps_samples) / len(decode_tps_samples) if decode_tps_samples else 0
        )

        print(f"\n{'='*60}")
        print("Results Summary:")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average TPS: {avg_tps:.2f} tok/s")
        print(f"  Average TTFT: {avg_ttft*1000:.1f}ms")
        print(f"  Decode TPS: {avg_decode_tps:.2f} tok/s")
        print(f"{'='*60}")

        return {
            "model": model_id,
            "avg_tps": avg_tps,
            "avg_ttft_ms": avg_ttft * 1000,
            "decode_tps": avg_decode_tps,
            "total_tokens": total_tokens,
        }

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DenseCore Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B-GGUF", help="Model ID")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per run")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of runs")
    args = parser.parse_args()

    run_benchmark(args.model, args.max_tokens, args.num_runs)
