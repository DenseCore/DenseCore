import time
import argparse
import densecore

def run_benchmark(model_id, duration_seconds=10):
    print(f"Benchmarking model: {model_id}")
    # Use native API
    try:
        model_path = densecore.download_model(model_id)
        # Use original repo for tokenizer
        tokenizer_id = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Loading model from {model_path} with tokenizer from {tokenizer_id}")
        model = densecore.DenseCore(model_path, hf_repo_id=tokenizer_id, threads=4)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Warming up...")
    try:
        model.generate("Hello", max_tokens=1) # Warmup
    except Exception as e:
        print(f"Warmup warning: {e}")

    print("Starting benchmark...")
    prompt = "Write a long essay about the history of computing."
    start_time = time.time()
    token_count = 0
    
    try:
        # Use generate for reliable token counting
        result = model.generate(prompt, max_tokens=256)
        # Rough token count based on output (space-separated)
        token_count = len(result.split())
    except Exception as e:
        print(f"Error during generation: {e}")

    elapsed = time.time() - start_time
    # Avoid div by zero
    if elapsed > 0:
        tps = token_count / elapsed
    else:
        tps = 0
    print(f"\nGenerated {token_count} tokens in {elapsed:.2f} seconds")
    print(f"TPS: {tps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct-GGUF", help="Model to benchmark")
    args = parser.parse_args()
    
    run_benchmark(args.model)
