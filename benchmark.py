import argparse
import time
import sys
import platform
import os
try:
    import psutil
except ImportError:
    psutil = None

try:
    import cpuinfo
except ImportError:
    cpuinfo = None

from typing import Generator

# Mock class for testing without a real model/engine build
class MockDenseCore:
    def __init__(self, model_path: str, verbose: bool = True):
        self.model_path = model_path
        if verbose:
            print(f"[Mock] Initialized MockDenseCore with {model_path}")

    def stream(self, prompt: str, max_tokens: int = 128) -> Generator[str, None, None]:
        # Simulate processing time for first token (latency)
        time.sleep(0.5)  # 500ms TTFT
        yield "MOCK_START"
        
        # Simulate generation (throughput)
        # 128 tokens at ~20 tokens/sec -> 50ms per token
        for i in range(max_tokens - 1):
            time.sleep(0.05) 
            yield f" token_{i}"

def get_system_info():
    info = {
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": platform.processor(),
    }
    
    if psutil:
        info["Cores"] = f"{psutil.cpu_count(logical=False)} physical / {psutil.cpu_count(logical=True)} logical"
        info["RAM"] = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
    else:
        info["Cores"] = f"{os.cpu_count()} (logical)"
        info["RAM"] = "Unknown (pip install psutil)"

    # Try to get detailed CPU info including specific flags
    if cpuinfo:
        cpu_details = cpuinfo.get_cpu_info()
        info["CPU Brand"] = cpu_details.get("brand_raw", info["CPU"])
        flags = cpu_details.get("flags", [])
        
        simd_support = []
        if "avx512f" in flags: simd_support.append("AVX-512")
        if "avx2" in flags: simd_support.append("AVX2")
        if "fma" in flags: simd_support.append("FMA")
        if "avx" in flags: simd_support.append("AVX")
        
        info["SIMD Support"] = ", ".join(simd_support) if simd_support else "Basic"
    else:
        # Fallback using /proc/cpuinfo on Linux
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
                if "avx512" in content: info["SIMD Support"] = "AVX-512 (Likely)"
                elif "avx2" in content: info["SIMD Support"] = "AVX2 (Likely)"
                else: info["SIMD Support"] = "Unknown (pip install py-cpuinfo for details)"
        except:
             info["SIMD Support"] = "Unknown"

    return info

def run_benchmark(model_path: str, n_generate: int, use_mock: bool):
    print("=" * 60)
    print(f"DenseCore Benchmark Tool v1.0")
    print("=" * 60)
    
    # 1. System Info
    print(f"\n[System Information]")
    sys_info = get_system_info()
    for k, v in sys_info.items():
        print(f"  {k:<12}: {v}")

    # 2. Initialize Engine
    print(f"\n[Initialization]")
    print(f"  Model Path  : {model_path}")
    print(f"  Mode        : {'MOCK' if use_mock else 'REAL ACCELERATED'}")
    
    start_init = time.time()
    if use_mock:
        engine = MockDenseCore(model_path)
    else:
        try:
            from densecore import DenseCore
            engine = DenseCore(model_path=model_path, verbose=True)
        except ImportError:
            print("Error: 'densecore' package not found. Please install it or use --mock.")
            return
        except Exception as e:
            print(f"Error initializing engine: {e}")
            return
            
    init_time = time.time() - start_init
    print(f"  Init Time   : {init_time:.4f}s")

    # 3. Running Benchmark
    prompt = "The quick brown fox jumps over the lazy dog." * 10 # ~90-100 tokens
    print(f"\n[Benchmark Scenario]")
    print(f"  Prompt Len  : ~{len(prompt.split())} words")
    print(f"  Gen Tokens  : {n_generate}")
    print(f"  Warming up...")
    
    # Warmup (optional, maybe 5 tokens)
    # _ = list(engine.stream("warmup", max_tokens=2))

    print(f"  Generating...")
    
    tokens_generated = 0
    start_time = time.time()
    first_token_time = start_time # Default to start time to avoid NoneType error
    end_time = start_time
    
    try:
        # Use stream() for precise token timing
        # Assuming engine.stream(prompt, max_tokens=...)
        stream_gen = engine.stream(prompt, max_tokens=n_generate)
        
        for i, token in enumerate(stream_gen):
            current_time = time.time()
            if i == 0:
                first_token_time = current_time
            tokens_generated += 1
            # print(token, end="", flush=True) # Optional: print output
            
        end_time = time.time()
        
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        return

    # 4. Metrics Calculation
    if tokens_generated == 0:
        print("\nError: No tokens generated.")
        return

    total_time = end_time - start_time
    ttft = (first_token_time - start_time) * 1000 # ms
    
    # Throughput: (N - 1) / (TotalTime - TTFT)
    # We exclude the first token from throughput because it includes prefill/encode latency
    generate_time = end_time - first_token_time
    if tokens_generated > 1 and generate_time > 0:
        tps = (tokens_generated - 1) / generate_time
    else:
        tps = tokens_generated / total_time # Fallback for short gen

    print(f"\n\n{'=' * 60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print(f"  Tokens Gen  : {tokens_generated}")
    print(f"  Total Time  : {total_time:.4f} s")
    print(f"  TTFT        : {ttft:.2f} ms  (Time To First Token / Latency)")
    print(f"  Decode TPS  : {tps:.2f} tok/s (Tokens Per Second / Throughput)")
    print(f"{'=' * 60}")

    # Success/Failure criteria
    if tps < 1.0:
        print("\nWARNING: Performance is extremely low (< 1 TPS). Check AVX/Acceleration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DenseCore Benchmark Script")
    parser.add_argument("--model", type=str, default="model.gguf", help="Path to GGUF model file")
    parser.add_argument("--tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--mock", action="store_true", help="Run with mock engine (no model required)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model) and not args.mock:
        print(f"Warning: Model file '{args.model}' not found.")
        print("Running in MOCK mode for demonstration (pass valid path to test real engine).")
        args.mock = True

    run_benchmark(args.model, args.tokens, args.mock)
