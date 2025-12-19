import argparse
import time
import sys
import os
import platform
import random
from typing import Optional, List, Any

# Mock classes for standalone testing
class MockDenseCore:
    def __init__(self, model_path: str):
        pass
    def generate(self, prompt: str, max_tokens: int = 128) -> str:
        return " Mock response" * (max_tokens // 2)
    def stream(self, prompt: str, max_tokens: int = 128):
        time.sleep(0.1) # TTFT
        yield "Start"
        for i in range(max_tokens):
            time.sleep(0.01) # TPS
            yield f" token_{i}"

try:
    import densecore
    from densecore.integrations.langchain import DenseCoreChatModel
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
except ImportError:
    densecore = None # type: ignore
    DenseCoreChatModel = None # type: ignore
    
def get_simd_level():
    # Attempt to detect SIMD level from CPU info or engine
    # In a real scenario, this might come from densecore.get_device_info()
    try:
        if densecore:
            return "AVX-512" if "avx512" in platform.processor().lower() else "AVX2"
    except:
        pass
    return "Unknown/Mock"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except:
        return "Error"

def benchmark_raw(model_path: str, n_tokens: int = 128, mock: bool = False):
    print(f"Running Scenario A: Raw Engine Throughput...")
    if mock:
        engine = MockDenseCore(model_path)
    else:
        engine = densecore.DenseCore(model_path=model_path, verbose=False)

    # TTFT and TPS measurement
    start_time = time.time()
    first_token_time = None
    count = 0
    
    for i, _ in enumerate(engine.stream("Hello", max_tokens=n_tokens)):
        if i == 0:
            first_token_time = time.time()
        count += 1
    
    end_time = time.time()
    
    if first_token_time:
        ttft = (first_token_time - start_time) * 1000
        gen_time = end_time - first_token_time
        tps = (count - 1) / gen_time if count > 1 else 0
    else:
        ttft = 0
        tps = 0
        
    return ttft, tps

def benchmark_langchain(model_path: str, mock: bool = False):
    print(f"Running Scenario B: LangChain Agent Latency...")
    if mock:
        return 1234.5 # Mock latency ms
        
    if not DenseCoreChatModel:
        print("LangChain not installed, skipping.")
        return 0.0

    chat = DenseCoreChatModel(model_path=model_path, temperature=0)
    chat_with_tools = chat.bind_tools([calculator])
    
    start_time = time.time()
    try:
        # Simple math query that forces tool use
        response = chat_with_tools.invoke([HumanMessage(content="What is 25 * 4?")])
        # In a real agent loop, we'd execute the tool, but here we measure Model Latency for tool call generation
    except Exception as e:
        print(f"LangChain Error: {e}")
    
    end_time = time.time()
    return (end_time - start_time) * 1000

def main():
    parser = argparse.ArgumentParser(description="DenseCore Throughput & Latency Benchmark")
    parser.add_argument("--model", type=str, default="model.gguf", help="Path to GGUF model")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    args = parser.parse_args()
    
    # Header
    print(f"{'='*60}")
    print(f"DenseCore Performance Benchmark (SIMD: {get_simd_level()})")
    print(f"{'='*60}")
    
    # Run Scenario A
    ttft, tps = benchmark_raw(args.model, mock=args.mock or (densecore is None))
    
    # Run Scenario B
    agent_latency = benchmark_langchain(args.model, mock=args.mock or (densecore is None))
    
    # Output Table
    print(f"\n{'-'*60}")
    print(f"{'Metric':<25} | {'Value':<15} | {'Unit':<10}")
    print(f"{'-'*60}")
    print(f"{'Time To First Token':<25} | {ttft:>10.2f}      | ms")
    print(f"{'Generation Throughput':<25} | {tps:>10.2f}      | tok/s")
    print(f"{'Agent Response Latency':<25} | {agent_latency:>10.2f}      | ms")
    print(f"{'-'*60}")

if __name__ == "__main__":
    main()
