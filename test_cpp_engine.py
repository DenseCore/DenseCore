
import sys
import os
import time

# Skip torch import in generate_output
import densecore.generate_output
densecore.generate_output.TORCH_AVAILABLE = False
densecore.generate_output.torch = None

from densecore import DenseCore

def test_engine():
    model_path = "/home/jaewook/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    print(f"Initializing engine with model: {model_path}")
    print("Skipping tokenizer (hf_repo_id=None)")
    
    # Initialize without tokenizer
    model = DenseCore(model_path, threads=8, hf_repo_id=None, verbose=True)
    
    print("Engine initialized successfully.")
    
    # Manual token IDs (Qwen encoding for "Hello")
    # Roughly: [1, 5, 2] idk. Just random tokens in range.
    prompt_ids = [128, 55, 1024, 77]
    
    print(f"Generating from prompt IDs: {prompt_ids}")
    
    output = model.generate(input_ids=prompt_ids, max_tokens=10)
    
    print(f"Generation output: {output}")
    print("Test passed.")

if __name__ == "__main__":
    test_engine()
