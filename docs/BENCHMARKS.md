# DenseCore Model Benchmarks

Performance benchmarks for DenseCore inference engine on Intel i7-10870H (8 cores, 16 threads, AVX2).

## Summary Table

| Model | Size | Quantization | TTFT (ms) | Speed (tok/s) |
| :--- | :--- | :--- | ---: | ---: |
| **Qwen3-0.6B** | 0.6B | Q8_0 | 56.58 | 22.81 |
| **Qwen3-4B** | 4B | Q4_K_M | 186.41 | 8.38 |
| **Qwen3-8B** | 8B | Q4_K_M | 346.75 | 5.11 |
| Qwen2.5-0.5B | 0.5B | Q4_K_M | 50.85 | 29.66 |
| Qwen2.5-1.5B | 1.5B | Q4_K_M | 74.42 | 16.18 |
| **Llama-3.2-1B** | 1B | Q8_0 | 71.46 | 17.05 |
| TinyLlama-1.1B | 1.1B | Q4_K_M | 43.06 | 24.68 |

## Model Family Details

### Qwen3 Series ✅

Qwen3 models are fully supported after the GEMV op refactor (v0.3.1+).

| Variant | TTFT (ms) | Decode (tok/s) | Memory |
|---------|-----------|----------------|--------|
| Qwen3-0.6B (Q8_0) | 56.58 | 22.81 | ~600 MB |
| Qwen3-4B (Q4_K_M) | 186.41 | 8.38 | ~2.5 GB |
| Qwen3-8B (Q4_K_M) | 346.75 | 5.11 | ~4.5 GB |

### Llama 3.2 Series ✅

Meta's latest efficient Llama models with improved instruction following.

| Variant | TTFT (ms) | Decode (tok/s) | Memory |
|---------|-----------|----------------|--------|
| Llama-3.2-1B (Q8_0) | 71.46 | 17.05 | ~1.1 GB |

### Legacy Models ✅

| Variant | TTFT (ms) | Decode (tok/s) | Memory |
|---------|-----------|----------------|--------|
| Qwen2.5-0.5B (Q4_K_M) | 50.85 | 29.66 | ~350 MB |
| Qwen2.5-1.5B (Q4_K_M) | 74.42 | 16.18 | ~900 MB |
| TinyLlama-1.1B (Q4_K_M) | 43.06 | 24.68 | ~670 MB |

## Test Environment

- **CPU**: Intel Core i7-10870H @ 2.20GHz (8 cores / 16 threads)
- **RAM**: 32 GB DDR4
- **OS**: Ubuntu 24.04 (WSL2)
- **SIMD**: AVX2 (AVX-512 not available on this CPU)
- **Threads**: 8 (physical cores only)
- **Test Prompt**: "The capital of France is" (100 tokens generated)

## Notes

- **TTFT (Time to First Token)**: Includes model loading, prefill phase, and first token generation
- **Speed (tok/s)**: Sustained decode throughput after first token
- All benchmarks use HuggingFace cached GGUF models
- Qwen3 models require the GEMV refactor (v0.3.1+) using `GGML_OP_CUSTOM` for dimension-changing projections
- RoPE uses GGML native `ggml_rope_ext` for stability

## Supported Architectures

| Architecture | Status | Notes |
|--------------|--------|-------|
| Llama (1/2/3) | ✅ Stable | Full support including GQA |
| Qwen (2/2.5/3) | ✅ Stable | QK-norm, tied embeddings |
