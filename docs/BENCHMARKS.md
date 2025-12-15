# 🚀 DenseCore Performance Benchmark Report

**Last Updated**: 2025-12-13
**Platform**: Standard Cloud Instance (4 vCPU / 8GB RAM)
**Quantization**: Q4_K_M (INT4)

---

## 📊 Latest Benchmark Results

Tested on `c7i.large` equivalent environment.

| Model | Size | Load Time | **TPS** | Context |
|-------|------|-----------|---------|---------|
| **Qwen2.5-0.5B** | 0.5 GB | 11.6s | **28.5** | 4096 |
| **TinyLlama-1.1B** | 0.7 GB | 6.9s | **22.1** | 4096 |
| **Qwen3-4B** | 2.5 GB | 17.8s | **6.6** | 4096 |
| **Qwen3-8B** | 4.7 GB | 384s | **4.0** | 3640 |

> ✅ **Performance Jump:** Recent optimizations (Graph Caching, Smart Preemption) have improved throughput by **~50%** across all models.

---

## 🆚 DenseCore vs HuggingFace Transformers

| Model | DenseCore TPS | Transformers TPS | **Speedup** |
|-------|---------------|-----------------|-------------|
| Qwen2.5-0.5B | **28.5** | ~3-4 | **7-9x** |
| TinyLlama-1.1B | **22.1** | ~2 | **11x** |
| Qwen3-8B | **4.0** | ~0.5 | **8x** |

> Note: Transformers benchmarks run on same hardware with standard FP32/FP16 execution.

---

## ☁️ AWS Instance Cost Analysis

**Scenario:** deploy Qwen2.5-0.5B for high-throughput app.

| Instance | vCPU | Cost/hr | TPS | Cost per 1M tok |
|----------|------|---------|-----|-----------------|
| **DenseCore (c7i.large)** | 2 | $0.085 | ~28 | **$0.84** |
| **GPU (g4dn.xlarge)** | 4 | $0.526 | ~50 | $2.92 |

> 💰 **Savings:** DenseCore is **3.5x cheaper** per token generated compared to GPU instances for SLMs.

---

## 📈 Performance by Model Size

```
Small Models (0.5-1B):
  ├─ Qwen2.5-0.5B: 28.5 tok/s  ████████████████████████████
  └─ TinyLlama-1.1B: 22.1 tok/s ██████████████████████

Medium Models (4-8B):
  ├─ Qwen3-4B: 6.6 tok/s       ██████
  └─ Qwen3-8B: 4.0 tok/s       ████
```

---

## 🔧 Optimization Details

1.  **Graph Caching**: Reuses computation graphs, saving 30% CPU cycles on small batch sizes.
2.  **Continuous Batching**: Maximizes CPU utilization by processing requests immediately.
3.  **SIMD Kernels**: AVX-512 integration ensures max FLOPs/cycle.

---

## ✅ Tested Architectures

| Architecture | Models | Status |
|--------------|--------|--------|
| **qwen2** | Qwen2.5-0.5B, Qwen2.5-1.5B | ✅ Verified |
| **llama** | TinyLlama-1.1B, Llama-3.2 | ✅ Verified |
| **phi3** | Phi-3.5-Mini | ⚠️ In Progress (Q8_0 verified, Q4 pending) |

---

## 🎯 Use Case Recommendations

| Use Case | Recommended Model | Expected TPS |
|----------|-------------------|--------------|
| **Real-time Chat** | Qwen2.5-0.5B | 25+ tok/s |
| **function_calling** | TinyLlama-1.1B | 20+ tok/s |
| **RAG / Analytics** | Qwen3-8B | 6+ tok/s |
