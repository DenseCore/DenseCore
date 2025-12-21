# DenseCore: High-Performance AVX2/AVX-512 LLM Inference

DenseCore is a specialized C++ inference engine optimized for Intel CPUs, delivering state-of-the-art performance for LLMs on consumer hardware (e.g., i7-10870H).

## 🚀 Key Features

- **Mixed-Precision Inference**: Q4_K_M optimized kernels.
- **Smart MatMul Dispatching**: Hybrid GEMV (Decode) / GEMM (Prefill) threading strategies.
- **Robust GGUF Support**: Auto-detection and patching of anomalous model tensors (e.g. Mixed MHA/GQA layers).
- **Zero-Dependency Core**: Pure C++ implementation with minimal external deps.

## 📊 Performance (AVX2)

| Model | Status | TPS (Decode) | Note |
|-------|--------|--------------|------|
| **TinyLlama-1.1B** | ✅ Verified | **~15.0** | Stable & Optimized |
| **Qwen2.5-0.5B** | ✅ Verified | **~25.0** | Recommended |
| **Qwen3-4B** | ⚠️ Experimental | - | Known issues on AVX2 (Bias Mismatch) |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full report.

## 🛠️ Usage

```python
from densecore import DenseCore

model = DenseCore("models/tinyllama.gguf")
print(model.generate("Hello world", max_tokens=64))
```

## 📦 Installation

```bash
pip install densecore
```

## 🏗️ Build from Source

```bash
mkdir build && cd build
cmake ..
make -j
```
