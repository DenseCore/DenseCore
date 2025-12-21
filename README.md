# DenseCore: High-Performance CPU Inference for LLMs

DenseCore is a specialized C++ inference engine optimized for Intel/AMD CPUs, delivering state-of-the-art performance for LLMs on consumer and server hardware.

## 🚀 Key Features

- **SIMD Optimized**: AVX2, AVX-512, and ARM NEON support
- **Quantization**: Q4_K_M, Q8_0 with native vec_dot kernels
- **Smart Dispatching**: Hybrid GEMV (Decode) / GEMM (Prefill) strategies
- **Cloud Native**: Kubernetes, Helm charts, Docker multi-stage builds
- **GGUF Native**: Direct HuggingFace model loading

## 📊 Performance Benchmarks

*Intel i7-10870H (8 cores, AVX2)*

| Model | Quantization | TTFT (ms) | Speed (tok/s) |
|-------|--------------|-----------|---------------|
| **Qwen3-0.6B** | Q8_0 | 56.58 | 22.81 |
| **Qwen3-4B** | Q4_K_M | 186.41 | 8.38 |
| **Qwen3-8B** | Q4_K_M | 346.75 | 5.11 |
| **Llama-3.2-1B** | Q8_0 | 71.46 | 17.05 |
| Qwen2.5-0.5B | Q4_K_M | 50.85 | 29.66 |
| TinyLlama-1.1B | Q4_K_M | 43.06 | 24.68 |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full report.

## 🛠️ Quick Start

```python
from densecore import DenseCore

# Load from HuggingFace
model = DenseCore(hf_repo_id="Qwen/Qwen3-0.6B-GGUF")

# Generate text
for token in model.generate("The capital of France is", max_tokens=64, stream=True):
    print(token, end="", flush=True)
```

## 📦 Installation

```bash
pip install densecore
```

## 🐳 Docker

```bash
docker pull densecore/densecore:latest
docker run -p 8080:8080 -v /models:/app/models densecore/densecore
```

## ☸️ Kubernetes

```bash
helm install densecore ./charts/densecore \
  --set model.repository=Qwen/Qwen3-0.6B-GGUF
```

## 🏗️ Build from Source

```bash
# Build C++ core
cd core && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install Python bindings
cd ../../python && pip install -e .
```

## 🎯 Supported Models

| Family | Status | Examples |
|--------|--------|----------|
| **Llama** | ✅ Stable | Llama-2, Llama-3, Llama-3.2 |
| **Qwen** | ✅ Stable | Qwen2.5, Qwen3 (0.6B-8B) |
| **TinyLlama** | ✅ Stable | TinyLlama-1.1B |

## 📄 License

Apache 2.0
