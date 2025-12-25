# DenseCore: High-Performance CPU Inference for LLMs

**Cloud-Native, GPU-Free LLM Inference Engine**

DenseCore is a production-grade inference engine optimized for CPU and Apple Silicon, delivering enterprise-level performance without expensive GPUs.

## Why DenseCore?

| Feature | DenseCore | llama.cpp | vLLM |
|---------|-----------|-----------|------|
| **Cloud Native** | ✅ K8s/Helm/Docker | △ Server mode | ✅ Native |
| **GPU Required** | ❌ CPU-first | ❌ | ⚠️ NVIDIA only |
| **Continuous Batching** | ✅ vLLM-style | ❌ Static | ✅ |
| **Apple Silicon** | ✅ Metal + ANE + AMX | ✅ Metal | ❌ |
| **ARM (Graviton)** | ✅ SVE + DotProd | △ NEON only | ❌ |

## Key Features

- **Multi-Platform Optimization**
  - Intel/AMD: AVX-512, AVX2, AMX (Sapphire Rapids)
  - Apple M-series: Metal GPU, ANE (Neural Engine), Accelerate AMX
  - AWS Graviton: SVE, NEON DotProd, FP16
- **Production Ready**
  - Continuous Batching scheduler (vLLM-style)
  - Prometheus metrics, OpenTelemetry tracing
  - Helm charts, Docker multi-stage builds
- **Developer Friendly**
  - Python SDK with LangChain/LangGraph integration
  - OpenAI-compatible REST API
  - Direct HuggingFace GGUF loading

## Performance

*Intel i7-10870H (8 cores, AVX2)*

| Model | Quant | TTFT (ms) | Speed (tok/s) |
|-------|-------|-----------|---------------|
| **Qwen3-0.6B** | Q8_0 | 56 | 22.8 |
| **Qwen3-4B** | Q4_K_M | 186 | 8.4 |
| **Llama-3.2-1B** | Q8_0 | 71 | 17.1 |

## Quick Start

```python
from densecore import DenseCore

model = DenseCore(hf_repo_id="Qwen/Qwen3-0.6B-GGUF")

for token in model.generate("The capital of France is", max_tokens=64, stream=True):
    print(token, end="", flush=True)
```

## Installation

```bash
pip install densecore                # Basic
pip install densecore[langchain]     # + LangChain support
```

## Deployment

**Docker:**
```bash
docker run -p 8080:8080 densecore/densecore:latest
```

**Kubernetes:**
```bash
helm install densecore ./charts/densecore \
  --set model.repository=Qwen/Qwen3-0.6B-GGUF
```

## Build from Source

```bash
cd core && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Apple Silicon Guide](docs/APPLE_SILICON.md)
- [Benchmarks](docs/BENCHMARKS.md)

## License

Apache 2.0
