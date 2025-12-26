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

## Hardware Support

### Apple Silicon
- **Metal GPU**: SIMD-group optimized compute shaders, FlashAttention prefill kernel
- **ANE (Neural Engine)**: CoreML-backed MatMul with dynamic bucketing (1-32K tokens)
- **Accelerate AMX**: Apple's matrix coprocessor via BLAS (M1-M4)
- **NEON FP16**: Native `vfmaq_f16` for 2x throughput on M3/M4

### x86 (Intel/AMD)
- **AVX-512**: 512-bit vectors with 8-way unrolling (Skylake-X+)
- **AVX-512 VNNI**: `vpdpbusd` INT8 dot products (Ice Lake+, Zen4+)
- **AMX**: BF16 tile matrix operations (Sapphire Rapids+)
- **AVX2 + FMA**: 256-bit vectors (Haswell+, Zen+)
- **AVX**: 256-bit vectors (Sandy Bridge+)
- **SSE4.1**: 128-bit vectors (Penryn+)

### ARM64 (AWS Graviton, Qualcomm)
- **SVE (256-bit+)**: Scalable vectors with `svdot_s32` (Graviton 3/4)
- **NEON DOTPROD**: `vdotq_s32` INT8 dot products (Graviton 2+)
- **NEON FP16**: `vfmaq_f16` for 2x throughput (Graviton 3+)
- **NEON**: Fixed 128-bit vectors (all ARM64)

### Runtime Detection
DenseCore automatically detects and selects the optimal kernel at runtime:
```
Intel Xeon (Sapphire Rapids) → AMX > AVX-512 VNNI > AVX-512
Apple M3 Max                 → Metal GPU + ANE + NEON FP16
AWS Graviton3                → SVE DotProd > NEON DOTPROD > NEON
AMD Zen4                     → AVX-512 VNNI > AVX2
```

## Supported Backends

| Backend | Target Devices | Status |
|---------|----------------|--------|
| **CPU** | All (x86, ARM64) | ✅ Production |
| **Metal** | Apple Silicon GPU | ✅ Production |
| **ANE** | Apple Neural Engine (M1-M4) | ✅ Production |
| **Accelerate** | Apple AMX (via BLAS) | ✅ Production |
| **Hybrid Scheduler** | Apple Silicon (CPU+GPU+ANE) | ✅ Production |

> **Design Philosophy**: DenseCore focuses on **CPU-first** inference.


## Key Features

- **Production Ready**
  - Continuous Batching scheduler (vLLM-style)
  - Prometheus metrics, OpenTelemetry tracing
  - Helm charts, Docker multi-stage builds
  - **Redis**: Distributed rate limiting & API key storage
  - **KEDA**: Queue-based autoscaling
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

### Homebrew (macOS/Linux)

```bash
brew tap Jake-Network/densecore
brew install densecore

# Start chatting immediately
densecore run
```

### Python

```bash
pip install densecore                # Basic
pip install densecore[langchain]     # + LangChain support
```

## CLI Usage

The `densecore` CLI provides a beautiful terminal interface for local inference:

```bash
# Interactive chat with TUI (downloads model automatically)
densecore run Qwen/Qwen2.5-0.5B-Instruct-GGUF

# Production HTTP server
densecore serve --model ./model.gguf --port 8080
```

See [CLI Documentation](docs/CLI.md) for full details.

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

- [CLI Guide](docs/CLI.md)
- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Cloud-Native Guide](docs/CLOUD_NATIVE.md)
- [Apple Silicon Guide](docs/APPLE_SILICON.md)
- [Benchmarks](docs/BENCHMARKS.md)

## License

Apache 2.0
