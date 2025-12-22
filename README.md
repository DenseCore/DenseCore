# DenseCore

**High-Performance CPU Inference Engine for LLMs**

DenseCore is a C++ inference engine optimized for consumer hardware (AVX2/AVX-512) and Apple Silicon. It provides an OpenAI-compatible server and a Pythonic SDK.

## Key Features

- **Efficient**: Paged KV Cache and continuous batching.
- **Optimized**: AVX-512/AMX on x86, Metal/ANE on macOS.
- **Accessible**: Native Python SDK (`pip install densecore`).
- **Production-Ready**: Go server with metrics and health checks.

## Performance

Tested on Intel i7-10870H (8 cores, AVX2):

| Model | Quantization | TTFT (ms) | Speed (tok/s) |
|-------|--------------|-----------|---------------|
| **Qwen3-0.6B** | Q8_0 | 56.58 | **22.81** |
| **Qwen3-4B** | Q4_K_M | 186.41 | **8.38** |
| **Llama-3.2-1B** | Q8_0 | 71.46 | **17.05** |

> See [BENCHMARKS.md](docs/BENCHMARKS.md) for full report.

## Quick Start

```bash
pip install densecore
```

```python
from densecore import DenseCore

# Download and load model (auto-detects hardware)
model = DenseCore(hf_repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF")

# Stream generation
for token in model.generate("Explain quantum computing", stream=True):
    print(token, end="", flush=True)
```

## Hardware Support

| Platform | Acceleration | Status |
|----------|--------------|--------|
| **Linux/WSL** | AVX2, AVX-512 | ✅ Stable |
| **macOS** | Metal GPU, ANE | ✅ Stable |

## Deployment

**Docker**:
```bash
docker run -p 8080:8080 densecore/densecore:latest
```

**Kubernetes**:
```bash
helm install densecore ./charts/densecore
```

## Documentation

- [Python SDK Guide](python/README.md)
- [Architecture Internals](docs/ARCHITECTURE.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Contributing](CONTRIBUTING.md)

## License
Apache 2.0
