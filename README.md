<div align="center">

# 🚀 DenseCore

### **Run LLMs 10x Faster on CPU. No GPU Required.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/densecore?color=blue)](https://pypi.org/project/densecore/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](python/)
[![Go 1.24+](https://img.shields.io/badge/go-1.24+-00ADD8.svg)](server/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)
[![CI](https://github.com/Jake-Network/DenseCore/actions/workflows/ci.yml/badge.svg)](https://github.com/Jake-Network/DenseCore/actions)

**The high-performance open-source inference engine that makes Small Language Models production-ready on CPUs.**

[Quick Start](#-quick-start) • [Why DenseCore](#-why-densecore) • [Benchmarks](#-benchmarks) • [Roadmap](#-roadmap) • [Docs](docs/)

</div>

---

## ⚡ Quick Start

> [!NOTE]
> **Linux Only (Recommended)**: DenseCore is optimized for **Linux (Ubuntu 22.04+)**.

### 1. Python SDK

Designed for developers who want a "just works" experience.

```bash
pip install densecore
```

```python
import densecore

# 🚀 One line to load from HuggingFace (No manual conversion needed!)
model = densecore.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct-GGUF")

# ✨ Generate with industry-leading speed
print(model.generate("Explain quantum computing in one sentence."))
```

### 2. Docker Server (OpenAI-Compatible)

Deploy a production-ready API server in seconds.

```bash
docker run -p 8080:8080 \
  -v ./models:/models \
  -e MAIN_MODEL_PATH=/models/qwen.gguf \
  densecore/densecore:latest
```

Test it with `curl`:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## 🎯 Why DenseCore?

We built DenseCore because **GPUs are scarce and expensive**, but CPUs are everywhere.

<table>
<tr>
<td width="50%">

### 🔥 Blazing Fast
C++ core with **AVX-512 & AVX2** hand-tuned kernels. 
- **Runtime SIMD Dispatch**: Detects CPU at runtime (CPUID) and uses best available (AVX-512 → AVX2 → Scalar).
- **INT4 Quantization**: 7× compression, 5-6× faster inference.
- **Continuous Batching**: Maximizes throughput.
- **Graph Caching**: Reduces overhead by 40%.
- **OpenMP Threading**: Full CPU core utilization (16 threads on i7-10870H).


</td>
<td width="50%">

### 💰 Cost Efficiency
Slash your cloud bills by **90%**.
- Run production workloads on commodity VMs (`c7i.large`).
- Cost as low as **$0.01/hr**.
- No specialized hardware required.

</td>
</tr>
<tr>
<td width="50%">

### 🐍 Developer First
- **Native Python SDK**: No complex compilation.
- **HuggingFace Integration**: Direct downloads.
- **OpenAI API**: Drop-in replacement for existing apps.

</td>
<td width="50%">

### 🛡️ Production Ready
- **Kubernetes Native**: Helm charts & manifests included.
- **Observability**: Prometheus metrics & OpenTelemetry tracing.
- **Security**: Enterprise-grade API key management.
- **NUMA-Aware**: Optimized for multi-socket servers.

</td>
</tr>
</table>

---

## 📊 Benchmarks

**Environment:** 4-core vCPU, 8GB RAM (AWS c7i.large equivalent) — **No GPU**.

| Model | DenseCore | Transformers | Speedup |
|-------|-----------|--------------|---------| 
| **Qwen2.5-0.5B** | **28.5 tok/s** | ~3-4 tok/s | **🚀 8x** |
| **TinyLlama-1.1B** | **22.1 tok/s** | ~2 tok/s | **🔥 11x** |
| **Qwen3-4B** | **6.6 tok/s** | ~1.5 tok/s | **4x** |
| **Qwen3-8B** | **4.0 tok/s** | ~0.5 tok/s | **8x** |

> See [full benchmarks](docs/BENCHMARKS.md) for methodology.

---

## 🛠️ Model Optimization Tools

Includes built-in tools to compress models (Quantization & Pruning) for edge deployment.

```python
from densecore.quantize import quantize_model, Q4_K_M_CFG, INT4_PAPER_CFG

# Standard GGML quantization
quantize_model("model.gguf", "model-q4.gguf", config=Q4_K_M_CFG)

# 🚀 NEW: Custom INT4 with AVX512-optimized kernels (5-6× faster inference!)
quantize_model("model.gguf", "model-int4.gguf", config=INT4_PAPER_CFG(block_size=128))
```

📖 **[Optimization Guide →](docs/MODEL_OPTIMIZATION.md)** | **[INT4 Quantization →](docs/INT4_QUANTIZATION.md)**

---

## 📚 Documentation

| Component | Description |
|-----------|-------------|
| **[Python SDK](python/README.md)** | Full guide for Python developers. |
| **[API Reference](docs/API_REFERENCE.md)** | Detailed API docs for Python, C, and Go. |
| **[INT4 Quantization](docs/INT4_QUANTIZATION.md)** | AVX512-optimized INT4 with 5-6× speedup. |
| **[NUMA Optimization](docs/NUMA_OPTIMIZATION.md)** | Multi-socket server optimization guide. |
| **[Architecture](docs/ARCHITECTURE.md)** | Deep dive into the C++ internal design. |
| **[Deployment](docs/DEPLOYMENT.md)** | Docker and Kubernetes setup guides. |
| **[Contributing](CONTRIBUTING.md)** | How to build and contribute to DenseCore. |

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/Jake-Network/DenseCore.git
cd DenseCore
make lib && cd python && pip install -e ".[dev]"
pytest
```

---

<div align="center">

**Apache 2.0 License** • [Documentation](docs/) • [Discord](https://discord.gg/densecore) • [Twitter](https://twitter.com/densecore)

Made with ❤️ for the CPU-first AI era

</div>
