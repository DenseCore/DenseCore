<div align="center">

# 🚀 DenseCore

### **High-Performance CPU Inference Engine for LLMs**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/densecore?color=blue)](https://pypi.org/project/densecore/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](python/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

**Make Small Language Models production-ready on CPUs with AVX-512 optimization.**

[Quick Start](#-quick-start) • [Why DenseCore](#-why-densecore) • [LangChain](#-langchain-integration) • [Benchmarks](#-benchmarks)

</div>

---

## ⚡ Quick Start

### Installation

```bash
# Install from source (requires C++ compiler)
pip install .

# Or install from PyPI
pip install densecore
```

### Python SDK

```python
from densecore import AutoModel

# Auto-download and load model
model = AutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct-GGUF")

# Generate text
print(model.generate("Hello world!"))
```

### 🦜🔗 LangChain Integration

Build agents in 5 lines of code:

```python
from densecore.langchain import DenseCoreChatModel
from langchain_core.messages import HumanMessage

chat = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF")
chat_with_tools = chat.bind_tools([calculator_tool])

response = chat_with_tools.invoke([HumanMessage(content="What is 25 * 4?")])
print(response.content)
```

---

## 🎯 Why DenseCore?

We built DenseCore because **GPUs are scarce**, but CPUs are everywhere.

<table>
<tr>
<td width="50%">

### 🔥 High-Performance Core
- **AVX-512 & AMX Optimized**: Hand-tuned Assembly kernels.
- **PagedAttention on CPU**: Zero-copy KV cache management.
- **Runtime SIMD Dispatch**: Auto-selects AVX-512/AVX2/Scalar.
- **Continuous Batching**: High throughput for server workloads.

</td>
<td width="50%">

### 🛠️ Developer Ready
- **OpenAI API Compatible**: Drop-in replacement server.
- **LangChain/LangGraph Ready**: Native tool calling support.
- **HuggingFace Integration**: Seamless model downloading.
- **Easy Installation**: `pip install .` works out of the box.

</td>
</tr>
</table>

---

## 📊 Benchmarks

**Environment:** 4-core vCPU, 8GB RAM (AWS c7i.large) — **No GPU**.

| Model | DenseCore (AVX-512) | Transformers (Standard) | Speedup |
|-------|---------------------|--------------------------|---------| 
| **Qwen2.5-0.5B** | **Pending** | ~3-4 tok/s | **-** |
| **Qwen3-4B** | **Pending** | ~1.5 tok/s | **-** |
| **TinyLlama-1.1B** | **22.1** | ~2 tok/s | **11x** |

> **Status Update (2025-12-20):** AVX2 kernel hardening applied. All SIMD unit tests pass (32/32).
> Multi-threaded inference on AVX2 hardware (Intel Comet Lake) has a known threading issue being investigated.
> Single-thread mode (`--threads 1`) or AVX-512 hardware works correctly.

> Run the benchmark script to test your machine: `python benchmarks/benchmark_throughput.py --model model.gguf`

---

## 🛠️ Model Optimization Tools

Includes built-in tools to compress models (Quantization & Pruning) for edge deployment.

```python
from densecore.quantize import quantize_model, INT4_PAPER_CFG

# Custom INT4 with AVX512-optimized kernels (5-6× faster inference!)
quantize_model("model.gguf", "model-int4.gguf", config=INT4_PAPER_CFG(block_size=128))
```

📖 **[Optimization Guide →](docs/MODEL_OPTIMIZATION.md)**

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
