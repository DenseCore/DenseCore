# DenseCore Documentation

Welcome to the DenseCore documentation — your guide to **high-performance CPU inference for LLMs**.

---

## 📚 Documentation

### Getting Started
- **[Python SDK Guide](../python/README.md)** - Installation, quick start, and Python API
- **[LangChain Guide](../python/docs/LANGCHAIN_GUIDE.md)** - LangChain & LangGraph integration with tool calling
- **[API Reference](API_REFERENCE.md)** - Complete Python, C, and REST API reference

### Performance
- **[Benchmarks](BENCHMARKS.md)** - Performance data across Qwen3, Llama, and more
- **[Architecture](ARCHITECTURE.md)** - System design and SIMD optimizations
- **[NUMA Optimization](NUMA_OPTIMIZATION.md)** - Multi-socket server tuning

### Production
- **[Deployment Guide](DEPLOYMENT.md)** - Docker, Kubernetes, and cloud deployment
- **[Model Optimization](MODEL_OPTIMIZATION.md)** - Quantization and pruning techniques
- **[HuggingFace to GGUF](HF_TO_GGUF.md)** - Convert any HuggingFace model

### Contributing
- **[Contributing Guide](../CONTRIBUTING.md)** - Development setup and guidelines
- **[Code of Conduct](../CODE_OF_CONDUCT.md)** - Community standards

---

## ⚡ Quick Start

```python
from densecore import DenseCore

model = DenseCore(hf_repo_id="Qwen/Qwen3-0.6B-GGUF")
for token in model.generate("Hello, world!", max_tokens=64, stream=True):
    print(token, end="", flush=True)
```

---

## 🗂️ Documentation Map

| I want to... | Read this |
|--------------|-----------| 
| Get started quickly | [Python SDK](../python/README.md) |
| See benchmark results | [Benchmarks](BENCHMARKS.md) |
| Use with LangChain | [LangChain Guide](../python/docs/LANGCHAIN_GUIDE.md) |
| Deploy to production | [Deployment](DEPLOYMENT.md) |
| Use KEDA autoscaling | [Deployment → KEDA](DEPLOYMENT.md#keda-queue-based-autoscaling-recommended) |
| Optimize for multi-socket | [NUMA Optimization](NUMA_OPTIMIZATION.md) |
| Make models faster | [Model Optimization](MODEL_OPTIMIZATION.md) |
| Understand internals | [Architecture](ARCHITECTURE.md) |
| Contribute code | [Contributing](../CONTRIBUTING.md) |

---

## 🆘 Getting Help

1. **Search docs** using Ctrl+F
2. **Check [Python SDK troubleshooting](../python/README.md#troubleshooting)**
3. **Open a [GitHub Issue](https://github.com/Jake-Network/DenseCore/issues)**

---

**Apache 2.0 License** • [GitHub](https://github.com/Jake-Network/DenseCore) • [Discord](https://discord.gg/densecore)
