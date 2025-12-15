# DenseCore Documentation

Welcome to the DenseCore documentation — your guide to **blazing fast CPU inference for LLMs**.

---

## 📚 Documentation

### Getting Started
- **[Python SDK Guide](../python/README.md)** - Installation, quick start, and Python API
- **[LangChain Guide](../python/docs/LANGCHAIN_GUIDE.md)** - LangChain & LangGraph integration with tool calling
- **[API Reference](API_REFERENCE.md)** - Complete Python, C, and REST API reference

### Production
- **[Deployment Guide](DEPLOYMENT.md)** - Docker, Kubernetes, and cloud deployment
- **[Model Optimization](MODEL_OPTIMIZATION.md)** - Quantization and pruning techniques
- **[Benchmarks](BENCHMARKS.md)** - Performance data and methodology

### Reference
- **[Architecture](ARCHITECTURE.md)** - System design and internals
- **[HuggingFace to GGUF](HF_TO_GGUF.md)** - Convert any HuggingFace model
- **[Contributing](../CONTRIBUTING.md)** - Development setup and guidelines

---

## ⚡ Quick Start

```python
import densecore

model = densecore.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct-GGUF")
print(model.generate("Hello, world!"))
```

---

## 🗂️ Documentation Map

| I want to... | Read this |
|--------------|-----------| 
| Get started quickly | [Python SDK](../python/README.md) |
| Use with LangChain | [LangChain Guide](../python/docs/LANGCHAIN_GUIDE.md) |
| Deploy to production | [Deployment](DEPLOYMENT.md) |
| Use KEDA autoscaling | [Deployment → KEDA](DEPLOYMENT.md#keda-queue-based-autoscaling-recommended) |
| Hot-swap LoRA adapters | [Python SDK → LoRA](../python/README.md#-lora-runtime-switching) |
| Make models faster | [Model Optimization](MODEL_OPTIMIZATION.md) |
| Understand internals | [Architecture](ARCHITECTURE.md) |
| Contribute code | [Contributing](../CONTRIBUTING.md) |

---

## 🆘 Getting Help

1. **Search docs** using Ctrl+F
2. **Check [Python SDK troubleshooting](../python/README.md#troubleshooting)**
3. **Open a [GitHub Issue](https://github.com/Jake-Network/DenseCore/issues)**

---

**MIT License** • [GitHub](https://github.com/Jake-Network/DenseCore) • [Discord](https://discord.gg/densecore)
