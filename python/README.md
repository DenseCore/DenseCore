# DenseCore Python SDK

High-performance CPU inference client for Python.

## Installation

```bash
pip install densecore
pip install densecore[langchain]  # Optional: LangChain support
```

## Quick Start

```python
from densecore import DenseCore

model = DenseCore(hf_repo_id="Qwen/Qwen3-0.6B-GGUF")

# Streaming generation
for token in model.generate("Hello world", stream=True):
    print(token, end="", flush=True)
```

## Features

- **Text Generation**: `model.generate()`
- **Chat**: `model.chat(messages)`
- **Embeddings**: `model.embed(text)`
- **Async Support**: `await model.stream_async()`

## LangChain Integration

```python
from densecore.integrations import DenseCoreChatModel

chat = DenseCoreChatModel(hf_repo_id="Qwen/Qwen3-0.6B-GGUF")
chat.invoke("Hello!")
```

## Troubleshooting

- **Memory Error**: Try a quantized model (e.g., `Q8_0` or `Q4_K_M`).
- **Illegal Instruction**: Your CPU may lack AVX2 support. Rebuild with `cmake -DDENSECORE_AVX2=OFF`.
