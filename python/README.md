# DenseCore Python SDK

The official Python client for DenseCore — **high-performance CPU & Apple Silicon inference for LLMs**.

[![PyPI](https://img.shields.io/pypi/v/densecore)](https://pypi.org/project/densecore/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE)

**Key Features:**
- **Multi-Platform**: Intel (AVX-512), AMD (AVX2), Apple Silicon (Metal/ANE/AMX), AWS Graviton (SVE/DotProd)
- **Production Ready**: Continuous Batching, OpenAI-compatible API
- **LangChain Native**: First-class LangChain/LangGraph support with tool calling


---

## Installation

```bash
pip install densecore
```

**Optional extras:**
```bash
pip install densecore[langchain]  # LangChain/LangGraph support
pip install densecore[full]       # All optional dependencies
```

---

## Quick Start

```python
from densecore import DenseCore

# Load from HuggingFace (auto-downloads GGUF)
model = DenseCore(hf_repo_id="Qwen/Qwen3-0.6B-GGUF")

# Generate with streaming
for token in model.generate("Explain quantum computing:", max_tokens=100, stream=True):
    print(token, end="", flush=True)
```

---

## Core Features

### Text Generation

```python
response = model.generate(
    prompt="Write a haiku about AI:",
    max_tokens=100,
    temperature=0.8,
    top_p=0.95
)
```

### Streaming

```python
# Sync
for token in model.stream("Count to 10:"):
    print(token, end="", flush=True)

# Async
async for token in model.stream_async("Count to 10:"):
    print(token, end="", flush=True)
```

### Chat Completions

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]
response = model.chat(messages)
```

### Embeddings

```python
# Single
embedding = model.embed("Hello world", normalize=True)

# Batch
embeddings = model.embed_batch(["Hello", "World"], normalize=True)
```

---

## 🤖 LangChain Integration

DenseCore has first-class LangChain and LangGraph support with **tool calling**.

### Basic Usage

```python
from densecore.integrations import DenseCoreChatModel
from langchain_core.messages import HumanMessage

chat = DenseCoreChatModel(hf_repo_id="Qwen/Qwen3-0.6B-GGUF")
response = chat.invoke([HumanMessage("Hello!")])
print(response.content)
```

### Tool Calling

```python
from densecore.integrations import DenseCoreChatModel
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

chat = DenseCoreChatModel(hf_repo_id="Qwen/Qwen3-0.6B-GGUF")
chat_with_tools = chat.bind_tools([calculator])

response = chat_with_tools.invoke([HumanMessage("What is 25 * 4?")])
if response.tool_calls:
    print(f"Tool call: {response.tool_calls}")
```

📖 **[Full LangChain Guide →](docs/LANGCHAIN_GUIDE.md)**

---

## 🔄 LoRA Runtime Switching

Hot-swap LoRA adapters without reloading the base model:

```python
model = DenseCore(hf_repo_id="Qwen/Qwen3-0.6B-GGUF")

# Load multiple adapters
model.load_lora("customer_support.gguf", scale=0.8, name="support")
model.load_lora("code_assistant.gguf", scale=1.0, name="code")

# Switch adapters at runtime (instant, no model reload)
model.enable_lora("code")
print(model.generate("def quicksort(arr):"))

model.enable_lora("support")
print(model.generate("How can I help you today?"))
```

---

## API Reference

### DenseCore

```python
class DenseCore:
    def __init__(main_model_path, threads=0, hf_repo_id=None)

    @classmethod
    def from_pretrained(repo_id, filename=None) -> DenseCore

    def generate(prompt, max_tokens=256, temperature=0.8, ...) -> str
    def stream(prompt, **kwargs) -> Iterator[str]
    async def stream_async(prompt, **kwargs) -> AsyncIterator[str]
    def chat(messages, **kwargs) -> str
    def embed(text, normalize=True) -> np.ndarray
    def embed_batch(texts, normalize=True) -> np.ndarray
    def get_metrics() -> Dict[str, Any]
```

### LangChain

```python
from densecore.integrations import (
    DenseCoreLLM,           # LangChain LLM
    DenseCoreChatModel,     # LangChain ChatModel with tool calling
    create_react_agent,     # LangGraph ReAct agent
)
```

Full API docs: [API Reference](../docs/API_REFERENCE.md)

---

## Troubleshooting

### "Memory allocation failed"
Use a quantized model: `Qwen3-0.6B-Q8_0.gguf` instead of FP16

### "Illegal instruction"
Your CPU doesn't support AVX2. Rebuild with `cmake -DDENSECORE_AVX2=OFF`

### "Garbage output"
Use HuggingFace tokenizer: `DenseCore(hf_repo_id="Qwen/Qwen3-0.6B")`

---

## Links

- [Main Repository](https://github.com/Jake-Network/DenseCore)
- [API Reference](../docs/API_REFERENCE.md)
- [LangChain Guide](docs/LANGCHAIN_GUIDE.md)
- [Deployment Guide](../docs/DEPLOYMENT.md)

---

**Apache 2.0 License** • [Issues](https://github.com/Jake-Network/DenseCore/issues)
