# DenseCore Python SDK

The official Python client for DenseCore—**blazing fast CPU inference for LLMs**.

[![PyPI](https://img.shields.io/pypi/v/densecore)](https://pypi.org/project/densecore/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE)
---
## Installation
```bash
pip install densecore
```
**Optional extras:**

...

**Apache 2.0 License** • [Issues](https://github.com/Jake-Network/DenseCore/issues) • [Discord](https://discord.gg/densecore)
```bash
pip install densecore[langchain]  # LangChain/LangGraph support
pip install densecore[full]       # All optional dependencies
```

---

## Quick Start

```python
import densecore

# Load from HuggingFace (auto-downloads GGUF)
model = densecore.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct-GGUF")

# Generate
response = model.generate("Explain quantum computing in simple terms.")
print(response)
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

chat = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
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

chat = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
chat_with_tools = chat.bind_tools([calculator])

response = chat_with_tools.invoke([HumanMessage("What is 25 * 4?")])
if response.tool_calls:
    print(f"Tool call: {response.tool_calls}")
```

### ReAct Agents

```python
from densecore.integrations import DenseCoreChatModel, create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

llm = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
agent = create_react_agent(llm, [search])

result = agent.invoke({"messages": [HumanMessage("Find info about DenseCore")]})
```

📖 **[Full LangChain Guide →](docs/LANGCHAIN_GUIDE.md)**

---

## 🔄 LoRA Runtime Switching

Hot-swap LoRA adapters without reloading the base model—ideal for multi-tenant serving:

```python
model = densecore.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct-GGUF")

# Load multiple adapters
model.load_lora("customer_support.gguf", scale=0.8, name="support")
model.load_lora("code_assistant.gguf", scale=1.0, name="code")

# Switch adapters at runtime (instant, no model reload)
model.enable_lora("code")
print(model.generate("def quicksort(arr):"))

model.enable_lora("support")
print(model.generate("How can I help you today?"))

# Use base model (disable all adapters)
model.disable_lora()
```

---

## Advanced Features

### Model Quantization

```python
from densecore.quantize import quantize_model, Q4_K_M_CFG

quantize_model(
    input_path="model-fp16.gguf",
    output_path="model-q4km.gguf",
    config=Q4_K_M_CFG  # 4x smaller, ~5% quality loss
)
```

### Model Pruning

```python
from densecore.prune import prune_model, DEPTH_PRUNE_50_CFG

prune_model(
    input_path="llama-7b.gguf",
    output_path="llama-3.5b.gguf",
    config=DEPTH_PRUNE_50_CFG  # Remove 50% of layers
)
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
    create_tool_node,       # LangGraph tool execution
    create_densecore_node,  # LangGraph node factory
)
```

Full API docs: [API Reference](../docs/API_REFERENCE.md)

---

## Troubleshooting

### "Memory allocation failed"
Use a quantized model: `qwen-7b-q4.gguf` instead of `qwen-7b-fp16.gguf`

### "Illegal instruction"  
Your CPU doesn't support AVX2. Rebuild with `cmake -DDENSECORE_AVX2=OFF`

### "Garbage output"
Use HuggingFace tokenizer: `DenseCore("model.gguf", hf_repo_id="Qwen/Qwen2.5")`

---

## Links

- [Main Repository](https://github.com/Jake-Network/DenseCore)
- [API Reference](../docs/API_REFERENCE.md)
- [LangChain Guide](docs/LANGCHAIN_GUIDE.md)
- [Deployment Guide](../docs/DEPLOYMENT.md)

---

**Apache 2.0 License** • [Issues](https://github.com/Jake-Network/DenseCore/issues) • [Discord](https://discord.gg/densecore)
