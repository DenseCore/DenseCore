# API Reference

Complete API documentation for DenseCore's C, Python, and REST interfaces.

---

## Table of Contents

- [Python SDK API](#python-sdk-api)
- [C API Reference](#c-api-reference)
- [REST API (Go Server)](#rest-api-go-server)
- [Configuration Objects](#configuration-objects)
- [Error Handling](#error-handling)

---

## Python SDK API

### Installation

```bash
pip install densecore
```

### DenseCore Class

Main entry point for the Python SDK.

#### Constructor

```python
from densecore import DenseCore

model = DenseCore(
    main_model_path: str,
    threads: int = 0,
    hf_repo_id: Optional[str] = None
)
```

**Parameters:**
- `main_model_path` (str): Path to the main/target GGUF model file
- `draft_model_path` (str, optional): Path to draft model for speculative decoding
- `threads` (int, default=0): Number of CPU threads (0 = auto-detect)
- `hf_repo_id` (str, optional): HuggingFace repo ID for tokenizer

**Example:**
```python
# Local model
model = DenseCore("./models/qwen-7b-q4.gguf")

# With HF tokenizer
model = DenseCore(
    main_model_path="./model.gguf",
    hf_repo_id="Qwen/Qwen2.5-7B-Instruct"
)
```

---

#### from_pretrained()

Factory method to download and load models from HuggingFace Hub.

```python
@classmethod
def from_pretrained(
    cls,
    repo_id: str,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    threads: int = 0
) -> DenseCore
```

**Parameters:**
- `repo_id` (str): HuggingFace repository ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
- `filename` (str, optional): Specific GGUF file to download
- `draft_repo_id` (str, optional): Repo ID for draft model (speculative decoding)
- `cache_dir` (str, optional): Custom cache directory
- `threads` (int): Number of threads

**Returns:**
- DenseCore instance with downloaded model

**Example:**
```python
# Auto-download from HF Hub
model = DenseCore.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct-GGUF")

# Specific file
model = DenseCore.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf"
)
```

---

#### generate()

Generate text completion synchronously.

```python
def generate(
    self,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    stop_sequences: Optional[List[str]] = None,
    config: Optional[GenerationConfig] = None
) -> str
```

**Parameters:**
- `prompt` (str): Input text prompt
- `max_tokens` (int): Maximum tokens to generate
- `temperature` (float): Sampling temperature (0.0 = deterministic)
- `top_p` (float): Nucleus sampling threshold
- `top_k` (int): Top-K sampling
- `repetition_penalty` (float): Penalty for repeating tokens
- `stop_sequences` (List[str], optional): Stop generation when these appear
- `config` (GenerationConfig, optional): Advanced configuration object

**Returns:**
- str: Generated text

**Example:**
```python
# Simple generation
response = model.generate("Explain AI in one sentence.")

# With parameters
response = model.generate(
    "Write a poem about stars",
    max_tokens=100,
    temperature=0.9,
    stop_sequences=["\n\n", "---"]
)

# With config object
from densecore import GenerationConfig
config = GenerationConfig(temperature=0.7, max_tokens=500)
response = model.generate("Once upon a time", config=config)
```

---

#### stream()

Stream generated tokens synchronously.

```python
def stream(
    self,
    prompt: str,
    **kwargs
) -> Iterator[str]
```

**Parameters:**
- `prompt` (str): Input text prompt
- `**kwargs`: Same as `generate()`

**Yields:**
- str: Individual tokens as they're generated

**Example:**
```python
for token in model.stream("Tell me a story about dragons"):
    print(token, end="", flush=True)
print()  # Newline at end
```

---

#### stream_async()

Stream generated tokens asynchronously.

```python
async def stream_async(
    self,
    prompt: str,
    **kwargs
) -> AsyncIterator[str]
```

**Parameters:**
- Same as `stream()`

**Yields:**
- str: Individual tokens

**Example:**
```python
import asyncio

async def main():
    async for token in model.stream_async("Write a haiku"):
        print(token, end="", flush=True)

asyncio.run(main())
```

---

#### chat()

OpenAI-style chat completion.

```python
def chat(
    self,
    messages: List[Dict[str, str]],
    **kwargs
) -> str
```

**Parameters:**
- `messages` (List[Dict]): List of message dicts with "role" and "content"
- `**kwargs`: Same as `generate()`

**Returns:**
- str: Assistant's response

**Example:**
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "Give me an example."}
]

response = model.chat(messages, max_tokens=200)
print(response)
```

---

#### embed()

Generate embedding for a single text.

```python
def embed(
    self,
    text: str,
    pooling: str = "mean",
    normalize: bool = True
) -> np.ndarray
```

**Parameters:**
- `text` (str): Input text
- `pooling` (str): Pooling strategy ("mean", "cls", "last", "max")
- `normalize` (bool): L2 normalize the embedding

**Returns:**
- np.ndarray: Embedding vector (shape: [embedding_dim])

**Example:**
```python
embedding = model.embed("The quick brown fox", normalize=True)
print(f"Shape: {embedding.shape}")  # (768,)
print(f"Norm: {np.linalg.norm(embedding)}")  # 1.0 (normalized)
```

---

#### embed_batch()

Generate embeddings for multiple texts.

```python
def embed_batch(
    self,
    texts: List[str],
    pooling: str = "mean",
    normalize: bool = True
) -> np.ndarray
```

**Parameters:**
- `texts` (List[str]): List of input texts
- `pooling` (str): Pooling strategy
- `normalize` (bool): L2 normalize

**Returns:**
- np.ndarray: Embeddings matrix (shape: [batch_size, embedding_dim])

**Example:**
```python
texts = [
    "The cat sat on the mat.",
    "A quick brown fox jumps.",
    "Machine learning is cool."
]

embeddings = model.embed_batch(texts, normalize=True)
print(embeddings.shape)  # (3, 768)

# Compute similarity
from scipy.spatial.distance import cosine
similarity = 1 - cosine(embeddings[0], embeddings[1])
print(f"Similarity: {similarity:.3f}")
```

---

#### tokenize()

Convert text to token IDs.

```python
def tokenize(self, text: str) -> List[int]
```

**Returns:**
- List[int]: Token IDs

**Example:**
```python
token_ids = model.tokenize("Hello, world!")
print(token_ids)  # [9906, 11, 1917, 0]
```

---

#### detokenize()

Convert token IDs to text.

```python
def detokenize(self, token_ids: List[int]) -> str
```

**Returns:**
- str: Decoded text

**Example:**
```python
text = model.detokenize([9906, 11, 1917, 0])
print(text)  # "Hello, world!"
```

---

#### get_metrics()

Get current inference metrics.

```python
def get_metrics(self) -> Dict[str, Any]
```

**Returns:**
- dict: Metrics including TPS, latency, cache stats

**Example:**
```python
metrics = model.get_metrics()
print(f"Tokens/sec: {metrics['tokens_per_second']:.2f}")
print(f"Active requests: {metrics['active_requests']}")
print(f"Cache usage: {metrics['kv_cache_usage_percent']:.1f}%")
```

---

### GenerationConfig

Configuration object for advanced generation control.

```python
from densecore import GenerationConfig

config = GenerationConfig(
    max_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
    stop_sequences: Optional[List[str]] = None,
    json_mode: bool = False
)
```

**Example:**
```python
config = GenerationConfig(
    temperature=0.7,
    max_tokens=500,
    stop_sequences=["User:", "\n\n"],
    repetition_penalty=1.15
)

response = model.generate("Write a story", config=config)
```

---

### Exceptions

```python
from densecore.exceptions import (
    DenseCoreError,           # Base exception
    ModelLoadError,           # Failed to load model
    InferenceError,           # Inference failure
    InvalidConfigError,       # Invalid configuration
    TimeoutError,             # Request timeout
)
```

**Example:**
```python
from densecore import DenseCore
from densecore.exceptions import ModelLoadError

try:
    model = DenseCore("nonexistent.gguf")
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
```

---

## C API Reference

Low-level C API defined in `densecore.h`.

### Engine Initialization

#### InitEngine

```c
DenseCoreHandle InitEngine(
    const char *model_path,
    const char *reserved,       // NULL (reserved for future use)
    int threads,
    int numa_node_id,
    int pinning_policy
);
```

Initialize the inference engine with NUMA-aware configuration.

**Parameters:**
- `model_path`: Path to GGUF model file
- `reserved`: Reserved for future use (pass NULL)
- `threads`: Number of threads (0 = auto-detect)
- `numa_node_id`: NUMA node for memory/thread binding (-1 = auto)
- `pinning_policy`: Thread pinning strategy
  - `0` = SCATTER (distribute across cores for max bandwidth)
  - `1` = COMPACT (pack threads, share L2 cache)

**Returns:**
- Opaque handle to engine, or NULL on failure

**Example:**
```c
// Default: auto NUMA, SCATTER pinning
DenseCoreHandle engine = InitEngine("model.gguf", NULL, 8, -1, 0);
if (!engine) {
    fprintf(stderr, "Failed to initialize engine\n");
    return -1;
}

// Multi-socket: bind to NUMA node 1 with COMPACT pinning
DenseCoreHandle engine2 = InitEngine("model.gguf", NULL, 8, 1, 1);
```

> 📖 See [NUMA Optimization Guide](NUMA_OPTIMIZATION.md) for advanced configuration.

---

#### FreeEngine

```c
void FreeEngine(DenseCoreHandle handle);
```

Free engine and release all resources.

---

### Request Submission

#### SubmitRequest

```c
int SubmitRequest(
    DenseCoreHandle handle,
    const char *prompt,
    int max_tokens,
    TokenCallback callback,
    void *user_data
);
```

Submit text generation request.

**Returns:**
- Request ID (positive integer) on success
- Negative error code on failure

**Example:**
```c
void on_token(const char *token, int is_finished, void *user_data) {
    printf("%s", token);
    if (is_finished) printf("\n");
}

int req_id = SubmitRequest(engine, "Hello!", 100, on_token, NULL);
if (req_id < 0) {
    fprintf(stderr, "Request failed: %d\n", req_id);
}
```

---

#### SubmitRequestIds

```c
int SubmitRequestIds(
    DenseCoreHandle handle,
    const int *tokens,
    int n_tokens,
    int max_tokens,
    TokenCallback callback,
    void *user_data
);
```

Submit request with pre-tokenized input.

---

#### SubmitEmbeddingRequest

```c
int SubmitEmbeddingRequest(
    DenseCoreHandle handle,
    const char *prompt,
    EmbeddingCallback callback,
    void *user_data
);
```

Submit embedding request.

**Example:**
```c
void on_embedding(const float *embedding, int size, void *user_data) {
    printf("Embedding dimension: %d\n", size);
    for (int i = 0; i < size && i < 10; i++) {
        printf("%.4f ", embedding[i]);
    }
    printf("...\n");
}

SubmitEmbeddingRequest(engine, "Hello world", on_embedding, NULL);
```

---

### Metrics

#### GetMetrics

```c
DenseCoreMetrics GetMetrics(DenseCoreHandle handle);
```

Get basic metrics.

**Returns:**
```c
typedef struct {
    float requests_per_second;
    float tokens_per_second;
    int active_requests;
    long total_tokens_generated;
} DenseCoreMetrics;
```

---

#### GetDetailedMetrics

```c
DetailedMetrics GetDetailedMetrics(DenseCoreHandle handle);
```

Get detailed metrics including latency percentiles.

---

## REST API (Go Server)

OpenAI-compatible REST API.

### Base URL

```
http://localhost:8080
```

### Authentication

Include API key in header (if `AUTH_ENABLED=true`):

```bash
curl -H "Authorization: Bearer sk-your-api-key" ...
```

---

### POST /v1/chat/completions

OpenAI-compatible chat completion endpoint.

**Request:**
```json
{
  "model": "qwen",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "qwen",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 9,
    "total_tokens": 22
  }
}
```

**Streaming:**
Set `"stream": true` for Server-Sent Events:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

---

### POST /v1/embeddings

Generate embeddings.

**Request:**
```json
{
  "model": "bge-small",
  "input": "The cat sat on the mat.",
  "encoding_format": "float"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": [0.123, -0.456, ...],
    "index": 0
  }],
  "model": "bge-small",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

---

### GET /v1/models

List available models.

**Response:**
```json
{
  "object": "list",
  "data": [{
    "id": "qwen",
    "object": "model",
    "created": 1677652288,
    "owned_by": "densecore"
  }]
}
```

---

### Health Endpoints

#### GET /health

Overall health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-09T12:00:00Z"
}
```

#### GET /health/live

Liveness probe (Kubernetes).

#### GET /health/ready

Readiness probe.

#### GET /health/startup

Startup probe.

---

### GET /metrics

Prometheus metrics.

**Response:**
```prometheus
# HELP densecore_requests_total Total number of requests
# TYPE densecore_requests_total counter
densecore_requests_total 1234

# HELP densecore_tokens_per_second Current tokens per second
# TYPE densecore_tokens_per_second gauge
densecore_tokens_per_second 32.5
```

---

## Configuration Objects

### QuantConfig

```python
from densecore.quantize import QuantConfig

config = QuantConfig(
    format: str = "q4_k_m",    # q4_k_m, q5_k_m, q8_0, q4_0
    algorithm: str = "q4_k_m", # q4_k_m, q5_k_m, q8_0, q4_0
    block_size: int = 32
)
```

### PruneConfig

```python
from densecore.prune import PruneConfig

config = PruneConfig(
    strategy: str = "depth",           # depth, width, attention, combined
    target_n_layer: Optional[int] = None,
    importance_method: str = "magnitude"  # magnitude, l2_norm, activation
)
```

---

## Error Handling

### Python Error Codes

| Exception | Cause | Solution |
|-----------|-------|----------|
| `ModelLoadError` | Invalid GGUF file | Check file path and format |
| `InferenceError` | Inference failed | Check model compatibility |
| `TimeoutError` | Request timeout | Increase timeout or reduce max_tokens |
| `InvalidConfigError` | Bad config | Validate config parameters |

### C Error Codes

| Code | Meaning |
|------|---------|
| `-1` | Generic error |
| `-2` | Out of memory |
| `-3` | Invalid model |
| `-4` | Timeout |

---

**Next Steps:**
- [Architecture Deep Dive](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Model Optimization](MODEL_OPTIMIZATION.md)
