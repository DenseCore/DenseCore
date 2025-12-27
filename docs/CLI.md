# DenseCore CLI Documentation

The DenseCore CLI provides a beautiful terminal interface for running LLM inference locally. It's designed for developers who want a quick, interactive way to chat with AI models without setting up complex infrastructure.

## Installation

### Homebrew (Recommended for macOS/Linux)

```bash
# Add the DenseCore tap
brew tap Jake-Network/densecore

# Install DenseCore CLI
brew install densecore
```

### From Source

```bash
cd server
CGO_ENABLED=1 go build -o densecore cmd/densecore/main.go

# Move to PATH
sudo mv densecore /usr/local/bin/
```

### Pre-built Binaries

Download from [GitHub Releases](https://github.com/Jake-Network/DenseCore/releases):

| Platform | Architecture | Download |
|----------|-------------|----------|
| macOS | Apple Silicon (M1/M2/M3) | `densecore-darwin-arm64.tar.gz` |
| macOS | Intel | `densecore-darwin-amd64.tar.gz` |
| Linux | x86_64 | `densecore-linux-amd64.tar.gz` |
| Linux | ARM64 (Graviton) | `densecore-linux-arm64.tar.gz` |

## Commands

### `densecore run` - Interactive Chat (Hero Feature 🌟)

The easiest way to chat with an AI model. One command does everything:

1. ✅ Checks if model exists locally
2. 📥 Downloads from HuggingFace if not found
3. ⚡ Starts inference server in background
4. 💬 Opens beautiful TUI chat interface

```bash
# Run with default model (Qwen 0.5B)
densecore run

# Run with specific model
densecore run Qwen/Qwen2.5-0.5B-Instruct-GGUF

# Specify filename for models with multiple files
densecore run TheBloke/Llama-2-7B-Chat-GGUF --filename llama-2-7b-chat.Q4_K_M.gguf

# Use different port
densecore run --port 9090
```

**TUI Controls:**
- `Enter` - Send message
- `Ctrl+C` - Exit (graceful shutdown)
- `Esc` - Exit

**Screenshot:**
```
🚀 DenseCore AI
Qwen/Qwen2.5-0.5B-Instruct-GGUF

You: What is the capital of France?

AI: The capital of France is Paris. Paris is not only the capital but also 
the largest city in France, known for its rich history, art, culture, and 
iconic landmarks like the Eiffel Tower.

> Type a message...▌

Enter ↵ send • Ctrl+C exit • Port: 8080
```

### `densecore serve` - Production Server

Start the HTTP API server for production deployments. Outputs structured JSON logs suitable for Kubernetes and log aggregation systems.

```bash
# Basic usage
densecore serve --model ./models/qwen2.5-0.5b-instruct-q4_k_m.gguf

# With custom port
densecore serve --model ./model.gguf --port 9090

# With authentication
AUTH_ENABLED=true API_KEYS=sk-xxx:user1:enterprise densecore serve --model ./model.gguf

# Verbose logging
densecore serve --model ./model.gguf --verbose
```

**Flags:**
| Flag | Short | Description |
|------|-------|-------------|
| `--model` | `-m` | Path to GGUF model file |
| `--port` | | Server port (default: 8080) |
| `--host` | | Bind address (default: 0.0.0.0) |
| `--threads` | `-t` | Inference threads (0 = auto-detect) |
| `--auth` | | Enable API key authentication |
| `--verbose` | `-v` | Enable debug logging |

**API Endpoints:**
- `POST /v1/chat/completions` - Chat completion (OpenAI-compatible)
- `POST /v1/embeddings` - Generate embeddings
- `GET /v1/models` - List loaded models
- `GET /health/live` - Liveness probe
- `GET /health/ready` - Readiness probe
- `GET /metrics` - Prometheus metrics

### `densecore help`

Display help for any command:

```bash
densecore --help
densecore run --help
densecore serve --help
```

## Configuration

### Model Cache Directory

Downloaded models are stored in `~/.densecore/models/`:

```bash
~/.densecore/
└── models/
    ├── qwen2.5-0.5b-instruct-q4_k_m.gguf
    └── llama-2-7b-chat.Q4_K_M.gguf
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port | `8080` |
| `HOST` | Bind address | `0.0.0.0` |
| `AUTH_ENABLED` | Enable API key auth | `false` |
| `API_KEYS` | API keys (format: `key:user:tier,...`) | - |
| `RATE_LIMIT_RPS` | Rate limit (requests/second) | `100` |
| `LOG_FORMAT` | Log format (`json` or `text`) | `json` |
| `REDIS_URL` | Redis URL for distributed rate limiting | - |

## Examples

### Quick Chat Session

```bash
# Start chatting immediately
densecore run

# The model will be downloaded automatically on first use
```

### Development Server

```bash
# Run server with hot-reload model loading
densecore serve --port 8080

# Load model via API
curl -X POST http://localhost:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/model.gguf"}'
```

### Production Deployment

```bash
# With all production settings
AUTH_ENABLED=true \
API_KEYS="sk-prod-key123:prod-user:enterprise" \
RATE_LIMIT_RPS=50 \
LOG_FORMAT=json \
densecore serve --model /models/production-model.gguf --port 8080
```

### Using with Docker

```bash
# Build the CLI into a Docker image
docker build -t densecore-cli -f Dockerfile .

# Run interactive chat
docker run -it --rm densecore-cli run
```

## Troubleshooting

### Port Already in Use

DenseCore automatically finds an available port if the default is busy:

```
⚡ Starting inference server on port 8081...
   (Port 8080 was in use)
```

To specify a different port:

```bash
densecore run --port 9090
```

### Model Download Fails

Check if the HuggingFace repository and filename are correct:

```bash
# List files in a HuggingFace repo
curl https://huggingface.co/api/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/main
```

For gated models, set `HF_TOKEN`:

```bash
export HF_TOKEN=hf_xxx
densecore run meta-llama/Llama-2-7b-chat-hf-GGUF
```

### Library Not Found (Linux)

If you get `libdensecore.so not found`:

```bash
export LD_LIBRARY_PATH=/path/to/build:$LD_LIBRARY_PATH
densecore run
```

### macOS dylib Issues

```bash
export DYLD_LIBRARY_PATH=/path/to/build:$DYLD_LIBRARY_PATH
densecore run
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    densecore CLI                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │  run cmd    │  │  serve cmd  │  │  other cmds...  │ │
│  │  (TUI Chat) │  │  (HTTP API) │  │                 │ │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────┘ │
│         │                │                              │
│         v                v                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │           internal/server (Go HTTP)             │   │
│  │  • OpenAI-compatible API                        │   │
│  │  • Rate limiting, Authentication                │   │
│  │  • Prometheus metrics                           │   │
│  └──────────────────────┬──────────────────────────┘   │
│                         │ CGO                          │
│  ┌──────────────────────v──────────────────────────┐   │
│  │           libdensecore.so (C++ Core)            │   │
│  │  • AVX-512/AMX/NEON kernels                     │   │
│  │  • Continuous batching                          │   │
│  │  • KV Cache management                          │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## See Also

- [API Reference](API_REFERENCE.md) - Full API documentation
- [Deployment Guide](DEPLOYMENT.md) - Kubernetes/Docker deployment
- [Architecture](ARCHITECTURE.md) - System internals
