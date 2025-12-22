# Changelog

All notable changes to DenseCore will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2025-12-21

### Added
- **Full Qwen3 Support**: Fixed GEMV dimension mismatch for projections.
- **Llama 3.2 Support**: Verified Llama-3.2-1B-Instruct compatibility.

### Fixed
- **GEMV Memory Corruption**: Refactored to `GGML_OP_CUSTOM` for dimension-changing ops.
- **CPU Backend Build**: Fixed `__attribute__` placement in `cpu_backend_opt.cpp`.

---

## [0.2.0] - 2025-12-16

### Added
- **Universal Model Support:** Added Qwen2.5, TinyLlama, SmolLM.
- **Speculative Decoding:** 1.5-2.5x speedup capable.
- **Paged KV Cache:** vLLM-inspired memory management.
- **Async Streaming:** Python `stream_async()` support.
- **Go Server:** OpenAI-compatible API with SSE streaming.

### Fixed
- **QK-Norm Support:** Fixed compatibility for new Qwen architectures.
- **Memory Leaks:** Resolved leaks in long-running sessions.

---

## [0.1.0] - 2025-12-15

- Initial Release with C++ Core, Python bindings, and basic Go server.
