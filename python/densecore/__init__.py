"""
DenseCore - High-Performance CPU Inference Engine for LLMs

DenseCore is a production-ready inference engine optimized for running
large language models on CPU with HuggingFace integration.

Quick Start:
    >>> import densecore

    # Load from HuggingFace Hub (recommended)
    >>> model = densecore.from_pretrained("TheBloke/Llama-2-7B-GGUF")
    >>> response = model.generate("Hello, how are you?")

    # Or load from local file
    >>> model = densecore.DenseCore("./model.gguf")
    >>> response = model.generate("Hello!", max_tokens=100)

    # Streaming generation
    >>> for token in model.stream("Tell me a story"):
    ...     print(token, end="", flush=True)

    # Async support
    >>> async for token in model.stream_async("Hello"):
    ...     print(token, end="", flush=True)

Features:
    - 🚀 High-performance CPU inference with SIMD optimization
    - 🤗 Native HuggingFace Hub integration
    - 📦 GGUF format support (llama.cpp compatible)
    - 🔗 LangChain & LangGraph integration with tool calling
    - 🔄 Streaming and async support
    - 💾 Automatic model caching
"""

from typing import TYPE_CHECKING

# Core exports
from .engine import DenseCore
from .config import GenerationConfig, ModelConfig, SamplingParams
from .hub import from_pretrained, list_gguf_files, download_model
from .embedding import EmbeddingModel, EmbeddingConfig, embed
from .convert import convert_from_hf, quick_convert
from .smart_loader import smart_load, recommend_quantization, get_system_resources
from .lora import LoRAConfig, LoRAAdapterInfo, LoRAManager
from .auto import AutoModel, AutoModelForCausalLM, AutoTokenizer
from .generate_output import (
    GenerateOutput,
    GenerateBeamOutput,
    StoppingCriteria,
    StoppingCriteriaList,
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    EosTokenCriteria,
)

# Version info
__version__ = "2.0.0"
__author__ = "DenseCore Team"

# Type annotations for IDE support
if TYPE_CHECKING:
    from .engine import DenseCore as DenseCoreType

__all__ = [
    # Main class
    "DenseCore",
    # Auto classes (HuggingFace-style)
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoTokenizer",
    # Generation output (HuggingFace-compatible)
    "GenerateOutput",
    "GenerateBeamOutput",
    # Stopping criteria (HuggingFace-compatible)
    "StoppingCriteria",
    "StoppingCriteriaList",
    "MaxLengthCriteria",
    "MaxNewTokensCriteria",
    "EosTokenCriteria",
    # Embedding
    "EmbeddingModel",
    "EmbeddingConfig",
    "embed",
    # Config classes
    "GenerationConfig",
    "ModelConfig",
    "SamplingParams",
    # HuggingFace Hub functions
    "from_pretrained",
    "list_gguf_files",
    "download_model",
    # Smart loading
    "smart_load",
    "recommend_quantization",
    "get_system_resources",
    # LoRA adapters
    "LoRAConfig",
    "LoRAAdapterInfo",
    "LoRAManager",
    # Conversion functions
    "convert_from_hf",
    "quick_convert",
    # Version
    "__version__",
    # Integrations (lazy-loaded)
    "integrations",
]


def get_version() -> str:
    """Return the package version."""
    return __version__


def get_device_info() -> dict:
    """Get information about available compute devices."""
    import platform
    import os

    return {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
    }
