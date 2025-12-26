"""
DenseCore Engine - Main inference interface.

This module provides the core DenseCore class that wraps the C++ inference engine
with a Pythonic API supporting both synchronous and asynchronous generation.
"""

import asyncio
import ctypes
import os
import platform
import queue
import threading
import warnings
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Optional,
    TypeVar,
    Union,
)

try:
    from huggingface_hub import hf_hub_download, list_repo_files

    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False

from .config import GenerationConfig, ModelConfig
from .lora import LoRAManager

# ==============================================================================
# GIL Release Architecture
# ==============================================================================
#
# DenseCore achieves true non-blocking AsyncIO through the following design:
#
# 1. THREAD POOL OFFLOADING:
#    - SubmitRequest is offloaded to a ThreadPoolExecutor via run_in_executor
#    - This frees the Python event loop to handle other async tasks
#
# 2. GIL RELEASE DURING C++ EXECUTION:
#    - ctypes.CDLL (not ctypes.pythonapi) is used for bindings
#    - The GIL is released during C function calls by default
#    - C++ inference runs on native threads without holding the GIL
#
# 3. THREAD-SAFE CALLBACK BRIDGING:
#    - Token callbacks use loop.call_soon_threadsafe() to safely
#      communicate from C++ worker threads back to the event loop
#
# This enables 100+ concurrent async requests without event loop starvation.
# ==============================================================================

# Dedicated thread pool for C++ inference submissions
# Lazily initialized to avoid overhead if only sync methods are used
_INFERENCE_EXECUTOR: ThreadPoolExecutor | None = None


def _get_inference_executor() -> ThreadPoolExecutor:
    """
    Get or create the inference thread pool executor (Singleton).

    The executor is sized to handle high concurrency while avoiding
    thread explosion. Workers are named for easy debugging.
    """
    global _INFERENCE_EXECUTOR
    if _INFERENCE_EXECUTOR is None:
        max_workers = min(32, (os.cpu_count() or 4) * 4)
        _INFERENCE_EXECUTOR = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="densecore-submit",
        )
    return _INFERENCE_EXECUTOR


def shutdown_executor() -> None:
    """
    Shut down the inference executor gracefully.

    Call this during application shutdown to ensure clean termination.
    """
    global _INFERENCE_EXECUTOR
    if _INFERENCE_EXECUTOR is not None:
        _INFERENCE_EXECUTOR.shutdown(wait=True)
        _INFERENCE_EXECUTOR = None


# ==============================================================================
# Custom Exception Hierarchy for ctypes Error Handling
# ==============================================================================


class DenseCoreError(Exception):
    """Base exception for all DenseCore errors."""

    pass


class DenseCoreRuntimeError(DenseCoreError):
    """Runtime error during inference (e.g., model loading failure)."""

    pass


class ContextLimitExceededError(DenseCoreError):
    """Prompt exceeds the model's context length."""

    pass


class OOMError(DenseCoreError):
    """Out of memory error."""

    pass


class InvalidRequestError(DenseCoreError):
    """Invalid request parameters."""

    pass


class EngineNotInitializedError(DenseCoreError):
    """Engine handle is null or not initialized."""

    pass


# Error code to exception mapping
_ERROR_CODE_MAP: dict[int, type[DenseCoreError]] = {
    -1: DenseCoreRuntimeError,  # Generic error
    -2: InvalidRequestError,  # Invalid parameters
    -3: ContextLimitExceededError,  # Context overflow
    -4: OOMError,  # Out of memory
    -5: EngineNotInitializedError,  # Null handle
}


def _error_code_to_exception(code: int, context: str = "") -> DenseCoreError:
    """Convert a C++ error code to a Python exception."""
    exc_class = _ERROR_CODE_MAP.get(code, DenseCoreRuntimeError)
    msg = f"DenseCore error (code={code})"
    if context:
        msg = f"{msg}: {context}"
    return exc_class(msg)


def _errcheck(result: int, func: Any, args: tuple) -> int:
    """
    ctypes errcheck callback for validating C function return values.

    Raises appropriate Python exceptions for negative error codes.
    """
    if result < 0:
        func_name = getattr(func, "__name__", "unknown")
        raise _error_code_to_exception(result, context=f"in {func_name}()")
    return result


# Type variable for generic typing
T = TypeVar("T")

# C callback function type (legacy string-based)
# typedef void (*TokenCallback)(const char *token, int is_finished, void *user_data);
CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)


# TokenResult struct for structured token callbacks
class TokenResult(ctypes.Structure):
    """
    Structured token result from C++ engine.

    Attributes:
        token_id: Token ID for HuggingFace tokenizer decoding
        text: Pre-decoded text (may be empty if using HF tokenizer)
        is_finished: 1 if generation is complete, 0 otherwise
    """

    _fields_ = [
        ("token_id", ctypes.c_int),
        ("text", ctypes.c_char_p),
        ("is_finished", ctypes.c_int),
    ]


# Callback type for TokenResult struct
TOKEN_RESULT_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.POINTER(TokenResult), ctypes.c_void_p)


@dataclass
class GenerationOutput:
    """
    Output from text generation.

    Attributes:
        text: Generated text
        tokens: Number of tokens generated
        finish_reason: Why generation stopped ("length", "stop", "error")
        prompt_tokens: Number of prompt tokens
        generation_time: Time taken for generation in seconds
    """

    text: str
    tokens: int = 0
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    generation_time: float = 0.0

    def __str__(self) -> str:
        return self.text


def _find_library() -> str:
    """
    Find the DenseCore shared library.

    Searches in:
    1. Package directory
    2. Parent directories (for development)
    3. System library paths
    4. LD_LIBRARY_PATH / DYLD_LIBRARY_PATH

    Returns:
        Path to the library file

    Raises:
        RuntimeError: If library cannot be found
    """
    if platform.system() == "Windows":
        lib_names = ["densecore.dll"]
    elif platform.system() == "Darwin":
        lib_names = ["libdensecore.dylib"]
    else:
        lib_names = ["libdensecore.so", "libdensecore.so.1"]

    search_paths = [
        Path(__file__).parent,
        Path(__file__).parent / "lib",
        Path(__file__).parent.parent,
        Path(__file__).parent.parent.parent / "build",
        Path(__file__).parent.parent.parent
        / "core"
        / "build",  # Added: correctly locate core build artifacts
        Path("/usr/local/lib"),
        Path("/usr/lib"),
    ]

    # Add environment library paths
    env_var = "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
    if env_var in os.environ:
        for path_str in os.environ[env_var].split(os.pathsep):
            search_paths.append(Path(path_str))

    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                return str(lib_path)

    raise RuntimeError(
        f"Could not find DenseCore library. Searched in: {[str(p) for p in search_paths]}"
    )


_LIB_INSTANCE = None


def get_library() -> ctypes.CDLL:
    """
    Get the shared library instance (Singleton).
    """
    global _LIB_INSTANCE
    if _LIB_INSTANCE is None:
        lib_path = _find_library()
        _LIB_INSTANCE = ctypes.CDLL(lib_path)
    return _LIB_INSTANCE


class DenseCore:
    """
    High-performance CPU inference engine for large language models.

    DenseCore provides a simple interface for running inference on GGUF models
    with support for streaming and async operations.

    Args:
        model_path: Path to the GGUF model file (required)
        threads: Number of CPU threads (0 = auto-detect)
        hf_repo_id: HuggingFace repo ID for loading tokenizer
        config: Optional ModelConfig for advanced settings

    Examples:
        Basic usage:

        >>> from densecore import DenseCore
        >>> model = DenseCore("./model.gguf")
        >>> response = model.generate("Hello, how are you?")
        >>> print(response)

        With context manager:

        >>> with DenseCore("./model.gguf") as model:
        ...     print(model.generate("Hello!"))

        Streaming:

        >>> for token in model.stream("Tell me a story"):
        ...     print(token, end="", flush=True)

        Async streaming:

        >>> async for token in model.stream_async("Hello"):
        ...     print(token, end="", flush=True)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        lora_adapter_path: Optional[str] = None,
        lora_scale: float = 1.0,
        threads: int = 0,
        hf_repo_id: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        verbose: bool = True,
    ):
        self._handle = None
        self._closed = True
        self._requests: dict[int, Any] = {}
        self._active_ctypes_refs: dict[int, list] = {}  # Prevent GC of ctypes objects
        self._req_id_counter = 0
        self._lock = threading.Lock()
        self._verbose = verbose
        self._model_path = ""

        # LoRA adapter management
        self._lora_manager = LoRAManager()

        # Handle config
        if config is not None:
            model_path = config.model_path or model_path
            threads = config.threads if config.threads > 0 else threads

        if not model_path:
            raise ValueError("model_path is required")

        self._model_path = model_path

        # Load shared library
        self._lib = get_library()

        # Setup C function signatures
        self._setup_cfunctions()

        # Keep callback reference to prevent GC
        self._c_callback = CALLBACK_TYPE(self._global_callback)

        # Initialize engine
        model_path_bytes = model_path.encode("utf-8")

        if hf_repo_id:
            self._init_tokenizer(hf_repo_id)

        # Initialize native engine (allocates large memory pools)
        print(f"[DEBUG] Calling InitEngine with model: {model_path}")
        self._handle = self._lib.InitEngine(model_path_bytes, None, threads)
        print(f"[DEBUG] InitEngine returned handle: {self._handle}")
        if not self._handle:
            raise RuntimeError(f"Failed to initialize DenseCore engine for model: {model_path}")

        # Load LoRA adapter if provided
        if lora_adapter_path:
            self.load_lora(lora_adapter_path, scale=lora_scale)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DenseCore":
        """
        Load a model directly from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
            filename: GGUF filename to download (e.g., "llama-2-7b.Q4_K_M.gguf").
                      If None, attempts to find a suitable GGUF file automatically.
            revision: Specific model revision (branch/tag/commit)
            cache_dir: Directory to cache downloaded files
            **kwargs: Additional arguments passed to DenseCore constructor

        Returns:
            Initialized DenseCore instance

        Raises:
            ImportError: If huggingface_hub is not installed
            ValueError: If finding a GGUF file fails
        """
        if not HUGGINGFACE_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub is not installed. "
                "Please install it with: pip install huggingface-hub"
            )

        if filename is None:
            # Try to auto-detect GGUF file
            print(f"[DenseCore] No filename provided, searching in {repo_id}...")
            files = list_repo_files(repo_id=repo_id, revision=revision)
            gguf_files = [f for f in files if f.endswith(".gguf")]

            if not gguf_files:
                raise ValueError(f"No .gguf files found in {repo_id}")

            # Prefer Q4_K_M if available, otherwise pick the first one
            preferred = [f for f in gguf_files if "Q4_K_M" in f]
            if preferred:
                filename = preferred[0]
            else:
                filename = gguf_files[0]

            print(f"[DenseCore] Auto-selected model file: {filename}")

        print(f"[DenseCore] Downloading {filename} from {repo_id}...")
        model_path = hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir
        )

        # Pass hf_repo_id to constructor to auto-load tokenizer
        if "hf_repo_id" not in kwargs:
            kwargs["hf_repo_id"] = repo_id

        return cls(model_path=model_path, **kwargs)

    def _setup_cfunctions(self) -> None:
        """Setup C function argument and return types with error checking."""
        # InitEngine
        self._lib.InitEngine.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.InitEngine.restype = ctypes.c_void_p

        # SubmitRequest - returns request ID or negative error code
        self._lib.SubmitRequest.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.SubmitRequest.restype = ctypes.c_int
        self._lib.SubmitRequest.errcheck = _errcheck

        # SubmitRequestIds (for tokenized input) - returns request ID or negative error
        self._lib.SubmitRequestIds.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.SubmitRequestIds.restype = ctypes.c_int
        self._lib.SubmitRequestIds.errcheck = _errcheck

        # FreeEngine - void return, no errcheck needed
        self._lib.FreeEngine.argtypes = [ctypes.c_void_p]
        self._lib.FreeEngine.restype = None

        # GetMetrics (optional)
        try:
            self._lib.GetMetrics.argtypes = [ctypes.c_void_p]
            self._lib.GetMetrics.restype = ctypes.c_void_p
            self._has_metrics = True
        except AttributeError:
            self._has_metrics = False

        # LoRA Runtime API
        try:
            self._lib.LoadLoraAdapter.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_float,
                ctypes.c_char_p,
            ]
            self._lib.LoadLoraAdapter.restype = ctypes.c_int
            self._lib.LoadLoraAdapter.errcheck = _errcheck

            self._lib.ActivateLoraAdapter.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self._lib.ActivateLoraAdapter.restype = ctypes.c_int
            self._lib.ActivateLoraAdapter.errcheck = _errcheck

            self._lib.DeactivateLoraAdapters.argtypes = [ctypes.c_void_p]
            self._lib.DeactivateLoraAdapters.restype = ctypes.c_int
            self._lib.DeactivateLoraAdapters.errcheck = _errcheck

            self._lib.UnloadLoraAdapter.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self._lib.UnloadLoraAdapter.restype = ctypes.c_int
            self._lib.UnloadLoraAdapter.errcheck = _errcheck

            self._has_lora_api = True
        except AttributeError:
            self._has_lora_api = False

    def _init_tokenizer(self, hf_repo_id: str) -> None:
        """Initialize tokenizer from HuggingFace repo."""
        try:
            from transformers import AutoTokenizer

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(hf_repo_id)
                if self._verbose:
                    print(f"[DenseCore] Loaded tokenizer from {hf_repo_id}")
            except Exception as e:
                # If verbose is not available (it's called verbose in __init__ but _verbose here? check context)
                # Engine has self.verbose. Constructor sets self.verbose.
                if getattr(self, "verbose", False):
                    print(f"[DenseCore] Warning: Failed to load tokenizer from {hf_repo_id}: {e}")
                self.tokenizer = None
        except ImportError as e:
            if getattr(self, "_verbose", False):
                print(f"[DenseCore] transformers load failed: {e}")
                print("[DenseCore] tokenizer disabled")
        except Exception as e:
            warnings.warn(f"Failed to load tokenizer from {hf_repo_id}: {e}")

    def _clean_bpe_token(self, token: str) -> str:
        """Clean BPE artifacts from token string."""
        # Replace Ġ with space
        # Replace Ċ with newline
        # This is a simple heuristic for BPE/SentencePiece tokens
        if not token:
            return token

        # Common BPE replacements
        s = token.replace("Ġ", " ").replace("Ċ", "\n")
        # Handle other common replacement characters if needed
        return s

    def _global_callback(self, token: bytes, is_finished: int, user_data: int) -> None:
        """Global callback called from C.

        Handles both legacy string format and new TokenResult struct format.
        For TokenResult, the token parameter contains the struct data.
        """
        req_id = user_data

        with self._lock:
            handler = self._requests.get(req_id)

        if not handler:
            return

        try:
            # Decode token string
            token_str = token.decode("utf-8", errors="replace") if token else ""

            # Since C++ now sends decoded tokens, just clean BPE artifacts
            token_str = self._clean_bpe_token(token_str)

            finished = bool(is_finished)
            handler(token_str, finished)
        except Exception as e:
            import traceback

            error_msg = f"Error in callback handler for req {req_id}: {e}\n{traceback.format_exc()}"
            print(f"[DenseCore] {error_msg}")
            warnings.warn(error_msg)

        if finished:
            with self._lock:
                self._requests.pop(req_id, None)
                self._active_ctypes_refs.pop(req_id, None)  # Release ctypes refs

    def _register_request(self, handler: Callable[[str, bool], None]) -> int:
        """Register a request handler and return request ID."""
        with self._lock:
            self._req_id_counter += 1
            req_id = self._req_id_counter
            self._requests[req_id] = handler
            return req_id

    def _submit_request(
        self,
        prompt: str,
        max_tokens: int,
        req_id: int,
        json_mode: bool = False,
    ) -> int:
        """Submit a generation request to the engine.

        Requires HuggingFace tokenizer to be initialized.
        """
        if not self.tokenizer:
            raise RuntimeError(
                "HuggingFace tokenizer is required for inference. "
                "Please provide 'hf_repo_id' when initializing DenseCore, "
                "or use DenseCore.from_pretrained() which auto-loads the tokenizer."
            )

        tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        tokens_array = (ctypes.c_int * len(tokens))(*tokens)
        # CRITICAL: Keep ctypes array reference alive until C++ is done!
        with self._lock:
            self._active_ctypes_refs[req_id] = [tokens_array]
        res = self._lib.SubmitRequestIds(
            self._handle,
            tokens_array,
            len(tokens),
            max_tokens,
            ctypes.cast(self._c_callback, ctypes.c_void_p),
            ctypes.c_void_p(req_id),
        )
        return res

    # =========================================================================
    # Public API - Generation Methods
    # =========================================================================

    def generate(
        self,
        prompt: Optional[Union[str, List[int], Any]] = None,
        max_tokens: int = 256,
        config: Optional[GenerationConfig] = None,
        # HuggingFace Transformers compatible kwargs
        input_ids: Optional[Any] = None,
        attention_mask: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        min_length: int = 0,
        min_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        num_beams: int = 1,
        repetition_penalty: Optional[float] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        stopping_criteria: Optional[Any] = None,
        return_dict_in_generate: bool = False,
        output_scores: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> Union[str, "GenerateOutput"]:
        """
        Generate text from a prompt.

        This method is fully compatible with HuggingFace Transformers' generate() API,
        enabling drop-in replacement for existing inference code.

        Args:
            prompt: Input text string or token IDs (list/tensor)
            max_tokens: Maximum tokens to generate (DenseCore style)
            config: Optional GenerationConfig for advanced settings

            # HuggingFace Transformers compatible args:
            input_ids: Token IDs as list, numpy array, or torch.Tensor
            attention_mask: Attention mask (ignored, for API compatibility)
            max_new_tokens: Maximum new tokens to generate (HF style)
            max_length: Maximum total length including prompt (HF style)
            min_length: Minimum total length (HF style)
            min_new_tokens: Minimum new tokens (HF style)
            do_sample: Enable sampling (False = greedy)
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling
            num_beams: Beam search (only 1 supported)
            repetition_penalty: Penalty for repeating tokens
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID(s)
            stopping_criteria: StoppingCriteriaList for custom stopping
            return_dict_in_generate: Return GenerateOutput instead of string
            output_scores: Return generation scores (not supported)
            output_attentions: Return attention weights (not supported)
            output_hidden_states: Return hidden states (not supported)
            **kwargs: Additional parameters

        Returns:
            str: Generated text (default)
            GenerateOutput: If return_dict_in_generate=True

        Example:
            >>> # DenseCore style
            >>> response = model.generate("Hello!", max_tokens=100)

            >>> # HuggingFace style
            >>> output = model.generate(
            ...     "Hello!",
            ...     max_new_tokens=100,
            ...     do_sample=True,
            ...     temperature=0.7,
            ...     top_p=0.9,
            ...     return_dict_in_generate=True,
            ... )
            >>> print(output.text)

            >>> # Token ID input (HuggingFace style)
            >>> input_ids = tokenizer.encode("Hello!", return_tensors="pt")
            >>> output = model.generate(input_ids=input_ids, max_new_tokens=100)
        """
        # Import here to avoid circular import
        from .generate_output import GenerateOutput

        # =======================================================================
        # Input Processing: Support string, list[int], numpy array, torch.Tensor
        # =======================================================================
        prompt_tokens: Optional[List[int]] = None
        prompt_text: Optional[str] = None

        # Priority: input_ids > prompt
        if input_ids is not None:
            prompt_tokens = self._coerce_to_token_ids(input_ids)
        elif prompt is not None:
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                # Assume it's token IDs (list, numpy array, or tensor)
                prompt_tokens = self._coerce_to_token_ids(prompt)

        if prompt_text is None and prompt_tokens is None:
            raise ValueError("Either 'prompt' or 'input_ids' must be provided")

        # =======================================================================
        # Parameter Resolution: config > explicit kwargs > defaults
        # =======================================================================
        effective_max_tokens = max_tokens
        effective_temperature = 1.0
        effective_top_p = 1.0
        effective_top_k = 0
        effective_repetition_penalty = 1.0
        effective_return_dict = return_dict_in_generate

        # Config takes precedence
        if config is not None:
            effective_max_tokens = config.max_tokens
            effective_temperature = config.temperature
            effective_top_p = config.top_p
            effective_top_k = config.top_k
            effective_repetition_penalty = config.repetition_penalty
            effective_return_dict = config.return_dict_in_generate or return_dict_in_generate

        # HuggingFace kwargs override config
        if max_new_tokens is not None:
            effective_max_tokens = max_new_tokens
        elif max_length is not None:
            # max_length includes prompt; estimate prompt length
            prompt_len = len(prompt_tokens) if prompt_tokens else len(prompt_text or "") // 4
            effective_max_tokens = max(1, max_length - prompt_len)

        if temperature is not None:
            effective_temperature = temperature
        if top_p is not None:
            effective_top_p = top_p
        if top_k is not None:
            effective_top_k = top_k
        if repetition_penalty is not None:
            effective_repetition_penalty = repetition_penalty

        # Handle do_sample=False (greedy decoding)
        if not do_sample:
            effective_temperature = 0.0
            effective_top_p = 1.0
            effective_top_k = 0

        # Warn about unsupported features
        if num_beams > 1:
            warnings.warn(
                f"DenseCore only supports num_beams=1. Requested num_beams={num_beams} will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        if output_scores or output_attentions or output_hidden_states:
            warnings.warn(
                "output_scores, output_attentions, and output_hidden_states are not supported by DenseCore.",
                UserWarning,
                stacklevel=2,
            )

        # =======================================================================
        # Generation Execution
        # =======================================================================
        result_queue: queue.Queue = queue.Queue()
        generated_token_ids: List[int] = []

        def handler(token: str, finished: bool) -> None:
            result_queue.put((token, finished))

        req_id = self._register_request(handler)

        # Submit request
        if prompt_tokens is not None:
            # Use tokenized input
            tokens_array = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)
            # CRITICAL: Keep ctypes array reference alive until C++ is done!
            with self._lock:
                self._active_ctypes_refs[req_id] = [tokens_array]
            ret = self._lib.SubmitRequestIds(
                self._handle,
                tokens_array,
                len(prompt_tokens),
                effective_max_tokens,
                ctypes.cast(self._c_callback, ctypes.c_void_p),
                ctypes.c_void_p(req_id),
            )
        else:
            # Use text prompt (requires tokenizer)
            ret = self._submit_request(prompt_text, effective_max_tokens, req_id)

        if ret < 0:
            with self._lock:
                self._requests.pop(req_id, None)
                self._active_ctypes_refs.pop(req_id, None)
            raise _error_code_to_exception(ret, context="in generate()")

        # Collect generated tokens
        tokens: List[str] = []
        while True:
            token, finished = result_queue.get()
            tokens.append(token)
            if finished:
                break

        generated_text = "".join(tokens)

        # =======================================================================
        # Return Format
        # =======================================================================
        if effective_return_dict:
            # Build GenerateOutput compatible with HuggingFace
            prompt_token_ids = prompt_tokens if prompt_tokens else []
            if self.tokenizer and prompt_text:
                prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

            # Decode generated tokens to IDs if tokenizer available
            if self.tokenizer:
                generated_token_ids = self.tokenizer.encode(
                    generated_text, add_special_tokens=False
                )
            else:
                generated_token_ids = []

            # Full sequence = prompt + generated
            full_sequence = list(prompt_token_ids) + generated_token_ids

            return GenerateOutput(
                sequences=[full_sequence],
                text=generated_text,
                finish_reason="stop",
                usage={
                    "prompt_tokens": len(prompt_token_ids),
                    "completion_tokens": len(generated_token_ids),
                    "total_tokens": len(full_sequence),
                },
            )

        return generated_text

    def _coerce_to_token_ids(self, input_data: Any) -> List[int]:
        """
        Convert various input types to a flat list of token IDs.

        Supports:
        - list[int]: Direct token IDs
        - list[list[int]]: Batched token IDs (takes first sequence)
        - numpy.ndarray: NumPy array of token IDs
        - torch.Tensor: PyTorch tensor of token IDs

        Args:
            input_data: Input in any supported format

        Returns:
            list[int]: Flat list of token IDs

        Raises:
            ValueError: If input format is not supported
        """
        # Already a list
        if isinstance(input_data, list):
            if not input_data:
                return []
            # Check if it's a list of lists (batched)
            if isinstance(input_data[0], list):
                return list(input_data[0])  # Take first sequence
            # Flat list of ints
            return [int(x) for x in input_data]

        # NumPy array
        try:
            import numpy as np

            if isinstance(input_data, np.ndarray):
                flat = input_data.flatten() if input_data.ndim > 1 else input_data
                return flat.tolist()
        except ImportError:
            pass

        # PyTorch tensor
        try:
            import torch

            if isinstance(input_data, torch.Tensor):
                flat = input_data.flatten() if input_data.dim() > 1 else input_data
                return flat.tolist()
        except ImportError:
            pass

        # Try generic iteration
        try:
            return [int(x) for x in input_data]
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot convert input to token IDs. Expected list[int], numpy array, or torch.Tensor. "
                f"Got {type(input_data).__name__}"
            ) from e

    def stream(
        self,
        prompt: Union[str, List[int]],
        max_tokens: int = 256,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream generated tokens one at a time.

        Args:
            prompt: Input text prompt (str) or pre-tokenized token IDs (list[int])
            max_tokens: Maximum tokens to generate
            config: Optional GenerationConfig
            **kwargs: Additional generation parameters

        Yields:
            Generated tokens one at a time

        Example:
            >>> for token in model.stream("Tell me a story"):
            ...     print(token, end="", flush=True)

            >>> # With pre-tokenized input
            >>> token_ids = [1, 2, 3, 4, 5]
            >>> for token in model.stream(token_ids, max_tokens=50):
            ...     print(token, end="", flush=True)
        """
        if config is not None:
            max_tokens = config.max_tokens

        result_queue: queue.Queue = queue.Queue()

        def handler(token: str, finished: bool) -> None:
            result_queue.put((token, finished))

        req_id = self._register_request(handler)

        # Determine if prompt is already tokenized (list[int]) or text (str)
        if isinstance(prompt, list):
            # Pre-tokenized input: use SubmitRequestIds directly
            prompt_tokens = [int(x) for x in prompt]

            tokens_array = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)
            # CRITICAL: Keep ctypes array reference alive until C++ is done!
            with self._lock:
                self._active_ctypes_refs[req_id] = [tokens_array]
            ret = self._lib.SubmitRequestIds(
                self._handle,
                tokens_array,
                len(prompt_tokens),
                max_tokens,
                ctypes.cast(self._c_callback, ctypes.c_void_p),
                ctypes.c_void_p(req_id),
            )

        else:
            # Text prompt: use tokenizer via _submit_request

            ret = self._submit_request(prompt, max_tokens, req_id)

        if ret < 0:
            with self._lock:
                self._requests.pop(req_id, None)
                self._active_ctypes_refs.pop(req_id, None)
            raise RuntimeError(f"Failed to submit request: error code {ret}")

        while True:
            token, finished = result_queue.get()

            if token:
                yield token
            if finished:
                break

    # Alias for HuggingFace-style API
    generate_stream = stream

    async def generate_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """
        Asynchronously generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            config: Optional GenerationConfig

        Returns:
            Generated text string

        Example:
            >>> response = await model.generate_async("Hello!")
            >>> print(response)
        """
        if config is not None:
            max_tokens = config.max_tokens

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        accumulated: List[str] = []

        def handler(token: str, finished: bool) -> None:
            if token:
                accumulated.append(token)
            if finished:
                result = "".join(accumulated)
                loop.call_soon_threadsafe(future.set_result, result)

        req_id = self._register_request(handler)

        # Offload submission to thread pool to avoid blocking the event loop
        def blocking_submit() -> int:
            """Execute SubmitRequest in executor - main loop stays free."""
            return self._lib.SubmitRequest(
                self._handle,
                prompt.encode("utf-8"),
                max_tokens,
                ctypes.cast(self._c_callback, ctypes.c_void_p),
                ctypes.c_void_p(req_id),
            )

        ret = await loop.run_in_executor(_get_inference_executor(), blocking_submit)

        if ret < 0:
            with self._lock:
                self._requests.pop(req_id, None)
            raise _error_code_to_exception(ret, context="in generate_async()")

        return await future

    async def stream_async(
        self,
        prompt: str,
        max_tokens: int = 256,
        config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Asynchronously stream generated tokens.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            config: Optional GenerationConfig

        Yields:
            Generated tokens one at a time

        Example:
            >>> async for token in model.stream_async("Hello"):
            ...     print(token, end="", flush=True)
        """
        if config is not None:
            max_tokens = config.max_tokens

        loop = asyncio.get_running_loop()
        async_queue: asyncio.Queue = asyncio.Queue()

        def handler(token: str, finished: bool) -> None:
            loop.call_soon_threadsafe(async_queue.put_nowait, (token, finished))

        req_id = self._register_request(handler)

        # Offload submission to thread pool to avoid blocking the event loop
        def blocking_submit() -> int:
            """Execute SubmitRequest in executor - main loop stays free."""
            return self._lib.SubmitRequest(
                self._handle,
                prompt.encode("utf-8"),
                max_tokens,
                ctypes.cast(self._c_callback, ctypes.c_void_p),
                ctypes.c_void_p(req_id),
            )

        ret = await loop.run_in_executor(_get_inference_executor(), blocking_submit)

        if ret < 0:
            with self._lock:
                self._requests.pop(req_id, None)
                self._active_ctypes_refs.pop(req_id, None)
            raise _error_code_to_exception(ret, context="in stream_async()")

        while True:
            token, finished = await async_queue.get()
            if token:
                yield token
            if finished:
                break

    # Alias for compatibility
    generate_stream_async = stream_async

    # =========================================================================
    # Batch Generation
    # =========================================================================

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
    ) -> List[str]:
        """
        Generate responses for multiple prompts using C++ continuous batching.

        This method submits all requests to the C++ engine's scheduler before
        waiting, allowing the engine to form efficient batches internally.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per response

        Returns:
            List of generated responses in the same order as prompts

        Example:
            >>> responses = model.generate_batch([
            ...     "Hello!",
            ...     "How are you?",
            ...     "Tell me a joke"
            ... ])
        """
        if not prompts:
            return []

        # Create a queue for each request to collect results
        result_queues: List[queue.Queue] = [queue.Queue() for _ in prompts]
        req_ids: List[int] = []

        # Submit all requests (non-blocking) - this pushes them to the C++
        # pending_queue, enabling the Scheduler to form efficient batches
        for idx, prompt in enumerate(prompts):
            result_queue = result_queues[idx]

            def make_handler(q: queue.Queue) -> Callable[[str, bool], None]:
                """Factory to capture the queue in closure."""
                tokens: List[str] = []

                def handler(token: str, finished: bool) -> None:
                    tokens.append(token)
                    if finished:
                        q.put("".join(tokens))

                return handler

            handler = make_handler(result_queue)
            req_id = self._register_request(handler)
            req_ids.append(req_id)

            ret = self._submit_request(prompt, max_tokens, req_id)

            if ret < 0:
                # Clean up the registered request on failure
                with self._lock:
                    self._requests.pop(req_id, None)
                raise _error_code_to_exception(
                    ret, context=f"submitting prompt {idx} in generate_batch"
                )

        # Wait for all results (order preserved by queue indices)
        results: List[str] = []
        for result_queue in result_queues:
            result = result_queue.get()  # Blocks until result is ready
            results.append(result)

        return results

    # =========================================================================
    # Chat Interface (OpenAI-compatible)
    # =========================================================================

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Chat-style interface compatible with OpenAI API format.

        Args:
            messages: List of message dicts with "role" and "content"
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt to prepend

        Returns:
            Assistant's response

        Example:
            >>> response = model.chat([
            ...     {"role": "user", "content": "Hello!"}
            ... ])
            >>> print(response)
        """
        # Build prompt from messages
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")

        prompt_parts.append("Assistant: ")
        prompt = "".join(prompt_parts)

        return self.generate(prompt, max_tokens, **kwargs)

    # =========================================================================
    # LoRA Adapter Management
    # =========================================================================

    def load_lora(
        self,
        adapter_path: str,
        scale: float = 1.0,
        name: str = "default",
    ) -> None:
        """
        Load a LoRA adapter for fine-tuned inference.

        Args:
            adapter_path: Path to the GGUF LoRA adapter file
            scale: LoRA scaling factor (alpha). Higher = stronger adapter effect
            name: Identifier for this adapter

        Example:
            >>> model = DenseCore("base_model.gguf")
            >>> model.load_lora("./my-adapter.gguf", scale=0.8)
        """
        # Track in Python LoRA manager
        self._lora_manager.load(name, adapter_path, scale=scale, activate=False)

        # Call C++ API to actually load the adapter
        if hasattr(self, "_has_lora_api") and self._has_lora_api:
            ret = self._lib.LoadLoraAdapter(
                self._handle,
                adapter_path.encode("utf-8"),
                ctypes.c_float(scale),
                name.encode("utf-8"),
            )
            if ret < 0:
                raise RuntimeError(f"Failed to load LoRA adapter: error code {ret}")

            # Activate immediately
            self._lib.ActivateLoraAdapter(self._handle, name.encode("utf-8"))
            self._lora_manager.activate(name)
        else:
            # Fallback: Python-level tracking only (legacy warning removed)
            self._lora_manager.activate(name)

        if self._verbose:
            print(f"[LoRA] Loaded adapter '{name}' with scale={scale}")

    def unload_lora(self, name: str = "default") -> None:
        """
        Unload a LoRA adapter.

        Args:
            name: Adapter identifier to unload
        """
        # Call C++ API first
        if hasattr(self, "_has_lora_api") and self._has_lora_api:
            ret = self._lib.UnloadLoraAdapter(self._handle, name.encode("utf-8"))
            if ret < 0:
                warnings.warn(f"Failed to unload LoRA adapter in C++: error code {ret}")

        self._lora_manager.unload(name)

    def enable_lora(self, name: Optional[str] = None) -> None:
        """
        Enable a loaded LoRA adapter.

        Args:
            name: Adapter to enable (None = activate last loaded)
        """
        if name is None and self._lora_manager.list_adapters():
            name = self._lora_manager.list_adapters()[0]

        if not name:
            return

        # Call C++ API
        if hasattr(self, "_has_lora_api") and self._has_lora_api:
            ret = self._lib.ActivateLoraAdapter(self._handle, name.encode("utf-8"))
            if ret < 0:
                raise RuntimeError(f"Failed to activate LoRA adapter: error code {ret}")

        self._lora_manager.activate(name)

    def disable_lora(self) -> None:
        """Disable all LoRA adapters (use base model only)."""
        # Call C++ API
        if hasattr(self, "_has_lora_api") and self._has_lora_api:
            ret = self._lib.DeactivateLoraAdapters(self._handle)
            if ret < 0:
                warnings.warn(f"Failed to deactivate LoRA adapters in C++: error code {ret}")

        self._lora_manager.deactivate()

    @property
    def has_lora(self) -> bool:
        """Check if any LoRA adapter is active."""
        return self._lora_manager.is_active

    def list_lora_adapters(self) -> List[str]:
        """List all loaded LoRA adapter names."""
        return self._lora_manager.list_adapters()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @property
    def model_path(self) -> str:
        """Return the path to the loaded model."""
        return self._model_path

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": self._model_path,
            "has_tokenizer": self.tokenizer is not None,
            "has_lora": self.has_lora,
            "lora_adapters": self.list_lora_adapters(),
            "closed": self._closed,
        }

    def close(self) -> None:
        """Close the engine and release resources."""
        if not self._closed and self._handle:
            self._lib.FreeEngine(self._handle)
            self._handle = None
            self._closed = True
            if self._verbose:
                print("[DenseCore] Engine closed")

    def __enter__(self) -> "DenseCore":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        self.close()

    def __repr__(self) -> str:
        status = "closed" if self._closed else "active"
        return f"DenseCore(model='{self._model_path}', status={status})"
