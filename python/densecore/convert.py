"""
HuggingFace to GGUF Conversion Module

Provides Python API for converting HuggingFace models to GGUF format
without requiring llama.cpp command-line tools.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

# Type hints for quantization levels
QuantType = Literal["Q4_0", "Q4_1", "Q4_K_M", "Q5_K_M", "Q8_0", "F16", "F32"]


def convert_from_hf(
    model_id: str,
    output_path: Optional[str] = None,
    quantization: QuantType = "Q4_K_M",
    cache_dir: Optional[str] = None,
    use_fast_tokenizer: bool = True,
) -> str:
    """
    Convert a HuggingFace model directly to GGUF format.
    
    This is a convenience wrapper that:
    1. Downloads the HF model (if not cached)
    2. Converts to GGUF using llama.cpp converter
    3. Quantizes to specified format
    
    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-4B")
        output_path: Output GGUF file path. If None, saves to current directory
        quantization: Quantization type (Q4_K_M recommended)
        cache_dir: HuggingFace cache directory
        use_fast_tokenizer: Use fast tokenizer if available
        
    Returns:
        Path to the converted GGUF file
        
    Example:
        >>> from densecore import convert_from_hf
        >>> gguf_path = convert_from_hf("Qwen/Qwen3-4B", quantization="Q4_K_M")
        >>> print(f"Model saved to: {gguf_path}")
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "Converting from HuggingFace requires transformers and torch.\n"
            "Install with: pip install transformers torch"
        )
    
    logger.info(f"Converting {model_id} to GGUF with {quantization} quantization")
    
    # Generate output path if not provided
    if output_path is None:
        model_name = model_id.split("/")[-1].lower()
        quant_suffix = quantization.lower().replace("_", "-")
        output_path = f"{model_name}-{quant_suffix}.gguf"
    
    output_path = Path(output_path).absolute()
    
    # Check if llama.cpp converter is available
    converter_path = _find_llamacpp_converter()
    if converter_path:
        # Use llama.cpp converter (faster, more reliable)
        return _convert_via_llamacpp(
            model_id, output_path, quantization, converter_path
        )
    else:
        # Native Python conversion (slower, but works without llama.cpp)
        return _convert_native(
            model_id, output_path, quantization, cache_dir, use_fast_tokenizer
        )


def _find_llamacpp_converter() -> Optional[Path]:
    """Try to find llama.cpp converter script."""
    # Check common locations
    possible_paths = [
        Path("llama.cpp/convert_hf_to_gguf.py"),  # Current directory
        Path("../llama.cpp/convert_hf_to_gguf.py"),  # Parent directory (DenseCore structure)
        Path.home() / "llama.cpp/convert_hf_to_gguf.py",  # Home directory
        Path("/usr/local/bin/convert_hf_to_gguf.py"),  # System-wide
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found llama.cpp converter at: {path}")
            return path.absolute()
    
    # Try to find in PATH
    try:
        result = subprocess.run(
            ["which", "convert_hf_to_gguf.py"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except Exception:
        pass
    
    return None


def _convert_via_llamacpp(
    model_id: str,
    output_path: Path,
    quantization: QuantType,
    converter_path: Path,
) -> str:
    """Convert using llama.cpp converter (recommended)."""
    logger.info(f"Using llama.cpp converter at {converter_path}")
    
    # Step 1: Download model from HuggingFace
    try:
        from huggingface_hub import snapshot_download
        logger.info(f"Downloading {model_id} from HuggingFace...")
        model_dir = snapshot_download(repo_id=model_id, repo_type="model")
        logger.info(f"Model downloaded to: {model_dir}")
    except ImportError:
        raise ImportError(
            "huggingface_hub required for model download.\n"
            "Install with: pip install huggingface_hub"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        f16_path = Path(tmpdir) / "model-f16.gguf"
        
        # Step 2: Convert to F16 GGUF
        logger.info(f"Converting {model_id} to F16 GGUF...")
        cmd = [
            "python3",
            str(converter_path),
            model_dir,  # Use downloaded directory path, not model ID
            "--outfile",
            str(f16_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed: {result.stderr}")
        
        # Step 3: Quantize
        if quantization in ["F16", "F32"]:
            # No quantization needed, just rename
            f16_path.rename(output_path)
        else:
            logger.info(f"Quantizing to {quantization}...")
            _quantize_gguf(f16_path, output_path, quantization)
    
    logger.info(f"✅ Conversion complete: {output_path}")
    return str(output_path)


def _convert_native(
    model_id: str,
    output_path: Path,
    quantization: QuantType,
    cache_dir: Optional[str],
    use_fast_tokenizer: bool,
) -> str:
    """
    Native Python conversion (fallback when llama.cpp not available).
    
    Note: This is a simplified implementation. For production use,
    install llama.cpp for more robust conversion.
    """
    logger.warning(
        "llama.cpp converter not found. Using native Python conversion.\n"
        "For better results, install llama.cpp:\n"
        "  git clone https://github.com/ggerganov/llama.cpp"
    )
    
    raise NotImplementedError(
        "Native Python conversion not yet implemented.\n"
        "Please install llama.cpp and try again.\n\n"
        "Quick install:\n"
        "  git clone https://github.com/ggerganov/llama.cpp\n"
        "  cd llama.cpp && make\n\n"
        "Then run convert_from_hf() again."
    )


def _quantize_gguf(input_path: Path, output_path: Path, qtype: QuantType) -> None:
    """Quantize a GGUF file using DenseCore's quantize tool."""
    # Find quantize binary
    quantize_bin = _find_quantize_binary()
    
    if not quantize_bin:
        raise FileNotFoundError(
            "DenseCore quantize binary not found.\n"
            "Please build DenseCore first: make lib"
        )
    
    cmd = [str(quantize_bin), str(input_path), str(output_path), qtype]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Quantization failed: {result.stderr}")


def _find_quantize_binary() -> Optional[Path]:
    """Find quantize binary (preferring llama.cpp's llama-quantize)."""
    possible_paths = [
        # llama.cpp paths (prioritize these for modern quant types like Q4_K_M)
        Path("../llama.cpp/build/bin/llama-quantize"),
        Path("llama.cpp/build/bin/llama-quantize"),
        Path.home() / "llama.cpp/build/bin/llama-quantize",
        
        # DenseCore older paths
        Path(__file__).parent.parent.parent / "core/bin/quantize",
        Path(__file__).parent.parent.parent / "core/build/quantize",
        Path("/usr/local/bin/densecore-quantize"),
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found quantize binary at: {path}")
            return path.absolute()
    
    return None


# Convenience function for quick conversion with defaults
def quick_convert(model_id: str) -> str:
    """
    Quick conversion with recommended settings.
    
    Converts to Q4_K_M (best balance of size/quality) and saves to current directory.
    
    Example:
        >>> from densecore import quick_convert
        >>> path = quick_convert("Qwen/Qwen3-4B")
    """
    return convert_from_hf(model_id, quantization="Q4_K_M")
