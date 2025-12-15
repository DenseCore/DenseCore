#ifndef DENSECORE_QUANTIZATION_CONFIG_H
#define DENSECORE_QUANTIZATION_CONFIG_H

#include <cstdint>
#include <string>

namespace densecore {

/**
 * Quantization algorithms.
 *
 * Note: Advanced algorithms like AWQ and SmoothQuant require special kernel
 * support to un-scale activations at runtime. Without this infrastructure,
 * they damage model accuracy. For MVP, we use reliable GGML-native methods.
 */
enum class QuantAlgorithm {
  // Reliable GGML-native quantization (recommended)
  GGML_Q4_K_M, // 4-bit K-quants with medium accuracy (best quality/size ratio)
  GGML_Q5_K_M, // 5-bit K-quants with medium accuracy (higher quality)
  GGML_Q8_0,   // 8-bit symmetric quantization (highest quality)
  GGML_Q4_0,   // 4-bit basic quantization (fastest, lowest quality)

  // Legacy (kept for API compatibility, maps to safe defaults)
  MAX,         // Maps to GGML_Q4_0
  SMOOTHQUANT, // Maps to GGML_Q8_0 (deprecated: true SmoothQuant not
               // implemented)
  AWQ_LITE,    // Maps to GGML_Q4_K_M (deprecated: true AWQ not implemented)
  AWQ_CLIP,    // Maps to GGML_Q4_K_M (deprecated: true AWQ not implemented)
};

/**
 * Quantization formats (GGML types).
 */
enum class QuantFormat {
  FP16,   // No quantization (baseline, GGML_TYPE_F16)
  Q4_0,   // 4-bit basic quantization (GGML_TYPE_Q4_0)
  Q4_K_M, // 4-bit K-quants medium (GGML_TYPE_Q4_K, recommended for INT4)
  Q5_K_M, // 5-bit K-quants medium (GGML_TYPE_Q5_K)
  Q8_0,   // 8-bit symmetric (GGML_TYPE_Q8_0, recommended for INT8)

  // Legacy aliases (kept for API compatibility)
  FP8_E4M3 = FP16, // Not supported, falls back to FP16
  INT8 = Q8_0,
  INT4_BLOCKWISE = Q4_K_M,
};

/**
 * Quantization configuration.
 */
struct QuantConfig {
  QuantFormat format = QuantFormat::Q4_K_M;
  QuantAlgorithm algorithm = QuantAlgorithm::GGML_Q4_K_M;

  // Blockwise quantization parameters
  int block_size = 32; // Standard GGML block size

  // Quantization targets
  bool quantize_weights = true;
  bool quantize_activations =
      false; // Weight-only (activation quant not supported)

  // Layer selection
  bool skip_output_layer = true; // Skip lm_head for better accuracy
  bool skip_embeddings = true;   // Keep embeddings in FP16

  // Calibration parameters (unused in current implementation)
  int calib_size = 0;
  std::string calib_dataset = "";

  // Helper methods
  std::string GetFormatName() const {
    switch (format) {
    case QuantFormat::FP16:
      return "fp16";
    case QuantFormat::Q4_0:
      return "q4_0";
    case QuantFormat::Q4_K_M:
      return "q4_k_m";
    case QuantFormat::Q5_K_M:
      return "q5_k_m";
    case QuantFormat::Q8_0:
      return "q8_0";
    default:
      return "unknown";
    }
  }

  std::string GetAlgorithmName() const {
    switch (algorithm) {
    case QuantAlgorithm::GGML_Q4_K_M:
      return "q4_k_m";
    case QuantAlgorithm::GGML_Q5_K_M:
      return "q5_k_m";
    case QuantAlgorithm::GGML_Q8_0:
      return "q8_0";
    case QuantAlgorithm::GGML_Q4_0:
      return "q4_0";
    case QuantAlgorithm::MAX:
      return "max";
    case QuantAlgorithm::SMOOTHQUANT:
      return "smoothquant_compat";
    case QuantAlgorithm::AWQ_LITE:
    case QuantAlgorithm::AWQ_CLIP:
      return "awq_compat";
    default:
      return "unknown";
    }
  }
};

// ============================================================================
// Predefined Configurations (Recommended)
// ============================================================================

/**
 * High-quality INT4 quantization (recommended for most use cases).
 * Uses Q4_K_M which provides excellent quality/size tradeoff.
 */
inline QuantConfig Q4_K_M_CFG() {
  QuantConfig cfg;
  cfg.format = QuantFormat::Q4_K_M;
  cfg.algorithm = QuantAlgorithm::GGML_Q4_K_M;
  cfg.quantize_weights = true;
  cfg.skip_output_layer = true;
  cfg.skip_embeddings = true;
  return cfg;
}

/**
 * Higher-quality INT5 quantization (when accuracy is critical).
 */
inline QuantConfig Q5_K_M_CFG() {
  QuantConfig cfg;
  cfg.format = QuantFormat::Q5_K_M;
  cfg.algorithm = QuantAlgorithm::GGML_Q5_K_M;
  cfg.quantize_weights = true;
  cfg.skip_output_layer = true;
  cfg.skip_embeddings = true;
  return cfg;
}

/**
 * INT8 quantization (highest quality, larger model size).
 */
inline QuantConfig Q8_0_CFG() {
  QuantConfig cfg;
  cfg.format = QuantFormat::Q8_0;
  cfg.algorithm = QuantAlgorithm::GGML_Q8_0;
  cfg.quantize_weights = true;
  cfg.skip_output_layer = false; // Q8 is accurate enough for output layer
  cfg.skip_embeddings = false;
  return cfg;
}

// ============================================================================
// Legacy Configurations (Deprecated, kept for API compatibility)
// ============================================================================

/**
 * @deprecated Use Q4_K_M_CFG() instead.
 * AWQ is not properly implemented; this uses standard Q4_K_M quantization.
 */
inline QuantConfig INT4_AWQ_CFG() { return Q4_K_M_CFG(); }

/**
 * @deprecated Use Q8_0_CFG() instead.
 * SmoothQuant requires runtime activation scaling which is not implemented.
 */
inline QuantConfig INT8_SMOOTHQUANT_CFG() { return Q8_0_CFG(); }

/**
 * @deprecated FP8 is not supported. Falls back to FP16.
 */
inline QuantConfig FP8_DEFAULT_CFG() {
  QuantConfig cfg;
  cfg.format = QuantFormat::FP16;
  cfg.quantize_weights = false;
  return cfg;
}

} // namespace densecore

#endif // DENSECORE_QUANTIZATION_CONFIG_H
