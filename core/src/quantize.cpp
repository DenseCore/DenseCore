/**
 * @file quantize.cpp
 * @brief DenseCore Model Quantization CLI Tool
 *
 * Supports both GGML-native quantization and custom INT4_BLOCKWISE format.
 * For INT4_BLOCKWISE, uses the DenseCore Quantizer abstraction instead
 * of raw ggml_quantize_chunk calls.
 */

#include "ggml.h"
#include "gguf.h"
#include "quantization/int4_quantizer.h"
#include "quantization/int4_types.h"
#include "quantization_config.h"
#include "quantizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace densecore;

// ============================================================================
// Tensor Selection Logic
// ============================================================================

/**
 * Check if a tensor should be quantized based on its name.
 */
bool ShouldQuantize(const std::string &name, const QuantConfig &config) {
  // Skip embeddings if configured
  if (config.skip_embeddings) {
    if (name.find("tok_embed") != std::string::npos ||
        name.find("token_embed") != std::string::npos ||
        name.find("wte") != std::string::npos) {
      return false;
    }
  }

  // Skip output layer (LM head) if configured
  if (config.skip_output_layer) {
    if (name.find("output") != std::string::npos &&
        name.find("layer") == std::string::npos) {
      return false; // Skip lm_head but not layer outputs
    }
    if (name.find("lm_head") != std::string::npos) {
      return false;
    }
  }

  // Quantize weights (2D), but skip norms and biases
  if (name.find("weight") != std::string::npos) {
    if (name.find("norm") != std::string::npos)
      return false; // Skip normalization weights (keep F32)
    if (name.find("bias") != std::string::npos)
      return false; // Skip biases (keep F32)
    return true;    // Quantize attention/ffn weights
  }

  return false;
}

/**
 * Convert QuantFormat to GGML type for GGML-native quantization.
 */
ggml_type FormatToGGMLType(QuantFormat format) {
  switch (format) {
  case QuantFormat::Q4_0:
    return GGML_TYPE_Q4_0;
  case QuantFormat::Q4_K_M:
    return GGML_TYPE_Q4_K;
  case QuantFormat::Q5_K_M:
    return GGML_TYPE_Q5_K;
  case QuantFormat::Q8_0:
    return GGML_TYPE_Q8_0;
  case QuantFormat::FP16:
    return GGML_TYPE_F16;
  default:
    return GGML_TYPE_Q4_0;
  }
}

/**
 * Parse command line to create QuantConfig.
 */
QuantConfig ParseConfig(int argc, char **argv) {
  std::string type_str = (argc > 3) ? argv[3] : "q4_0";
  int block_size = (argc > 4) ? std::atoi(argv[4]) : 128;

  // Parse format
  if (type_str == "int4" || type_str == "int4_blockwise" ||
      type_str == "int4_paper") {
    return INT4_PAPER_CFG(block_size);
  } else if (type_str == "q4_k_m" || type_str == "q4_k") {
    return Q4_K_M_CFG();
  } else if (type_str == "q5_k_m" || type_str == "q5_k") {
    return Q5_K_M_CFG();
  } else if (type_str == "q8_0" || type_str == "int8") {
    return Q8_0_CFG();
  } else if (type_str == "q4_0") {
    QuantConfig cfg;
    cfg.format = QuantFormat::Q4_0;
    cfg.algorithm = QuantAlgorithm::GGML_Q4_0;
    return cfg;
  } else if (type_str == "q4_1") {
    // Q4_1 uses Q4_0 config but different GGML type
    QuantConfig cfg;
    cfg.format = QuantFormat::Q4_0;
    cfg.algorithm = QuantAlgorithm::GGML_Q4_0;
    return cfg;
  } else if (type_str == "f16" || type_str == "fp16") {
    QuantConfig cfg;
    cfg.format = QuantFormat::FP16;
    cfg.quantize_weights = false;
    return cfg;
  }

  // Default
  QuantConfig cfg;
  cfg.format = QuantFormat::Q4_0;
  return cfg;
}

// ============================================================================
// GGML-Native Quantization Path
// ============================================================================

/**
 * Quantize model using GGML-native quantization (q4_0, q8_0, etc.)
 */
int QuantizeGGML(const char *input_path, const char *output_path,
                 const QuantConfig &config) {
  ggml_type qtype = FormatToGGMLType(config.format);

  std::cout << "[Quantize] Loading model '" << input_path << "'..."
            << std::endl;

  struct gguf_init_params params = {
      .no_alloc = false,
      .ctx = nullptr,
  };

  struct ggml_context *ctx_in = nullptr;
  params.ctx = &ctx_in;

  struct gguf_context *ctx_gguf = gguf_init_from_file(input_path, params);
  if (!ctx_gguf) {
    std::cerr << "Error: Failed to load GGUF file: " << input_path << std::endl;
    return 1;
  }

  std::cout << "[Quantize] Model loaded. Quantizing to "
            << config.GetFormatName() << "..." << std::endl;

  // Create output GGUF context
  struct gguf_context *ctx_out = gguf_init_empty();

  // Copy KV pairs (Metadata)
  int n_kv = gguf_get_n_kv(ctx_gguf);
  for (int i = 0; i < n_kv; ++i) {
    const char *key = gguf_get_key(ctx_gguf, i);
    gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    if (type == GGUF_TYPE_ARRAY) {
      gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
      int n = gguf_get_arr_n(ctx_gguf, i);
      const void *data = gguf_get_arr_data(ctx_gguf, i);
      gguf_set_arr_data(ctx_out, key, arr_type, data, n);
    } else if (type == GGUF_TYPE_STRING) {
      const char *val = gguf_get_val_str(ctx_gguf, i);
      gguf_set_val_str(ctx_out, key, val);
    } else {
      switch (type) {
      case GGUF_TYPE_UINT8:
        gguf_set_val_u8(ctx_out, key, gguf_get_val_u8(ctx_gguf, i));
        break;
      case GGUF_TYPE_INT8:
        gguf_set_val_i8(ctx_out, key, gguf_get_val_i8(ctx_gguf, i));
        break;
      case GGUF_TYPE_UINT16:
        gguf_set_val_u16(ctx_out, key, gguf_get_val_u16(ctx_gguf, i));
        break;
      case GGUF_TYPE_INT16:
        gguf_set_val_i16(ctx_out, key, gguf_get_val_i16(ctx_gguf, i));
        break;
      case GGUF_TYPE_UINT32:
        gguf_set_val_u32(ctx_out, key, gguf_get_val_u32(ctx_gguf, i));
        break;
      case GGUF_TYPE_INT32:
        gguf_set_val_i32(ctx_out, key, gguf_get_val_i32(ctx_gguf, i));
        break;
      case GGUF_TYPE_FLOAT32:
        gguf_set_val_f32(ctx_out, key, gguf_get_val_f32(ctx_gguf, i));
        break;
      case GGUF_TYPE_UINT64:
        gguf_set_val_u64(ctx_out, key, gguf_get_val_u64(ctx_gguf, i));
        break;
      case GGUF_TYPE_INT64:
        gguf_set_val_i64(ctx_out, key, gguf_get_val_i64(ctx_gguf, i));
        break;
      case GGUF_TYPE_FLOAT64:
        gguf_set_val_f64(ctx_out, key, gguf_get_val_f64(ctx_gguf, i));
        break;
      case GGUF_TYPE_BOOL:
        gguf_set_val_bool(ctx_out, key, gguf_get_val_bool(ctx_gguf, i));
        break;
      default:
        std::cerr << "Warning: Skipping unknown KV type for key: " << key
                  << std::endl;
      }
    }
  }

  // Initialize quantization tables
  ggml_quantize_init(qtype);

  // Process tensors (Pass 1: Add tensor info)
  int n_tensors = gguf_get_n_tensors(ctx_gguf);
  std::vector<struct ggml_tensor *> out_tensors;
  out_tensors.reserve(n_tensors);

  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_in, name);

    bool quantize = ShouldQuantize(name, config) && ggml_n_dims(tensor) == 2;
    ggml_type target_type = quantize ? qtype : tensor->type;

    struct ggml_tensor *out_t =
        ggml_new_tensor(ctx_in, target_type, ggml_n_dims(tensor), tensor->ne);
    ggml_set_name(out_t, name);
    gguf_add_tensor(ctx_out, out_t);
    out_tensors.push_back(out_t);
  }

  // Write header
  std::cout << "[Quantize] Writing header to " << output_path << "..."
            << std::endl;
  gguf_write_to_file(ctx_out, output_path, true);

  // Re-open for appending
  FILE *f = fopen(output_path, "ab");
  if (!f) {
    std::cerr << "Error: Failed to open output file for appending."
              << std::endl;
    return 1;
  }

  // Padding helper
  auto pad_file = [&](size_t alignment) {
    long pos = ftell(f);
    size_t padding = (alignment - (pos % alignment)) % alignment;
    if (padding > 0) {
      char buf[32] = {0};
      fwrite(buf, 1, padding, f);
    }
  };

  // Process tensors (Pass 2: Write data)
  std::cout << "[Quantize] Writing tensors..." << std::endl;

  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_in, name);
    struct ggml_tensor *out_t = out_tensors[i];

    bool quantize = ShouldQuantize(name, config) && ggml_n_dims(tensor) == 2;

    pad_file(GGUF_DEFAULT_ALIGNMENT);

    if (quantize) {
      int64_t nelements = ggml_nelements(tensor);
      size_t row_size = ggml_row_size(out_t->type, tensor->ne[0]);
      size_t data_size = row_size * tensor->ne[1];

      std::vector<uint8_t> qdata(data_size);
      std::vector<float> f32_data;
      const float *src_data = nullptr;

      if (tensor->type == GGML_TYPE_F32) {
        src_data = (const float *)tensor->data;
      } else if (tensor->type == GGML_TYPE_F16) {
        f32_data.resize(nelements);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)tensor->data,
                              f32_data.data(), nelements);
        src_data = f32_data.data();
      }

      ggml_quantize_chunk(out_t->type, src_data, qdata.data(), 0, tensor->ne[1],
                          tensor->ne[0], nullptr);

      fwrite(qdata.data(), 1, data_size, f);

      std::cout << "  Converted " << name << " [" << tensor->ne[0] << "x"
                << tensor->ne[1] << "]" << std::endl;
    } else {
      size_t size = ggml_nbytes(tensor);
      fwrite(tensor->data, 1, size, f);
    }
  }

  fclose(f);

  gguf_free(ctx_out);
  gguf_free(ctx_gguf);
  ggml_free(ctx_in);

  std::cout << "[Quantize] Done. Output saved to " << output_path << std::endl;
  return 0;
}

// ============================================================================
// Custom INT4 Quantization Path (DenseCore Quantizer)
// ============================================================================

/**
 * Check if a tensor has been quantized to INT4 format.
 */
inline bool IsINT4Quantized(const struct ggml_tensor *tensor) {
  if (!tensor || !tensor->extra)
    return false;
  const TensorInt4 *int4 = static_cast<const TensorInt4 *>(tensor->extra);
  return (int4 && int4->q_data && int4->scales && int4->zero_points);
}

/**
 * Quantize model using DenseCore's custom INT4_BLOCKWISE format.
 *
 * Uses the "Split Tensor" approach for GGUF serialization:
 * - {tensor_name}        - Packed INT4 weights (raw bytes)
 * - {tensor_name}_scales - Per-block scale factors (FP32)
 * - {tensor_name}_zeros  - Per-block zero points (FP32)
 * - {tensor_name}_meta   - Metadata (group_size, num_blocks) as KV
 */
int QuantizeINT4Custom(const char *input_path, const char *output_path,
                       const QuantConfig &config) {
  std::cout << "[Quantize] Loading model '" << input_path << "'..."
            << std::endl;

  struct gguf_init_params params = {
      .no_alloc = false,
      .ctx = nullptr,
  };

  struct ggml_context *ctx_in = nullptr;
  params.ctx = &ctx_in;

  struct gguf_context *ctx_gguf = gguf_init_from_file(input_path, params);
  if (!ctx_gguf) {
    std::cerr << "Error: Failed to load GGUF file: " << input_path << std::endl;
    return 1;
  }

  std::cout << "[Quantize] Model loaded. Using custom INT4 quantization "
            << "(block_size=" << config.block_size << ")..." << std::endl;

  // Create the quantizer
  std::unique_ptr<Quantizer> quantizer;
  try {
    quantizer = CreateQuantizer(config);
  } catch (const std::exception &e) {
    std::cerr << "Error: Failed to create quantizer: " << e.what() << std::endl;
    gguf_free(ctx_gguf);
    ggml_free(ctx_in);
    return 1;
  }

  // Create output GGUF context
  struct gguf_context *ctx_out = gguf_init_empty();

  // Copy KV pairs (Metadata) from input
  int n_kv = gguf_get_n_kv(ctx_gguf);
  for (int i = 0; i < n_kv; ++i) {
    const char *key = gguf_get_key(ctx_gguf, i);
    gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    if (type == GGUF_TYPE_ARRAY) {
      gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
      int n = gguf_get_arr_n(ctx_gguf, i);
      const void *data = gguf_get_arr_data(ctx_gguf, i);
      gguf_set_arr_data(ctx_out, key, arr_type, data, n);
    } else if (type == GGUF_TYPE_STRING) {
      const char *val = gguf_get_val_str(ctx_gguf, i);
      gguf_set_val_str(ctx_out, key, val);
    } else {
      switch (type) {
      case GGUF_TYPE_UINT8:
        gguf_set_val_u8(ctx_out, key, gguf_get_val_u8(ctx_gguf, i));
        break;
      case GGUF_TYPE_INT8:
        gguf_set_val_i8(ctx_out, key, gguf_get_val_i8(ctx_gguf, i));
        break;
      case GGUF_TYPE_UINT16:
        gguf_set_val_u16(ctx_out, key, gguf_get_val_u16(ctx_gguf, i));
        break;
      case GGUF_TYPE_INT16:
        gguf_set_val_i16(ctx_out, key, gguf_get_val_i16(ctx_gguf, i));
        break;
      case GGUF_TYPE_UINT32:
        gguf_set_val_u32(ctx_out, key, gguf_get_val_u32(ctx_gguf, i));
        break;
      case GGUF_TYPE_INT32:
        gguf_set_val_i32(ctx_out, key, gguf_get_val_i32(ctx_gguf, i));
        break;
      case GGUF_TYPE_FLOAT32:
        gguf_set_val_f32(ctx_out, key, gguf_get_val_f32(ctx_gguf, i));
        break;
      case GGUF_TYPE_UINT64:
        gguf_set_val_u64(ctx_out, key, gguf_get_val_u64(ctx_gguf, i));
        break;
      case GGUF_TYPE_INT64:
        gguf_set_val_i64(ctx_out, key, gguf_get_val_i64(ctx_gguf, i));
        break;
      case GGUF_TYPE_FLOAT64:
        gguf_set_val_f64(ctx_out, key, gguf_get_val_f64(ctx_gguf, i));
        break;
      case GGUF_TYPE_BOOL:
        gguf_set_val_bool(ctx_out, key, gguf_get_val_bool(ctx_gguf, i));
        break;
      default:
        break;
      }
    }
  }

  // Add INT4 format metadata
  gguf_set_val_str(ctx_out, "densecore.quantization_format", "int4_blockwise");
  gguf_set_val_u32(ctx_out, "densecore.block_size", config.block_size);

  // Pass 1: Quantize tensors and collect info
  int n_tensors = gguf_get_n_tensors(ctx_gguf);
  std::vector<std::string> tensor_names;
  std::vector<bool> is_quantized;
  int quantized_count = 0;

  std::cout << "[Quantize] Processing " << n_tensors << " tensors..."
            << std::endl;

  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_in, name);

    if (!tensor) {
      std::cerr << "Warning: Tensor not found: " << name << std::endl;
      tensor_names.push_back(name);
      is_quantized.push_back(false);
      continue;
    }

    bool should_quantize =
        ShouldQuantize(name, config) && ggml_n_dims(tensor) == 2;

    if (should_quantize) {
      std::cout << "  Quantizing " << name << " [" << tensor->ne[0] << "x"
                << tensor->ne[1] << "]..." << std::endl;

      quantizer->QuantizeWeight(tensor);
      quantized_count++;
    }

    tensor_names.push_back(name);
    is_quantized.push_back(should_quantize && IsINT4Quantized(tensor));
  }

  // Pass 2: Add tensor definitions to GGUF context
  // For INT4 tensors, we add 3 tensors: qweight, scales, zeros
  for (int i = 0; i < n_tensors; ++i) {
    const char *name = tensor_names[i].c_str();
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_in, name);

    if (!tensor)
      continue;

    if (is_quantized[i] && IsINT4Quantized(tensor)) {
      const TensorInt4 *int4 = static_cast<const TensorInt4 *>(tensor->extra);

      // Calculate sizes
      int64_t K = int4->ne[0]; // cols (inner dim)
      int64_t N = int4->ne[1]; // rows (output dim)
      int num_groups = K / int4->group_size;
      int64_t packed_size = N * (K / 2); // Packed INT4 bytes

      // 1. Packed INT4 weights: [N, K/2] as UINT8
      // We store as 1D for simplicity (can't use UINT4 in GGML)
      struct ggml_tensor *qw =
          ggml_new_tensor_1d(ctx_in, GGML_TYPE_I8, packed_size);
      std::string qw_name = std::string(name);
      ggml_set_name(qw, qw_name.c_str());
      gguf_add_tensor(ctx_out, qw);

      // 2. Scales: [N * num_groups] as FP32
      int64_t meta_size = N * num_groups;
      struct ggml_tensor *scales_t =
          ggml_new_tensor_1d(ctx_in, GGML_TYPE_F32, meta_size);
      std::string scales_name = std::string(name) + "_scales";
      ggml_set_name(scales_t, scales_name.c_str());
      gguf_add_tensor(ctx_out, scales_t);

      // 3. Zeros: [N * num_groups] as FP32
      struct ggml_tensor *zeros_t =
          ggml_new_tensor_1d(ctx_in, GGML_TYPE_F32, meta_size);
      std::string zeros_name = std::string(name) + "_zeros";
      ggml_set_name(zeros_t, zeros_name.c_str());
      gguf_add_tensor(ctx_out, zeros_t);

      // Add per-tensor metadata as KV
      std::string meta_key = std::string("densecore.int4.") + name;
      gguf_set_val_u32(ctx_out, (meta_key + ".group_size").c_str(),
                       int4->group_size);
      gguf_set_val_u32(ctx_out, (meta_key + ".num_blocks").c_str(),
                       int4->num_blocks);
      gguf_set_val_i64(ctx_out, (meta_key + ".K").c_str(), K);
      gguf_set_val_i64(ctx_out, (meta_key + ".N").c_str(), N);
    } else {
      // Standard tensor - keep as is
      gguf_add_tensor(ctx_out, tensor);
    }
  }

  // Write header
  std::cout << "[Quantize] Writing header to " << output_path << "..."
            << std::endl;
  gguf_write_to_file(ctx_out, output_path, true);

  // Re-open for appending
  FILE *f = fopen(output_path, "ab");
  if (!f) {
    std::cerr << "Error: Failed to open output file for appending."
              << std::endl;
    gguf_free(ctx_out);
    gguf_free(ctx_gguf);
    ggml_free(ctx_in);
    return 1;
  }

  // Padding helper
  auto pad_file = [&](size_t alignment) {
    long pos = ftell(f);
    size_t padding = (alignment - (pos % alignment)) % alignment;
    if (padding > 0) {
      char buf[32] = {0};
      fwrite(buf, 1, padding, f);
    }
  };

  // Pass 3: Write tensor data
  std::cout << "[Quantize] Writing tensors..." << std::endl;

  for (int i = 0; i < n_tensors; ++i) {
    const char *name = tensor_names[i].c_str();
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_in, name);

    if (!tensor)
      continue;

    if (is_quantized[i] && IsINT4Quantized(tensor)) {
      const TensorInt4 *int4 = static_cast<const TensorInt4 *>(tensor->extra);

      int64_t K = int4->ne[0];
      int64_t N = int4->ne[1];
      int num_groups = K / int4->group_size;
      int64_t packed_size = N * (K / 2);
      int64_t meta_size = N * num_groups;

      // Write packed INT4 weights
      pad_file(GGUF_DEFAULT_ALIGNMENT);
      fwrite(int4->q_data, 1, packed_size, f);

      // Write scales
      pad_file(GGUF_DEFAULT_ALIGNMENT);
      fwrite(int4->scales, sizeof(float), meta_size, f);

      // Write zeros
      pad_file(GGUF_DEFAULT_ALIGNMENT);
      fwrite(int4->zero_points, sizeof(float), meta_size, f);

      std::cout << "  Wrote INT4 tensor " << name << " (" << packed_size
                << " + " << meta_size * 4 * 2 << " bytes)" << std::endl;
    } else {
      // Standard tensor - write raw data
      pad_file(GGUF_DEFAULT_ALIGNMENT);
      size_t size = ggml_nbytes(tensor);
      fwrite(tensor->data, 1, size, f);
    }
  }

  fclose(f);

  gguf_free(ctx_out);
  gguf_free(ctx_gguf);

  // Free INT4 quantized data before freeing ggml context
  // (ggml_free does not manage tensor->extra allocations)
  for (int i = 0; i < n_tensors; ++i) {
    struct ggml_tensor *tensor =
        ggml_get_tensor(ctx_in, tensor_names[i].c_str());
    if (tensor && is_quantized[i]) {
      INT4Quantizer::FreeINT4Data(tensor);
    }
  }
  ggml_free(ctx_in);

  std::cout << std::endl;
  std::cout << "[Quantize] Done! Quantized " << quantized_count << "/"
            << n_tensors << " tensors to INT4_BLOCKWISE." << std::endl;
  std::cout << "[Quantize] Output saved to " << output_path << std::endl;
  std::cout << std::endl;
  std::cout << "INT4 tensors saved with split format:" << std::endl;
  std::cout << "  {name}        - Packed 4-bit weights" << std::endl;
  std::cout << "  {name}_scales - Per-block scale factors" << std::endl;
  std::cout << "  {name}_zeros  - Per-block zero points" << std::endl;

  return 0;
}

// ============================================================================
// Main Entry Point
// ============================================================================

void PrintUsage(const char *prog) {
  std::cerr << "Usage: " << prog
            << " <input.gguf> <output.gguf> [type] [block_size]" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Quantization types:" << std::endl;
  std::cerr << "  q4_0        - 4-bit basic GGML quantization (default)"
            << std::endl;
  std::cerr << "  q4_k_m      - 4-bit K-quants medium (recommended for quality)"
            << std::endl;
  std::cerr << "  q5_k_m      - 5-bit K-quants medium (higher quality)"
            << std::endl;
  std::cerr << "  q8_0        - 8-bit symmetric (highest quality)" << std::endl;
  std::cerr << "  f16         - Keep FP16 (no quantization)" << std::endl;
  std::cerr << "  int4_paper  - Custom INT4 (DenseCore optimized kernels)"
            << std::endl;
  std::cerr << std::endl;
  std::cerr << "Block size (for int4_paper only): 32, 64, or 128 (default: 128)"
            << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    PrintUsage(argv[0]);
    return 1;
  }

  const char *input_path = argv[1];
  const char *output_path = argv[2];

  // Parse configuration
  QuantConfig config = ParseConfig(argc, argv);

  std::cout << "[Quantize] Format: " << config.GetFormatName() << std::endl;
  std::cout << "[Quantize] Algorithm: " << config.GetAlgorithmName()
            << std::endl;

  // Dispatch to appropriate quantization path
  if (config.IsCustomFormat()) {
    return QuantizeINT4Custom(input_path, output_path, config);
  } else {
    return QuantizeGGML(input_path, output_path, config);
  }
}
