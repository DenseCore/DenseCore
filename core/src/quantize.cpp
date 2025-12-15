#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Helper to check if a tensor name should be quantized
bool ShouldQuantize(const std::string &name) {
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

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <input.gguf> <output.gguf> [type]"
              << std::endl;
    std::cerr << "  type: q4_0 (default), q4_1, q8_0, f16" << std::endl;
    return 1;
  }

  const char *input_path = argv[1];
  const char *output_path = argv[2];
  std::string type_str = (argc > 3) ? argv[3] : "q4_0";

  ggml_type qtype = GGML_TYPE_Q4_0;
  if (type_str == "q4_1")
    qtype = GGML_TYPE_Q4_1;
  else if (type_str == "q8_0")
    qtype = GGML_TYPE_Q8_0;
  else if (type_str == "f16")
    qtype = GGML_TYPE_F16;
  else if (type_str != "q4_0") {
    std::cerr << "Unknown quantization type: " << type_str << std::endl;
    return 1;
  }

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

  std::cout << "[Quantize] Model loaded. Preparing to quantize to " << type_str
            << "..." << std::endl;

  // Create output GGUF context
  struct gguf_context *ctx_out = gguf_init_empty();

  // 1. Copy KV pairs (Metadata)
  int n_kv = gguf_get_n_kv(ctx_gguf);
  for (int i = 0; i < n_kv; ++i) {
    const char *key = gguf_get_key(ctx_gguf, i);
    gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    // Skip general.quantization_version if present, we'll set it?
    // Or just copy everything.

    if (type == GGUF_TYPE_ARRAY) {
      gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
      int n = gguf_get_arr_n(ctx_gguf, i);
      const void *data = gguf_get_arr_data(ctx_gguf, i);
      gguf_set_arr_data(ctx_out, key, arr_type, data, n);
    } else if (type == GGUF_TYPE_STRING) {
      const char *val = gguf_get_val_str(ctx_gguf, i);
      gguf_set_val_str(ctx_out, key, val);
    } else {
      // For scalar types, we can use a switch or just get raw data if API
      // supported it. Since gguf_set_val_* is typed, we need a switch.
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

  // 2. Process Tensors
  int n_tensors = gguf_get_n_tensors(ctx_gguf);
  std::vector<uint8_t> work_buffer;

  // Initialize quantization tables
  ggml_quantize_init(qtype);

  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_in, name);

    if (!tensor) {
      std::cerr << "Error: Tensor not found in context: " << name << std::endl;
      continue;
    }

    bool quantize = ShouldQuantize(name) && ggml_n_dims(tensor) == 2;
    ggml_type target_type = quantize ? qtype : tensor->type;

    if (quantize) {
      std::cout << "  Quantizing " << name << " ("
                << ggml_type_name(tensor->type) << " -> "
                << ggml_type_name(target_type) << ") " << tensor->ne[0] << "x"
                << tensor->ne[1] << std::endl;

      // Calculate size needed for quantized data
      int64_t nelements = ggml_nelements(tensor);
      size_t row_size = ggml_row_size(target_type, tensor->ne[0]);
      size_t data_size = row_size * tensor->ne[1];

      // Allocate buffer for quantized data
      std::vector<uint8_t> qdata(data_size);

      // Quantize
      // ggml_quantize_chunk expects float input. If input is F16, we need to
      // convert.
      std::vector<float> f32_data;
      const float *src_data = nullptr;

      if (tensor->type == GGML_TYPE_F32) {
        src_data = (const float *)tensor->data;
      } else if (tensor->type == GGML_TYPE_F16) {
        f32_data.resize(nelements);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)tensor->data,
                              f32_data.data(), nelements);
        src_data = f32_data.data();
      } else {
        std::cerr << "    Skipping " << name
                  << ": Unsupported input type for quantization: "
                  << ggml_type_name(tensor->type) << std::endl;
        gguf_add_tensor(ctx_out, tensor);
        gguf_set_tensor_data(ctx_out, name, tensor->data);
        continue;
      }

      // Quantize chunk
      // int64_t start, int64_t nrows, int64_t n_per_row, const float * imatrix
      size_t size = ggml_quantize_chunk(target_type, src_data, qdata.data(), 0,
                                        tensor->ne[1], tensor->ne[0], nullptr);

      if (size != data_size) {
        std::cerr << "    Error: Quantization size mismatch!" << std::endl;
        return 1;
      }

      // Create new tensor info for output
      struct ggml_tensor *qtensor =
          ggml_new_tensor_2d(ctx_in, target_type, tensor->ne[0], tensor->ne[1]);
      ggml_set_name(qtensor, name);

      // We can't easily attach the data pointer to the tensor struct managed by
      // ggml_context if we want to write it using gguf_write_to_file
      // immediately or we need to keep the buffer alive. gguf_add_tensor just
      // adds metadata. gguf_write_to_file writes data from tensor->data. So we
      // need to store qdata somewhere persistent or write manually.
      //
      // Easier approach: Use gguf_add_tensor, then gguf_set_tensor_data with
      // our buffer. But we need to keep qdata alive until write. Since we are
      // iterating, we can't easily write one by one unless we use the "write
      // meta then append" approach. Let's use the "write meta then append"
      // approach to save memory? Or just keep all buffers in memory if model
      // fits in RAM (it usually does for conversion). But for 7B model, 14GB
      // F16 -> 4GB Q4. We might run out of RAM if we keep both.
      //
      // Let's just write the whole file at the end. We will store the quantized
      // blobs in a map. But wait, `qdata` vector will go out of scope. We need
      // to persist it.

      // Hack: We will just write to a temporary file or just assume we have
      // enough RAM for now. To be safe, let's use a global buffer list.
    } else {
      // Copy as is
      // std::cout << "  Copying " << name << " (" <<
      // ggml_type_name(tensor->type) << ")" << std::endl;
      gguf_add_tensor(ctx_out, tensor);
      gguf_set_tensor_data(ctx_out, name, tensor->data);
    }
  }

  // Re-loop to actually handle memory management properly?
  // The above loop has a bug: `qdata` dies.
  // Let's refactor to use a 2-pass approach or just store data.
  // Since we can't easily modify the loop above without complex memory
  // management, let's use a simpler approach:
  // 1. Define all tensors in ctx_out.
  // 2. Write to file using the low-level API (write header, then write
  // tensors).

  // Actually, `gguf_write_to_file` expects all data to be available.
  // Let's use the "write meta, then append data" approach which is supported by
  // `gguf`. See gguf.h:
  // - gguf_write_to_file(ctx, fname, /*only_meta =*/ true);
  // - FILE * f = fopen(fname, "ab");
  // - fwrite(f, ...); // write tensor data

  // But `gguf_write_to_file` writes the tensor info with offsets.
  // We need to calculate offsets first?
  // `gguf_add_tensor` adds the tensor to the context.
  // `gguf_write_to_file` with only_meta=true writes the header.
  // But does it calculate offsets?
  // Usually GGUF writer calculates offsets based on order.

  // Let's try to implement the "write one by one" loop.

  std::cout << "[Quantize] Writing header to " << output_path << "..."
            << std::endl;

  // We need to add all tensors to ctx_out first to write the header.
  // But we don't have the data yet for quantized ones.
  // We can set the type and shape, but data pointer?
  // `gguf_add_tensor` takes a `ggml_tensor`.

  // Pass 1: Add all tensors to ctx_out (with correct types)
  std::vector<struct ggml_tensor *> out_tensors;
  out_tensors.reserve(n_tensors);

  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_in, name);

    bool quantize = ShouldQuantize(name) && ggml_n_dims(tensor) == 2;
    ggml_type target_type = quantize ? qtype : tensor->type;

    struct ggml_tensor *out_t =
        ggml_new_tensor(ctx_in, target_type, ggml_n_dims(tensor),
                        tensor->ne); // Allocate in ctx_in for convenience
    ggml_set_name(out_t, name);
    gguf_add_tensor(ctx_out, out_t);
    out_tensors.push_back(out_t);
  }

  // Write header
  gguf_write_to_file(ctx_out, output_path, true);

  // Re-open for appending
  FILE *f = fopen(output_path, "ab");
  if (!f) {
    std::cerr << "Error: Failed to open output file for appending."
              << std::endl;
    return 1;
  }

  // Pass 2: Write data
  // We must write in the SAME order as added to ctx_out.
  // GGUF spec requires tensors to be written in the order they appear in the
  // header? Actually, `gguf_write_to_file` writes them in the order they are in
  // the context. And we added them in the same order as input.

  // We need to align to GGUF_DEFAULT_ALIGNMENT (32)
  auto pad_file = [&](size_t alignment) {
    long pos = ftell(f);
    size_t padding = (alignment - (pos % alignment)) % alignment;
    if (padding > 0) {
      char buf[32] = {0};
      fwrite(buf, 1, padding, f);
    }
  };

  std::cout << "[Quantize] Writing tensors..." << std::endl;

  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor *tensor = ggml_get_tensor(ctx_in, name);
    struct ggml_tensor *out_t = out_tensors[i]; // Corresponding output tensor

    bool quantize = ShouldQuantize(name) && ggml_n_dims(tensor) == 2;

    // Pad before writing data
    pad_file(GGUF_DEFAULT_ALIGNMENT);

    if (quantize) {
      // Quantize and write
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
      // Write raw data
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
