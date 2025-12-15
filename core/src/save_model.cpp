#include "save_model.h"
#include <cstring>
#include <ggml.h>
#include <gguf.h>
#include <iostream>

namespace densecore {

// Helper to map GGMT tensor type to GGUF tensor type string
static const char *GetGGUFTypeName(enum ggml_type type) {
  switch (type) {
  case GGML_TYPE_F32:
    return "F32";
  case GGML_TYPE_F16:
    return "F16";
  case GGML_TYPE_Q4_0:
    return "Q4_0";
  case GGML_TYPE_Q4_1:
    return "Q4_1";
  case GGML_TYPE_Q5_0:
    return "Q5_0";
  case GGML_TYPE_Q5_1:
    return "Q5_1";
  case GGML_TYPE_Q8_0:
    return "Q8_0";
  case GGML_TYPE_Q8_1:
    return "Q8_1";
  default:
    return "UNKNOWN";
  }
}

// Helper to add a tensor to GGUF context
static void AddTensorToGGUF(gguf_context *ctx, const ggml_tensor *tensor,
                            const std::string &name) {
  if (!tensor || !tensor->data)
    return;

  gguf_add_tensor(ctx, tensor);
}

bool SaveGGUFModel(const TransformerModel &model,
                   const std::string &output_path) {
  return SaveGGUFModelWithProgress(model, output_path, nullptr);
}

bool SaveGGUFModelWithProgress(const TransformerModel &model,
                               const std::string &output_path,
                               SaveProgressCallback callback) {
  std::cout << "[SaveModel] Saving model to: " << output_path << std::endl;

  // Create new GGUF context
  gguf_context *ctx = gguf_init_empty();
  if (!ctx) {
    std::cerr << "[SaveModel] Failed to create GGUF context" << std::endl;
    return false;
  }

  // ===== 1. Add model metadata (hyperparameters) =====
  const auto &hp = model.hparams;

  // Architecture type (assuming LLaMA-like)
  gguf_set_val_str(ctx, "general.architecture", "llama");
  gguf_set_val_str(ctx, "general.name", "DenseCore Optimized Model");

  // LLaMA-specific hyperparameters
  gguf_set_val_u32(ctx, "llama.context_length", hp.n_ctx);
  gguf_set_val_u32(ctx, "llama.embedding_length", hp.n_embd);
  gguf_set_val_u32(ctx, "llama.block_count", hp.n_layer);
  gguf_set_val_u32(ctx, "llama.attention.head_count", hp.n_head);
  gguf_set_val_u32(ctx, "llama.attention.head_count_kv", hp.n_head_kv);
  gguf_set_val_u32(ctx, "llama.vocab_size", hp.n_vocab);
  gguf_set_val_f32(ctx, "llama.attention.layer_norm_rms_epsilon",
                   hp.f_norm_rms_eps);
  gguf_set_val_f32(ctx, "llama.rope.freq_base", hp.rope_freq_base);

  if (callback)
    callback(1, 10, "Metadata written");

  // ===== 2. Add token embeddings =====
  int total_tensors = 3 + (model.layers.size() *
                           12); // approx (embed, norm, output + per-layer)
  int current_tensor = 0;

  if (model.tok_embeddings) {
    AddTensorToGGUF(ctx, model.tok_embeddings, "token_embd.weight");
    current_tensor++;
    if (callback)
      callback(current_tensor, total_tensors, "Token embeddings");
  }

  // ===== 3. Add output norm =====
  if (model.output_norm) {
    AddTensorToGGUF(ctx, model.output_norm, "output_norm.weight");
    current_tensor++;
    if (callback)
      callback(current_tensor, total_tensors, "Output norm");
  }

  // ===== 4. Add output (lm_head) =====
  if (model.output && !model.tied_embeddings) {
    AddTensorToGGUF(ctx, model.output, "output.weight");
    current_tensor++;
    if (callback)
      callback(current_tensor, total_tensors, "Output projection");
  }

  // ===== 5. Add layer tensors =====
  for (size_t i = 0; i < model.layers.size(); ++i) {
    const auto &layer = model.layers[i];
    std::string prefix = "blk." + std::to_string(i) + ".";

    // Attention weights
    if (layer.wq)
      AddTensorToGGUF(ctx, layer.wq, prefix + "attn_q.weight");
    if (layer.wk)
      AddTensorToGGUF(ctx, layer.wk, prefix + "attn_k.weight");
    if (layer.wv)
      AddTensorToGGUF(ctx, layer.wv, prefix + "attn_v.weight");
    if (layer.wo)
      AddTensorToGGUF(ctx, layer.wo, prefix + "attn_output.weight");

    // Attention bias (if present)
    if (layer.bq)
      AddTensorToGGUF(ctx, layer.bq, prefix + "attn_q.bias");
    if (layer.bk)
      AddTensorToGGUF(ctx, layer.bk, prefix + "attn_k.bias");
    if (layer.bv)
      AddTensorToGGUF(ctx, layer.bv, prefix + "attn_v.bias");
    if (layer.bo)
      AddTensorToGGUF(ctx, layer.bo, prefix + "attn_output.bias");

    // QK-Norm (if present, e.g., Qwen3)
    if (layer.attn_q_norm)
      AddTensorToGGUF(ctx, layer.attn_q_norm, prefix + "attn_q_norm.weight");
    if (layer.attn_k_norm)
      AddTensorToGGUF(ctx, layer.attn_k_norm, prefix + "attn_k_norm.weight");

    // Norm layers
    if (layer.attention_norm)
      AddTensorToGGUF(ctx, layer.attention_norm, prefix + "attn_norm.weight");
    if (layer.ffn_norm)
      AddTensorToGGUF(ctx, layer.ffn_norm, prefix + "ffn_norm.weight");

    // FFN weights
    if (layer.w1)
      AddTensorToGGUF(ctx, layer.w1, prefix + "ffn_gate.weight");
    if (layer.w2)
      AddTensorToGGUF(ctx, layer.w2, prefix + "ffn_down.weight");
    if (layer.w3)
      AddTensorToGGUF(ctx, layer.w3, prefix + "ffn_up.weight");

    current_tensor += 12; // approximate
    if (callback && (i % 4 == 0)) {
      callback(current_tensor, total_tensors,
               ("Layer " + std::to_string(i) + " written").c_str());
    }
  }

  // ===== 6. Write to file =====
  std::cout << "[SaveModel] Writing " << current_tensor << " tensors to "
            << output_path << "..." << std::endl;

  // Use gguf_write_to_file
  bool success = gguf_write_to_file(ctx, output_path.c_str(), false);

  if (!success) {
    std::cerr << "[SaveModel] Failed to write GGUF file" << std::endl;
    gguf_free(ctx);
    return false;
  }

  if (callback)
    callback(total_tensors, total_tensors, "Complete");

  std::cout << "[SaveModel] Successfully saved model to: " << output_path
            << std::endl;

  // Count actual tensor sizes
  size_t total_bytes = 0;
  for (size_t i = 0; i < model.layers.size(); ++i) {
    const auto &layer = model.layers[i];
    auto count_tensor = [&](const ggml_tensor *t) {
      if (t && t->data)
        total_bytes += ggml_nbytes(t);
    };
    count_tensor(layer.wq);
    count_tensor(layer.wk);
    count_tensor(layer.wv);
    count_tensor(layer.wo);
    count_tensor(layer.w1);
    count_tensor(layer.w2);
    count_tensor(layer.w3);
  }
  if (model.tok_embeddings)
    total_bytes += ggml_nbytes(model.tok_embeddings);
  if (model.output && !model.tied_embeddings)
    total_bytes += ggml_nbytes(model.output);
  if (model.output_norm)
    total_bytes += ggml_nbytes(model.output_norm);

  std::cout << "[SaveModel] Total tensor data: " << (total_bytes / 1024 / 1024)
            << " MB" << std::endl;

  gguf_free(ctx);
  return true;
}

} // namespace densecore

// C API wrapper for bridge
extern "C" {

int SaveModel(const char *model_path, const char *output_path) {
  // Note: This requires loading the model first, so we need the handle
  // For now, return error - the main entry point is via
  // QuantizeModel/PruneModel
  std::cerr << "[SaveModel] Direct SaveModel not supported - use QuantizeModel "
               "or PruneModel"
            << std::endl;
  return -1;
}
}
