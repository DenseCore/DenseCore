#ifndef DENSECORE_MODEL_TYPES_H
#define DENSECORE_MODEL_TYPES_H

#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Forward declaration for RoPE table (defined in simd_ops.h)
namespace densecore {
namespace simd {
struct RoPETable;
} // namespace simd
} // namespace densecore

// Transformer hyperparameters
struct TransformerHParams {
  uint32_t n_vocab = 32000;
  uint32_t n_ctx = 512;
  uint32_t n_embd = 4096;
  uint32_t n_head = 32;
  uint32_t n_head_kv = 32;
  uint32_t n_layer = 32;
  uint32_t n_rot = 64;

  // llama.cpp style: separate head dimensions for K and V
  // Allows models like Qwen3 where head_dim != n_embd/n_head
  // If 0, will be auto-computed from weight tensor shapes
  uint32_t n_embd_head_k = 0; // K head dimension
  uint32_t n_embd_head_v = 0; // V head dimension

  float f_norm_rms_eps = 1e-5f;
  float rope_freq_base = 10000.0f;
  float rope_freq_scale = 1.0f;
};

// Single transformer layer weights
struct TransformerLayer {
  // Attention
  struct ggml_tensor *wq;
  struct ggml_tensor *wk;
  struct ggml_tensor *wv;
  struct ggml_tensor *wo;

  // Attention Bias
  struct ggml_tensor *bq = nullptr;
  struct ggml_tensor *bk = nullptr;
  struct ggml_tensor *bv = nullptr;
  struct ggml_tensor *bo = nullptr;

  // QK-Norm (Qwen3 uses RMS norm on Q and K instead of bias)
  struct ggml_tensor *attn_q_norm = nullptr;
  struct ggml_tensor *attn_k_norm = nullptr;

  // Normalization
  struct ggml_tensor *attention_norm;
  struct ggml_tensor *ffn_norm;

  // Feed-Forward
  struct ggml_tensor *w1; // gate
  struct ggml_tensor *w2; // down
  struct ggml_tensor *w3; // up
};

// KV Cache Structure
struct KVCache {
  struct ggml_context *ctx = nullptr;
  struct ggml_tensor *k = nullptr; // [head_dim, n_head_kv, n_ctx, n_layer]
  struct ggml_tensor *v = nullptr; // [head_dim, n_head_kv, n_ctx, n_layer]

  int head_dim = 0;
  int n_ctx = 0;
  int n_layer = 0;
  int n_tokens = 0; // Current number of tokens in cache

  ~KVCache();

  void Reset();
  void RemoveLast(int n);
};

// Complete transformer model
struct TransformerModel {
  TransformerHParams hparams;

  struct ggml_tensor *tok_embeddings;
  struct ggml_tensor *output_norm;
  struct ggml_tensor *output;

  std::vector<TransformerLayer> layers;

  // Context & Backend
  struct ggml_context *ctx_w = nullptr; // weight context
  ggml_backend_t backend = nullptr;     // compute backend
  // Mock flag
  bool is_mock = false;
  // Tied embeddings flag (output = tok_embeddings)
  bool tied_embeddings = false;
  struct gguf_context *ctx_gguf = nullptr;

  // Tokenizer data
  std::vector<std::string> vocab_tokens;
  std::vector<float> token_scores; // BPE merge scores (lower = higher priority)
  std::map<std::string, int> token_to_id;
  int32_t bos_token_id = 1;
  int32_t eos_token_id = 2;

  // Pre-computed RoPE cos/sin table (interleaved: [cos, sin, cos, sin, ...])
  // Layout: [max_seq_len, head_dim] where each pair is (cos, sin)
  // Initialized during model loading based on hparams
  std::vector<float> rope_cos_sin;
  int rope_head_dim = 0; // Head dim used for RoPE table

  // =========================================================================
  // NUMA-Aware Memory Tracking
  // =========================================================================
  // Stores (ptr, size) pairs for tensor data that was rebind to NUMA nodes.
  // These buffers are NOT owned by ggml_context and MUST be freed manually
  // in the destructor. Failing to do so will cause memory leaks.
  // =========================================================================
  std::vector<std::pair<void *, size_t>> numa_buffers;

  ~TransformerModel();
};

// Engine handle (opaque pointer)
struct DenseCoreHandle_t {
  TransformerModel *model;
  KVCache *kv_cache;
};

#endif // DENSECORE_MODEL_TYPES_H
