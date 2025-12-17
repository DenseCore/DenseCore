#include "inference.h"
#include "flash_attention.h"
#include "ggml.h"              // Required for ggml_tensor definition
#include "hardware_topology.h" // For compute thread affinity

#ifndef GGML_KQ_MASK_PAD
#define GGML_KQ_MASK_PAD 32
#endif
#include "kv_cache.h" // Added for KV cache
#include "memory_pool.h"
#include "quantization/int4_types.h" // For TensorInt4
#include "simd_ops.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

// User data for KV cache operations
struct KVCacheUserData {
  PagedKVCache *cache;
  const BatchSpec *batch;
  int layer;
  int head_dim_kv; // Store dynamically detected head dim
  bool is_k;       // True if processing K, false if processing V
};

// Thread-local pool of KVCacheUserData to avoid memory allocation per layer
// Max layers supported: 128 (enough for any current model)
// Each layer needs 2 entries (K and V), so 256 total slots
static constexpr int kMaxKVCacheUserDataSlots = 256;
static thread_local KVCacheUserData
    g_kv_userdata_pool[kMaxKVCacheUserDataSlots];

// Helper to get a userdata slot (no allocation, no leak)
inline KVCacheUserData *GetKVCacheUserData(int layer, bool is_k) {
  int idx = layer * 2 + (is_k ? 0 : 1);
  if (idx >= kMaxKVCacheUserDataSlots) {
    idx = idx % kMaxKVCacheUserDataSlots; // Wrap for safety
  }
  return &g_kv_userdata_pool[idx];
}

// ============================================================================
// Custom RoPE Callback (AVX-512 Optimized)
// ============================================================================

/**
 * User data for custom RoPE operation
 */
struct RoPEUserData {
  const float *cos_sin_table; ///< Pre-computed [max_seq, head_dim] table
  const int *positions;       ///< Token positions [n_tokens]
  int n_tokens;               ///< Number of tokens
  int n_heads;                ///< Number of heads (Q or KV)
  int head_dim;               ///< Dimension per head
  int rope_dim;               ///< Number of dimensions to rotate
};

// Thread-local pool for RoPE user data (avoids allocation per layer)
static constexpr int kMaxRoPEUserDataSlots = 512;
static thread_local RoPEUserData g_rope_userdata_pool[kMaxRoPEUserDataSlots];
static thread_local int g_rope_userdata_index = 0;

inline RoPEUserData *GetRoPEUserData() {
  int idx = g_rope_userdata_index++;
  if (idx >= kMaxRoPEUserDataSlots) {
    g_rope_userdata_index = 0;
    idx = 0;
  }
  return &g_rope_userdata_pool[idx];
}

/**
 * Custom RoPE callback using AVX-512 kernel
 *
 * Input tensor shape: [head_dim, n_heads, n_tokens] (GGML standard for Q/K)
 * Applies RoPE rotation in-place using pre-computed cos/sin tables.
 *
 * Threading: Work is partitioned across n_heads dimension.
 */
void cb_rope_avx512(struct ggml_tensor *dst, const struct ggml_tensor *src,
                    int ith, int nth, void *userdata) {
  auto *ud = (RoPEUserData *)userdata;
  if (!ud || !ud->cos_sin_table || !ud->positions)
    return;

  // Tensor layout: [head_dim, n_heads, n_tokens]
  const int head_dim = ud->head_dim;
  const int n_heads = ud->n_heads;
  const int n_tokens = ud->n_tokens;
  const int rope_dim = ud->rope_dim;

  // Partition work across heads
  const int heads_per_thread = (n_heads + nth - 1) / nth;
  const int h_start = ith * heads_per_thread;
  const int h_end = std::min(h_start + heads_per_thread, n_heads);

  if (h_start >= n_heads)
    return;

  // Process assigned heads
  for (int h = h_start; h < h_end; h++) {
    for (int t = 0; t < n_tokens; t++) {
      const int pos = ud->positions[t];
      const float *cs_ptr = ud->cos_sin_table + pos * head_dim;

      // Input/output pointers for this head and token
      // Layout: [head_dim, n_heads, n_tokens] -> offset = head_dim * (h +
      // n_heads * t)
      const float *in_ptr =
          (const float *)src->data + head_dim * (h + n_heads * t);
      float *out_ptr = (float *)dst->data + head_dim * (h + n_heads * t);

      // Apply RoPE to pairs
      for (int d = 0; d < rope_dim; d += 2) {
        float x0 = in_ptr[d];
        float x1 = in_ptr[d + 1];
        float cos_val = cs_ptr[d];
        float sin_val = cs_ptr[d + 1];

        out_ptr[d] = x0 * cos_val - x1 * sin_val;
        out_ptr[d + 1] = x0 * sin_val + x1 * cos_val;
      }

      // Copy dimensions beyond rope_dim unchanged
      for (int dd = rope_dim; dd < head_dim; dd++) {
        out_ptr[dd] = in_ptr[dd];
      }
    }
  }
}

// ============================================================================
// RoPE Table Initialization
// ============================================================================

/**
 * @brief Initialize pre-computed RoPE cos/sin table for the model
 *
 * Populates model->rope_cos_sin with values for all positions and dimensions.
 * Layout: [pos * head_dim + d] = cos/sin pair for position 'pos', dimension 'd'
 * Interleaved format: [cos0, sin0, cos1, sin1, ...]
 *
 * @param model Model to initialize RoPE table for
 */
void InitRoPETable(TransformerModel *model) {
  if (!model)
    return;

  const int n_ctx = model->hparams.n_ctx;
  const int head_dim = model->hparams.n_embd / model->hparams.n_head;
  const float freq_base = model->hparams.rope_freq_base;

  // Allocate table: [n_ctx, head_dim] interleaved [cos, sin] pairs
  model->rope_cos_sin.resize(static_cast<size_t>(n_ctx) * head_dim);
  model->rope_head_dim = head_dim;

  // Pre-compute frequencies: theta[d] = 1 / (freq_base ** (2d / head_dim))
  std::vector<float> freqs(head_dim / 2);
  for (int d = 0; d < head_dim / 2; d++) {
    float exp_val = (2.0f * d) / static_cast<float>(head_dim);
    freqs[d] = 1.0f / std::pow(freq_base, exp_val);
  }

  // Compute cos/sin for all positions
  for (int pos = 0; pos < n_ctx; pos++) {
    for (int d = 0; d < head_dim / 2; d++) {
      float angle = static_cast<float>(pos) * freqs[d];
      // Interleaved storage for cache efficiency
      model->rope_cos_sin[pos * head_dim + 2 * d] = std::cos(angle);
      model->rope_cos_sin[pos * head_dim + 2 * d + 1] = std::sin(angle);
    }
  }
}

// Custom callback to load K/V history from cache and append current K/V
// This gathers the full context (history + current) into the destination tensor
// AND writes the current K/V into the cache for future steps.
void cb_kv_manage(struct ggml_tensor *dst, const struct ggml_tensor *src,
                  int ith, int nth, void *userdata) {
  // Prevent race conditions: only run on the first thread
  if (ith != 0)
    return;

  auto *ud = (KVCacheUserData *)userdata;
  if (!ud || !ud->cache || !ud->batch)
    return;

  // src is [head_dim, n_head_kv, n_total] (PADDED Kcur)
  // dst is [head_dim, n_head_kv, n_total] (History + Current)

  const int head_dim = ud->head_dim_kv;
  const int n_head_kv = ud->cache->n_head_kv;

  // Logic: src contains Kcur in the FIRST N positions (ggml_pad pads at end)
  // We need to:
  // 1. Write src[0..N] to cache (Update Step)
  // 2. Read cache[0..n_past] to dst[0..n_past] (History Step)
  // 3. Copy src[0..N] to dst[n_past..n_total] (Current Step)

  const int N = ud->batch->tokens.size(); // Original batch size
  const int n_total = src->ne[2];
  const int n_past = n_total - N;

  const size_t bytes_per_elem = sizeof(float);
  const size_t head_block_size = head_dim * n_head_kv;
  const size_t head_block_bytes = head_block_size * bytes_per_elem;

  // 1. Write current tokens to cache
  for (int i = 0; i < N; i++) {
    int seq_id = ud->batch->seq_id[i];
    int pos = ud->batch->pos[i];

    if (seq_id >= (int)ud->batch->block_tables.size())
      continue;

    const auto &block_table = ud->batch->block_tables[seq_id];
    int logical_block = pos / BLOCK_SIZE;
    int slot = pos % BLOCK_SIZE;

    if (logical_block < (int)block_table.size()) {
      int block_id = block_table[logical_block];
      // src has data at [i]
      const float *src_data = (const float *)src->data + i * head_block_size;

      if (ud->is_k)
        ud->cache->WriteKSlot(ud->layer, block_id, slot, src_data);
      else
        ud->cache->WriteVSlot(ud->layer, block_id, slot, src_data);
    }
  }

  // 2. Read full history
  if (n_past > 0) {
    int seq_id = ud->batch->seq_id[0];
    const auto &block_table = ud->batch->block_tables[seq_id];

    for (int i = 0; i < n_past; i++) {
      int pos = i;
      int logical_block = pos / BLOCK_SIZE;
      int slot = pos % BLOCK_SIZE;

      if (logical_block < (int)block_table.size()) {
        int block_id = block_table[logical_block];
        float *dst_data = (float *)dst->data + i * head_block_size;

        if (ud->is_k)
          ud->cache->ReadKSlot(ud->layer, block_id, slot, dst_data);
        else
          ud->cache->ReadVSlot(ud->layer, block_id, slot, dst_data);
      } else {
        memset((float *)dst->data + i * head_block_size, 0, head_block_bytes);
      }
    }
  }

  // 3. Copy current tokens to the END of dst
  // src[0..N] -> dst[n_past..n_total]
  memcpy((float *)dst->data + n_past * head_block_size, src->data,
         N * head_block_bytes);
}

// ============================================================================
// INT4 GEMM Integration
// ============================================================================

/**
 * User data for INT4 GEMM custom operation
 */
struct INT4GemmUserData {
  const densecore::TensorInt4 *int4_weight; // INT4 quantized weight metadata
  int M;                                    // Output rows
  int N;                                    // Output columns
  int K;                                    // Inner dimension
};

/**
 * Custom callback for INT4 GEMM operation (MULTI-THREADED)
 *
 * Computes: dst = src * weight^T
 * where weight is INT4 quantized
 *
 * Threading Strategy:
 * - Parallelize along N dimension (output columns/features)
 * - Each thread computes a subset of output columns
 * - M dimension (batch size) is usually small, so not worth parallelizing
 */
void cb_int4_gemm(struct ggml_tensor *dst, const struct ggml_tensor *src,
                  int ith, int nth, void *userdata) {
  // Pin this GGML worker thread on first invocation (thread-local, O(1) after
  // first call)
  densecore::HardwareTopology::GetInstance().PinComputeThread(ith);

  auto *ud = (INT4GemmUserData *)userdata;
  if (!ud || !ud->int4_weight)
    return;

  const int M = ud->M;
  const int N = ud->N;
  const int K = ud->K;
  const auto *w = ud->int4_weight;
  const int num_groups = K / w->group_size;

  // Work partitioning along N dimension
  // Each thread computes a range of output columns: [n_start, n_end)
  const int n_per_thread = (N + nth - 1) / nth; // Ceiling division
  const int n_start = ith * n_per_thread;
  const int n_end = std::min(n_start + n_per_thread, N);

  // Early exit if this thread has no work
  if (n_start >= N || n_start >= n_end)
    return;

  const int n_local = n_end - n_start;

  // Input/output pointers
  const float *A = (const float *)src->data; // [M × K] - shared by all threads
  float *C = (float *)dst->data;             // [M × N] - partitioned output

  // Offset pointers for this thread's partition
  // Weights are stored as [N × K/2] (packed INT4, row-major)
  // Thread i processes weight rows [n_start, n_end)
  const size_t weight_row_size = K / 2; // Bytes per weight row (packed)
  const uint8_t *W_int4_local =
      (const uint8_t *)w->q_data + n_start * weight_row_size;

  // Scales and zero-points are stored as [N × num_groups]
  // Thread i needs entries [n_start * num_groups, n_end * num_groups)
  const float *scales_local = w->scales + n_start * num_groups;
  const float *zeros_local = w->zero_points + n_start * num_groups;

  // Thread-local workspace buffer (reused across calls, no heap allocation)
  // Max supported: M=64 (batch size), n_local=8192 (large hidden dim)
  // 64 * 8192 * 4 = 2MB per thread
  static constexpr size_t kMaxWorkspaceSize = 64 * 8192;
  static thread_local float s_workspace[kMaxWorkspaceSize];

  const size_t required_size = static_cast<size_t>(M) * n_local;
  float *C_local = s_workspace;

  // Fallback to heap allocation if workspace is too small (rare edge case)
  std::vector<float> C_local_heap;
  if (required_size > kMaxWorkspaceSize) {
    C_local_heap.resize(required_size);
    C_local = C_local_heap.data();
  }

  // Call the INT4 GEMM kernel for this thread's partition
  densecore::simd::GemmInt4Fp32_AVX512(
      C_local,       // Temporary output [M × n_local]
      A,             // Full activations [M × K]
      W_int4_local,  // Subset of weights [n_local × K]
      scales_local,  // Subset of scales [n_local × num_groups]
      zeros_local,   // Subset of zero-points [n_local × num_groups]
      M, n_local, K, // n_local instead of N
      w->group_size);

  // Copy thread-local results to final output with proper stride
  // C[m, n] = C[m * N + n] (row-major)
  // We write to columns [n_start, n_end) for all rows
  for (int m = 0; m < M; m++) {
    memcpy(C + m * N + n_start,      // Destination: C[m, n_start]
           C_local + m * n_local,    // Source: thread's local buffer
           n_local * sizeof(float)); // Copy n_local elements
  }
}

/**
 * Check if a tensor is quantized to INT4
 */
inline bool IsINT4Quantized(const struct ggml_tensor *tensor) {
  if (!tensor || !tensor->extra)
    return false;

  // Check if extra data contains TensorInt4 metadata
  // This is set by the INT4Quantizer during quantization
  const densecore::TensorInt4 *int4 =
      static_cast<const densecore::TensorInt4 *>(tensor->extra);

  // Validate it's actually INT4 data
  return (int4 && int4->q_data && int4->scales && int4->zero_points);
}

/**
 * Create a custom GGML operation for INT4 GEMM
 *
 * This replaces ggml_mul_mat when weights are INT4 quantized.
 */
inline struct ggml_tensor *
ggml_mul_mat_int4(struct ggml_context *ctx,
                  struct ggml_tensor *weight, // INT4 quantized
                  struct ggml_tensor *input,  // FP32 activations
                  INT4GemmUserData *userdata) {

  // Create output tensor
  // ggml_mul_mat(weight, input) computes input^T * weight^T
  // For weight [N × K] and input [K × M], result is [N × M]
  const int M = input->ne[1];  // Batch size / sequence length
  const int N = weight->ne[1]; // Output dimension
  const int K = weight->ne[0]; // Input dimension

  struct ggml_tensor *result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, M);

  // Populate user data
  userdata->int4_weight =
      static_cast<const densecore::TensorInt4 *>(weight->extra);
  userdata->M = M;
  userdata->N = N;
  userdata->K = K;

  // Create custom operation
  result = ggml_map_custom1(ctx, input, cb_int4_gemm, 1, userdata);

  return result;
}

// Thread-local pool for INT4 GEMM user data (avoid allocations)
static constexpr int kMaxINT4GemmUserDataSlots = 64;
static thread_local INT4GemmUserData
    g_int4_gemm_userdata_pool[kMaxINT4GemmUserDataSlots];
static thread_local int g_int4_gemm_userdata_index = 0;

inline INT4GemmUserData *GetINT4GemmUserData() {
  int idx = g_int4_gemm_userdata_index++;
  if (idx >= kMaxINT4GemmUserDataSlots) {
    g_int4_gemm_userdata_index = 0;
    idx = 0;
  }
  return &g_int4_gemm_userdata_pool[idx];
}

/**
 * Smart matrix multiplication dispatcher
 *
 * Uses INT4 GEMM kernel if weights are quantized, otherwise falls back to
 * ggml_mul_mat
 */
inline struct ggml_tensor *smart_mul_mat(struct ggml_context *ctx,
                                         struct ggml_tensor *weight,
                                         struct ggml_tensor *input) {

  // Check if INT4 quantization is available and should be used
  static const bool use_int4_kernel = (densecore::simd::DetectSimdLevel() >=
                                       densecore::simd::SimdLevel::AVX512);

  if (use_int4_kernel && IsINT4Quantized(weight)) {
    // Use custom INT4 GEMM kernel
    INT4GemmUserData *ud = GetINT4GemmUserData();
    return ggml_mul_mat_int4(ctx, weight, input, ud);
  } else {
    // Fallback to standard GGML matrix multiplication
    return ggml_mul_mat(ctx, weight, input);
  }
}

// ============================================================================
// SIMPLIFIED UNIVERSAL ATTENTION (llama.cpp style)
// This version trades the complex paged KV cache for correctness and clarity.
// Once working, KV cache can be re-added following the proven llama.cpp
// pattern.
// ============================================================================

struct ggml_tensor *BuildTransformerGraph(
    TransformerModel *model, PagedKVCache *cache, struct ggml_context *ctx_c,
    const BatchSpec &batch, bool embedding_mode, struct ggml_cgraph *gf,
    struct ggml_tensor **out_embd, struct ggml_tensor **out_pos) {

  const int N = batch.tokens.size();
  const int n_embd = model->hparams.n_embd;
  const int n_head = model->hparams.n_head;
  const int n_head_kv = model->hparams.n_head_kv;
  const int n_layer = model->hparams.n_layer;
  const int n_ctx = model->hparams.n_ctx;

  // =========================================================================
  // 1. Token Embedding Lookup
  // =========================================================================
  struct ggml_tensor *embd_inp = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, N);
  ggml_set_name(embd_inp, "embd_inp");
  if (embd_inp->data) {
    memcpy(embd_inp->data, batch.tokens.data(), N * sizeof(int));
  }
  if (out_embd)
    *out_embd = embd_inp;

  struct ggml_tensor *cur =
      ggml_get_rows(ctx_c, model->tok_embeddings, embd_inp);

  // Position tensor for RoPE
  struct ggml_tensor *pos = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, N);
  ggml_set_name(pos, "pos");
  if (pos->data) {
    memcpy(pos->data, batch.pos.data(), N * sizeof(int));
  }
  if (out_pos)
    *out_pos = pos;

  // =========================================================================
  // 2. Transformer Layers
  // =========================================================================
  for (int il = 0; il < n_layer; ++il) {
    struct ggml_tensor *inpL = cur;

    // Attention Norm
    cur = ggml_rms_norm(ctx_c, cur, model->hparams.f_norm_rms_eps);
    cur = ggml_mul(ctx_c, cur, model->layers[il].attention_norm);

    // Q/K/V Projections (using smart dispatcher for INT4 support)
    struct ggml_tensor *Qcur = smart_mul_mat(ctx_c, model->layers[il].wq, cur);
    struct ggml_tensor *Kcur = smart_mul_mat(ctx_c, model->layers[il].wk, cur);
    struct ggml_tensor *Vcur = smart_mul_mat(ctx_c, model->layers[il].wv, cur);

    // Add Bias if present (for Qwen2 and some other models)
    if (model->layers[il].bq)
      Qcur = ggml_add(ctx_c, Qcur, model->layers[il].bq);
    if (model->layers[il].bk)
      Kcur = ggml_add(ctx_c, Kcur, model->layers[il].bk);
    if (model->layers[il].bv)
      Vcur = ggml_add(ctx_c, Vcur, model->layers[il].bv);

    // Dynamically infer head dimensions from the actual projected tensors
    // This allows universal support regardless of what GGUF metadata says
    int dim_q = Qcur->ne[0];
    int dim_k = Kcur->ne[0];
    int dim_v = Vcur->ne[0];

    int head_dim_q = dim_q / n_head;
    int head_dim_kv = dim_k / n_head_kv;

    // Reshape to [head_dim, n_heads, N]
    Qcur = ggml_reshape_3d(ctx_c, Qcur, head_dim_q, n_head, N);
    Kcur = ggml_reshape_3d(ctx_c, Kcur, head_dim_kv, n_head_kv, N);
    Vcur = ggml_reshape_3d(ctx_c, Vcur, head_dim_kv, n_head_kv, N);

    // QK-Norm: Apply RMSNorm + weight for Q and K vectors (required for Qwen3)
    // This normalizes Q/K before attention for numerical stability with FP16
    if (model->layers[il].attn_q_norm) {
      Qcur = ggml_rms_norm(ctx_c, Qcur, model->hparams.f_norm_rms_eps);
      Qcur = ggml_mul(ctx_c, Qcur, model->layers[il].attn_q_norm);
    }
    if (model->layers[il].attn_k_norm) {
      Kcur = ggml_rms_norm(ctx_c, Kcur, model->hparams.f_norm_rms_eps);
      Kcur = ggml_mul(ctx_c, Kcur, model->layers[il].attn_k_norm);
    }

    // Apply RoPE
    // Use n_rot from model params if specified (e.g. for partial RoPE or
    // specific dim) Fallback to full head_dim_q if n_rot is 0
    int rope_dim = model->hparams.n_rot;
    if (rope_dim <= 0) {
      rope_dim = head_dim_q;
    }

    // Ensure rope_dim is valid (<= head_dim)
    if (rope_dim > head_dim_q)
      rope_dim = head_dim_q;

    // Use custom RoPE if pre-computed table is available
    if (!model->rope_cos_sin.empty()) {
      // Apply custom RoPE using pre-computed table
      // Q: [head_dim_q, n_head, N]
      RoPEUserData *q_ud = GetRoPEUserData();
      *q_ud = {model->rope_cos_sin.data(),
               batch.pos.data(),
               N,
               n_head,
               head_dim_q,
               rope_dim};
      Qcur = ggml_map_custom1(ctx_c, Qcur, cb_rope_avx512, 1, q_ud);

      // K: [head_dim_kv, n_head_kv, N]
      RoPEUserData *k_ud = GetRoPEUserData();
      *k_ud = {model->rope_cos_sin.data(),
               batch.pos.data(),
               N,
               n_head_kv,
               head_dim_kv,
               rope_dim};
      Kcur = ggml_map_custom1(ctx_c, Kcur, cb_rope_avx512, 1, k_ud);
    } else {
      // Fallback to standard GGML RoPE
      Qcur =
          ggml_rope_ext(ctx_c, Qcur, pos, nullptr, rope_dim, 0, n_ctx,
                        model->hparams.rope_freq_base,
                        model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
      Kcur =
          ggml_rope_ext(ctx_c, Kcur, pos, nullptr, rope_dim, 0, n_ctx,
                        model->hparams.rope_freq_base,
                        model->hparams.rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    // =========================================================================
    // KV CACHE INTEGRATION (Universal Paged Attention)
    // =========================================================================
    struct ggml_tensor *K_all = Kcur; // Default to current K
    struct ggml_tensor *V_all = Vcur; // Default to current V

    const bool use_cache = (cache != nullptr);
    int n_past_val = 0;
    if (use_cache && batch.num_seqs > 0 && batch.n_past.size() > 0) {
      n_past_val = batch.n_past[0];
    }
    const int n_total_tokens = n_past_val + N;

    if (use_cache) {
      // Only need fancy logic if we have history.
      // If n_past = 0 (Prefill), K_all == Kcur is mostly fine,
      // BUT we still need to WRITE to cache.
      // The 'ggml_pad' trick updates cache as side effect.
      // So we act always if use_cache is true.

      // Use ggml_pad to create a tensor of correct size (N + n_past)
      // ggml_pad(ctx, a, pad_0, pad_1, pad_2, pad_3)
      // We pad dimension 2 (sequence) by n_past_val.
      // Result shape: [head_dim, n_head, N + n_past]
      struct ggml_tensor *K_padded = Kcur;
      struct ggml_tensor *V_padded = Vcur;

      if (n_past_val > 0) {
        K_padded = ggml_pad(ctx_c, Kcur, 0, 0, n_past_val, 0);
        V_padded = ggml_pad(ctx_c, Vcur, 0, 0, n_past_val, 0);
      }

      KVCacheUserData *k_ud = GetKVCacheUserData(il, true);
      *k_ud = {cache, &batch, il, head_dim_kv, true};
      KVCacheUserData *v_ud = GetKVCacheUserData(il, false);
      *v_ud = {cache, &batch, il, head_dim_kv, false};

      K_all = ggml_map_custom1(ctx_c, K_padded, cb_kv_manage, 1, k_ud);
      V_all = ggml_map_custom1(ctx_c, V_padded, cb_kv_manage, 1, v_ud);
    }

    // After projection and reshape:

    // Q: [head_dim_q, n_head, N]
    // K_all: [head_dim_kv, n_head_kv, n_past + N]
    // V_all: [head_dim_kv, n_head_kv, n_past + N]

    // For GQA: repeat K_all and V_all to match Q head count BEFORE permute
    struct ggml_tensor *K = K_all;
    struct ggml_tensor *V = V_all;

    if (n_head_kv != n_head) {
      // GQA: n_head = n_head_kv * n_rep
      const int n_rep = n_head / n_head_kv;
      const int n_total_kv = K->ne[2]; // Use actual shape from K_all

      // K: [head_dim_kv, n_head_kv, n_total_kv] -> [head_dim_kv, n_head,
      // n_total_kv]
      K = ggml_reshape_4d(ctx_c, K, head_dim_kv, 1, n_head_kv, n_total_kv);
      struct ggml_tensor *K_target = ggml_new_tensor_4d(
          ctx_c, GGML_TYPE_F32, head_dim_kv, n_rep, n_head_kv, n_total_kv);
      K = ggml_repeat(ctx_c, K, K_target);
      K = ggml_reshape_3d(ctx_c, K, head_dim_kv, n_head, n_total_kv);
      K = ggml_cont(ctx_c, K);

      // V
      V = ggml_reshape_4d(ctx_c, V, head_dim_kv, 1, n_head_kv, n_total_kv);
      struct ggml_tensor *V_target = ggml_new_tensor_4d(
          ctx_c, GGML_TYPE_F32, head_dim_kv, n_rep, n_head_kv, n_total_kv);
      V = ggml_repeat(ctx_c, V, V_target);
      V = ggml_reshape_3d(ctx_c, V, head_dim_kv, n_head, n_total_kv);
      V = ggml_cont(ctx_c, V);
    }

    // =========================================================================
    // ATTENTION (llama.cpp style - corrected tensor layouts)
    // =========================================================================
    // After projection and reshape:
    //   Q: [head_dim, n_head, N]
    //   K: [head_dim, n_head, N] (after GQA expansion)
    //   V: [head_dim, n_head, N] (after GQA expansion)
    //
    // llama.cpp uses permute(0, 2, 1, 3) for all Q/K/V to get:
    //   [head_dim, N, n_head]
    //
    // Then for KQV computation:
    //   KQ = K^T @ Q -> [N, N, n_head] (attention scores)
    //   V needs transpose before: V^T @ KQ -> [head_dim, N, n_head]
    // =========================================================================

    // =========================================================================
    // ATTENTION (llama.cpp style - corrected tensor layouts)
    // =========================================================================
    // After projection and reshape:
    //   Q: [head_dim_q, n_head, N]
    //   K: [head_dim_kv, n_head, N] (after GQA expansion)
    //   V: [head_dim_kv, n_head, N] (after GQA expansion)
    //
    // llama.cpp uses permute(0, 2, 1, 3) for all Q/K/V to get:
    //   [head_dim, N, n_head]
    //
    //   K: [head_dim_kv, n_head, n_total_tokens] (after GQA expansion)
    //   V: [head_dim_kv, n_head, n_total_tokens] (after GQA expansion)
    //
    // llama.cpp uses permute(0, 2, 1, 3) for all Q/K/V to get:
    //   [head_dim, N, n_head]
    //
    // Then for KQV computation:
    //   KQ = K^T @ Q -> [N, N, n_head] (attention scores)
    //   V needs transpose before: V^T @ KQ -> [head_dim, N, n_head]
    // =========================================================================

    // =========================================================================
    // ATTENTION DISPATCH (Runtime selection based on CPU capabilities)
    // - AVX-512+: Use Flash Attention (ggml_flash_attn_ext) for efficiency
    // - Other: Use standard Q*K^T -> softmax -> V for compatibility
    // =========================================================================
    static const bool use_flash_attention =
        (densecore::simd::DetectSimdLevel() >=
         densecore::simd::SimdLevel::AVX512);

    struct ggml_tensor *KQV = nullptr;

    if (use_flash_attention) {
      // -----------------------------------------------------------------------
      // FLASH ATTENTION PATH (AVX-512 only)
      // -----------------------------------------------------------------------
      // Q: [head_dim, N, n_head]
      struct ggml_tensor *Q = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);

      // K/V after GQA expansion: [head_dim, n_head, n_total]
      // Permute to [head_dim, n_total, n_head] for Flash Attention
      struct ggml_tensor *K_fa = ggml_permute(ctx_c, K, 0, 2, 1, 3);
      struct ggml_tensor *V_fa = ggml_permute(ctx_c, V, 0, 2, 1, 3);

      // Create mask [n_total, N_padded, 1, 1] as required by
      // ggml_flash_attn_ext 0.0f = can attend, -INFINITY = cannot attend
      // (masked)
      int N_padded = (N + GGML_KQ_MASK_PAD - 1) & ~(GGML_KQ_MASK_PAD - 1);
      struct ggml_tensor *KQ_mask = ggml_new_tensor_4d(
          ctx_c, GGML_TYPE_F32, n_total_tokens, N_padded, 1, 1);

      // Fill causal mask (column-major: element (k, q) is at k + q * n_kv)
      float *mask_data = (float *)KQ_mask->data;
      for (int q = 0; q < N_padded; q++) {
        for (int k = 0; k < n_total_tokens; k++) {
          int query_pos = n_past_val + q;
          int key_pos = k;
          int idx = k + q * n_total_tokens;

          if (q >= N) {
            mask_data[idx] = 0.0f; // Padding row
          } else if (key_pos <= query_pos) {
            mask_data[idx] = 0.0f; // Can attend
          } else {
            mask_data[idx] = -INFINITY; // Masked (future position)
          }
        }
      }

      // Ensure contiguity for Flash Attention
      Q = ggml_cont(ctx_c, Q);
      K_fa = ggml_cont(ctx_c, K_fa);
      V_fa = ggml_cont(ctx_c, V_fa);

      // Scale factor: 1/sqrt(head_dim)
      float scale = 1.0f / sqrtf((float)head_dim_q);

      // Flash Attention: fused Q*K^T, scale, mask, softmax, *V
      // Result: [head_dim, N, n_head]
      KQV =
          ggml_flash_attn_ext(ctx_c, Q, K_fa, V_fa, KQ_mask, scale, 0.0f, 0.0f);

      // Permute to [head_dim, n_head, N] for projection
      KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
    } else {
      // -----------------------------------------------------------------------
      // STANDARD ATTENTION PATH (Fallback for non-AVX512 CPUs)
      // -----------------------------------------------------------------------
      // Q: [head_dim, N, n_head]
      struct ggml_tensor *Q = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);

      // K/V: [head_dim, n_total, n_head]
      struct ggml_tensor *K_fa = ggml_permute(ctx_c, K, 0, 2, 1, 3);
      struct ggml_tensor *V_fa = ggml_permute(ctx_c, V, 0, 2, 1, 3);

      // Ensure contiguity
      Q = ggml_cont(ctx_c, Q);
      K_fa = ggml_cont(ctx_c, K_fa);
      V_fa = ggml_cont(ctx_c, V_fa);

      // Step 1: Q @ K^T -> [n_total, N, n_head]
      struct ggml_tensor *KQ = ggml_mul_mat(ctx_c, K_fa, Q);

      // Step 2: Scale by 1/sqrt(head_dim)
      float scale = 1.0f / sqrtf((float)head_dim_q);
      KQ = ggml_scale(ctx_c, KQ, scale);

      // Step 3: Apply causal mask (only for prefill with N > 1)
      if (N > 1) {
        KQ = ggml_diag_mask_inf(ctx_c, KQ, n_past_val);
      }

      // Step 4: Softmax
      KQ = ggml_soft_max(ctx_c, KQ);

      // Step 5: KQ @ V -> [head_dim, N, n_head]
      // V_fa is [head_dim, n_total, n_head], need to permute for matmul
      struct ggml_tensor *V_t =
          ggml_permute(ctx_c, V_fa, 1, 0, 2, 3); // [n_total, head_dim, n_head]
      V_t = ggml_cont(ctx_c, V_t);
      KQV = ggml_mul_mat(ctx_c, V_t, KQ); // [head_dim, N, n_head]

      // Permute to [head_dim, n_head, N] for projection
      KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
    }

    // Must be contiguous before reshape
    struct ggml_tensor *KQV_merged = ggml_cont(ctx_c, KQV);

    cur = ggml_reshape_2d(ctx_c, KQV_merged, head_dim_q * n_head, N);

    // Output Projection (using smart dispatcher for INT4 support)
    cur = smart_mul_mat(ctx_c, model->layers[il].wo, cur);
    if (model->layers[il].bo)
      cur = ggml_add(ctx_c, cur, model->layers[il].bo);

    // Residual Connection
    cur = ggml_add(ctx_c, cur, inpL);
    // Usually output projection expects n_embd input.
    // wo: [n_embd, n_embd] (or [n_embd, n_head*head_dim])
    // The standard transformer expects concatenation of all heads to be n_embd.
    // If n_head * head_dim_kv != n_embd, we have a mismatch.
    // Qwen3 has n_head=16, head_dim_kv=128 => 2048 != 1024.
    // This implies wo expects 2048 input!

    // KQV = ggml_reshape_2d(ctx_c, KQV, n_head * head_dim_kv, N);

    // Output projection
    // cur = ggml_mul_mat(ctx_c, model->layers[il].wo, KQV);
    // if (model->layers[il].bo)
    //   cur = ggml_add(ctx_c, cur, model->layers[il].bo);

    // Residual connection
    // cur = ggml_add(ctx_c, cur, inpL);

    // =========================================================================
    // FFN
    // =========================================================================
    struct ggml_tensor *inpFF = cur;
    cur = ggml_rms_norm(ctx_c, cur, model->hparams.f_norm_rms_eps);
    cur = ggml_mul(ctx_c, cur, model->layers[il].ffn_norm);

    // SwiGLU FFN (using smart dispatcher for INT4 support)
    struct ggml_tensor *w1 = smart_mul_mat(ctx_c, model->layers[il].w1, cur);
    struct ggml_tensor *w3 = smart_mul_mat(ctx_c, model->layers[il].w3, cur);
    cur = ggml_mul(ctx_c, ggml_silu(ctx_c, w1), w3);
    cur = smart_mul_mat(ctx_c, model->layers[il].w2, cur);

    // Residual connection
    cur = ggml_add(ctx_c, cur, inpFF);
  }

  // =========================================================================
  // 3. Final Layer Norm and LM Head
  // =========================================================================
  cur = ggml_rms_norm(ctx_c, cur, model->hparams.f_norm_rms_eps);
  cur = ggml_mul(ctx_c, cur, model->output_norm);

  if (embedding_mode) {
    return cur;
  }

  // LM Head projection: [n_embd, N] -> [n_vocab, N]
  // ggml_mul_mat(A, B) computes B^T * A^T = (A * B)^T, result is [A->ne[1],
  // B->ne[1]] For tied embeddings: tok_embeddings is [n_embd, n_vocab] We want
  // [n_vocab, N] output. ggml_mul_mat(tok_emb, cur) gives [n_vocab, N] ✓
  cur = ggml_mul_mat(ctx_c, model->output, cur);
  ggml_set_name(cur, "output");

  // Add to graph
  if (gf) {
    ggml_build_forward_expand(gf, cur);
  }

  return cur;
}

// ============================================================================
// Grammar-Based Sampling Implementation
// ============================================================================

void InitGrammarConstraint(GrammarConstraint *grammar,
                           const std::vector<std::string> &vocab) {
  if (!grammar)
    return;

  // Find token IDs for JSON special characters
  for (size_t i = 0; i < vocab.size(); i++) {
    const std::string &token = vocab[i];
    if (token == "{" || token == " {")
      grammar->token_lbrace = i;
    else if (token == "}" || token == " }")
      grammar->token_rbrace = i;
    else if (token == "[" || token == " [")
      grammar->token_lbracket = i;
    else if (token == "]" || token == " ]")
      grammar->token_rbracket = i;
    else if (token == "\"" || token == " \"")
      grammar->token_quote = i;
    else if (token == ":" || token == " :")
      grammar->token_colon = i;
    else if (token == "," || token == " ,")
      grammar->token_comma = i;
  }
}

void GrammarConstraint::UpdateState(const std::string &token_text) {
  if (!enabled || !is_json_mode)
    return;

  accumulated += token_text;

  // Trim leading whitespace for state transitions
  std::string trimmed = token_text;
  size_t start = trimmed.find_first_not_of(" \t\n\r");
  if (start != std::string::npos) {
    trimmed = trimmed.substr(start);
  }

  if (trimmed.empty())
    return;

  char first_char = trimmed[0];

  switch (state) {
  case JSONState::EXPECT_OBJECT_START:
    if (first_char == '{') {
      state = JSONState::EXPECT_KEY_OR_END;
      brace_depth = 1;
    }
    break;

  case JSONState::EXPECT_KEY_OR_END:
    if (first_char == '"') {
      state = JSONState::IN_KEY;
    } else if (first_char == '}') {
      brace_depth--;
      if (brace_depth == 0) {
        state = JSONState::COMPLETED;
      }
    }
    break;

  case JSONState::IN_KEY:
    if (first_char == '"' && !in_escape) {
      state = JSONState::EXPECT_COLON;
    } else if (first_char == '\\') {
      in_escape = !in_escape;
    } else {
      in_escape = false;
    }
    break;

  case JSONState::EXPECT_COLON:
    if (first_char == ':') {
      state = JSONState::EXPECT_VALUE;
    }
    break;

  case JSONState::EXPECT_VALUE:
    if (first_char == '"') {
      state = JSONState::IN_STRING_VALUE;
    } else if (first_char == '{') {
      brace_depth++;
      state = JSONState::EXPECT_KEY_OR_END;
    } else if (first_char == '[') {
      bracket_depth++;
      state = JSONState::IN_ARRAY;
    } else if (isdigit(first_char) || first_char == '-') {
      state = JSONState::IN_NUMBER;
    } else if (trimmed.find("true") == 0 || trimmed.find("false") == 0 ||
               trimmed.find("null") == 0) {
      state = JSONState::EXPECT_COMMA_OR_END;
    }
    break;

  case JSONState::IN_STRING_VALUE:
    if (first_char == '"' && !in_escape) {
      state = JSONState::EXPECT_COMMA_OR_END;
    } else if (first_char == '\\') {
      in_escape = !in_escape;
    } else {
      in_escape = false;
    }
    break;

  case JSONState::IN_NUMBER:
    if (first_char == ',' || first_char == '}' || first_char == ']') {
      state = JSONState::EXPECT_COMMA_OR_END;
      if (first_char == ',') {
        state = brace_depth > 0 ? JSONState::EXPECT_KEY_OR_END
                                : JSONState::EXPECT_VALUE;
      } else if (first_char == '}') {
        brace_depth--;
        if (brace_depth == 0)
          state = JSONState::COMPLETED;
      } else if (first_char == ']') {
        bracket_depth--;
        state = JSONState::EXPECT_COMMA_OR_END;
      }
    }
    break;

  case JSONState::EXPECT_COMMA_OR_END:
    if (first_char == ',') {
      state = brace_depth > 0 ? JSONState::EXPECT_KEY_OR_END
                              : JSONState::EXPECT_VALUE;
    } else if (first_char == '}') {
      brace_depth--;
      if (brace_depth == 0) {
        state = JSONState::COMPLETED;
      }
    } else if (first_char == ']') {
      bracket_depth--;
      if (bracket_depth == 0) {
        state = JSONState::EXPECT_COMMA_OR_END;
      }
    }
    break;

  case JSONState::IN_ARRAY:
    if (first_char == ']') {
      bracket_depth--;
      if (bracket_depth == 0) {
        state = JSONState::EXPECT_COMMA_OR_END;
      }
    } else if (first_char == ',') {
      // Stay in array
    } else if (first_char == '"') {
      state = JSONState::IN_STRING_VALUE;
    } else if (first_char == '{') {
      brace_depth++;
      state = JSONState::EXPECT_KEY_OR_END;
    }
    break;

  case JSONState::COMPLETED:
    break;
  }
}

bool IsDigitToken(const std::string &token) {
  if (token.empty())
    return false;
  for (char c : token) {
    if (!isdigit(c) && c != '.' && c != '-' && c != 'e' && c != 'E' &&
        c != '+' && c != ' ')
      return false;
  }
  return true;
}

bool IsWhitespaceToken(const std::string &token) {
  if (token.empty())
    return false;
  for (char c : token) {
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
      return false;
  }
  return true;
}

bool ContainsChar(const std::string &token, char ch) {
  return token.find(ch) != std::string::npos;
}

void ApplyGrammarMask(float *logits, int n_vocab,
                      const GrammarConstraint *grammar,
                      const std::vector<std::string> &vocab) {
  if (!grammar || !grammar->enabled || !grammar->is_json_mode) {
    return;
  }

  const float NEG_INF = -INFINITY;
  std::vector<bool> allowed(n_vocab, false);

  // Always allow whitespace
  for (int i = 0; i < n_vocab; i++) {
    if (IsWhitespaceToken(vocab[i])) {
      allowed[i] = true;
    }
  }

  switch (grammar->state) {
  case JSONState::EXPECT_OBJECT_START:
    for (int i = 0; i < n_vocab; i++) {
      if (ContainsChar(vocab[i], '{')) {
        allowed[i] = true;
      }
    }
    break;

  case JSONState::EXPECT_KEY_OR_END:
    for (int i = 0; i < n_vocab; i++) {
      if (ContainsChar(vocab[i], '"') || ContainsChar(vocab[i], '}')) {
        allowed[i] = true;
      }
    }
    break;

  case JSONState::IN_KEY:
  case JSONState::IN_STRING_VALUE:
    for (int i = 0; i < n_vocab; i++) {
      const std::string &token = vocab[i];
      bool has_control = false;
      for (char c : token) {
        if (c < 32 && c != '\t' && c != '\n') {
          has_control = true;
          break;
        }
      }
      if (!has_control) {
        allowed[i] = true;
      }
    }
    break;

  case JSONState::EXPECT_COLON:
    for (int i = 0; i < n_vocab; i++) {
      if (ContainsChar(vocab[i], ':')) {
        allowed[i] = true;
      }
    }
    break;

  case JSONState::EXPECT_VALUE:
    for (int i = 0; i < n_vocab; i++) {
      const std::string &token = vocab[i];
      if (ContainsChar(token, '"') || ContainsChar(token, '{') ||
          ContainsChar(token, '[') || IsDigitToken(token) ||
          token.find("true") != std::string::npos ||
          token.find("false") != std::string::npos ||
          token.find("null") != std::string::npos) {
        allowed[i] = true;
      }
    }
    break;

  case JSONState::IN_NUMBER:
    for (int i = 0; i < n_vocab; i++) {
      const std::string &token = vocab[i];
      if (IsDigitToken(token) || ContainsChar(token, ',') ||
          ContainsChar(token, '}') || ContainsChar(token, ']')) {
        allowed[i] = true;
      }
    }
    break;

  case JSONState::EXPECT_COMMA_OR_END:
    for (int i = 0; i < n_vocab; i++) {
      if (ContainsChar(vocab[i], ',') || ContainsChar(vocab[i], '}') ||
          ContainsChar(vocab[i], ']')) {
        allowed[i] = true;
      }
    }
    break;

  case JSONState::IN_ARRAY:
    for (int i = 0; i < n_vocab; i++) {
      const std::string &token = vocab[i];
      if (ContainsChar(token, '"') || ContainsChar(token, '{') ||
          ContainsChar(token, '[') || ContainsChar(token, ']') ||
          ContainsChar(token, ',') || IsDigitToken(token)) {
        allowed[i] = true;
      }
    }
    break;

  case JSONState::COMPLETED:
    break;
  }

  for (int i = 0; i < n_vocab; i++) {
    if (!allowed[i]) {
      logits[i] = NEG_INF;
    }
  }
}

// ============================================================================
// Token Sampling
// ============================================================================

int SampleToken(struct ggml_tensor *logits, int idx,
                const SamplingParams &params) {
  float *logits_data = (float *)logits->data;
  int n_vocab = logits->ne[0];
  float *last_logits = logits_data + idx * n_vocab;

  std::vector<float> working_logits(last_logits, last_logits + n_vocab);

  if (params.grammar && params.vocab) {
    ApplyGrammarMask(working_logits.data(), n_vocab, params.grammar,
                     *params.vocab);
  }

  if (params.repetition_penalty != 1.0f && params.token_history &&
      !params.token_history->empty()) {
    for (int token : *params.token_history) {
      if (token >= 0 && token < n_vocab) {
        if (working_logits[token] < 0) {
          working_logits[token] *= params.repetition_penalty;
        } else {
          working_logits[token] /= params.repetition_penalty;
        }
      }
    }
  }

  if ((params.frequency_penalty != 0.0f || params.presence_penalty != 0.0f) &&
      params.token_history && !params.token_history->empty()) {
    std::map<int, int> token_counts;
    for (int token : *params.token_history) {
      if (token >= 0 && token < n_vocab) {
        token_counts[token]++;
      }
    }

    for (auto &kv : token_counts) {
      int token = kv.first;
      int count = kv.second;
      float penalty = params.frequency_penalty * count +
                      params.presence_penalty * (count > 0 ? 1.0f : 0.0f);
      working_logits[token] -= penalty;
    }
  }

  if (params.temperature != 1.0f && params.temperature > 0.0f) {
    for (int i = 0; i < n_vocab; i++) {
      working_logits[i] /= params.temperature;
    }
  }

  float max_logit =
      *std::max_element(working_logits.begin(), working_logits.end());
  std::vector<float> probs(n_vocab);
  float sum_exp = 0.0f;
  for (int i = 0; i < n_vocab; i++) {
    probs[i] = std::exp(working_logits[i] - max_logit);
    sum_exp += probs[i];
  }
  for (int i = 0; i < n_vocab; i++) {
    probs[i] /= sum_exp;
  }

  std::vector<std::pair<float, int>> prob_idx;
  prob_idx.reserve(n_vocab);
  for (int i = 0; i < n_vocab; i++) {
    prob_idx.push_back({probs[i], i});
  }

  std::sort(prob_idx.begin(), prob_idx.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

  int k = std::min(params.top_k, n_vocab);
  if (k > 0 && k < n_vocab) {
    prob_idx.resize(k);
  }

  if (params.min_p > 0.0f && !prob_idx.empty()) {
    float max_prob = prob_idx[0].first;
    float threshold = params.min_p * max_prob;
    auto it = std::remove_if(
        prob_idx.begin(), prob_idx.end(),
        [threshold](const auto &p) { return p.first < threshold; });
    prob_idx.erase(it, prob_idx.end());
  }

  if (params.top_p < 1.0f && !prob_idx.empty()) {
    float cumulative = 0.0f;
    size_t cutoff = 0;
    for (size_t i = 0; i < prob_idx.size(); i++) {
      cumulative += prob_idx[i].first;
      cutoff = i + 1;
      if (cumulative >= params.top_p) {
        break;
      }
    }
    prob_idx.resize(cutoff);
  }

  if (prob_idx.empty()) {
    auto max_it =
        std::max_element(working_logits.begin(), working_logits.end());
    return std::distance(working_logits.begin(), max_it);
  }

  float total = 0.0f;
  for (const auto &p : prob_idx) {
    total += p.first;
  }
  for (auto &p : prob_idx) {
    p.first /= total;
  }

  float random_val = (float)rand() / RAND_MAX;
  float cumulative = 0.0f;
  for (const auto &p : prob_idx) {
    cumulative += p.first;
    if (random_val <= cumulative) {
      return p.second;
    }
  }

  return prob_idx[0].second;
}
