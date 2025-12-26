#include "inference.h"

#include "flash_attention.h"
#include "ggml-cpu.h"             // For ggml_get_type_traits_cpu (vec_dot)
#include "ggml.h"                 // Required for ggml_tensor definition
#include "hardware_topology.h"    // For compute thread affinity
#include "optimization_bridge.h"  // Runtime SIMD dispatch

#ifndef GGML_KQ_MASK_PAD
#define GGML_KQ_MASK_PAD 32
#endif
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "kv_cache.h"  // Added for KV cache
#include "memory_pool.h"
#include "quantization/int4_types.h"  // For TensorInt4
#include "simd_ops.h"

// ============================================================================
// InferenceContext Implementation ("Rebuild Graph, Reuse Memory" Strategy)
// ============================================================================

void InferenceContext::Init(size_t buffer_size) {
    if (initialized) {
        return;  // Already initialized
    }

    // Allocate aligned buffer for GGML context
    // Use 64-byte alignment for AVX-512 compatibility
    compute_buffer.resize(buffer_size);

    struct ggml_init_params params = {
        .mem_size = buffer_size,
        .mem_buffer = compute_buffer.data(),
        .no_alloc = false,
    };
    ctx_compute = ggml_init(params);

    if (ctx_compute) {
        initialized = true;
        std::cerr << "[InferenceContext] Initialized with " << (buffer_size / (1024 * 1024))
                  << " MB persistent buffer" << std::endl;
    } else {
        std::cerr << "[InferenceContext] ERROR: Failed to initialize GGML context!" << std::endl;
    }
}

void InferenceContext::Reset() {
    if (!initialized || compute_buffer.empty()) {
        return;
    }

    // GGML doesn't expose a public ggml_reset_pool() API.
    // Workaround: Free and re-init with the SAME memory buffer.
    // This is effectively O(1) since:
    //   - No malloc/free syscalls (buffer is reused)
    //   - ggml_init just sets up internal allocator state
    if (ctx_compute) {
        ggml_free(ctx_compute);
    }

    struct ggml_init_params params = {
        .mem_size = compute_buffer.size(),
        .mem_buffer = compute_buffer.data(),
        .no_alloc = false,
    };
    ctx_compute = ggml_init(params);
}

void InferenceContext::Free() {
    if (ctx_compute) {
        ggml_free(ctx_compute);
        ctx_compute = nullptr;
    }
    compute_buffer.clear();
    compute_buffer.shrink_to_fit();
    initialized = false;
}

// ============================================================================
// GLOBAL BATCH CONTEXT FOR KV CACHE CALLBACKS
// ============================================================================
// This solves the stale pointer problem with graph caching.
// Instead of storing a batch pointer in userdata (which becomes stale when
// the graph is reused), we store a pointer to this global context which is
// updated BEFORE each graph execution.
// ============================================================================
struct GlobalBatchContext {
    const BatchSpec* batch = nullptr;  // Updated before each graph execution
};

// Thread-local global batch context (one per worker thread)
static thread_local GlobalBatchContext g_batch_context;

// Update global batch context before each graph execution
// NOTE: Not inline - needs external linkage for worker.cpp to call
void SetCurrentBatch(const BatchSpec* batch) {
    g_batch_context.batch = batch;
}

// Get current batch from global context (used in callbacks)
inline const BatchSpec* GetCurrentBatch() {
    return g_batch_context.batch;
}

// User data for KV cache operations
// NOTE: Uses global batch context instead of direct batch pointer
struct KVCacheUserData {
    PagedKVCache* cache;
    // Removed: const BatchSpec *batch; -- now uses GetCurrentBatch()
    int layer;
    int head_dim_kv;  // Store dynamically detected head dim
    bool is_k;        // True if processing K, false if processing V
};

// Thread-local pool of KVCacheUserData to avoid memory allocation per layer
// Max layers supported: 128 (enough for any current model)
// Each layer needs 2 entries (K and V), so 256 total slots
static constexpr int kMaxKVCacheUserDataSlots = 256;
static thread_local KVCacheUserData g_kv_userdata_pool[kMaxKVCacheUserDataSlots];

// Helper to get a userdata slot (no allocation, no leak)
inline KVCacheUserData* GetKVCacheUserData(int layer, bool is_k) {
    int idx = layer * 2 + (is_k ? 0 : 1);
    if (idx >= kMaxKVCacheUserDataSlots) {
        idx = idx % kMaxKVCacheUserDataSlots;  // Wrap for safety
    }
    return &g_kv_userdata_pool[idx];
}

// ============================================================================
// NEW: Robust KV Cache Update and Gather UserData
// ============================================================================
// This replaces the fragile cb_kv_manage approach that relied on ggml_pad
// assumptions. The new approach explicitly:
//   1. Writes current K/V to the PagedKVCache
//   2. Reads history from the cache into destination tensor
//   3. Appends current K/V to destination tensor
// ============================================================================

struct KVUpdateGatherUserData {
    PagedKVCache* cache;             // KV cache instance
    const BatchSpec* batch;          // Batch specification with block tables
    int layer;                       // Current transformer layer
    int head_dim;                    // Dimension per head
    int n_head_kv;                   // Number of KV heads
    int N;                           // Current batch size (new tokens)
    int n_past;                      // Number of past/history tokens
    bool is_k;                       // True for K tensor, false for V tensor
    struct ggml_tensor* src_tensor;  // Pointer to Kcur/Vcur tensor (data accessed at runtime)
};

// Thread-local pool for KVUpdateGatherUserData
static constexpr int kMaxKVUpdateGatherSlots = 256;
static thread_local KVUpdateGatherUserData g_kv_update_gather_pool[kMaxKVUpdateGatherSlots];

inline KVUpdateGatherUserData* GetKVUpdateGatherUserData(int layer, bool is_k) {
    int idx = layer * 2 + (is_k ? 0 : 1);
    if (idx >= kMaxKVUpdateGatherSlots) {
        idx = idx % kMaxKVUpdateGatherSlots;
    }
    return &g_kv_update_gather_pool[idx];
}

// ============================================================================
// Custom RoPE Callback (AVX-512 Optimized)
// ============================================================================

/**
 * User data for custom RoPE operation
 */
struct RoPEUserData {
    const float* cos_sin_table;  ///< Pre-computed [max_seq, head_dim] table
    const int* positions;        ///< Token positions [n_tokens]
    int n_tokens;                ///< Number of tokens
    int n_heads;                 ///< Number of heads (Q or KV)
    int head_dim;                ///< Dimension per head
    int rope_dim;                ///< Number of dimensions to rotate
};

// Thread-local pool for RoPE user data (avoids allocation per layer)
static constexpr int kMaxRoPEUserDataSlots = 512;
static thread_local RoPEUserData g_rope_userdata_pool[kMaxRoPEUserDataSlots];
static thread_local int g_rope_userdata_index = 0;

inline RoPEUserData* GetRoPEUserData() {
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
void cb_rope_avx512(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                    void* userdata) {
    auto* ud = (RoPEUserData*)userdata;
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
            const float* cs_ptr = ud->cos_sin_table + pos * head_dim;

            // Input/output pointers for this head and token
            // Layout: [head_dim, n_heads, n_tokens] -> offset = head_dim * (h +
            // n_heads * t)
            const float* in_ptr = (const float*)src->data + head_dim * (h + n_heads * t);
            float* out_ptr = (float*)dst->data + head_dim * (h + n_heads * t);

#if defined(__AVX512F__)
            // AVX-512 path: process 16 floats (8 pairs) at a time
            int d = 0;
            for (; d + 16 <= rope_dim; d += 16) {
                // Load input: [x0, x1, x2, x3, ..., x14, x15]
                __m512 x = _mm512_loadu_ps(in_ptr + d);

                // Load cos/sin: [c0, s0, c1, s1, c2, s2, ...]
                __m512 cs = _mm512_loadu_ps(cs_ptr + d);

                // Deinterleave cos and sin using permute
                // cos = [c0, c0, c1, c1, ...]  sin = [s0, s0, s1, s1, ...]
                const __m512i idx_cos =
                    _mm512_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
                const __m512i idx_sin =
                    _mm512_setr_epi32(1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15);

                __m512 cos_vec = _mm512_permutexvar_ps(idx_cos, cs);
                __m512 sin_vec = _mm512_permutexvar_ps(idx_sin, cs);

                // Create swapped x: [x1, x0, x3, x2, x5, x4, ...]
                const __m512i idx_swap =
                    _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
                __m512 x_swap = _mm512_permutexvar_ps(idx_swap, x);

                // RoPE formula:
                //   x'_{2d}   = x_{2d} * cos - x_{2d+1} * sin  (even positions)
                //   x'_{2d+1} = x_{2d} * sin + x_{2d+1} * cos  (odd positions)
                // Alternating sign: [-1, +1, -1, +1, ...]
                const __m512 sign_mask =
                    _mm512_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
                                  1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);

                // out = x * cos + x_swap * sign * sin
                __m512 result = _mm512_mul_ps(x, cos_vec);
                __m512 term2 = _mm512_mul_ps(x_swap, sin_vec);
                term2 = _mm512_mul_ps(term2, sign_mask);
                result = _mm512_add_ps(result, term2);

                _mm512_storeu_ps(out_ptr + d, result);
            }

            // Handle remainder (less than 16 floats)
            for (; d < rope_dim; d += 2) {
                float x0 = in_ptr[d];
                float x1 = in_ptr[d + 1];
                float cos_val = cs_ptr[d];
                float sin_val = cs_ptr[d + 1];

                out_ptr[d] = x0 * cos_val - x1 * sin_val;
                out_ptr[d + 1] = x0 * sin_val + x1 * cos_val;
            }
#else
            // Scalar fallback for non-AVX512 builds
            for (int d = 0; d < rope_dim; d += 2) {
                float x0 = in_ptr[d];
                float x1 = in_ptr[d + 1];
                float cos_val = cs_ptr[d];
                float sin_val = cs_ptr[d + 1];

                out_ptr[d] = x0 * cos_val - x1 * sin_val;
                out_ptr[d + 1] = x0 * sin_val + x1 * cos_val;
            }
#endif

            // Copy dimensions beyond rope_dim unchanged
            for (int dd = rope_dim; dd < head_dim; dd++) {
                out_ptr[dd] = in_ptr[dd];
            }
        }
    }
}

// ============================================================================
// Fused Add + RMSNorm Callback (AVX-512 Optimized)
// ============================================================================
// Combines residual connection (x += residual) and RMSNorm in a single pass
// to reduce memory bandwidth by loading/storing data once instead of twice.
// ============================================================================

/**
 * User data for fused Add+RMSNorm operation
 */
struct AddRMSNormUserData {
    const float* residual;    ///< Residual tensor data [n_embd, N]
    const float* rms_weight;  ///< RMSNorm weight [n_embd]
    int n_embd;               ///< Embedding dimension
    int n_tokens;             ///< Number of tokens
    float eps;                ///< RMSNorm epsilon
};

// Thread-local pool for AddRMSNorm user data
static constexpr int kMaxAddRMSNormSlots = 256;
static thread_local AddRMSNormUserData g_add_rmsnorm_pool[kMaxAddRMSNormSlots];
static thread_local int g_add_rmsnorm_index = 0;

inline AddRMSNormUserData* GetAddRMSNormUserData() {
    int idx = g_add_rmsnorm_index++;
    if (idx >= kMaxAddRMSNormSlots) {
        g_add_rmsnorm_index = 0;
        idx = 0;
    }
    return &g_add_rmsnorm_pool[idx];
}

/**
 * Custom callback for fused Add + RMSNorm
 *
 * Input tensor (src): Current tensor to add residual to and normalize
 * Output tensor (dst): Result of (src + residual) normalized with RMSNorm
 *
 * Uses AVX-512 fused kernel for optimal memory bandwidth utilization.
 */
void cb_residual_rmsnorm_fused(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith,
                               int nth, void* userdata) {
    auto* ud = (AddRMSNormUserData*)userdata;
    if (!ud || !ud->residual || !ud->rms_weight)
        return;

    const int n_embd = ud->n_embd;
    const int n_tokens = ud->n_tokens;
    const float eps = ud->eps;

    // Partition work across tokens
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

    if (t_start >= n_tokens)
        return;

    // Process assigned tokens
    for (int t = t_start; t < t_end; t++) {
        const float* x_ptr = (const float*)src->data + t * n_embd;
        const float* res_ptr = ud->residual + t * n_embd;
        float* out_ptr = (float*)dst->data + t * n_embd;

        // Use unified AddRMSNorm dispatcher (Runtime AVX512/AVX2/Scalar)
        densecore::simd::AddRMSNorm(out_ptr, x_ptr, res_ptr, ud->rms_weight,
                                    static_cast<size_t>(n_embd), eps);
    }
}

// ============================================================================
// Fused QKV Projection Callback (Tensor-Level Parallelism)
// ============================================================================
// Computes Q, K, V projections in a single pass with intra-operator parallelism
// across the output dimension (dim_q + dim_k + dim_v). This enables
// multi-thread utilization during decode when batch_size=1.
// ============================================================================

/**
 * User data for fused Q/K/V projection operation
 */
struct QKVUserData {
    const float* x;    ///< Input tensor [n_tokens, n_embd]
    const float* w_q;  ///< Q weight [dim_q, n_embd] (row-major)
    const float* w_k;  ///< K weight [dim_k, n_embd] (row-major)
    const float* w_v;  ///< V weight [dim_v, n_embd] (row-major)
    float* q_out;      ///< Q output [n_tokens, dim_q]
    float* k_out;      ///< K output [n_tokens, dim_k]
    float* v_out;      ///< V output [n_tokens, dim_v]
    int n_embd;        ///< Input embedding dimension
    int dim_q;         ///< Q output dimension (n_head * head_dim)
    int dim_k;         ///< K output dimension (n_head_kv * head_dim)
    int dim_v;         ///< V output dimension (n_head_kv * head_dim)
    int n_tokens;      ///< Number of tokens in batch
};

// Thread-local pool for QKV userdata
static constexpr int kMaxQKVUserDataSlots = 256;
static thread_local QKVUserData g_qkv_userdata_pool[kMaxQKVUserDataSlots];
static thread_local int g_qkv_userdata_index = 0;

inline QKVUserData* GetQKVUserData() {
    int idx = g_qkv_userdata_index++;
    if (idx >= kMaxQKVUserDataSlots) {
        g_qkv_userdata_index = 0;
        idx = 0;
    }
    return &g_qkv_userdata_pool[idx];
}

/**
 * Custom callback for fused Q/K/V projection with HYBRID parallelism
 *
 * Implements a smart dispatch strategy based on batch size:
 *
 * CASE A - DECODE (Small Batch: n_tokens < nth):
 *   - All threads iterate through ALL tokens
 *   - Pass real ith/nth to ComputeQKV for TENSOR PARALLELISM
 *   - Multiple threads collaborate on each token's output dimensions
 *   - Fixes single-threaded bottleneck when batch_size=1
 *
 * CASE B - PREFILL (Large Batch: n_tokens >= nth):
 *   - Partition tokens across threads (TOKEN PARALLELISM)
 *   - Each thread computes FULL dimensions for its token subset
 *   - Pass ith=0, nth=1 to ComputeQKV to disable dimension splitting
 *   - More cache-friendly since each thread works on contiguous data
 */
void cb_compute_qkv(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                    void* userdata) {
    // ===========================================================================
    // BARRIER SAFETY CONTRACT:
    // ===========================================================================
    // This callback is invoked by GGML's thread pool. ALL threads must reach the
    // end of this function cleanly, even if they have no work to do.
    //
    // - ComputeQKV handles `start >= end` by doing nothing and returning early
    // - Early returns here are safe ONLY in PREFILL case (token partitioning)
    // - In DECODE case, all threads iterate all tokens (no early return)
    // ===========================================================================
    auto* ud = static_cast<QKVUserData*>(userdata);
    if (!ud || !ud->x || !ud->w_q || !ud->w_k || !ud->w_v)
        return;

    const int n_tokens = ud->n_tokens;

    // ==========================================================================
    // HYBRID DISPATCH: Choose parallelism strategy based on batch size
    // ==========================================================================

    if (n_tokens < nth) {
        // ========================================================================
        // CASE A: DECODE (Tensor Parallelism)
        // ========================================================================
        // Few tokens (typically 1 during generation), many threads
        // All threads collaborate on ALL tokens, each computing a SLICE of dims
        // ========================================================================
        for (int t = 0; t < n_tokens; t++) {
            const float* x_t = ud->x + t * ud->n_embd;
            float* q_t = ud->q_out + t * ud->dim_q;
            float* k_t = ud->k_out + t * ud->dim_k;
            float* v_t = ud->v_out + t * ud->dim_v;

            // Each thread computes slice [start_col, end_col) of output dimensions
            // ComputeQKV internally partitions: total_cols = dim_q + dim_k + dim_v
            densecore::simd::ComputeQKV(q_t, k_t, v_t, x_t, ud->w_q, ud->w_k, ud->w_v, ud->n_embd,
                                        ud->dim_q, ud->dim_k, ud->dim_v, ith,
                                        nth  // Enable tensor parallelism
            );
        }
    } else {
        // ========================================================================
        // CASE B: PREFILL (Token Parallelism)
        // ========================================================================
        // Many tokens (prompt processing), partition tokens across threads
        // Each thread computes FULL dimensions for its subset of tokens
        // More cache-friendly: each thread touches contiguous weight rows
        // ========================================================================
        const int tokens_per_thread = (n_tokens + nth - 1) / nth;  // Ceiling div
        const int t_start = ith * tokens_per_thread;
        const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

        // Early exit if this thread has no tokens to process
        if (t_start >= n_tokens)
            return;

        for (int t = t_start; t < t_end; t++) {
            const float* x_t = ud->x + t * ud->n_embd;
            float* q_t = ud->q_out + t * ud->dim_q;
            float* k_t = ud->k_out + t * ud->dim_k;
            float* v_t = ud->v_out + t * ud->dim_v;

            // Compute FULL dimensions for this token (no dimension splitting)
            // Pass ith=0, nth=1 to disable tensor parallelism within ComputeQKV
            densecore::simd::ComputeQKV(
                q_t, k_t, v_t, x_t, ud->w_q, ud->w_k, ud->w_v, ud->n_embd, ud->dim_q, ud->dim_k,
                ud->dim_v, 0,
                1  // Disable tensor parallelism (single-threaded kernel call)
            );
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
void InitRoPETable(TransformerModel* model) {
    if (!model)
        return;

    const int n_ctx = model->hparams.n_ctx;
    int head_dim = model->hparams.n_embd / model->hparams.n_head;
    if (model->hparams.n_embd_head_k > 0) {
        head_dim = model->hparams.n_embd_head_k;
    }
    const float freq_base = model->hparams.rope_freq_base;

    // Reuse RoPETable from simd_ops.h to avoid code duplication
    densecore::simd::RoPETable table;
    table.Init(n_ctx, head_dim, freq_base);

    // Move the computed data to the model
    model->rope_cos_sin = std::move(table.cos_sin);
    model->rope_head_dim = head_dim;
}

// Custom callback to load K/V history from cache and append current K/V
// This gathers the full context (history + current) into the destination tensor
// AND writes the current K/V into the cache for future steps.
//
// CRITICAL: Uses GetCurrentBatch() instead of storing batch pointer in
// userdata. This allows graph caching to work correctly because the batch is
// read from a global thread-local context that is updated before each graph
// execution.
void cb_kv_manage(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                  void* userdata) {
    // Prevent race conditions: only run on the first thread
    if (ith != 0)
        return;

    auto* ud = (KVCacheUserData*)userdata;
    if (!ud || !ud->cache)
        return;

    // Get current batch from global context (updated before each graph execution)
    const BatchSpec* batch = GetCurrentBatch();
    if (!batch)
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

    // Read batch tokens size and determine past/total
    const int N = batch->tokens.size();
    const int n_total = src->ne[2];
    const int n_past = n_total - N;

    const size_t bytes_per_elem = sizeof(float);
    const size_t head_block_size = head_dim * n_head_kv;
    const size_t head_block_bytes = head_block_size * bytes_per_elem;

    // 1. Write current tokens to cache
    for (int i = 0; i < N; i++) {
        int seq_id = batch->seq_id[i];
        int pos = batch->pos[i];

        if (seq_id >= (int)batch->block_tables.size())
            continue;

        const auto& block_table = batch->block_tables[seq_id];
        int logical_block = pos / BLOCK_SIZE;
        int slot = pos % BLOCK_SIZE;

        if (logical_block < (int)block_table.size()) {
            int block_id = block_table[logical_block];
            // src has data at [i]
            const float* src_data = (const float*)src->data + i * head_block_size;

            if (ud->is_k)
                ud->cache->WriteKSlot(ud->layer, block_id, slot, src_data);
            else
                ud->cache->WriteVSlot(ud->layer, block_id, slot, src_data);
        }
    }

    // 2. Read full history
    if (n_past > 0) {
        int seq_id = batch->seq_id[0];
        const auto& block_table = batch->block_tables[seq_id];

        for (int i = 0; i < n_past; i++) {
            int pos = i;
            int logical_block = pos / BLOCK_SIZE;
            int slot = pos % BLOCK_SIZE;

            if (logical_block < (int)block_table.size()) {
                int block_id = block_table[logical_block];
                float* dst_data = (float*)dst->data + i * head_block_size;

                if (ud->is_k)
                    ud->cache->ReadKSlot(ud->layer, block_id, slot, dst_data);
                else
                    ud->cache->ReadVSlot(ud->layer, block_id, slot, dst_data);
            } else {
                memset((float*)dst->data + i * head_block_size, 0, head_block_bytes);
            }
        }
    }

    // 3. Copy current tokens to the END of dst
    // src[0..N] -> dst[n_past..n_total]
    memcpy((float*)dst->data + n_past * head_block_size, src->data, N * head_block_bytes);
}

// ============================================================================
// NEW: Robust KV Cache Update and Gather Callback
// ============================================================================
/**
 * @brief Unified KV Cache Update and Gather Callback
 *
 * This callback performs three operations atomically:
 *   Step A: Write current K/V tokens into PagedKVCache
 *   Step B: Read historical K/V from cache into destination tensor
 *   Step C: Append current K/V to destination tensor after history
 *
 * Key improvements over cb_kv_manage:
 *   - Does NOT rely on ggml_pad assumptions about data placement
 *   - Explicitly controls all memory operations
 *   - Uses src_data pointer from userdata (not src tensor)
 *   - Clear, sequential steps with bounds checking
 *
 * Threading: ith == 0 only to avoid race conditions on cache writes.
 *
 * Input: dst is pre-allocated [head_dim * n_head_kv, n_past + N]
 * Output: dst filled with [history (0..n_past) | current (n_past..n_total)]
 */
void cb_kv_update_and_gather(struct ggml_tensor* dst, const struct ggml_tensor* /* src - unused */,
                             int ith, int nth, void* userdata) {
    (void)nth;  // Unused - single thread execution

    // Only thread 0 performs the work to avoid race conditions
    if (ith != 0)
        return;

    auto* ud = static_cast<KVUpdateGatherUserData*>(userdata);
    if (!ud || !ud->cache || !ud->batch || !ud->src_tensor)
        return;

    // Get source data pointer from tensor at runtime (after GGML backend
    // allocates memory)
    const float* src_data = reinterpret_cast<const float*>(ud->src_tensor->data);
    if (!src_data)
        return;

    const int layer = ud->layer;
    const int head_dim = ud->head_dim;
    const int n_head_kv = ud->n_head_kv;
    const int N = ud->N;
    const int n_past = ud->n_past;
    const bool is_k = ud->is_k;

    const size_t head_block_size = static_cast<size_t>(head_dim) * n_head_kv;
    const size_t head_block_bytes = head_block_size * sizeof(float);
    const int n_total = n_past + N;

    // ===========================================================================
    // BOUNDS CHECK: Verify destination tensor has sufficient size
    // ===========================================================================
    const size_t expected_bytes = head_block_size * n_total * sizeof(float);
    if (ggml_nbytes(dst) < expected_bytes) {
        // Tensor too small - this indicates a graph construction error
        // Log and return to avoid buffer overflow
        return;
    }

    // ===========================================================================
    // STEP A: Write current tokens to cache
    // ===========================================================================
    // For each new token in the batch, write its K/V to the PagedKVCache
    // ===========================================================================
    for (int i = 0; i < N; i++) {
        // Validate seq_id bounds
        if (i >= static_cast<int>(ud->batch->seq_id.size()))
            continue;

        int seq_id = ud->batch->seq_id[i];
        int pos = ud->batch->pos[i];

        // Validate block_table bounds
        if (seq_id < 0 || seq_id >= static_cast<int>(ud->batch->block_tables.size()))
            continue;

        const auto& block_table = ud->batch->block_tables[seq_id];
        int logical_block = pos / BLOCK_SIZE;
        int slot = pos % BLOCK_SIZE;

        // Validate logical block exists
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size()))
            continue;

        int block_id = block_table[logical_block];
        const float* token_data = src_data + i * head_block_size;

        if (is_k) {
            ud->cache->WriteKSlot(layer, block_id, slot, token_data);
        } else {
            ud->cache->WriteVSlot(layer, block_id, slot, token_data);
        }
    }

    // ===========================================================================
    // STEP B: Gather history from cache into destination tensor [0, n_past)
    // ===========================================================================
    // Read all historical K/V values from the PagedKVCache into dst
    // ===========================================================================
    if (n_past > 0) {
        // Use seq_id from first token (all tokens in batch share same sequence for
        // decode)
        int seq_id = ud->batch->seq_id[0];

        if (seq_id >= 0 && seq_id < static_cast<int>(ud->batch->block_tables.size())) {
            const auto& block_table = ud->batch->block_tables[seq_id];

            for (int i = 0; i < n_past; i++) {
                int logical_block = i / BLOCK_SIZE;
                int slot = i % BLOCK_SIZE;

                float* dst_slot = reinterpret_cast<float*>(dst->data) + i * head_block_size;

                if (logical_block >= 0 && logical_block < static_cast<int>(block_table.size())) {
                    int block_id = block_table[logical_block];
                    if (is_k) {
                        ud->cache->ReadKSlot(layer, block_id, slot, dst_slot);
                    } else {
                        ud->cache->ReadVSlot(layer, block_id, slot, dst_slot);
                    }
                } else {
                    // Block doesn't exist - zero-fill this slot
                    memset(dst_slot, 0, head_block_bytes);
                }
            }
        } else {
            // Invalid sequence - zero-fill entire history section
            memset(dst->data, 0, n_past * head_block_bytes);
        }
    }

    // ===========================================================================
    // STEP C: Append current tokens to destination tensor [n_past, n_total)
    // ===========================================================================
    // Copy the current K/V data after the history section
    // ===========================================================================
    float* dst_current = reinterpret_cast<float*>(dst->data) + n_past * head_block_size;
    memcpy(dst_current, src_data, N * head_block_bytes);
}

// ============================================================================
// Parallel GEMV Callback for Decode-Phase (N=1)
// ============================================================================
// GGML's ggml_mul_mat parallelizes along batch dimension.
// During decode (batch_size=1), there's NO parallelism opportunity.
// This callback uses GemvParallel to parallelize along output dimension.
// ============================================================================

/**
 * User data for parallel GEMV operation
 */
struct GemvUserData {
    struct ggml_tensor* weight_tensor;  // Weight tensor (data accessed at runtime)
    int N;                              // Input dimension
    int K;                              // Output dimension
    ggml_type weight_type;              // Tensor type (F32, Q4_K, Q8_0, etc.)
    ggml_type input_quant_type;         // Quantization type for input (Q8_K, Q8_0, or F32)
    const void* quantized_input;        // Pointer to pre-quantized input buffer (nullptr
                                        // = use F32)
};

// Thread-local pool for GEMV userdata
// MUST be large enough for all mul_mat ops in a single graph (deep models >
// 128)
static constexpr int kMaxGemvUserDataSlots = 2048;
static thread_local GemvUserData g_gemv_userdata_pool[kMaxGemvUserDataSlots];
static thread_local int g_gemv_userdata_index = 0;

// =============================================================================
// Thread-local buffer for pre-quantized input (Q8_K format)
// =============================================================================
static constexpr size_t kMaxQuantInputBufferSize = 65536;  // 64KB for large N
alignas(64) static thread_local uint8_t g_quant_input_buffer[kMaxQuantInputBufferSize];

inline GemvUserData* GetGemvUserData() {
    int idx = g_gemv_userdata_index++;
    if (idx >= kMaxGemvUserDataSlots) {
        g_gemv_userdata_index = 0;
        idx = 0;
    }
    return &g_gemv_userdata_pool[idx];
}

/**
 * Custom callback for parallel GEMV (decode-phase)
 *
 * REFACTORED: Uses GGML_OP_CUSTOM signature to allow output tensor shape
 * to be independent of input tensor shape. This fixes memory corruption
 * when Qwen3 projections change dimensions (e.g., 1024 -> 2048).
 *
 * Signature: void (*)(struct ggml_tensor *dst, int ith, int nth, void
 * *userdata) Input tensor accessed via dst->src[0]
 */
void cb_gemv_custom(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto* ud = static_cast<GemvUserData*>(userdata);
    if (!ud || !ud->weight_tensor)
        return;

    // Extract input tensor from dst->src[0] (GGML_OP_CUSTOM convention)
    const struct ggml_tensor* src = dst->src[0];
    if (!src)
        return;

    // Extract weight tensor from dst->src[1] (reliable graph topology, not
    // userdata)
    const struct ggml_tensor* weight_tensor = dst->src[1];
    if (!weight_tensor || !weight_tensor->data) {
        fprintf(stderr, "CRITICAL: GEMV weight tensor is null or has no data\n");
        return;
    }

    // Get data pointers at runtime (guaranteed valid after GGML allocates)
    const float* x_f32 = reinterpret_cast<const float*>(src->data);
    const void* weight_data = weight_tensor->data;
    float* output = reinterpret_cast<float*>(dst->data);

    if (!x_f32 || !weight_data || !output)
        return;

    // ==========================================================================
    // DIMENSION VALIDATION (using weight tensor from graph, not stale userdata)
    // ==========================================================================
    const int K = static_cast<int>(weight_tensor->ne[1]);  // Output dimension from weight
    const int N = static_cast<int>(weight_tensor->ne[0]);  // Input dimension from weight

    // Validate output tensor matches expected K
    if (dst->ne[0] != K) {
        fprintf(stderr,
                "CRITICAL: GEMV buffer mismatch! dst->ne[0](%ld) != weight->ne[1](%d). "
                "Output tensor was sized incorrectly.\n",
                (long)dst->ne[0], K);
        return;
    }

    const ggml_type weight_type = weight_tensor->type;

    // ==========================================================================
    // DEFERRED PRE-QUANTIZATION:
    // Thread 0 quantizes the input vector once; other threads reuse the buffer.
    // ==========================================================================
    const void* quant_input = ud->quantized_input;
    if (quant_input && ith == 0 && ud->input_quant_type != GGML_TYPE_F32) {
        const auto* input_type_traits = ggml_get_type_traits_cpu(ud->input_quant_type);
        if (input_type_traits && input_type_traits->from_float) {
            input_type_traits->from_float(x_f32, const_cast<void*>(quant_input),
                                          static_cast<int64_t>(N));
        }
    }

    // Partition output dimension across threads
    const int k_per_thread = (K + nth - 1) / nth;
    const int k_start = ith * k_per_thread;
    const int k_end = std::min(k_start + k_per_thread, K);

    if (k_start >= K)
        return;

    // ==========================================================================
    // CASE A: FP32 weights - use optimized simd::GemvParallel
    // ==========================================================================
    if (weight_type == GGML_TYPE_F32) {
        const float* weight = reinterpret_cast<const float*>(weight_data);
        densecore::simd::GemvParallel(output, x_f32, weight, N, K, ith, nth);
        return;
    }

    // ==========================================================================
    // CASE B: Quantized weights with pre-quantized input - use native vec_dot
    // ==========================================================================
    const size_t row_stride = weight_tensor->nb[1];  // Bytes per row
    const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight_type);

    if (quant_input && type_traits_cpu && type_traits_cpu->vec_dot) {
        for (int k = k_start; k < k_end; k++) {
            const void* row_ptr = reinterpret_cast<const char*>(weight_data) + k * row_stride;
            type_traits_cpu->vec_dot(N, &output[k], 0, row_ptr, 0, quant_input, 0, 1);
        }
        return;
    }

    // ==========================================================================
    // CASE C: Fallback - dequantize weights (no pre-quantized input available)
    // ==========================================================================
    static constexpr size_t kMaxDequantBufferSize = 16384;
    static thread_local float dequant_buffer[kMaxDequantBufferSize];
    const auto* type_traits = ggml_get_type_traits(weight_type);
    if (!type_traits || !type_traits->to_float || N > static_cast<int>(kMaxDequantBufferSize)) {
        for (int k = k_start; k < k_end; k++)
            output[k] = 0.0f;
        return;
    }

    for (int k = k_start; k < k_end; k++) {
        const void* row_ptr = reinterpret_cast<const char*>(weight_data) + k * row_stride;
        type_traits->to_float(row_ptr, dequant_buffer, N);
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += x_f32[i] * dequant_buffer[i];
        }
        output[k] = sum;
    }
}

/**
 * Create a custom GGML operation for parallel GEMV
 *
 * REFACTORED: Uses GGML_OP_CUSTOM instead of GGML_OP_MAP_CUSTOM1.
 * GGML_OP_MAP_CUSTOM1 assumes output shape == input shape, which causes
 * buffer overflows when Qwen3 projections change dimensions (e.g., 1024->2048).
 * GGML_OP_CUSTOM allows the output tensor shape to be independent of inputs.
 */
inline struct ggml_tensor* ggml_mul_mat_gemv(struct ggml_context* ctx, struct ggml_tensor* weight,
                                             struct ggml_tensor* input, GemvUserData* userdata) {
    const int K = weight->ne[1];  // Output dimension
    const int N = weight->ne[0];  // Input dimension

    userdata->weight_tensor = weight;
    userdata->N = N;
    userdata->K = K;
    userdata->weight_type = weight->type;
    userdata->quantized_input = nullptr;
    userdata->input_quant_type = GGML_TYPE_F32;

    const ggml_type wtype = weight->type;
    if (ggml_is_quantized(wtype)) {
        const auto* type_traits_cpu = ggml_get_type_traits_cpu(wtype);
        if (type_traits_cpu && type_traits_cpu->vec_dot) {
            const ggml_type vec_dot_type = type_traits_cpu->vec_dot_type;
            const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
            if (input_type_traits && input_type_traits->from_float) {
                const size_t quant_input_size = ggml_row_size(vec_dot_type, N);
                if (quant_input_size > 0 && quant_input_size <= kMaxQuantInputBufferSize) {
                    userdata->input_quant_type = vec_dot_type;
                    userdata->quantized_input = g_quant_input_buffer;
                }
            }
        }
    }

    int n_threads = InferenceConfig::Instance().num_threads;
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
        if (n_threads <= 0)
            n_threads = 4;
    }

    int physical_cores = densecore::HardwareTopology::GetInstance().GetPhysicalCoreCount();
    if (physical_cores <= 0)
        physical_cores = 4;

    constexpr int kMaxDecodeThreads = 8;
    n_threads = std::min(n_threads, std::min(kMaxDecodeThreads, physical_cores));

    if (K < 256)
        n_threads = std::min(n_threads, 4);
    if (K < 64)
        n_threads = 1;

    // ===========================================================================
    // Create output tensor with correct dimension K (INDEPENDENT of input shape)
    // This is the critical fix: GGML_OP_CUSTOM allows explicit output dimensions
    // ===========================================================================
    const int64_t ne_res[4] = {K, 1, 1, 1};
    struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_res);

    // ===========================================================================
    // Configure GGML_OP_CUSTOM (NOT MAP_CUSTOM1 which assumes shape preservation)
    // ===========================================================================
    result->op = GGML_OP_CUSTOM;
    result->src[0] = input;   // Input tensor accessible via dst->src[0] in callback
    result->src[1] = weight;  // Weight tensor accessible via dst->src[1] in callback

    // Custom op params (layout must match ggml_custom_op_params)
    // Signature: { ggml_custom_op_t fun, int n_tasks, void *userdata }
    // NOTE: userdata is still passed for pre-quantized input buffer pointer,
    //       but dimensions/weight are read directly from dst->src[1] for
    //       reliability
    struct {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    } params = {cb_gemv_custom, n_threads, userdata};
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "params too large");
    memcpy(result->op_params, &params, sizeof(params));

    return result;
}

/**
 * Smart matrix multiplication dispatcher
 *
 * CRITICAL: For decode-phase (batch=1), uses optimized GEMV path but ONLY when
 * dimensions are compatible. The custom GEMV kernel expects weight->ne[0] ==
 * input->ne[0]. If incompatible (e.g., Qwen3 transposed weights), falls back
 * to ggml_mul_mat which handles stride/transpose correctly.
 */
inline struct ggml_tensor* smart_mul_mat(struct ggml_context* ctx, struct ggml_tensor* weight,
                                         struct ggml_tensor* input) {
    const int input_cols = input->ne[1];

    // STRICT COMPATIBILITY CHECK (zero-overhead: evaluated at graph build time)
    const bool is_compatible = (weight->ne[0] == input->ne[0]);
    const bool is_gemv_candidate =
        (input_cols == 1) && (weight->type == GGML_TYPE_F32 || ggml_is_quantized(weight->type));

    if (is_gemv_candidate && is_compatible) {
        // Optimized GEMV path - dimensions
        if (input->type != GGML_TYPE_F32) {
            fprintf(stderr, "CRITICAL: smart_mul_mat input type is %d! Tensor name: %s\n",
                    input->type, input->name);
        }
        GemvUserData* ud = GetGemvUserData();
        return ggml_mul_mat_gemv(ctx, weight, input, ud);
    }

    // Standard GGML fallback (handles transpose/stride correctly)
    return ggml_mul_mat(ctx, weight, input);
}

// ============================================================================
// SIMPLIFIED UNIVERSAL ATTENTION (llama.cpp style)
// This version trades the complex paged KV cache for correctness and
// clarity. Once working, KV cache can be re-added following the proven
// llama.cpp pattern.
// ============================================================================

struct ggml_tensor* BuildTransformerGraph(TransformerModel* model, PagedKVCache* cache,
                                          struct ggml_context* ctx_c, const BatchSpec& batch,
                                          bool embedding_mode, struct ggml_cgraph* gf,
                                          struct ggml_tensor** out_embd,
                                          struct ggml_tensor** out_pos) {
    // ENSURE: ctx_c must be initialized with sufficient memory (e.g. 128MB+)
    // to hold the compute graph nodes, especially for deep models like Qwen.
    // This initialization happens in worker.cpp (InitGraphCache or temp
    // context).

    // N is batch size
    const int N = batch.tokens.size();
    const int n_embd = model->hparams.n_embd;
    const int n_head = model->hparams.n_head;
    const int n_head_kv = model->hparams.n_head_kv;
    const int n_layer = model->hparams.n_layer;
    const int n_ctx = model->hparams.n_ctx;

    // =========================================================================
    // 1. Token Embedding Lookup
    // =========================================================================
    struct ggml_tensor* embd_inp = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, N);
    ggml_set_name(embd_inp, "embd_inp");
    if (embd_inp->data) {
        memcpy(embd_inp->data, batch.tokens.data(), N * sizeof(int));
    }
    if (out_embd)
        *out_embd = embd_inp;

    struct ggml_tensor* cur = ggml_get_rows(ctx_c, model->tok_embeddings, embd_inp);

    // Position tensor for RoPE
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I32, N);
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
        struct ggml_tensor* inpL = cur;

        // Attention Norm
        if (cur->type != GGML_TYPE_F32) {
            fprintf(stderr, "CRITICAL: Layer %d input type is %d! Tensor name: %s\n", il, cur->type,
                    cur->name);
        }
        cur = ggml_rms_norm(ctx_c, cur, model->hparams.f_norm_rms_eps);
        cur = ggml_mul(ctx_c, cur, model->layers[il].attention_norm);

        // Q/K/V Projections (using smart dispatcher for Parallel GEMV)
        struct ggml_tensor* Qcur = smart_mul_mat(ctx_c, model->layers[il].wq, cur);
        struct ggml_tensor* Kcur = smart_mul_mat(ctx_c, model->layers[il].wk, cur);
        struct ggml_tensor* Vcur = smart_mul_mat(ctx_c, model->layers[il].wv, cur);

        // Add Bias if present (for Qwen2 and some other models)
        if (model->layers[il].bq) {
            if (Qcur->ne[0] == model->layers[il].bq->ne[0]) {
                Qcur = ggml_add(ctx_c, Qcur, model->layers[il].bq);
            } else {
                // SKIP BIAS (Safe fallback)
                // std::cerr << "Skipping BQ mismatch L" << il << std::endl;
            }
        }
        if (model->layers[il].bk) {
            if (Kcur->ne[0] == model->layers[il].bk->ne[0]) {
                Kcur = ggml_add(ctx_c, Kcur, model->layers[il].bk);
            } else {
                // SKIP BIAS on mismatch to avoid graph complexity/hangs
                // std::cerr << "Skipping BK mismatch" << std::endl;
            }
        }
        if (model->layers[il].bv) {
            if (Vcur->ne[0] == model->layers[il].bv->ne[0]) {
                Vcur = ggml_add(ctx_c, Vcur, model->layers[il].bv);
            } else {
                // SKIP BIAS
            }
        }

        // Dynamically infer head dimensions from the actual projected tensors
        int dim_q = Qcur->ne[0];
        int dim_k = Kcur->ne[0];
        int dim_v = Vcur->ne[0];

        int n_head_kv = model->hparams.n_head_kv;
        int head_dim_q = dim_q / n_head;
        int head_dim_kv =
            (model->hparams.n_embd_head_k > 0) ? model->hparams.n_embd_head_k : (dim_k / n_head_kv);

        bool k_done = false;
        bool v_done = false;

        // Reshape Q (Standard)
        Qcur = ggml_reshape_3d(ctx_c, Qcur, head_dim_q, n_head, N);

        // Reshape K/V
        if (!k_done) {
            Kcur = ggml_reshape_3d(ctx_c, Kcur, head_dim_kv, n_head_kv, N);
        }
        if (!v_done) {
            Vcur = ggml_reshape_3d(ctx_c, Vcur, head_dim_kv, n_head_kv, N);
        }

        // =========================================================================
        // Per-Head QK-Norm (Architecture-flag based for Qwen3/Qwen2.5)
        // =========================================================================
        // Qwen3 requires RMS normalization applied per-head, not over the entire
        // embedding dimension. We reshape to [head_dim, n_heads * n_tokens] so that
        // ggml_rms_norm normalizes each head_dim vector independently.
        //
        // Using arch_flags for explicit requirement checking instead of implicit
        // null pointer guards. The tensor null check is kept for safety.
        //
        // Flow:
        //   1. Reshape Q from [head_dim, n_head, N] to [head_dim, n_head * N]
        //   2. Apply ggml_rms_norm (normalizes over ne[0] = head_dim)
        //   3. Multiply by weight [head_dim] (broadcasts across all head*token)
        //   4. Reshape back to [head_dim, n_head, N]
        // =========================================================================
        // Q Normalization (required for Qwen3, optional for others)
        if (model->arch_flags.requires_q_norm && model->layers[il].attn_q_norm) {
            const int64_t q_n_tokens = Qcur->ne[2];  // N (batch size)

            if (head_dim_q == model->layers[il].attn_q_norm->ne[0]) {
                // Reshape to 2D: [head_dim, n_head * n_tokens] for per-head norm
                struct ggml_tensor* Q_2d =
                    ggml_reshape_2d(ctx_c, Qcur, head_dim_q, n_head * q_n_tokens);

                // Apply RMS norm (normalizes over ne[0] = head_dim independently)
                if (Q_2d->type != GGML_TYPE_F32) {
                    fprintf(stderr, "CRITICAL: Layer %d Q_2d type is %d! Tensor name: %s\n", il,
                            Q_2d->type, Q_2d->name);
                }
                Q_2d = ggml_rms_norm(ctx_c, Q_2d, model->hparams.f_norm_rms_eps);

                // Multiply by weight [head_dim] - broadcasts across second dimension
                Q_2d = ggml_mul(ctx_c, Q_2d, model->layers[il].attn_q_norm);

                // Reshape back to original 3D: [head_dim, n_head, n_tokens]
                Qcur = ggml_reshape_3d(ctx_c, Q_2d, head_dim_q, n_head, q_n_tokens);
            } else if (il == 0) {
                static bool logged_q_mismatch = false;
                if (!logged_q_mismatch) {
                    std::cerr << "[DenseCore] WARN: attn_q_norm dimension mismatch! "
                              << "Qcur->ne[0]=" << Qcur->ne[0]
                              << " vs norm->ne[0]=" << model->layers[il].attn_q_norm->ne[0]
                              << ". Skipping Q normalization." << std::endl;
                    logged_q_mismatch = true;
                }
            }
        }

        // K Normalization (required for Qwen3, optional for others)
        if (model->arch_flags.requires_k_norm && model->layers[il].attn_k_norm) {
            const int64_t k_n_tokens = Kcur->ne[2];  // N (batch size)

            if (head_dim_kv == model->layers[il].attn_k_norm->ne[0]) {
                // Reshape to 2D: [head_dim, n_head_kv * n_tokens] for per-head norm
                struct ggml_tensor* K_2d =
                    ggml_reshape_2d(ctx_c, Kcur, head_dim_kv, n_head_kv * k_n_tokens);

                // Apply RMS norm (normalizes over ne[0] = head_dim independently)
                if (K_2d->type != GGML_TYPE_F32) {
                    fprintf(stderr, "CRITICAL: Layer %d K_2d type is %d! Tensor name: %s\n", il,
                            K_2d->type, K_2d->name);
                }
                K_2d = ggml_rms_norm(ctx_c, K_2d, model->hparams.f_norm_rms_eps);

                // Multiply by weight [head_dim] - broadcasts across second dimension
                K_2d = ggml_mul(ctx_c, K_2d, model->layers[il].attn_k_norm);

                // Reshape back to original 3D: [head_dim, n_head_kv, n_tokens]
                Kcur = ggml_reshape_3d(ctx_c, K_2d, head_dim_kv, n_head_kv, k_n_tokens);
            } else if (il == 0) {
                static bool logged_k_mismatch = false;
                if (!logged_k_mismatch) {
                    std::cerr << "[DenseCore] WARN: attn_k_norm dimension mismatch! "
                              << "Kcur->ne[0]=" << Kcur->ne[0]
                              << " vs norm->ne[0]=" << model->layers[il].attn_k_norm->ne[0]
                              << ". Skipping K normalization." << std::endl;
                    logged_k_mismatch = true;
                }
            }
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

        // SKIP RoPE if we detected anomaly and sliced (k_done)
        // Layer 0 anomaly (80 dim) is unsafe for 128-dim RoPE/Kernel which expects
        // 128
        bool skip_rope = k_done;
        if (skip_rope) {
            // std::cerr << "[DenseCore] Skipping RoPE for anomalous Layer " << il <<
            // std::endl;
        }

        // Use custom RoPE if pre-computed table is available
        // DISABLED: Custom RoPE callback causes NaN on multi-token generation
        // Root cause: thread_local userdata + position overflow issues
        // Using GGML native rope_ext which is stable and well-tested
        constexpr bool use_custom_rope = false;  // Disabled for stability
        if (!skip_rope && use_custom_rope && !model->rope_cos_sin.empty()) {
            // Apply custom RoPE using pre-computed table
            // Q: [head_dim_q, n_head, N]
            RoPEUserData* q_ud = GetRoPEUserData();
            *q_ud = {model->rope_cos_sin.data(), batch.pos.data(), N, n_head, head_dim_q, rope_dim};
            // n_tasks=1: thread_local userdata requires single-thread execution
            // GGML native ops (ggml_mul_mat) already use full thread pool via
            // ggml_backend_cpu_set_n_threads
            Qcur = ggml_map_custom1(ctx_c, Qcur, cb_rope_avx512, 1, q_ud);

            // K: [head_dim_kv, n_head_kv, N]
            RoPEUserData* k_ud = GetRoPEUserData();
            *k_ud = {
                model->rope_cos_sin.data(), batch.pos.data(), N, n_head_kv, head_dim_kv, rope_dim};
            // n_tasks=1: thread_local userdata requires single-thread execution
            Kcur = ggml_map_custom1(ctx_c, Kcur, cb_rope_avx512, 1, k_ud);
        } else if (!skip_rope) {
            // Fallback to standard GGML RoPE
            Qcur = ggml_rope_ext(ctx_c, Qcur, pos, nullptr, rope_dim, 0, n_ctx,
                                 model->hparams.rope_freq_base, model->hparams.rope_freq_scale,
                                 0.0f, 1.0f, 0.0f, 0.0f);
            Kcur = ggml_rope_ext(ctx_c, Kcur, pos, nullptr, rope_dim, 0, n_ctx,
                                 model->hparams.rope_freq_base, model->hparams.rope_freq_scale,
                                 0.0f, 1.0f, 0.0f, 0.0f);
        }

        // =========================================================================
        // KV CACHE INTEGRATION (Universal Paged Attention)
        // =========================================================================
        struct ggml_tensor* K_all = Kcur;  // Default to current K
        struct ggml_tensor* V_all = Vcur;  // Default to current V

        const bool use_cache = (cache != nullptr);
        int n_past_val = 0;
        if (use_cache && batch.num_seqs > 0 && batch.n_past.size() > 0) {
            n_past_val = batch.n_past[0];
        }
        const int n_total_tokens = n_past_val + N;

        if (use_cache) {  // Re-enabled old KV cache approach
            // Only need fancy logic if we have history.
            // If n_past = 0 (Prefill), K_all == Kcur is mostly fine,
            // BUT we still need to WRITE to cache.
            // The 'ggml_pad' trick updates cache as side effect.
            // So we act always if use_cache is true.

            // Use ggml_pad to create a tensor of correct size (N + n_past)
            // ggml_pad(ctx, a, pad_0, pad_1, pad_2, pad_3)
            // We pad dimension 2 (sequence) by n_past_val.
            // Result shape: [head_dim, n_head, N + n_past]
            struct ggml_tensor* K_padded = Kcur;
            struct ggml_tensor* V_padded = Vcur;

            if (n_past_val > 0) {
                K_padded = ggml_pad(ctx_c, Kcur, 0, 0, n_past_val, 0);
                V_padded = ggml_pad(ctx_c, Vcur, 0, 0, n_past_val, 0);
            }

            KVCacheUserData* k_ud = GetKVCacheUserData(il, true);
            *k_ud = {cache, il, head_dim_kv, true};  // batch accessed via GetCurrentBatch()
            KVCacheUserData* v_ud = GetKVCacheUserData(il, false);
            *v_ud = {cache, il, head_dim_kv, false};  // batch accessed via GetCurrentBatch()

            K_all = ggml_map_custom1(ctx_c, K_padded, cb_kv_manage, 1, k_ud);
            V_all = ggml_map_custom1(ctx_c, V_padded, cb_kv_manage, 1, v_ud);
        }

        // =========================================================================
        // NEW: Robust KV Cache Integration (Replaces ggml_pad approach)
        // =========================================================================
        // This approach explicitly:
        //   1. Allocates destination tensors with full size [head_dim, n_head_kv,
        //   n_total]
        //   2. Uses cb_kv_update_and_gather to write cache, read history, append
        //   current
        //   3. Does NOT rely on ggml_pad padding behavior which was causing
        //   NaN/hangs
        // =========================================================================
        if (false) {  // DISABLED: New approach has graph dependency bugs
            // Step 1: Explicitly allocate K_all and V_all with full context size
            // Shape: [head_dim_kv, n_head_kv, n_total_tokens]
            struct ggml_tensor* K_all_tensor =
                ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, head_dim_kv, n_head_kv, n_total_tokens);
            struct ggml_tensor* V_all_tensor =
                ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, head_dim_kv, n_head_kv, n_total_tokens);
            ggml_set_name(K_all_tensor, "K_all");
            ggml_set_name(V_all_tensor, "V_all");

            // Step 2: Force Kcur/Vcur to be contiguous before passing data pointers
            // This ensures src_data pointer is valid for memcpy in callback
            struct ggml_tensor* Kcur_contig = ggml_cont(ctx_c, Kcur);
            struct ggml_tensor* Vcur_contig = ggml_cont(ctx_c, Vcur);

            // Step 3: Setup userdata for K with src_tensor pointer
            // NOTE: src_tensor is set below after ggml_cont
            //       The tensor pointer is stable; data is populated at graph
            //       execution
            KVUpdateGatherUserData* k_gather_ud = GetKVUpdateGatherUserData(il, true);
            k_gather_ud->cache = cache;
            k_gather_ud->batch = &batch;
            k_gather_ud->layer = il;
            k_gather_ud->head_dim = head_dim_kv;
            k_gather_ud->n_head_kv = n_head_kv;
            k_gather_ud->N = N;
            k_gather_ud->n_past = n_past_val;
            k_gather_ud->is_k = true;
            k_gather_ud->src_tensor = nullptr;  // Set below

            // Step 4: Setup userdata for V
            KVUpdateGatherUserData* v_gather_ud = GetKVUpdateGatherUserData(il, false);
            v_gather_ud->cache = cache;
            v_gather_ud->batch = &batch;
            v_gather_ud->layer = il;
            v_gather_ud->head_dim = head_dim_kv;
            v_gather_ud->n_head_kv = n_head_kv;
            v_gather_ud->N = N;
            v_gather_ud->n_past = n_past_val;
            v_gather_ud->is_k = false;
            v_gather_ud->src_tensor = nullptr;  // Set below

            // Step 5: Create graph nodes that will execute the callbacks
            // The src_data will be populated from the contiguous tensor's data
            // pointer when the graph is executed (data is allocated by this point)
            //
            // WORKAROUND: ggml_map_custom1 passes its input tensor as 'src'.
            // We need to pass BOTH the destination and source data.
            // Solution: Store Kcur_contig as 'src' input, K_all_tensor->data is
            // 'dst'
            //
            // The callback signature is: cb(dst, src, ith, nth, userdata)
            // We set src_data = src->data in the callback if it's nullptr

            // For K: Map from Kcur_contig, output shape matches K_all_tensor
            // We need a custom callback wrapper that sets src_data from src tensor
            // For now, we'll pass the contiguous tensor and handle in callback

            // Actually, ggml_map_custom1(ctx, a, cb, n_tasks, userdata) creates:
            //   result tensor with same shape as 'a'
            //   callback receives: cb(result, a, ith, nth, userdata)
            //
            // So 'a' becomes 'src', and result is 'dst'
            // We need result to have shape [head_dim_kv, n_head_kv, n_total_tokens]
            // This means we should pass K_all_tensor as 'a', not Kcur!
            //
            // But then we need to access Kcur data via userdata.
            // Since Kcur_contig->data is available at graph execution time,
            // we can store its pointer now and it will be valid.

            // CRITICAL FIX: The tensor data pointers are only valid AFTER
            // ggml_backend allocates memory. At graph construction time, data may
            // be nullptr. We need to access the data through the tensor pointer in
            // the callback.

            // Store tensor pointers in userdata (not raw data pointers)
            // This requires modifying the struct to take ggml_tensor* instead of
            // float* For now, we'll use a simpler workaround: pass Kcur_contig as
            // input, create output tensor of correct size via ggml_new_tensor, then
            // use ggml_cpy

            // SIMPLER APPROACH: Use ggml_map_custom1 on K_all_tensor, pass Kcur as
            // extra userdata Since Kcur_contig is built into the graph, its data
            // pointer is stable
            k_gather_ud->src_tensor = Kcur_contig;
            v_gather_ud->src_tensor = Vcur_contig;

            // Create the combined tensors via callback
            K_all = ggml_map_custom1(ctx_c, K_all_tensor, cb_kv_update_and_gather, 1, k_gather_ud);
            V_all = ggml_map_custom1(ctx_c, V_all_tensor, cb_kv_update_and_gather, 1, v_gather_ud);

            // Mark as dependent on Kcur_contig and Vcur_contig for proper execution
            // order
            ggml_build_forward_expand(gf, Kcur_contig);
            ggml_build_forward_expand(gf, Vcur_contig);
        }

        // After projection and reshape:

        // Q: [head_dim_q, n_head, N]
        // K_all: [head_dim_kv, n_head_kv, n_past + N]
        // V_all: [head_dim_kv, n_head_kv, n_past + N]

        // =========================================================================
        // GQA (Grouped Query Attention): LOGICAL BROADCASTING
        // =========================================================================
        // For models like Qwen3 where n_head != n_head_kv (e.g., 32 Q heads, 4 KV
        // heads):
        //
        // OLD APPROACH (REMOVED - caused segfaults and was inefficient):
        //   Used ggml_repeat to physically expand K/V from n_head_kv to n_head.
        //   This allocated 8x more memory and caused OOM/crashes.
        //
        // NEW APPROACH (Logical Broadcasting):
        //   Keep K/V at their original [head_dim, n_head_kv, seq] shape.
        //   The attention kernel computes: kv_head = query_head / (n_head /
        //   n_head_kv) This is zero-copy and memory-efficient.
        //
        // Both ggml_flash_attn_ext and our custom FlashAttentionGQA support this.
        // =========================================================================
        struct ggml_tensor* K = K_all;
        struct ggml_tensor* V = V_all;

        // Compute GQA repetition factor for attention dispatch
        const int n_rep = (n_head_kv > 0) ? (n_head / n_head_kv) : 1;
        (void)n_rep;  // Used in attention mask/kernel setup

        // =========================================================================
        // ATTENTION (llama.cpp style - corrected tensor layouts)
        // =========================================================================
        // Tensor shapes at this point:
        //   Q: [head_dim_q, n_head, N]
        //   K: [head_dim_kv, n_head_kv, n_total_tokens]  (NOT expanded!)
        //   V: [head_dim_kv, n_head_kv, n_total_tokens]  (NOT expanded!)
        //
        // For GQA: The attention kernel handles broadcasting internally.
        // Query heads [0, n_rep) all attend to KV head 0, etc.
        // =========================================================================

        // =========================================================================
        // ATTENTION DISPATCH (Runtime selection based on CPU capabilities)
        // - AVX-512+: Use Flash Attention (ggml_flash_attn_ext) for efficiency
        // - Other: Use standard Q*K^T -> softmax -> V for compatibility
        // =========================================================================
        // Note: Flash Attention still requires AVX-512 for correctness
        // (uses GGML's ggml_flash_attn_ext which has AVX-512 requirement)
        static const bool use_flash_attention =
            densecore::OpsRegistry::IsInitialized() &&
            (densecore::simd::DetectSimdLevel() >= densecore::simd::SimdLevel::AVX512);

        struct ggml_tensor* KQV = nullptr;

        if (use_flash_attention) {
            // -----------------------------------------------------------------------
            // FLASH ATTENTION PATH (AVX-512 only)
            // -----------------------------------------------------------------------
            // ggml_flash_attn_ext natively supports GQA - it handles K/V with fewer
            // heads than Q. The kernel internally computes: kv_head = query_head /
            // n_rep
            //
            // Shapes: Q: [head_dim, N, n_head], K/V: [head_dim, n_total, n_head_kv]
            // -----------------------------------------------------------------------
            struct ggml_tensor* Q = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);

            // K/V: [head_dim, n_head_kv, n_total] -> [head_dim, n_total, n_head_kv]
            struct ggml_tensor* K_fa = ggml_permute(ctx_c, K, 0, 2, 1, 3);
            struct ggml_tensor* V_fa = ggml_permute(ctx_c, V, 0, 2, 1, 3);

            // Create mask [n_total, N_padded, 1, 1] as required by
            // ggml_flash_attn_ext 0.0f = can attend, -INFINITY = cannot attend
            // (masked)
            int N_padded = (N + GGML_KQ_MASK_PAD - 1) & ~(GGML_KQ_MASK_PAD - 1);
            struct ggml_tensor* KQ_mask =
                ggml_new_tensor_4d(ctx_c, GGML_TYPE_F32, n_total_tokens, N_padded, 1, 1);

            // Fill causal mask (column-major: element (k, q) is at k + q * n_kv)
            float* mask_data = (float*)KQ_mask->data;
            for (int q = 0; q < N_padded; q++) {
                for (int k = 0; k < n_total_tokens; k++) {
                    int query_pos = n_past_val + q;
                    int key_pos = k;
                    int idx = k + q * n_total_tokens;

                    if (q >= N) {
                        mask_data[idx] = 0.0f;  // Padding row
                    } else if (key_pos <= query_pos) {
                        mask_data[idx] = 0.0f;  // Can attend
                    } else {
                        mask_data[idx] = -INFINITY;  // Masked (future position)
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
            KQV = ggml_flash_attn_ext(ctx_c, Q, K_fa, V_fa, KQ_mask, scale, 0.0f, 0.0f);

            // Permute to [head_dim, n_head, N] for projection
            KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
        } else {
            // -----------------------------------------------------------------------
            // STANDARD ATTENTION PATH (AVX2/Fallback) - Tiled GQA Implementation
            // -----------------------------------------------------------------------
            // For GQA models (n_head != n_head_kv), we use a TILED approach:
            //   - Iterate over KV heads (h_kv = 0 to n_head_kv)
            //   - For each KV head, process n_rep query heads together
            //   - Use ggml_view to slice tensors without copying (O(1) memory)
            //
            // This avoids the massive memory bloat of ggml_repeat while maintaining
            // correctness on all hardware (AVX2, SSE, etc.)
            // -----------------------------------------------------------------------

            // Scale factor for attention
            float scale = 1.0f / sqrtf((float)head_dim_q);

            if (n_head_kv != n_head) {
// =====================================================================
// TILED GQA: Process query groups per KV head
// =====================================================================
// Q: [head_dim, n_head, N] - Permute to [head_dim, N, n_head]
// K: [head_dim, n_head_kv, n_total] - Permute to [head_dim, n_total,
// n_head_kv] V: [head_dim, n_head_kv, n_total] - Permute to [head_dim,
// n_total, n_head_kv]
//
// For each h_kv in [0, n_head_kv):
//   K_head = K[:, :, h_kv]              -> [head_dim, n_total]
//   V_head = V[:, :, h_kv]              -> [head_dim, n_total]
//   Q_group = Q[:, :, h_kv*n_rep : (h_kv+1)*n_rep] -> [head_dim, N,
//   n_rep] Compute attention and store to output
//
// MEMORY ALIGNMENT SAFETY:
// ggml_view requires aligned memory for SIMD operations.
// head_dim * sizeof(float) must be >= GGML_MEM_ALIGN (typically 32 bytes).
// Common head_dim values: 64 (256 bytes), 128 (512 bytes) - both safe.
// =====================================================================

// Runtime alignment check (debug builds only)
#ifndef NDEBUG
                const size_t row_bytes = head_dim_kv * sizeof(float);
                if (row_bytes < GGML_MEM_ALIGN) {
                    std::cerr << "[DenseCore] WARNING: head_dim=" << head_dim_kv
                              << " may cause unaligned SIMD access (row_bytes=" << row_bytes
                              << ", GGML_MEM_ALIGN=" << GGML_MEM_ALIGN << ")" << std::endl;
                }
#endif

                // Reshape to 3D for permutation
                struct ggml_tensor* Q_r = ggml_reshape_3d(ctx_c, Qcur, head_dim_q, n_head, N);
                struct ggml_tensor* K_r =
                    ggml_reshape_3d(ctx_c, K, head_dim_kv, n_head_kv, n_total_tokens);
                struct ggml_tensor* V_r =
                    ggml_reshape_3d(ctx_c, V, head_dim_kv, n_head_kv, n_total_tokens);

                // Permute tensors to [head_dim, seq, heads] layout
                struct ggml_tensor* Q_perm =
                    ggml_permute(ctx_c, Q_r, 0, 2, 1, 3);  // [head_dim, N, n_head]
                struct ggml_tensor* K_perm =
                    ggml_permute(ctx_c, K_r, 0, 2, 1, 3);  // [head_dim, n_total, n_head_kv]
                struct ggml_tensor* V_perm =
                    ggml_permute(ctx_c, V_r, 0, 2, 1, 3);  // [head_dim, n_total, n_head_kv]

                // Make contiguous for view operations
                Q_perm = ggml_cont(ctx_c, Q_perm);
                K_perm = ggml_cont(ctx_c, K_perm);
                V_perm = ggml_cont(ctx_c, V_perm);

                // Allocate output tensor [head_dim, N, n_head]
                struct ggml_tensor* KQV_out =
                    ggml_new_tensor_3d(ctx_c, GGML_TYPE_F32, head_dim_q, N, n_head);

                // Process each KV head and its associated query group
                for (int h_kv = 0; h_kv < n_head_kv; ++h_kv) {
                    // Slice this KV head: [head_dim, n_total]
                    // CRITICAL FIX: Use tensor strides (nb[2]) instead of manual byte
                    // calculation. Manual calculation assumes no padding, which may be
                    // incorrect with GGML alignment requirements.
                    size_t k_offset = h_kv * K_perm->nb[2];
                    size_t v_offset = h_kv * V_perm->nb[2];
                    struct ggml_tensor* K_head = ggml_view_2d(
                        ctx_c, K_perm, head_dim_kv, n_total_tokens, K_perm->nb[1], k_offset);
                    struct ggml_tensor* V_head = ggml_view_2d(
                        ctx_c, V_perm, head_dim_kv, n_total_tokens, V_perm->nb[1], v_offset);

                    // Slice query group: [head_dim, N, n_rep]
                    // Query heads [h_kv * n_rep, (h_kv + 1) * n_rep) share this KV head
                    // Use proper strides for Q_perm (nb[2] = stride per head group)
                    size_t q_offset = h_kv * n_rep * Q_perm->nb[2];
                    struct ggml_tensor* Q_group =
                        ggml_view_3d(ctx_c, Q_perm, head_dim_q, N, n_rep, Q_perm->nb[1],
                                     Q_perm->nb[2], q_offset);

                    // Validate mul_mat dimension compatibility: cols(A) == rows(B)
                    // K_head: [head_dim_kv, n_total_tokens] (A)
                    // Q_group: [head_dim_q, N, n_rep]       (B)
                    // std::cerr << "[DEBUG GQA Loop] h_kv=" << h_kv << " K_head=["
                    //           << K_head->ne[0] << "," << K_head->ne[1] << "]"
                    //           << " Q_group=[" << Q_group->ne[0] << "," <<
                    //           Q_group->ne[1]
                    //           << "," << Q_group->ne[2] << "]" << std::endl;

                    if (K_head->ne[0] != Q_group->ne[0]) {
                        std::cerr << "[DenseCore] ERROR: mul_mat dimension mismatch in "
                                     "GQA loop! "
                                  << std::endl;
                    }

                    // Compute attention for this group:
                    // KQ = Q_group @ K_head^T -> [n_total, N, n_rep]
                    struct ggml_tensor* KQ = ggml_mul_mat(ctx_c, K_head, Q_group);
                    KQ = ggml_scale(ctx_c, KQ, scale);

                    // Apply causal mask (prefill only)
                    if (N > 1) {
                        KQ = ggml_diag_mask_inf(ctx_c, KQ, n_past_val);
                    }

                    // Softmax
                    KQ = ggml_soft_max(ctx_c, KQ);

                    // V_head^T @ KQ -> [head_dim, N, n_rep]
                    struct ggml_tensor* V_t = ggml_permute(ctx_c, V_head, 1, 0, 2, 3);
                    V_t = ggml_cont(ctx_c, V_t);
                    struct ggml_tensor* Out_group = ggml_mul_mat(ctx_c, V_t, KQ);

                    // Copy output group to final tensor
                    // Output offset for this group - use proper tensor strides
                    size_t out_offset = h_kv * n_rep * KQV_out->nb[2];
                    struct ggml_tensor* Out_view =
                        ggml_view_3d(ctx_c, KQV_out, head_dim_q, N, n_rep, KQV_out->nb[1],
                                     KQV_out->nb[2], out_offset);
                    Out_view = ggml_cpy(ctx_c, Out_group, Out_view);
                    ggml_build_forward_expand(gf, Out_view);
                }

                // Permute to [head_dim, n_head, N] for output projection
                KQV = ggml_permute(ctx_c, KQV_out, 0, 2, 1, 3);

            } else {
                // =====================================================================
                // MHA PATH (n_head == n_head_kv): Standard attention
                // =====================================================================
                // Q: [head_dim, N, n_head]
                struct ggml_tensor* Q = ggml_permute(ctx_c, Qcur, 0, 2, 1, 3);

                // K/V: [head_dim, n_total, n_head]
                struct ggml_tensor* K_fa = ggml_permute(ctx_c, K, 0, 2, 1, 3);
                struct ggml_tensor* V_fa = ggml_permute(ctx_c, V, 0, 2, 1, 3);

                // Ensure contiguity
                Q = ggml_cont(ctx_c, Q);
                K_fa = ggml_cont(ctx_c, K_fa);
                V_fa = ggml_cont(ctx_c, V_fa);

                // Step 1: Q @ K^T -> [n_total, N, n_head]
                struct ggml_tensor* KQ = ggml_mul_mat(ctx_c, K_fa, Q);

                // Step 2: Scale
                KQ = ggml_scale(ctx_c, KQ, scale);

                // Step 3: Apply causal mask (only for prefill with N > 1)
                if (N > 1) {
                    KQ = ggml_diag_mask_inf(ctx_c, KQ, n_past_val);
                }

                // Step 4: Softmax
                KQ = ggml_soft_max(ctx_c, KQ);

                // Step 5: KQ @ V -> [head_dim, N, n_head]
                struct ggml_tensor* V_t =
                    ggml_permute(ctx_c, V_fa, 1, 0, 2, 3);  // [n_total, head_dim, n_head]
                V_t = ggml_cont(ctx_c, V_t);
                KQV = ggml_mul_mat(ctx_c, V_t, KQ);

                // Permute to [head_dim, n_head, N] for projection
                KQV = ggml_permute(ctx_c, KQV, 0, 2, 1, 3);
            }
        }

        // Must be contiguous before reshape
        struct ggml_tensor* KQV_merged = ggml_cont(ctx_c, KQV);

        cur = ggml_reshape_2d(ctx_c, KQV_merged, head_dim_q * n_head, N);

        // Output Projection (using smart dispatcher for Parallel GEMV)
        cur = smart_mul_mat(ctx_c, model->layers[il].wo, cur);
        if (model->layers[il].bo)
            cur = ggml_add(ctx_c, cur, model->layers[il].bo);

        // Residual Connection
        cur = ggml_add(ctx_c, cur, inpL);
        // Usually output projection expects n_embd input.
        // wo: [n_embd, n_embd] (or [n_embd, n_head*head_dim])
        // The standard transformer expects concatenation of all heads to be
        // n_embd. If n_head * head_dim_kv != n_embd, we have a mismatch. Qwen3
        // has n_head=16, head_dim_kv=128 => 2048 != 1024. This implies wo expects
        // 2048 input!

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
        struct ggml_tensor* inpFF = cur;
        if (cur->type != GGML_TYPE_F32) {
            fprintf(stderr, "CRITICAL: Layer %d input type is %d! Tensor name: %s\n", il, cur->type,
                    cur->name);
        }
        cur = ggml_rms_norm(ctx_c, cur, model->hparams.f_norm_rms_eps);
        cur = ggml_mul(ctx_c, cur, model->layers[il].ffn_norm);

        // SwiGLU FFN (using smart dispatcher for INT4 support)
        struct ggml_tensor* w1 = ggml_mul_mat(ctx_c, model->layers[il].w1, cur);
        struct ggml_tensor* w3 = ggml_mul_mat(ctx_c, model->layers[il].w3, cur);
        cur = ggml_mul(ctx_c, ggml_silu(ctx_c, w1), w3);
        cur = ggml_mul_mat(ctx_c, model->layers[il].w2, cur);

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
    // B->ne[1]] For tied embeddings: tok_embeddings is [n_embd, n_vocab] We
    // want [n_vocab, N] output. ggml_mul_mat(tok_emb, cur) gives [n_vocab, N] ✓
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

void InitGrammarConstraint(GrammarConstraint* grammar, const std::vector<std::string>& vocab) {
    if (!grammar)
        return;

    // Find token IDs for JSON special characters
    for (size_t i = 0; i < vocab.size(); i++) {
        const std::string& token = vocab[i];
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

void GrammarConstraint::UpdateState(const std::string& token_text) {
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
                state = brace_depth > 0 ? JSONState::EXPECT_KEY_OR_END : JSONState::EXPECT_VALUE;
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
            state = brace_depth > 0 ? JSONState::EXPECT_KEY_OR_END : JSONState::EXPECT_VALUE;
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

bool IsDigitToken(const std::string& token) {
    if (token.empty())
        return false;
    for (char c : token) {
        if (!isdigit(c) && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+' && c != ' ')
            return false;
    }
    return true;
}

bool IsWhitespaceToken(const std::string& token) {
    if (token.empty())
        return false;
    for (char c : token) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
            return false;
    }
    return true;
}

bool ContainsChar(const std::string& token, char ch) {
    return token.find(ch) != std::string::npos;
}

void ApplyGrammarMask(float* logits, int n_vocab, const GrammarConstraint* grammar,
                      const std::vector<std::string>& vocab) {
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
            const std::string& token = vocab[i];
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
            const std::string& token = vocab[i];
            if (ContainsChar(token, '"') || ContainsChar(token, '{') || ContainsChar(token, '[') ||
                IsDigitToken(token) || token.find("true") != std::string::npos ||
                token.find("false") != std::string::npos ||
                token.find("null") != std::string::npos) {
                allowed[i] = true;
            }
        }
        break;

    case JSONState::IN_NUMBER:
        for (int i = 0; i < n_vocab; i++) {
            const std::string& token = vocab[i];
            if (IsDigitToken(token) || ContainsChar(token, ',') || ContainsChar(token, '}') ||
                ContainsChar(token, ']')) {
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
            const std::string& token = vocab[i];
            if (ContainsChar(token, '"') || ContainsChar(token, '{') || ContainsChar(token, '[') ||
                ContainsChar(token, ']') || ContainsChar(token, ',') || IsDigitToken(token)) {
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

int SampleToken(struct ggml_tensor* logits, int idx, const SamplingParams& params) {
    float* logits_data = (float*)logits->data;
    int n_vocab = logits->ne[0];
    float* last_logits = logits_data + idx * n_vocab;

    std::vector<float> working_logits(last_logits, last_logits + n_vocab);

    if (params.grammar && params.vocab) {
        ApplyGrammarMask(working_logits.data(), n_vocab, params.grammar, *params.vocab);
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

        for (auto& kv : token_counts) {
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

    float max_logit = *std::max_element(working_logits.begin(), working_logits.end());
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
              [](const auto& a, const auto& b) { return a.first > b.first; });

    int k = std::min(params.top_k, n_vocab);
    if (k > 0 && k < n_vocab) {
        prob_idx.resize(k);
    }

    if (params.min_p > 0.0f && !prob_idx.empty()) {
        float max_prob = prob_idx[0].first;
        float threshold = params.min_p * max_prob;
        auto it = std::remove_if(prob_idx.begin(), prob_idx.end(),
                                 [threshold](const auto& p) { return p.first < threshold; });
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
        auto max_it = std::max_element(working_logits.begin(), working_logits.end());
        return std::distance(working_logits.begin(), max_it);
    }

    float total = 0.0f;
    for (const auto& p : prob_idx) {
        total += p.first;
    }
    for (auto& p : prob_idx) {
        p.first /= total;
    }

    float random_val = (float)rand() / RAND_MAX;
    float cumulative = 0.0f;
    for (const auto& p : prob_idx) {
        cumulative += p.first;
        if (random_val <= cumulative) {
            return p.second;
        }
    }

    return prob_idx[0].second;
}
