/**
 * @file cpu_backend.cpp
 * @brief CPU backend implementation using AVX2/AVX-512 kernels
 *
 * This file implements the CpuBackend class, wrapping the existing
 * SIMD-optimized kernels from simd_ops.h into the ComputeBackend interface.
 */

#include "cpu_backend.h"
#include "flash_attention.h"
#include "optimization_bridge.h"
#include "simd_platform.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

#if defined(_WIN32)
#include <malloc.h> // For _aligned_malloc/_aligned_free
#else
#include <cstdlib> // For aligned_alloc/free
#endif

namespace densecore {

// =============================================================================
// Constructor / Destructor
// =============================================================================

CpuBackend::CpuBackend() {
  // Detect SIMD level at runtime
  simd_level_ = simd::DetectSimdLevel();
  selected_isa_ = simd::SimdLevelName(simd_level_);

  std::cout << "[CpuBackend] Initialized with SIMD level: " << selected_isa_
            << std::endl;
}

CpuBackend::~CpuBackend() {
  // Nothing to clean up - all memory is externally managed
}

// =============================================================================
// Memory Management
// =============================================================================

void *CpuBackend::AllocateDevice(size_t size_bytes, size_t alignment) {
  if (size_bytes == 0) {
    return nullptr;
  }

  void *ptr = nullptr;

#if defined(_WIN32)
  ptr = _aligned_malloc(size_bytes, alignment);
#else
  // C11 aligned_alloc requires size to be multiple of alignment
  size_t aligned_size = ((size_bytes + alignment - 1) / alignment) * alignment;
  ptr = std::aligned_alloc(alignment, aligned_size);
#endif

  return ptr;
}

void CpuBackend::FreeDevice(void *ptr) {
  if (ptr == nullptr) {
    return;
  }

#if defined(_WIN32)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

void CpuBackend::CopyToDevice(void *dst, const void *src, size_t size_bytes) {
  if (dst && src && size_bytes > 0) {
    std::memcpy(dst, src, size_bytes);
  }
}

void CpuBackend::CopyFromDevice(void *dst, const void *src, size_t size_bytes) {
  if (dst && src && size_bytes > 0) {
    std::memcpy(dst, src, size_bytes);
  }
}

// =============================================================================
// Matrix Operations
// =============================================================================

void CpuBackend::MatMul(const Tensor &A, const Tensor &B, Tensor *C) {
  // Basic validation
  if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) {
    return;
  }

  // A: [M, K], B: [K, N], C: [M, N]
  const int M = static_cast<int>(A.shape[0]);
  const int K = static_cast<int>(A.shape[1]);
  const int N = static_cast<int>(B.shape[1]);

  // Use simd::MatMulTransB with transposed B for now
  // TODO: Add direct MatMul to simd_ops.h if needed
  const float *a_data = A.DataAs<float>();
  const float *b_data = B.DataAs<float>();
  float *c_data = C->DataAs<float>();

  // Simple nested loop for now - replace with optimized BLAS if available
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += a_data[m * K + k] * b_data[k * N + n];
      }
      c_data[m * N + n] = sum;
    }
  }
}

void CpuBackend::MatMulTransB(const Tensor &A, const Tensor &B, Tensor *C) {
  if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) {
    return;
  }

  // A: [M, K], B: [N, K] (stored row-major, will be transposed), C: [M, N]
  const int M = static_cast<int>(A.shape[0]);
  const int K = static_cast<int>(A.shape[1]);
  const int N = static_cast<int>(B.shape[0]);

  const float *a_data = A.DataAs<float>();
  const float *b_data = B.DataAs<float>();
  float *c_data = C->DataAs<float>();

  // Use simd::MatMulTransB which is optimized for this layout
  simd::MatMulTransB(c_data, a_data, b_data, M, N, K);
}

void CpuBackend::GemmInt4(const Tensor &A, const Tensor &W,
                          const Tensor &scales, const Tensor &zero_points,
                          Tensor *C, int group_size) {
  if (!A.IsValid() || !W.IsValid() || !C || !C->IsValid()) {
    return;
  }

  const int M = static_cast<int>(A.shape[0]);
  const int K = static_cast<int>(A.shape[1]);
  const int N = static_cast<int>(W.shape[0]);

  const float *a_data = A.DataAs<float>();
  const uint8_t *w_data = W.DataAs<uint8_t>();
  const float *scales_data = scales.DataAs<float>();
  const float *zeros_data = zero_points.DataAs<float>();
  float *c_data = C->DataAs<float>();

  // Use OpsRegistry for runtime dispatch to best available kernel
  if (OpsRegistry::IsInitialized()) {
    OpsRegistry::Instance().GemmInt4(c_data, a_data, w_data, scales_data,
                                     zeros_data, M, N, K, group_size);
  } else {
    // Fallback to simd dispatch
#if defined(__AVX512F__)
    simd::GemmInt4Fp32_AVX512(c_data, a_data, w_data, scales_data, zeros_data,
                              M, N, K, group_size);
#elif defined(__AVX2__)
    simd::GemmInt4Fp32_AVX2(c_data, a_data, w_data, scales_data, zeros_data, M,
                            N, K, group_size);
#else
    // Scalar fallback would go here
    std::cerr << "[CpuBackend] GemmInt4: No SIMD support, operation skipped"
              << std::endl;
#endif
  }
}

// =============================================================================
// Normalization Operations
// =============================================================================

void CpuBackend::RMSNorm(const Tensor &input, const Tensor &weight,
                         Tensor *output, float eps) {
  if (!input.IsValid() || !weight.IsValid() || !output || !output->IsValid()) {
    return;
  }

  const int64_t n_elements = input.NumElements();
  const int64_t hidden_dim = weight.shape[0];

  if (hidden_dim == 0) {
    return;
  }

  const int64_t n_tokens = n_elements / hidden_dim;
  const float *x = input.DataAs<float>();
  const float *w = weight.DataAs<float>();
  float *out = output->DataAs<float>();

  // Process each token
  for (int64_t t = 0; t < n_tokens; ++t) {
    const float *x_ptr = x + t * hidden_dim;
    float *out_ptr = out + t * hidden_dim;

    // Compute RMS
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < hidden_dim; ++i) {
      sum_sq += x_ptr[i] * x_ptr[i];
    }
    float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_dim) + eps);

    // Normalize and apply weight
    for (int64_t i = 0; i < hidden_dim; ++i) {
      out_ptr[i] = x_ptr[i] * rms * w[i];
    }
  }
}

void CpuBackend::AddRMSNorm(const Tensor &input, const Tensor &residual,
                            const Tensor &weight, Tensor *output, float eps) {
  if (!input.IsValid() || !residual.IsValid() || !weight.IsValid() || !output ||
      !output->IsValid()) {
    return;
  }

  const int64_t hidden_dim = weight.shape[0];
  const int64_t n_elements = input.NumElements();

  if (hidden_dim == 0) {
    return;
  }

  const int64_t n_tokens = n_elements / hidden_dim;
  const float *x = input.DataAs<float>();
  const float *res = residual.DataAs<float>();
  const float *w = weight.DataAs<float>();
  float *out = output->DataAs<float>();

  // Use simd::AddRMSNorm for each token (it handles one vector at a time)
  for (int64_t t = 0; t < n_tokens; ++t) {
    const float *x_ptr = x + t * hidden_dim;
    const float *res_ptr = res + t * hidden_dim;
    float *out_ptr = out + t * hidden_dim;

    simd::AddRMSNorm(out_ptr, x_ptr, res_ptr, w,
                     static_cast<size_t>(hidden_dim), eps);
  }
}

// =============================================================================
// Activation Operations
// =============================================================================

void CpuBackend::Softmax(const Tensor &input, Tensor *output) {
  if (!input.IsValid() || !output || !output->IsValid()) {
    return;
  }

  // Copy input to output first, then do in-place softmax
  const size_t size = input.SizeBytes();
  std::memcpy(output->data, input.data, size);

  SoftmaxInplace(output);
}

void CpuBackend::SoftmaxInplace(Tensor *data) {
  if (!data || !data->IsValid()) {
    return;
  }

  // Apply softmax along last dimension
  const int64_t n = data->shape[data->ndim - 1];
  int64_t batch_size = 1;
  for (int i = 0; i < data->ndim - 1; ++i) {
    batch_size *= data->shape[i];
  }

  float *ptr = data->DataAs<float>();
  for (int64_t b = 0; b < batch_size; ++b) {
    simd::SoftmaxF32(ptr + b * n, static_cast<size_t>(n));
  }
}

// =============================================================================
// Position Encoding Operations
// =============================================================================

void CpuBackend::RoPE(const Tensor &input, const Tensor &cos_sin,
                      const int *positions, Tensor *output, int rope_dim) {
  if (!input.IsValid() || !cos_sin.IsValid() || !positions || !output ||
      !output->IsValid()) {
    return;
  }

  // Determine dimensions from input shape
  // Supported layouts: [seq_len, head_dim] or [n_heads, seq_len, head_dim]
  int n_tokens, head_dim, n_heads;

  if (input.ndim == 2) {
    n_tokens = static_cast<int>(input.shape[0]);
    head_dim = static_cast<int>(input.shape[1]);
    n_heads = 1;
  } else if (input.ndim == 3) {
    n_heads = static_cast<int>(input.shape[0]);
    n_tokens = static_cast<int>(input.shape[1]);
    head_dim = static_cast<int>(input.shape[2]);
  } else {
    return; // Unsupported layout
  }

  if (rope_dim < 0) {
    rope_dim = head_dim;
  }

  const float *in = input.DataAs<float>();
  const float *cs = cos_sin.DataAs<float>();
  float *out = output->DataAs<float>();

  // Use simd::ApplyRoPE which handles the actual rotation
  simd::ApplyRoPE(out, in, cs, positions, n_tokens, head_dim, rope_dim);
}

// =============================================================================
// Fused Operations
// =============================================================================

void CpuBackend::FusedQKVProjection(const Tensor &input, const Tensor &wq,
                                    const Tensor &wk, const Tensor &wv,
                                    Tensor *q_out, Tensor *k_out,
                                    Tensor *v_out) {
  if (!input.IsValid() || !wq.IsValid() || !wk.IsValid() || !wv.IsValid() ||
      !q_out || !k_out || !v_out) {
    return;
  }

  const int n_tokens = static_cast<int>(input.shape[0]);
  const int n_embd = static_cast<int>(input.shape[1]);
  const int dim_q = static_cast<int>(wq.shape[0]);
  const int dim_k = static_cast<int>(wk.shape[0]);
  const int dim_v = static_cast<int>(wv.shape[0]);

  const float *x = input.DataAs<float>();
  const float *w_q = wq.DataAs<float>();
  const float *w_k = wk.DataAs<float>();
  const float *w_v = wv.DataAs<float>();
  float *q = q_out->DataAs<float>();
  float *k = k_out->DataAs<float>();
  float *v = v_out->DataAs<float>();

  // Process each token
  for (int t = 0; t < n_tokens; ++t) {
    const float *x_t = x + t * n_embd;
    float *q_t = q + t * dim_q;
    float *k_t = k + t * dim_k;
    float *v_t = v + t * dim_v;

    // Use simd::ComputeQKV for the actual computation
    // Single-threaded per token; parallelism handled at higher level
    simd::ComputeQKV(q_t, k_t, v_t, x_t, w_q, w_k, w_v, n_embd, dim_q, dim_k,
                     dim_v, 0, 1);
  }
}

void CpuBackend::FlashAttention(const Tensor &Q, const Tensor &K,
                                const Tensor &V, Tensor *output, float scale,
                                bool causal) {
  if (!Q.IsValid() || !K.IsValid() || !V.IsValid() || !output ||
      !output->IsValid()) {
    return;
  }

  // Expected layout: [batch, n_head, seq, head_dim]
  // For simplicity, we handle batch=1 and delegate to FlashAttentionBatched
  const int batch = static_cast<int>(Q.shape[0]);
  const int n_head = static_cast<int>(Q.shape[1]);
  const int seq_q = static_cast<int>(Q.shape[2]);
  const int head_dim = static_cast<int>(Q.shape[3]);
  const int seq_kv = static_cast<int>(K.shape[2]);

  const float *q_data = Q.DataAs<float>();
  const float *k_data = K.DataAs<float>();
  const float *v_data = V.DataAs<float>();
  float *o_data = output->DataAs<float>();

  FlashAttentionConfig config;
  config.scale = scale;
  config.causal = causal;

  // Use densecore::FlashAttentionBatched from flash_attention.h
  FlashAttentionBatched(q_data, k_data, v_data, o_data, batch, n_head, seq_q,
                        seq_kv, head_dim, config);
}

// =============================================================================
// Singleton Accessor
// =============================================================================

CpuBackend &GetCpuBackend() {
  static CpuBackend instance;
  return instance;
}

} // namespace densecore
