/**
 * @file compute_backend.h
 * @brief Abstract interface for compute backends
 *
 * Provides hardware-agnostic operations for LLM inference.
 * Each backend implements optimized kernels for its target hardware.
 *
 * Threading Model:
 * - All kernels are internally multi-threaded where beneficial
 * - Backends may provide thread hints through ThreadContext
 *
 * Memory Model:
 * - Backends manage their own memory pools
 * - `AllocateDevice` returns memory in the backend's address space
 * - CPU backend uses aligned malloc; ASIC backend uses SRAM pools
 */

#ifndef DENSECORE_COMPUTE_BACKEND_H
#define DENSECORE_COMPUTE_BACKEND_H

#include "tensor.h"
#include <cstddef>
#include <memory>
#include <string>

namespace densecore {

/**
 * @brief Thread context for parallel kernel execution
 *
 * Provides thread indexing for work partitioning within kernels.
 * Backends use this to distribute work across available cores.
 */
struct ThreadContext {
  int thread_id = 0;   ///< Current thread index (0-based)
  int num_threads = 1; ///< Total number of threads

  /// Default single-threaded context
  static ThreadContext Single() { return {0, 1}; }

  /// Create context for thread i of n
  static ThreadContext Make(int i, int n) { return {i, n}; }
};

/**
 * @brief Abstract interface for compute backends
 *
 * This is the core abstraction layer that decouples the inference graph
 * from hardware-specific implementations. Backends can be registered
 * at runtime and swapped without recompilation.
 *
 * Performance Guidelines:
 * - Minimize virtual function overhead by batching operations
 * - Use thread-local scratch buffers to avoid allocation overhead
 * - Prefer fused operations (e.g., AddRMSNorm) over separate calls
 */
class ComputeBackend {
public:
  virtual ~ComputeBackend() = default;

  // ===========================================================================
  // Backend Identification
  // ===========================================================================

  /**
   * @brief Get backend name (e.g., "CPU-AVX512", "ASIC-v1")
   */
  virtual const char *Name() const = 0;

  /**
   * @brief Get device type
   */
  virtual DeviceType Device() const = 0;

  // ===========================================================================
  // Memory Management
  // ===========================================================================

  /**
   * @brief Allocate device memory
   *
   * For CPU backend, this returns aligned host memory.
   * For accelerator backends, this allocates in device address space.
   *
   * @param size_bytes Number of bytes to allocate
   * @param alignment Required alignment in bytes (default: 64 for cache line)
   * @return Pointer to allocated memory, nullptr on failure
   */
  virtual void *AllocateDevice(size_t size_bytes, size_t alignment = 64) = 0;

  /**
   * @brief Free device memory
   * @param ptr Pointer previously returned by AllocateDevice
   */
  virtual void FreeDevice(void *ptr) = 0;

  /**
   * @brief Copy data to device memory
   *
   * For CPU backend, this is a memcpy.
   * For accelerator backends, this may involve DMA transfer.
   *
   * @param dst Destination pointer (device memory)
   * @param src Source pointer (host memory)
   * @param size_bytes Number of bytes to copy
   */
  virtual void CopyToDevice(void *dst, const void *src, size_t size_bytes) = 0;

  /**
   * @brief Copy data from device memory
   * @param dst Destination pointer (host memory)
   * @param src Source pointer (device memory)
   * @param size_bytes Number of bytes to copy
   */
  virtual void CopyFromDevice(void *dst, const void *src,
                              size_t size_bytes) = 0;

  // ===========================================================================
  // Core Linear Algebra Operations
  // ===========================================================================

  /**
   * @brief Matrix multiplication: C = A @ B
   *
   * Standard matrix multiplication for dense matrices.
   * A: [M, K], B: [K, N], C: [M, N]
   *
   * @param A Input tensor [M, K]
   * @param B Input tensor [K, N]
   * @param C Output tensor [M, N]
   */
  virtual void MatMul(const Tensor &A, const Tensor &B, Tensor *C) = 0;

  /**
   * @brief Matrix multiplication with transposed B: C = A @ B^T
   *
   * Optimized path for weight matrices stored in row-major format.
   * A: [M, K], B: [N, K], C: [M, N]
   *
   * @param A Input tensor [M, K]
   * @param B Input tensor [N, K] (will be transposed internally)
   * @param C Output tensor [M, N]
   */
  virtual void MatMulTransB(const Tensor &A, const Tensor &B, Tensor *C) = 0;

  /**
   * @brief INT4 quantized GEMM: C = A @ dequant(W_int4)^T
   *
   * Dequantizes INT4 weights on-the-fly and computes GEMM.
   * Scales/zeros are applied per group: w_f32 = scale * (w_int4 - zero)
   *
   * @param A Activations [M, K], F32
   * @param W Quantized weights [N, K/2], INT4 packed (2 weights per byte)
   * @param scales Per-group scales [N, num_groups]
   * @param zero_points Per-group zero points [N, num_groups]
   * @param C Output [M, N], F32
   * @param group_size Quantization group size (K must be divisible)
   */
  virtual void GemmInt4(const Tensor &A, const Tensor &W, const Tensor &scales,
                        const Tensor &zero_points, Tensor *C,
                        int group_size) = 0;

  // ===========================================================================
  // Normalization Operations
  // ===========================================================================

  /**
   * @brief RMS Normalization: out = (x / rms(x)) * weight
   *
   * Root Mean Square Layer Normalization (used in LLaMA, Qwen, etc.)
   * rms(x) = sqrt(mean(x^2) + eps)
   *
   * @param input Input tensor [*, hidden_dim]
   * @param weight Normalization weights [hidden_dim]
   * @param output Output tensor [*, hidden_dim]
   * @param eps Epsilon for numerical stability (default: 1e-5)
   */
  virtual void RMSNorm(const Tensor &input, const Tensor &weight,
                       Tensor *output, float eps = 1e-5f) = 0;

  /**
   * @brief Fused Add + RMSNorm: out = rmsnorm(x + residual) * weight
   *
   * Combines residual addition and RMSNorm into a single kernel to reduce
   * memory bandwidth by loading/storing data once instead of twice.
   *
   * @param input Current hidden state [*, hidden_dim]
   * @param residual Residual to add [*, hidden_dim]
   * @param weight Normalization weights [hidden_dim]
   * @param output Output tensor [*, hidden_dim]
   * @param eps Epsilon for numerical stability
   */
  virtual void AddRMSNorm(const Tensor &input, const Tensor &residual,
                          const Tensor &weight, Tensor *output,
                          float eps = 1e-5f) = 0;

  // ===========================================================================
  // Activation Operations
  // ===========================================================================

  /**
   * @brief Softmax: out[i] = exp(x[i]) / sum(exp(x))
   *
   * Applied along the last dimension of the input tensor.
   *
   * @param input Input tensor (last dimension is softmax dimension)
   * @param output Output tensor (same shape as input)
   */
  virtual void Softmax(const Tensor &input, Tensor *output) = 0;

  /**
   * @brief In-place softmax
   * @param data Input/output tensor
   */
  virtual void SoftmaxInplace(Tensor *data) = 0;

  // ===========================================================================
  // Position Encoding Operations
  // ===========================================================================

  /**
   * @brief Rotary Positional Embedding (RoPE)
   *
   * Applies rotation to pairs of elements using pre-computed cos/sin values.
   * Standard for LLaMA, Mistral, Qwen, and other modern LLMs.
   *
   * RoPE formula for pair (x_{2d}, x_{2d+1}):
   *   x'_{2d}   = x_{2d} * cos(θ) - x_{2d+1} * sin(θ)
   *   x'_{2d+1} = x_{2d} * sin(θ) + x_{2d+1} * cos(θ)
   *
   * @param input Input tensor [seq_len, head_dim] or [n_heads, seq_len,
   * head_dim]
   * @param cos_sin Pre-computed [max_pos, head_dim] interleaved cos/sin
   * @param positions Position indices for each token [seq_len]
   * @param output Output tensor (same shape as input)
   * @param rope_dim Number of dimensions to rotate (default: head_dim)
   */
  virtual void RoPE(const Tensor &input, const Tensor &cos_sin,
                    const int *positions, Tensor *output,
                    int rope_dim = -1) = 0;

  // ===========================================================================
  // Fused Operations (Performance Critical)
  // ===========================================================================

  /**
   * @brief Fused Q/K/V projection
   *
   * Computes Q = x @ Wq, K = x @ Wk, V = x @ Wv in a single pass.
   * Reduces memory bandwidth by loading input x into registers once
   * and writing three outputs in sequence.
   *
   * @param input Input tensor [seq_len, hidden_dim]
   * @param wq Q weight [dim_q, hidden_dim]
   * @param wk K weight [dim_k, hidden_dim]
   * @param wv V weight [dim_v, hidden_dim]
   * @param q_out Output Q [seq_len, dim_q]
   * @param k_out Output K [seq_len, dim_k]
   * @param v_out Output V [seq_len, dim_v]
   */
  virtual void FusedQKVProjection(const Tensor &input, const Tensor &wq,
                                  const Tensor &wk, const Tensor &wv,
                                  Tensor *q_out, Tensor *k_out,
                                  Tensor *v_out) = 0;

  /**
   * @brief Flash Attention with GQA Support
   *
   * Memory-efficient attention implementation that tiles computation
   * to fit in L2 cache. Essential for long sequence support.
   *
   * Supports Grouped Query Attention (GQA) where n_head_kv < n_head.
   * For MHA models, n_head_kv == n_head (or pass -1 for auto-detect).
   *
   * @param Q Query [batch, n_head, seq_q, head_dim]
   * @param K Key [batch, n_head_kv, seq_kv, head_dim]
   * @param V Value [batch, n_head_kv, seq_kv, head_dim]
   * @param output Output [batch, n_head, seq_q, head_dim]
   * @param scale Attention scale factor (typically 1/sqrt(head_dim))
   * @param causal Whether to apply causal masking
   * @param n_head_kv Number of KV heads (-1 = same as n_head, MHA mode)
   */
  virtual void FlashAttention(const Tensor &Q, const Tensor &K, const Tensor &V,
                              Tensor *output, float scale, bool causal = true,
                              int n_head_kv = -1) = 0;

  // ===========================================================================
  // Synchronization
  // ===========================================================================

  /**
   * @brief Synchronize all pending operations
   *
   * For CPU backend, this is a no-op.
   * For async backends (GPU/ASIC), blocks until all queued operations complete.
   */
  virtual void Synchronize() = 0;
};

/// Factory function type for backend creation
using BackendFactory = std::unique_ptr<ComputeBackend> (*)();

} // namespace densecore

#endif // DENSECORE_COMPUTE_BACKEND_H
