/**
 * @file cpu_backend.h
 * @brief CPU backend implementation using AVX2/AVX-512 kernels
 *
 * Features:
 * - Runtime SIMD detection (AVX512 -> AVX2 -> Scalar)
 * - 64-byte aligned memory allocation
 * - Cache-aware blocking for large operations
 * - Wraps existing simd_ops.h kernels
 */

#ifndef DENSECORE_CPU_BACKEND_H
#define DENSECORE_CPU_BACKEND_H

#include "compute_backend.h"
#include "simd_ops.h"
#include "tensor_view.h"

namespace densecore {

/**
 * @brief CPU backend implementation
 *
 * This backend wraps the existing AVX2/AVX-512 SIMD kernels from simd_ops.h
 * and provides them through the ComputeBackend interface.
 *
 * Threading:
 * - Uses OpenMP internally for parallelization
 * - Thread count is determined by OMP_NUM_THREADS or hardware concurrency
 *
 * Memory:
 * - All allocations are 64-byte aligned for AVX-512 compatibility
 * - Uses platform-specific aligned allocation (_aligned_malloc on Windows)
 */
class CpuBackend : public ComputeBackend {
public:
  CpuBackend();
  ~CpuBackend() override;

  // ===========================================================================
  // Backend Identification
  // ===========================================================================

  const char *Name() const override { return selected_isa_; }
  DeviceType Device() const override { return DeviceType::CPU; }

  // ===========================================================================
  // Memory Management
  // ===========================================================================

  void *AllocateDevice(size_t size_bytes, size_t alignment = 64) override;
  void FreeDevice(void *ptr) override;
  void CopyToDevice(void *dst, const void *src, size_t size_bytes) override;
  void CopyFromDevice(void *dst, const void *src, size_t size_bytes) override;

  // ===========================================================================
  // Core Operations
  // ===========================================================================

  void MatMul(const Tensor &A, const Tensor &B, Tensor *C) override;
  void MatMulTransB(const Tensor &A, const Tensor &B, Tensor *C) override;
  void GemmInt4(const Tensor &A, const Tensor &W, const Tensor &scales,
                const Tensor &zero_points, Tensor *C, int group_size) override;
  void RMSNorm(const Tensor &input, const Tensor &weight, Tensor *output,
               float eps = 1e-5f) override;
  void AddRMSNorm(const Tensor &input, const Tensor &residual,
                  const Tensor &weight, Tensor *output,
                  float eps = 1e-5f) override;
  void Softmax(const Tensor &input, Tensor *output) override;
  void SoftmaxInplace(Tensor *data) override;
  void RoPE(const Tensor &input, const Tensor &cos_sin, const int *positions,
            Tensor *output, int rope_dim = -1) override;
  void FusedQKVProjection(const Tensor &input, const Tensor &wq,
                          const Tensor &wk, const Tensor &wv, Tensor *q_out,
                          Tensor *k_out, Tensor *v_out) override;
  void FlashAttention(const Tensor &Q, const Tensor &K, const Tensor &V,
                      Tensor *output, float scale, bool causal = true,
                      int n_head_kv = -1) override;
  void Synchronize() override { /* No-op for CPU */ }

  // ===========================================================================
  // CPU-Specific Accessors
  // ===========================================================================

  /**
   * @brief Get detected SIMD level
   */
  simd::SimdLevel GetSimdLevel() const { return simd_level_; }

  // HPC Specialized Kernels
  // Using TensorView to leverage explicit byte strides
  void MatMulAMX(const TensorView &A, const TensorView &B, TensorView &C);
  void MatMulSVE(const TensorView &A, const TensorView &B, TensorView &C);

private:
  const char *selected_isa_;   ///< Human-readable ISA name
  simd::SimdLevel simd_level_; ///< Detected SIMD level
};

/**
 * @brief Get global CPU backend instance (singleton)
 *
 * Thread-safe initialization using C++11 magic statics.
 * The singleton is created on first use and destroyed at program exit.
 */
CpuBackend &GetCpuBackend();

} // namespace densecore

#endif // DENSECORE_CPU_BACKEND_H
