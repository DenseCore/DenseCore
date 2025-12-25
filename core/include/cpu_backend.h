/**
 * @file cpu_backend.h
 * @brief CPU backend implementation with UMA support for x86-64 and ARM64
 *
 * This backend provides an immediate-mode fallback for systems without
 * dedicated accelerators. All memory is inherently "unified" on CPU.
 *
 * **Features:**
 * - Runtime SIMD detection (AVX-512 → AVX2 → NEON → Scalar)
 * - 64-byte aligned memory allocation for cache efficiency
 * - Immediate-mode graph execution (records and replays ops)
 * - Hardware-accelerated quantization using SIMD intrinsics
 *
 * **UMA on CPU:**
 * Since all memory is accessible by the single processor, `AllocateUnified()`
 * is equivalent to `AllocateDevice()`. `SynchronizeMemory()` is a no-op due
 * to the strong memory model on x86 and appropriate barriers on ARM.
 *
 * @see compute_backend.h for the abstract interface
 * @see accelerator_traits.h for GenericCPU profile
 */

#ifndef DENSECORE_CPU_BACKEND_H
#define DENSECORE_CPU_BACKEND_H

#include "densecore/hal/compute_backend.h"

#include "simd_ops.h"
#include "tensor_view.h"

namespace densecore {

/**
 * @brief CPU backend implementation with UMA semantics
 *
 * This backend wraps the existing AVX2/AVX-512/NEON SIMD kernels from
 * simd_ops.h and provides them through the ComputeBackend interface.
 *
 * **Threading:**
 * - Uses internal thread pool for parallelization
 * - Thread count is determined by hardware concurrency or explicit config
 *
 * **Memory:**
 * - All allocations are 64-byte aligned for AVX-512 compatibility
 * - Uses platform-specific aligned allocation
 * - `AllocateUnified` == `AllocateDevice` (all CPU memory is unified)
 *
 * **Graph Capture:**
 * - Supports immediate-mode graph capture for API compatibility
 * - Records operation lambdas and replays them on `ExecuteGraph()`
 * - No compilation step (NPU-style optimization not available on CPU)
 */
class CpuBackend : public ComputeBackend {
public:
    CpuBackend();
    ~CpuBackend() override;

    // ===========================================================================
    // Backend Identification
    // ===========================================================================

    const char* Name() const override { return selected_isa_; }
    DeviceType Device() const override { return DeviceType::CPU; }

    /**
     * @brief Get CPU-specific hardware traits
     *
     * Returns GenericCPU profile:
     * - supports_unified_memory: true (all CPU memory is unified)
     * - supports_graph_execution: false (immediate mode)
     * - preferred_quantization: Q4_K (memory-bandwidth optimized)
     */
    AcceleratorTraits GetTraits() const override { return AcceleratorTraits::GenericCPU(); }

    // ===========================================================================
    // Unified Memory Management
    // ===========================================================================

    /**
     * @brief Allocate unified memory (same as device allocation on CPU)
     *
     * On CPU, all memory is inherently accessible by the single processor.
     * This method delegates to `AllocateDevice()`.
     *
     * @param size_bytes Number of bytes to allocate
     * @param alignment Alignment (default: 64 bytes for AVX-512)
     * @return Aligned memory pointer, nullptr on failure
     */
    void* AllocateUnified(size_t size_bytes, size_t alignment = 64) override {
        return AllocateDevice(size_bytes, alignment);
    }

    /**
     * @brief Synchronize memory (no-op on CPU)
     *
     * CPU has a strong memory model (x86) or uses appropriate barriers (ARM).
     * Memory writes are visible without explicit synchronization.
     */
    void SynchronizeMemory(void* ptr, size_t size_bytes, MemorySyncDirection direction) override {
        (void)ptr;
        (void)size_bytes;
        (void)direction;
        // No-op: CPU memory is always coherent
#if defined(__aarch64__) || defined(_M_ARM64)
        // ARM64: issue data memory barrier for completeness
        __asm__ __volatile__("dmb sy" ::: "memory");
#else
        // x86: compiler barrier is sufficient
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
    }

    void* AllocateDevice(size_t size_bytes, size_t alignment = 64) override;
    void FreeDevice(void* ptr) override;

    /**
     * @brief Copy to device (deprecated, just memcpy on CPU)
     */
    void CopyToDevice(void* dst, const void* src, size_t size_bytes) override;

    /**
     * @brief Copy from device (deprecated, just memcpy on CPU)
     */
    void CopyFromDevice(void* dst, const void* src, size_t size_bytes) override;

    // ===========================================================================
    // Graph Capture (Immediate Mode Fallback)
    // ===========================================================================

    /**
     * @brief Begin recording operations
     *
     * Creates an ImmediateModeGraph that stores operation callbacks.
     * Operations called after this will be recorded instead of executed.
     */
    void BeginCapture() override;

    /**
     * @brief End recording and return the graph
     *
     * @return ImmediateModeGraph containing recorded operation callbacks
     */
    std::unique_ptr<OperationGraph> EndCapture() override;

    /**
     * @brief Execute a recorded graph
     *
     * For CPU, this replays all recorded operations synchronously.
     *
     * @param graph The graph to execute (must be ImmediateModeGraph)
     */
    void ExecuteGraph(const OperationGraph& graph) override;

    // ===========================================================================
    // Quantization Operations
    // ===========================================================================

    /**
     * @brief Quantize tensor using SIMD-accelerated kernels
     *
     * Uses AVX2/AVX-512 (x86) or NEON (ARM) for fast quantization.
     * Computes optimal scale/zero-point if not provided.
     *
     * @param src Source FP32 tensor
     * @param dst Destination quantized tensor (pre-allocated)
     * @param type Target quantization type (INT8, Q4_K, etc.)
     */
    void Quantize(const Tensor& src, QuantizedTensorView* dst, QuantType type) override;

    /**
     * @brief Dequantize tensor back to FP32
     *
     * @param src Quantized source tensor
     * @param dst Destination FP32 tensor (pre-allocated)
     */
    void Dequantize(const QuantizedTensorView& src, Tensor* dst) override;

    // ===========================================================================
    // Core Operations
    // ===========================================================================

    void MatMul(const Tensor& A, const Tensor& B, Tensor* C) override;
    void MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) override;
    void GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales, const Tensor& zero_points,
                  Tensor* C, int group_size) override;
    void RMSNorm(const Tensor& input, const Tensor& weight, Tensor* output,
                 float eps = 1e-5f) override;
    void AddRMSNorm(const Tensor& input, const Tensor& residual, const Tensor& weight,
                    Tensor* output, float eps = 1e-5f) override;
    void Softmax(const Tensor& input, Tensor* output) override;
    void SoftmaxInplace(Tensor* data) override;
    void RoPE(const Tensor& input, const Tensor& cos_sin, const int* positions, Tensor* output,
              int rope_dim = -1) override;
    void FusedQKVProjection(const Tensor& input, const Tensor& wq, const Tensor& wk,
                            const Tensor& wv, Tensor* q_out, Tensor* k_out, Tensor* v_out) override;
    void FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output,
                        float scale, bool causal = true, int n_head_kv = -1) override;
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
    void MatMulAMX(const TensorView& A, const TensorView& B, TensorView& C);
    void MatMulSVE(const TensorView& A, const TensorView& B, TensorView& C);

private:
    const char* selected_isa_;    ///< Human-readable ISA name
    simd::SimdLevel simd_level_;  ///< Detected SIMD level
};

/**
 * @brief Get global CPU backend instance (singleton)
 *
 * Thread-safe initialization using C++11 magic statics.
 * The singleton is created on first use and destroyed at program exit.
 */
CpuBackend& GetCpuBackend();

}  // namespace densecore

#endif  // DENSECORE_CPU_BACKEND_H
