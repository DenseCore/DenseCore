/**
 * @file optimization_bridge.h
 * @brief Runtime SIMD kernel dispatch for CPU-optimal execution
 *
 * Provides a function pointer registry that gets populated at startup based
 * on detected CPU capabilities. This enables the same binary to run optimally
 * on different hardware (AVX512 vs AVX2 vs Scalar) without recompilation.
 *
 * Usage:
 *   // At engine startup
 *   densecore::OpsRegistry::Init();
 *
 *   // Use dispatched kernels
 *   densecore::Ops::RoPE(out, in, cos_sin, positions, n_tokens, head_dim,
 * rope_dim);
 */

#ifndef DENSECORE_OPTIMIZATION_BRIDGE_H
#define DENSECORE_OPTIMIZATION_BRIDGE_H

#include <cstdint>
#include <iostream>

namespace densecore {

// =============================================================================
// Function Pointer Type Definitions
// =============================================================================

/**
 * RoPE kernel signature
 * @param out Output tensor [n_tokens, head_dim]
 * @param in Input tensor [n_tokens, head_dim]
 * @param cos_sin Pre-computed [cos, sin] pairs [max_pos, head_dim]
 * @param positions Token positions array [n_tokens]
 * @param n_tokens Number of tokens
 * @param head_dim Head dimension
 * @param rope_dim Dimensions to apply RoPE (typically == head_dim)
 * @param ith Thread index for work partitioning
 * @param nth Total threads for work partitioning
 */
using RoPE_fn = void (*)(float* out, const float* in, const float* cos_sin, const int* positions,
                         int n_tokens, int head_dim, int rope_dim, int ith, int nth);

/**
 * INT4 GEMM kernel signature: C = A @ W^T (dequantized)
 * @param C Output [M, N]
 * @param A Input activations [M, K]
 * @param W_int4 Packed INT4 weights [N, K/2]
 * @param scales Per-group scales [N, num_groups]
 * @param zero_points Per-group zero points [N, num_groups]
 * @param M Batch dimension
 * @param N Output features
 * @param K Input features
 * @param group_size Quantization group size (K must be divisible)
 */
using GemmInt4_fn = void (*)(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                             const float* zero_points, int M, int N, int K, int group_size);

/**
 * Softmax kernel signature (in-place)
 * @param data Input/output array
 * @param n Length
 */
using Softmax_fn = void (*)(float* data, size_t n);

/**
 * Dot product kernel signature
 * @param a First vector
 * @param b Second vector
 * @param n Length
 * @return Dot product result
 */
using DotF32_fn = float (*)(const float* a, const float* b, size_t n);

// =============================================================================
// OpsRegistry: Singleton holding dispatched function pointers
// =============================================================================

/**
 * Central registry for runtime-dispatched SIMD kernels.
 *
 * Call Init() once at startup. After that, use the static Ops struct
 * to access the optimized kernels.
 */
class OpsRegistry {
public:
    // Kernel function pointers (populated by Init())
    RoPE_fn RoPE = nullptr;
    GemmInt4_fn GemmInt4 = nullptr;
    Softmax_fn Softmax = nullptr;
    DotF32_fn DotF32 = nullptr;

    // Which ISA was selected
    const char* selected_isa = "Unknown";

    /**
     * Initialize the registry based on detected CPU capabilities.
     * Must be called once before using any kernel.
     */
    static void Init();

    /**
     * Get the singleton instance.
     */
    static OpsRegistry& Instance() {
        static OpsRegistry instance;
        return instance;
    }

    /**
     * Check if initialization has been done.
     */
    static bool IsInitialized() { return Instance().RoPE != nullptr; }

private:
    OpsRegistry() = default;
    OpsRegistry(const OpsRegistry&) = delete;
    OpsRegistry& operator=(const OpsRegistry&) = delete;
};

// =============================================================================
// Ops: Convenience namespace for direct kernel access
// =============================================================================

/**
 * Convenience interface for accessing dispatched kernels.
 *
 * Usage:
 *   densecore::Ops::RoPE(out, in, cos_sin, positions, n, dim, rope_dim, 0, 1);
 */
struct Ops {
    static void RoPE(float* out, const float* in, const float* cos_sin, const int* positions,
                     int n_tokens, int head_dim, int rope_dim, int ith = 0, int nth = 1) {
        OpsRegistry::Instance().RoPE(out, in, cos_sin, positions, n_tokens, head_dim, rope_dim, ith,
                                     nth);
    }

    static void GemmInt4(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                         const float* zero_points, int M, int N, int K, int group_size) {
        OpsRegistry::Instance().GemmInt4(C, A, W_int4, scales, zero_points, M, N, K, group_size);
    }

    static void Softmax(float* data, size_t n) { OpsRegistry::Instance().Softmax(data, n); }

    static float DotF32(const float* a, const float* b, size_t n) {
        return OpsRegistry::Instance().DotF32(a, b, n);
    }
};

}  // namespace densecore

#endif  // DENSECORE_OPTIMIZATION_BRIDGE_H
