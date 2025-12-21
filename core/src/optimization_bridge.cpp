/**
 * @file optimization_bridge.cpp
 * @brief Runtime SIMD dispatch implementation
 *
 * Populates OpsRegistry with the best available kernel implementations
 * based on runtime CPU feature detection.
 */

#include "optimization_bridge.h"
#include "simd_ops.h"
#include <iostream>

namespace densecore {

// =============================================================================
// Scalar Fallback Implementations
// =============================================================================

/**
 * Scalar INT4 GEMM fallback (when AVX512/AVX2 not available at runtime)
 *
 * This is a standalone scalar implementation that's always available,
 * not guarded by __AVX512F__ or __AVX2__ preprocessor.
 */
static void GemmInt4Fp32_Scalar(float *C, const float *A, const uint8_t *W_int4,
                                const float *scales, const float *zero_points,
                                int M, int N, int K, int group_size) {
  if (K % group_size != 0)
    return;

  const int num_groups = K / group_size;

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float sum = 0.0f;

      for (int g = 0; g < num_groups; g++) {
        const float scale = scales[n * num_groups + g];
        const float zero = zero_points[n * num_groups + g];

        const int k_start = g * group_size;
        const uint8_t *w_packed = W_int4 + n * (K / 2) + g * (group_size / 2);

        for (int k = 0; k < group_size; k++) {
          // Unpack 4-bit weight
          const int byte_idx = k / 2;
          const int nibble_idx = k % 2;
          uint8_t packed_byte = w_packed[byte_idx];

          int8_t q;
          if (nibble_idx == 0) {
            q = static_cast<int8_t>(packed_byte & 0x0F);
          } else {
            q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
          }

          // Sign extend from 4-bit to 8-bit
          if (q & 0x08) {
            q |= static_cast<int8_t>(0xF0);
          }

          // Dequantize and accumulate
          float w_dequant = scale * (static_cast<float>(q) - zero);
          sum += A[m * K + k_start + k] * w_dequant;
        }
      }

      C[m * N + n] = sum;
    }
  }
}

// =============================================================================
// OpsRegistry Initialization
// =============================================================================

void OpsRegistry::Init() {
  auto &reg = Instance();

  // Detect CPU capabilities at runtime
  simd::SimdLevel level = simd::DetectSimdLevel();
  const char *level_name = simd::SimdLevelName(level);

  std::cout << "[OpsRegistry] Detected SIMD level: " << level_name << std::endl;

  // ---------------------------------------------------------------------
  // RoPE Dispatch
  // ---------------------------------------------------------------------
  // Dispatch chain: AVX512 -> AVX2 -> Scalar
  // NOTE: Runtime detection + compile-time guards for multi-ISA support

#if defined(__AVX512F__)
  // Build has AVX-512 support
  if (level >= simd::SimdLevel::AVX512) {
    reg.RoPE = simd::ApplyRoPE_AVX512;
    std::cout << "  [RoPE] -> AVX-512" << std::endl;
  } else if (level >= simd::SimdLevel::AVX2) {
    // WARN: ApplyRoPE_AVX2 causes Segfaults on some CPUs (e.g. i7-10870H).
    // Disabling it in favor of Scalar fallback which is safe and performant
    // enough.
    reg.RoPE = simd::ApplyRoPE_Scalar;
    std::cout
        << "  [RoPE] -> Scalar (AVX2 kernel disabled due to stability issues)"
        << std::endl;
  } else {
    reg.RoPE = simd::ApplyRoPE_Scalar;
    std::cout << "  [RoPE] -> Scalar (runtime: no AVX2)" << std::endl;
  }
#elif defined(__AVX2__)
  if (level >= simd::SimdLevel::AVX2) {
    reg.RoPE = simd::ApplyRoPE_Scalar;
    // Normalized to Scalar for stability (AVX2 kernel hangs on i7-10870H)
    std::cout
        << "  [RoPE] -> Scalar (AVX2 kernel disabled due to stability issues)"
        << std::endl;
  } else {
    reg.RoPE = simd::ApplyRoPE_Scalar;
    std::cout << "  [RoPE] -> Scalar (runtime: no AVX2)" << std::endl;
  }
#else
  // Build without AVX2/AVX512 support
  reg.RoPE = simd::ApplyRoPE_Scalar;
  std::cout << "  [RoPE] -> Scalar (build without AVX2/AVX-512)" << std::endl;
#endif

  // ---------------------------------------------------------------------
  // GemmInt4 Dispatch
  // ---------------------------------------------------------------------
  // Dispatch chain: AVX512 -> AVX2 -> Scalar

#if defined(__AVX512F__)
  // Build has AVX-512 support
  if (level >= simd::SimdLevel::AVX512) {
    reg.GemmInt4 = simd::GemmInt4Fp32_AVX512;
    std::cout << "  [GemmInt4] -> AVX-512" << std::endl;
  } else if (level >= simd::SimdLevel::AVX2) {
    // FIX: Use AVX2 kernel instead of Scalar fallback!
    reg.GemmInt4 = simd::GemmInt4Fp32_AVX2;
    std::cout << "  [GemmInt4] -> AVX2 (runtime: no AVX-512, build has AVX-512)"
              << std::endl;
  } else {
    reg.GemmInt4 = GemmInt4Fp32_Scalar;
    std::cout << "  [GemmInt4] -> Scalar (runtime: no AVX2)" << std::endl;
  }
#elif defined(__AVX2__)
  // Build has AVX2 support (no AVX-512)
  if (level >= simd::SimdLevel::AVX2) {
    reg.GemmInt4 = simd::GemmInt4Fp32_AVX2;
    std::cout << "  [GemmInt4] -> AVX2" << std::endl;
  } else {
    reg.GemmInt4 = GemmInt4Fp32_Scalar;
    std::cout << "  [GemmInt4] -> Scalar (runtime: no AVX2)" << std::endl;
  }
#else
  // Build without AVX2/AVX512 support
  reg.GemmInt4 = GemmInt4Fp32_Scalar;
  std::cout << "  [GemmInt4] -> Scalar (build without AVX2/AVX-512)"
            << std::endl;
#endif

  // ---------------------------------------------------------------------
  // Softmax Dispatch
  // ---------------------------------------------------------------------
  // Using SIMD softmax which has internal dispatch
  reg.Softmax = simd::SoftmaxF32;
  std::cout << "  [Softmax] -> SIMD (internal dispatch)" << std::endl;

  // ---------------------------------------------------------------------
  // DotF32 Dispatch
  // ---------------------------------------------------------------------
  // Using SIMD dot product which has internal dispatch
  reg.DotF32 = simd::DotF32;
  std::cout << "  [DotF32] -> SIMD (internal dispatch)" << std::endl;

  // Store selected ISA name
  reg.selected_isa = level_name;

  std::cout << "[OpsRegistry] Initialization complete. Using: " << level_name
            << std::endl;
}

} // namespace densecore
