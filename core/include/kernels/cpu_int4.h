/**
 * @file cpu_int4.h
 * @brief Decode-optimized INT4 GEMV kernel for token generation
 *
 * Provides high-performance GEMV (Matrix-Vector multiplication) for INT4
 * quantized weights, specifically optimized for the decode phase (M=1).
 *
 * Features:
 * - AVX-512 and AVX2 implementations
 * - Parallel dispatch across output dimension (N)
 * - Block-wise dequantization in registers
 * - Zero allocation during execution
 */

#ifndef DENSECORE_KERNELS_CPU_INT4_H
#define DENSECORE_KERNELS_CPU_INT4_H

#include <cstdint>

namespace densecore {
namespace kernels {

/**
 * @brief Decode-optimized INT4 GEMV using AVX-512
 *
 * Computes: output[n] = sum_k(input[k] * dequant(W[n,k]))
 * for a subset of output rows [n_start, n_end).
 *
 * Optimizations:
 * - 8-way unrolling along N dimension for maximum ILP
 * - 32 weights dequantized per iteration (64 packed bytes)
 * - Aggressive prefetching
 * - Horizontal reduction with _mm512_reduce_add_ps
 *
 * @param output Output vector [N] (only writes to [n_start, n_end))
 * @param input Input vector [K]
 * @param weights Packed INT4 weights [N, K/2]
 * @param scales Per-group scales [N, num_groups]
 * @param zeros Per-group zero points [N, num_groups]
 * @param K Input dimension
 * @param N Output dimension
 * @param group_size Quantization group size
 * @param n_start Start index in N dimension (inclusive)
 * @param n_end End index in N dimension (exclusive)
 */
void GemvInt4_AVX512(float* output, const float* input, const uint8_t* weights, const float* scales,
                     const float* zeros, int K, int N, int group_size, int n_start, int n_end);

/**
 * @brief Decode-optimized INT4 GEMV using AVX2
 *
 * Same as AVX-512 version but with:
 * - 4-way unrolling along N dimension
 * - 16 weights dequantized per iteration
 * - Manual horizontal sum (no _mm256_reduce_add_ps)
 */
void GemvInt4_AVX2(float* output, const float* input, const uint8_t* weights, const float* scales,
                   const float* zeros, int K, int N, int group_size, int n_start, int n_end);

/**
 * @brief Decode-optimized INT4 GEMV using ARM NEON
 *
 * Same as AVX2 version but with:
 * - 4-way unrolling along N dimension
 * - 16 weights (8 packed bytes) per iteration
 * - vmlaq_f32 for FMA accumulation
 * - vaddvq_f32 for horizontal reduction
 *
 * Target platforms: Apple M-series, AWS Graviton, Qualcomm Snapdragon
 */
void GemvInt4_NEON(float* output, const float* input, const uint8_t* weights, const float* scales,
                   const float* zeros, int K, int N, int group_size, int n_start, int n_end);

/**
 * @brief FP16-optimized INT4 GEMV using ARM NEON
 *
 * Uses FP16 compute for 2x throughput on modern ARM CPUs:
 * - Apple M3/M4, Snapdragon 8 Gen 3, AWS Graviton3+
 * - vfmaq_f16 for 2x wider vector FMA
 * - vcvt for FP32<->FP16 conversion
 *
 * Requires: __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
 */
void GemvInt4_NEON_FP16(float* output, const float* input, const uint8_t* weights,
                        const float* scales, const float* zeros, int K, int N, int group_size,
                        int n_start, int n_end);

/**
 * @brief DOTPROD-accelerated INT4 GEMV using ARM NEON
 *
 * Uses vdotq_s32 instructions for accelerated dot products on INT8 vectors,
 * avoiding per-element INT4->FP32 conversion overhead:
 * - AWS Graviton 2+, Apple M-series, Qualcomm Snapdragon 845+
 * - Processes 16 INT8 values per vdotq_s32 instruction
 * - ~4x bandwidth reduction vs FP32 path
 *
 * Requires: __ARM_FEATURE_DOTPROD
 */
void GemvInt4_NEON_DOTPROD(float* output, const float* input, const uint8_t* weights,
                           const float* scales, const float* zeros, int K, int N, int group_size,
                           int n_start, int n_end);

/**
 * @brief AVX512-VNNI accelerated INT4 GEMV (Ice Lake+, Zen4+)
 *
 * Uses vpdpbusd (_mm512_dpbusd_epi32) for 4x u8*s8 dot products per lane.
 * Keeps computation in integer domain until final scale/zero correction.
 * ~4x throughput improvement vs FP32 conversion path.
 *
 * Requires: __AVX512VNNI__
 */
void GemvInt4_AVX512_VNNI(float* output, const float* input, const uint8_t* weights,
                          const float* scales, const float* zeros, int K, int N, int group_size,
                          int n_start, int n_end);

/**
 * @brief SVE DotProd accelerated INT4 GEMV (AWS Graviton 3/4)
 *
 * Uses svdot_s32 for scalable vector dot products on ARM SVE hardware.
 * Supports 256-bit+ vectors for maximum throughput on Graviton processors.
 *
 * Requires: __ARM_FEATURE_SVE + runtime HWCAP_SVE detection
 */
void GemvInt4_SVE_DotProd(float* output, const float* input, const uint8_t* weights,
                          const float* scales, const float* zeros, int K, int N, int group_size,
                          int n_start, int n_end);

/**
 * @brief Scalar fallback for INT4 GEMV
 */
void GemvInt4_Scalar(float* output, const float* input, const uint8_t* weights, const float* scales,
                     const float* zeros, int K, int N, int group_size, int n_start, int n_end);

/**
 * @brief Unified entry point with runtime dispatch
 *
 * Automatically selects best available kernel:
 * - AVX-512 (Intel Skylake-X+)
 * - AVX2 (Intel Haswell+, AMD Zen+)
 * - ARM NEON (Apple M-series, AWS Graviton)
 * - Scalar fallback
 */
void GemvInt4(float* output, const float* input, const uint8_t* weights, const float* scales,
              const float* zeros, int K, int N, int group_size, int n_start, int n_end);

}  // namespace kernels
}  // namespace densecore

#endif  // DENSECORE_KERNELS_CPU_INT4_H
