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
void GemvInt4_AVX512(float *output, const float *input, const uint8_t *weights,
                     const float *scales, const float *zeros, int K, int N,
                     int group_size, int n_start, int n_end);

/**
 * @brief Decode-optimized INT4 GEMV using AVX2
 *
 * Same as AVX-512 version but with:
 * - 4-way unrolling along N dimension
 * - 16 weights dequantized per iteration
 * - Manual horizontal sum (no _mm256_reduce_add_ps)
 */
void GemvInt4_AVX2(float *output, const float *input, const uint8_t *weights,
                   const float *scales, const float *zeros, int K, int N,
                   int group_size, int n_start, int n_end);

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
void GemvInt4_NEON(float *output, const float *input, const uint8_t *weights,
                   const float *scales, const float *zeros, int K, int N,
                   int group_size, int n_start, int n_end);

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
void GemvInt4_NEON_FP16(float *output, const float *input,
                        const uint8_t *weights, const float *scales,
                        const float *zeros, int K, int N, int group_size,
                        int n_start, int n_end);

/**
 * @brief Scalar fallback for INT4 GEMV
 */
void GemvInt4_Scalar(float *output, const float *input, const uint8_t *weights,
                     const float *scales, const float *zeros, int K, int N,
                     int group_size, int n_start, int n_end);

/**
 * @brief Unified entry point with runtime dispatch
 *
 * Automatically selects AVX-512, AVX2, or scalar based on CPU capabilities.
 */
void GemvInt4(float *output, const float *input, const uint8_t *weights,
              const float *scales, const float *zeros, int K, int N,
              int group_size, int n_start, int n_end);

} // namespace kernels
} // namespace densecore

#endif // DENSECORE_KERNELS_CPU_INT4_H
