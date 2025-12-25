/**
 * @file cpu_int4.cpp
 * @brief Decode-optimized INT4 GEMV kernel implementation
 *
 * High-performance GEMV for decode phase (M=1, single token generation).
 * Features block-wise dequantization in registers and parallel N-dimension.
 */

#include "kernels/cpu_int4.h"

#include <cmath>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace densecore {
namespace kernels {

// =============================================================================
// AVX-512 Implementation
// =============================================================================

#if defined(__AVX512F__)

void GemvInt4_AVX512(float* output, const float* input, const uint8_t* weights, const float* scales,
                     const float* zeros, int K, int N, int group_size, int n_start, int n_end) {
    // Handle unaligned K: process full groups with SIMD, remainder with scalar
    const int num_full_groups = K / group_size;
    const int remainder = K % group_size;
    const int K_aligned = num_full_groups * group_size;
    const int packed_K = K / 2;

    // Process 8 output rows at a time for maximum ILP
    int n = n_start;
    for (; n + 8 <= n_end; n += 8) {
        // 8 accumulators for 8 output elements
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps();
        __m512 acc5 = _mm512_setzero_ps();
        __m512 acc6 = _mm512_setzero_ps();
        __m512 acc7 = _mm512_setzero_ps();

        // Loop over full quantization groups
        for (int g = 0; g < num_full_groups; g++) {
            const int k_offset = g * group_size;
            const int packed_offset = g * (group_size / 2);
            const float* a_ptr = input + k_offset;

            // Prefetch next group's activations
            if (g + 1 < num_full_groups) {
                _mm_prefetch(reinterpret_cast<const char*>(input + (g + 1) * group_size),
                             _MM_HINT_T0);
            }

            // Process 32 elements at a time
            for (int k = 0; k < group_size; k += 32) {
                // Load activation vector
                __m512 a0 = _mm512_loadu_ps(a_ptr + k);
                __m512 a1 = _mm512_loadu_ps(a_ptr + k + 16);

// Macro for processing one output row
#define PROCESS_ROW_INT4(idx)                                                      \
    do {                                                                           \
        const int row = n + (idx);                                                 \
        const float scale = scales[row * num_full_groups + g];                     \
        const float zero = zeros[row * num_full_groups + g];                       \
        const __m512 vscale = _mm512_set1_ps(scale);                               \
        const __m512 vzero = _mm512_set1_ps(zero);                                 \
                                                                                   \
        const uint8_t* w_ptr = weights + row * packed_K + packed_offset + k / 2;   \
        _mm_prefetch(reinterpret_cast<const char*>(w_ptr + 64), _MM_HINT_T0);      \
                                                                                   \
        /* Load 16 bytes = 32 packed weights */                                    \
        __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_ptr)); \
        __m256i packed_256 = _mm256_cvtepu8_epi16(packed);                         \
                                                                                   \
        /* Extract and sign-extend nibbles */                                      \
        __m256i low = _mm256_and_si256(packed_256, _mm256_set1_epi16(0x0F));       \
        __m256i high = _mm256_srli_epi16(packed_256, 4);                           \
        high = _mm256_and_si256(high, _mm256_set1_epi16(0x0F));                    \
                                                                                   \
        /* Sign extension via shift trick */                                       \
        low = _mm256_slli_epi16(low, 12);                                          \
        low = _mm256_srai_epi16(low, 12);                                          \
        high = _mm256_slli_epi16(high, 12);                                        \
        high = _mm256_srai_epi16(high, 12);                                        \
                                                                                   \
        /* Interleave to restore order */                                          \
        __m256i lo = _mm256_unpacklo_epi16(low, high);                             \
        __m256i hi = _mm256_unpackhi_epi16(low, high);                             \
        __m256i w16_0 = _mm256_permute2x128_si256(lo, hi, 0x20);                   \
        __m256i w16_1 = _mm256_permute2x128_si256(lo, hi, 0x31);                   \
                                                                                   \
        /* Convert to FP32 */                                                      \
        __m512 wf0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(w16_0));             \
        __m512 wf1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(w16_1));             \
                                                                                   \
        /* Dequantize: w = scale * (q - zero) */                                   \
        wf0 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf0, vzero));                    \
        wf1 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf1, vzero));                    \
                                                                                   \
        /* FMA accumulation */                                                     \
        acc##idx = _mm512_fmadd_ps(a0, wf0, acc##idx);                             \
        acc##idx = _mm512_fmadd_ps(a1, wf1, acc##idx);                             \
    } while (0)

                PROCESS_ROW_INT4(0);
                PROCESS_ROW_INT4(1);
                PROCESS_ROW_INT4(2);
                PROCESS_ROW_INT4(3);
                PROCESS_ROW_INT4(4);
                PROCESS_ROW_INT4(5);
                PROCESS_ROW_INT4(6);
                PROCESS_ROW_INT4(7);

#undef PROCESS_ROW_INT4
            }
        }

        // Horizontal reduction and store SIMD results
        float sums[8] = {_mm512_reduce_add_ps(acc0), _mm512_reduce_add_ps(acc1),
                         _mm512_reduce_add_ps(acc2), _mm512_reduce_add_ps(acc3),
                         _mm512_reduce_add_ps(acc4), _mm512_reduce_add_ps(acc5),
                         _mm512_reduce_add_ps(acc6), _mm512_reduce_add_ps(acc7)};

        // Handle remainder elements with scalar (if K % group_size != 0)
        if (remainder > 0) {
            const int last_group = num_full_groups;
            for (int i = 0; i < 8; i++) {
                const int row = n + i;
                // Use last full group's scale/zero for remainder (approximation)
                // In practice, remainder should form a partial group with its own scale
                const float scale =
                    (num_full_groups > 0) ? scales[row * num_full_groups + last_group - 1] : 1.0f;
                const float zero =
                    (num_full_groups > 0) ? zeros[row * num_full_groups + last_group - 1] : 0.0f;

                for (int k = K_aligned; k < K; k++) {
                    const int byte_idx = k / 2;
                    const int nibble_idx = k % 2;
                    uint8_t packed_byte = weights[row * packed_K + byte_idx];

                    int8_t q;
                    if (nibble_idx == 0) {
                        q = static_cast<int8_t>(packed_byte & 0x0F);
                    } else {
                        q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                    }
                    if (q & 0x08) {
                        q |= static_cast<int8_t>(0xF0);
                    }

                    float w_dequant = scale * (static_cast<float>(q) - zero);
                    sums[i] += input[k] * w_dequant;
                }
            }
        }

        output[n + 0] = sums[0];
        output[n + 1] = sums[1];
        output[n + 2] = sums[2];
        output[n + 3] = sums[3];
        output[n + 4] = sums[4];
        output[n + 5] = sums[5];
        output[n + 6] = sums[6];
        output[n + 7] = sums[7];
    }

    // Handle remaining rows (< 8)
    for (; n < n_end; n++) {
        __m512 acc = _mm512_setzero_ps();

        for (int g = 0; g < num_full_groups; g++) {
            const int k_offset = g * group_size;
            const float* a_ptr = input + k_offset;
            const float scale = scales[n * num_full_groups + g];
            const float zero = zeros[n * num_full_groups + g];
            const __m512 vscale = _mm512_set1_ps(scale);
            const __m512 vzero = _mm512_set1_ps(zero);
            const uint8_t* w_ptr = weights + n * packed_K + g * (group_size / 2);

            for (int k = 0; k < group_size; k += 32) {
                __m512 a0 = _mm512_loadu_ps(a_ptr + k);
                __m512 a1 = _mm512_loadu_ps(a_ptr + k + 16);

                __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_ptr + k / 2));
                __m256i packed_256 = _mm256_cvtepu8_epi16(packed);

                __m256i low = _mm256_and_si256(packed_256, _mm256_set1_epi16(0x0F));
                __m256i high = _mm256_srli_epi16(packed_256, 4);
                high = _mm256_and_si256(high, _mm256_set1_epi16(0x0F));

                low = _mm256_slli_epi16(low, 12);
                low = _mm256_srai_epi16(low, 12);
                high = _mm256_slli_epi16(high, 12);
                high = _mm256_srai_epi16(high, 12);

                __m256i lo = _mm256_unpacklo_epi16(low, high);
                __m256i hi = _mm256_unpackhi_epi16(low, high);
                __m256i w16_0 = _mm256_permute2x128_si256(lo, hi, 0x20);
                __m256i w16_1 = _mm256_permute2x128_si256(lo, hi, 0x31);

                __m512 wf0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(w16_0));
                __m512 wf1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(w16_1));

                wf0 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf0, vzero));
                wf1 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf1, vzero));

                acc = _mm512_fmadd_ps(a0, wf0, acc);
                acc = _mm512_fmadd_ps(a1, wf1, acc);
            }
        }

        float sum = _mm512_reduce_add_ps(acc);

        // Handle remainder with scalar
        if (remainder > 0) {
            const float scale =
                (num_full_groups > 0) ? scales[n * num_full_groups + num_full_groups - 1] : 1.0f;
            const float zero =
                (num_full_groups > 0) ? zeros[n * num_full_groups + num_full_groups - 1] : 0.0f;

            for (int k = K_aligned; k < K; k++) {
                const int byte_idx = k / 2;
                const int nibble_idx = k % 2;
                uint8_t packed_byte = weights[n * packed_K + byte_idx];

                int8_t q;
                if (nibble_idx == 0) {
                    q = static_cast<int8_t>(packed_byte & 0x0F);
                } else {
                    q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                }
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }

                float w_dequant = scale * (static_cast<float>(q) - zero);
                sum += input[k] * w_dequant;
            }
        }

        output[n] = sum;
    }
}

#else

void GemvInt4_AVX512(float* output, const float* input, const uint8_t* weights, const float* scales,
                     const float* zeros, int K, int N, int group_size, int n_start, int n_end) {
    // Fallback to scalar
    GemvInt4_Scalar(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
}

#endif  // __AVX512F__

// =============================================================================
// AVX512-VNNI Implementation (Ice Lake+, Sapphire Rapids, Zen4+)
// =============================================================================
// Uses vpdpbusd (_mm512_dpbusd_epi32) for 4x u8*s8 dot products per lane.
// Keeps computation in integer domain until final scale/zero correction.
//
// OPTIMIZATION: Input quantization hoisted outside N loop for O(K) vs O(N*K).
// SAFETY: Uses __attribute__((target(...))) instead of file-level flags.
// =============================================================================

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
#endif
void GemvInt4_AVX512_VNNI(float *output, const float *input,
                          const uint8_t *weights, const float *scales,
                          const float *zeros, int K, int N, int group_size,
                          int n_start, int n_end) {
#if defined(__x86_64__) || defined(_M_X64)
    const int num_full_groups = K / group_size;
    const int packed_K = K / 2;
    const float input_scale = 127.0f;

    // ==========================================================================
    // CRITICAL OPTIMIZATION: Hoist input quantization outside N loop
    // This reduces quantization from O(N*K) to O(K)
    // ==========================================================================

    // Allocate aligned buffer for quantized input (64-byte aligned for AVX-512)
    // Using heap allocation to avoid VLA (non-standard C++ extension)
    int8_t* quantized_input =
        static_cast<int8_t*>(_mm_malloc(static_cast<size_t>(K > 0 ? K : 1), 64));

    // Quantize entire input vector ONCE before the N loop
    for (int k = 0; k < K; k += 64) {
        const int remaining = K - k;
        const int count = remaining >= 64 ? 64 : remaining;

        if (count >= 64) {
            // AVX-512 vectorized quantization
            __m512 a0 = _mm512_loadu_ps(input + k + 0);
            __m512 a1 = _mm512_loadu_ps(input + k + 16);
            __m512 a2 = _mm512_loadu_ps(input + k + 32);
            __m512 a3 = _mm512_loadu_ps(input + k + 48);

            __m512 vscale_in = _mm512_set1_ps(input_scale);
            __m512i a0_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(a0, vscale_in));
            __m512i a1_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(a1, vscale_in));
            __m512i a2_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(a2, vscale_in));
            __m512i a3_i32 = _mm512_cvtps_epi32(_mm512_mul_ps(a3, vscale_in));

            // Pack to INT8 (saturating)
            __m512i a01_i16 = _mm512_packs_epi32(a0_i32, a1_i32);
            __m512i a23_i16 = _mm512_packs_epi32(a2_i32, a3_i32);
            __m512i a_s8 = _mm512_packs_epi16(a01_i16, a23_i16);

            _mm512_store_si512(reinterpret_cast<__m512i*>(quantized_input + k), a_s8);
        } else {
            // Scalar fallback for tail
            for (int i = 0; i < count; i++) {
                float scaled = input[k + i] * input_scale;
                int32_t clamped = static_cast<int32_t>(scaled);
                clamped = clamped > 127 ? 127 : (clamped < -127 ? -127 : clamped);
                quantized_input[k + i] = static_cast<int8_t>(clamped);
            }
        }
    }

    // ==========================================================================
    // Main GEMV loop: Use pre-quantized input for O(1) per-element access
    // ==========================================================================
    for (int n = n_start; n < n_end; n++) {
        float total_sum = 0.0f;

        for (int g = 0; g < num_full_groups; g++) {
            const int k_offset = g * group_size;
            const int packed_offset = g * (group_size / 2);
            const int8_t* a_ptr = quantized_input + k_offset;  // Pre-quantized!
            const uint8_t* w_ptr = weights + n * packed_K + packed_offset;
            const float scale = scales[n * num_full_groups + g];
            const float zero = zeros[n * num_full_groups + g];

            __m512i acc = _mm512_setzero_si512();
            __m512i isum = _mm512_setzero_si512();

            int k = 0;
            for (; k + 64 <= group_size; k += 64) {
                // Load 32 packed bytes = 64 weights
                __m256i packed =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_ptr + k / 2));

                // Unpack INT4 to INT8 (unsigned for dpbusd)
                __m256i lo_nibble = _mm256_and_si256(packed, _mm256_set1_epi8(0x0F));
                __m256i hi_nibble =
                    _mm256_and_si256(_mm256_srli_epi16(packed, 4), _mm256_set1_epi8(0x0F));
                __m256i w_lo = _mm256_unpacklo_epi8(lo_nibble, hi_nibble);
                __m256i w_hi = _mm256_unpackhi_epi8(lo_nibble, hi_nibble);
                __m512i w_u8 = _mm512_inserti64x4(_mm512_castsi256_si512(w_lo), w_hi, 1);

                // Input sum for zero-point correction
                // dpbusd: u8 * s8 -> s32. Use 1 (u8) * a_s8 (s8) = a_s8
                isum = _mm512_dpbusd_epi32(isum, _mm512_set1_epi8(1), a_s8);

                // Load pre-quantized input (already INT8!)
                __m512i a_s8 = _mm512_load_si512(reinterpret_cast<const __m512i*>(a_ptr + k));

                // VNNI dot product: acc += u8 * s8
                acc = _mm512_dpbusd_epi32(acc, w_u8, a_s8);
            }

            // Horizontal sum
            int32_t dot_result = _mm512_reduce_add_epi32(acc);
            int32_t input_sum = _mm512_reduce_add_epi32(isum);

            // Dequantize: scale * (dot/127 - (zero+8)*isum/127)
            // = (scale / 127) * (dot - (zero+8)*isum)
            float group_sum =
                (scale / input_scale) *
                (static_cast<float>(dot_result) - (zero + 8.0f) * static_cast<float>(input_sum));
            total_sum += group_sum;

            // Scalar tail (rare, only when group_size % 64 != 0)
            for (; k < group_size; k++) {
                const int byte_idx = k / 2;
                const int nibble_idx = k % 2;
                uint8_t packed_byte = w_ptr[byte_idx];
                int8_t q = (nibble_idx == 0) ? static_cast<int8_t>(packed_byte & 0x0F)
                                             : static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                if (q & 0x08)
                    q |= static_cast<int8_t>(0xF0);
                float w_dequant = scale * (static_cast<float>(q) - zero);
                total_sum += input[k_offset + k] * w_dequant;
            }
        }

        output[n] = total_sum;
    }

    _mm_free(quantized_input);  // Free aligned buffer
#else
    // Non-x86 fallback
    (void)N;  // Suppress unused parameter warning
    GemvInt4_Scalar(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
#endif
}

// =============================================================================
// AVX2 Implementation
// =============================================================================

#if defined(__AVX2__)

// Manual horizontal sum for AVX2
static inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void GemvInt4_AVX2(float* output, const float* input, const uint8_t* weights, const float* scales,
                   const float* zeros, int K, int N, int group_size, int n_start, int n_end) {
    // Handle unaligned K: process full groups with SIMD, remainder with scalar
    const int num_full_groups = K / group_size;
    const int remainder = K % group_size;
    const int K_aligned = num_full_groups * group_size;
    const int packed_K = K / 2;

    // Process 4 output rows at a time
    int n = n_start;
    for (; n + 4 <= n_end; n += 4) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int g = 0; g < num_full_groups; g++) {
            const int k_offset = g * group_size;
            const int packed_offset = g * (group_size / 2);
            const float* a_ptr = input + k_offset;

            for (int k = 0; k < group_size; k += 16) {
                __m256 a0 = _mm256_loadu_ps(a_ptr + k);
                __m256 a1 = _mm256_loadu_ps(a_ptr + k + 8);

#define PROCESS_ROW_AVX2(idx)                                                         \
    do {                                                                              \
        const int row = n + (idx);                                                    \
        const float scale = scales[row * num_full_groups + g];                        \
        const float zero = zeros[row * num_full_groups + g];                          \
        const __m256 vscale = _mm256_set1_ps(scale);                                  \
        const __m256 vzero = _mm256_set1_ps(zero);                                    \
                                                                                      \
        const uint8_t* w_ptr = weights + row * packed_K + packed_offset + k / 2;      \
                                                                                      \
        /* Load 8 bytes = 16 packed weights */                                        \
        __m128i packed_64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w_ptr)); \
        __m128i packed_16 = _mm_cvtepu8_epi16(packed_64);                             \
                                                                                      \
        __m128i low = _mm_and_si128(packed_16, _mm_set1_epi16(0x0F));                 \
        __m128i high = _mm_srli_epi16(packed_16, 4);                                  \
        high = _mm_and_si128(high, _mm_set1_epi16(0x0F));                             \
                                                                                      \
        low = _mm_slli_epi16(low, 12);                                                \
        low = _mm_srai_epi16(low, 12);                                                \
        high = _mm_slli_epi16(high, 12);                                              \
        high = _mm_srai_epi16(high, 12);                                              \
                                                                                      \
        __m128i lo = _mm_unpacklo_epi16(low, high);                                   \
        __m128i hi = _mm_unpackhi_epi16(low, high);                                   \
                                                                                      \
        __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo));                   \
        __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi));                   \
                                                                                      \
        wf0 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf0, vzero));                       \
        wf1 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf1, vzero));                       \
                                                                                      \
        acc##idx = _mm256_fmadd_ps(a0, wf0, acc##idx);                                \
        acc##idx = _mm256_fmadd_ps(a1, wf1, acc##idx);                                \
    } while (0)

                PROCESS_ROW_AVX2(0);
                PROCESS_ROW_AVX2(1);
                PROCESS_ROW_AVX2(2);
                PROCESS_ROW_AVX2(3);

#undef PROCESS_ROW_AVX2
            }
        }

        float sums[4] = {hsum_avx2(acc0), hsum_avx2(acc1), hsum_avx2(acc2), hsum_avx2(acc3)};

        // Handle remainder with scalar
        if (remainder > 0) {
            for (int i = 0; i < 4; i++) {
                const int row = n + i;
                const float scale = (num_full_groups > 0)
                                        ? scales[row * num_full_groups + num_full_groups - 1]
                                        : 1.0f;
                const float zero = (num_full_groups > 0)
                                       ? zeros[row * num_full_groups + num_full_groups - 1]
                                       : 0.0f;

                for (int k = K_aligned; k < K; k++) {
                    const int byte_idx = k / 2;
                    const int nibble_idx = k % 2;
                    uint8_t packed_byte = weights[row * packed_K + byte_idx];

                    int8_t q;
                    if (nibble_idx == 0) {
                        q = static_cast<int8_t>(packed_byte & 0x0F);
                    } else {
                        q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                    }
                    if (q & 0x08) {
                        q |= static_cast<int8_t>(0xF0);
                    }

                    float w_dequant = scale * (static_cast<float>(q) - zero);
                    sums[i] += input[k] * w_dequant;
                }
            }
        }

        output[n + 0] = sums[0];
        output[n + 1] = sums[1];
        output[n + 2] = sums[2];
        output[n + 3] = sums[3];
    }

    // Handle remaining rows with scalar
    for (; n < n_end; n++) {
        float sum = 0.0f;
        for (int g = 0; g < num_full_groups; g++) {
            const float scale = scales[n * num_full_groups + g];
            const float zero = zeros[n * num_full_groups + g];
            const int k_start = g * group_size;
            const uint8_t* w_packed = weights + n * packed_K + g * (group_size / 2);

            for (int k = 0; k < group_size; k++) {
                const int byte_idx = k / 2;
                const int nibble_idx = k % 2;
                uint8_t packed_byte = w_packed[byte_idx];

                int8_t q;
                if (nibble_idx == 0) {
                    q = static_cast<int8_t>(packed_byte & 0x0F);
                } else {
                    q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                }
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }

                float w_dequant = scale * (static_cast<float>(q) - zero);
                sum += input[k_start + k] * w_dequant;
            }
        }

        // Handle remainder
        if (remainder > 0) {
            const float scale =
                (num_full_groups > 0) ? scales[n * num_full_groups + num_full_groups - 1] : 1.0f;
            const float zero =
                (num_full_groups > 0) ? zeros[n * num_full_groups + num_full_groups - 1] : 0.0f;

            for (int k = K_aligned; k < K; k++) {
                const int byte_idx = k / 2;
                const int nibble_idx = k % 2;
                uint8_t packed_byte = weights[n * packed_K + byte_idx];

                int8_t q;
                if (nibble_idx == 0) {
                    q = static_cast<int8_t>(packed_byte & 0x0F);
                } else {
                    q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                }
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }

                float w_dequant = scale * (static_cast<float>(q) - zero);
                sum += input[k] * w_dequant;
            }
        }
        output[n] = sum;
    }
}

#else

void GemvInt4_AVX2(float* output, const float* input, const uint8_t* weights, const float* scales,
                   const float* zeros, int K, int N, int group_size, int n_start, int n_end) {
    GemvInt4_Scalar(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
}

#endif  // __AVX2__

// =============================================================================
// ARM NEON Implementation
// =============================================================================

#if defined(__aarch64__) || defined(__ARM_NEON)

// Helper: horizontal sum of float32x4
static inline float vaddvq_f32_compat(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    // ARMv7 fallback: pairwise add
    float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
#endif
}

// Helper: Dequantize int16x4_t to float32x4_t
// Encapsulates the common pattern: vmovl_s16 -> vcvtq_f32_s32 -> scale*(w-zero)
static inline float32x4_t dequant_s16x4_to_f32(int16x4_t w_s16, float32x4_t vscale,
                                               float32x4_t vzero) {
    int32x4_t w_s32 = vmovl_s16(w_s16);
    float32x4_t w_f32 = vcvtq_f32_s32(w_s32);
    return vmulq_f32(vscale, vsubq_f32(w_f32, vzero));
}

void GemvInt4_NEON(float* output, const float* input, const uint8_t* weights, const float* scales,
                   const float* zeros, int K, int N, int group_size, int n_start, int n_end) {
    const int num_full_groups = K / group_size;
    const int packed_K = K / 2;

    // Constants for nibble extraction
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
    const int8x16_t sign_bit = vdupq_n_s8(8);  // 0x08

    // Process one output row at a time (simpler, correct implementation)
    for (int n = n_start; n < n_end; n++) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        float32x4_t acc4 = vdupq_n_f32(0.0f);
        float32x4_t acc5 = vdupq_n_f32(0.0f);
        float32x4_t acc6 = vdupq_n_f32(0.0f);
        float32x4_t acc7 = vdupq_n_f32(0.0f);

        for (int g = 0; g < num_full_groups; g++) {
            const int k_offset = g * group_size;
            const int packed_offset = g * (group_size / 2);
            const float* a_ptr = input + k_offset;
            const uint8_t* w_ptr = weights + n * packed_K + packed_offset;
            const float scale = scales[n * num_full_groups + g];
            const float zero = zeros[n * num_full_groups + g];
            const float32x4_t vscale = vdupq_n_f32(scale);
            const float32x4_t vzero = vdupq_n_f32(zero);

            // Process 32 weights (16 bytes) at a time
            int k = 0;
            for (; k + 32 <= group_size; k += 32) {
                // Load 16 packed bytes = 32 nibbles = 32 weights
                uint8x16_t packed = vld1q_u8(w_ptr + k / 2);

                // Extract low nibbles (even indices: w0, w2, w4, ...)
                uint8x16_t w_low_u8 = vandq_u8(packed, mask_0f);
                // Extract high nibbles (odd indices: w1, w3, w5, ...)
                uint8x16_t w_high_u8 = vshrq_n_u8(packed, 4);

                // Interleave to get correct order: w0, w1, w2, w3, ...
                // vzip1: takes alternating elements from low halves
                // vzip2: takes alternating elements from high halves
                uint8x16_t w_ordered_0 = vzip1q_u8(w_low_u8, w_high_u8);  // Weights 0-15
                uint8x16_t w_ordered_1 = vzip2q_u8(w_low_u8, w_high_u8);  // Weights 16-31

                // Sign-extend INT4 to INT8: if bit 3 is set, set bits 7-4
                // For each nibble: if (val & 0x08) val |= 0xF0
                // Using NEON: compare with 8, then OR with 0xF0 if true
                int8x16_t w0_s8 = vreinterpretq_s8_u8(w_ordered_0);
                int8x16_t w1_s8 = vreinterpretq_s8_u8(w_ordered_1);

                // Sign extension: val >= 8 means negative in INT4
                // Subtract 16 if val >= 8 (equivalent to sign extension)
                uint8x16_t needs_sign_0 = vcgeq_s8(w0_s8, sign_bit);
                uint8x16_t needs_sign_1 = vcgeq_s8(w1_s8, sign_bit);
                int8x16_t offset = vdupq_n_s8(-16);
                w0_s8 = vaddq_s8(w0_s8, vandq_s8(vreinterpretq_s8_u8(needs_sign_0), offset));
                w1_s8 = vaddq_s8(w1_s8, vandq_s8(vreinterpretq_s8_u8(needs_sign_1), offset));

                // Widen INT8 to INT16, then to INT32, then to FP32
                // Process first 8 weights (w0_s8 low half)
                int16x8_t w0_lo_s16 = vmovl_s8(vget_low_s8(w0_s8));   // Weights 0-7
                int16x8_t w0_hi_s16 = vmovl_s8(vget_high_s8(w0_s8));  // Weights 8-15
                int16x8_t w1_lo_s16 = vmovl_s8(vget_low_s8(w1_s8));   // Weights 16-23
                int16x8_t w1_hi_s16 = vmovl_s8(vget_high_s8(w1_s8));  // Weights 24-31

                // Convert to float and dequantize using helper
                float32x4_t w_0_3_f32 =
                    dequant_s16x4_to_f32(vget_low_s16(w0_lo_s16), vscale, vzero);
                float32x4_t w_4_7_f32 =
                    dequant_s16x4_to_f32(vget_high_s16(w0_lo_s16), vscale, vzero);
                float32x4_t w_8_11_f32 =
                    dequant_s16x4_to_f32(vget_low_s16(w0_hi_s16), vscale, vzero);
                float32x4_t w_12_15_f32 =
                    dequant_s16x4_to_f32(vget_high_s16(w0_hi_s16), vscale, vzero);
                float32x4_t w_16_19_f32 =
                    dequant_s16x4_to_f32(vget_low_s16(w1_lo_s16), vscale, vzero);
                float32x4_t w_20_23_f32 =
                    dequant_s16x4_to_f32(vget_high_s16(w1_lo_s16), vscale, vzero);
                float32x4_t w_24_27_f32 =
                    dequant_s16x4_to_f32(vget_low_s16(w1_hi_s16), vscale, vzero);
                float32x4_t w_28_31_f32 =
                    dequant_s16x4_to_f32(vget_high_s16(w1_hi_s16), vscale, vzero);

                // Load input activations
                float32x4_t a_0_3 = vld1q_f32(a_ptr + k + 0);
                float32x4_t a_4_7 = vld1q_f32(a_ptr + k + 4);
                float32x4_t a_8_11 = vld1q_f32(a_ptr + k + 8);
                float32x4_t a_12_15 = vld1q_f32(a_ptr + k + 12);
                float32x4_t a_16_19 = vld1q_f32(a_ptr + k + 16);
                float32x4_t a_20_23 = vld1q_f32(a_ptr + k + 20);
                float32x4_t a_24_27 = vld1q_f32(a_ptr + k + 24);
                float32x4_t a_28_31 = vld1q_f32(a_ptr + k + 28);

                // FMA: acc += weight * activation
                acc0 = vmlaq_f32(acc0, w_0_3_f32, a_0_3);
                acc1 = vmlaq_f32(acc1, w_4_7_f32, a_4_7);
                acc2 = vmlaq_f32(acc2, w_8_11_f32, a_8_11);
                acc3 = vmlaq_f32(acc3, w_12_15_f32, a_12_15);
                acc4 = vmlaq_f32(acc4, w_16_19_f32, a_16_19);
                acc5 = vmlaq_f32(acc5, w_20_23_f32, a_20_23);
                acc6 = vmlaq_f32(acc6, w_24_27_f32, a_24_27);
                acc7 = vmlaq_f32(acc7, w_28_31_f32, a_28_31);
            }

            // Handle remaining elements in group (< 32) with scalar
            for (; k < group_size; k++) {
                const int byte_idx = k / 2;
                const int nibble_idx = k % 2;
                uint8_t packed_byte = w_ptr[byte_idx];

                int8_t q;
                if (nibble_idx == 0) {
                    q = static_cast<int8_t>(packed_byte & 0x0F);
                } else {
                    q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                }
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }

                float w_dequant = scale * (static_cast<float>(q) - zero);
                // Add to first accumulator (will be summed at the end)
                acc0 = vsetq_lane_f32(vgetq_lane_f32(acc0, 0) + input[k_offset + k] * w_dequant,
                                      acc0, 0);
            }
        }

        // Horizontal reduction of all accumulators
        float32x4_t sum_01 = vaddq_f32(acc0, acc1);
        float32x4_t sum_23 = vaddq_f32(acc2, acc3);
        float32x4_t sum_45 = vaddq_f32(acc4, acc5);
        float32x4_t sum_67 = vaddq_f32(acc6, acc7);
        float32x4_t sum_0123 = vaddq_f32(sum_01, sum_23);
        float32x4_t sum_4567 = vaddq_f32(sum_45, sum_67);
        float32x4_t sum_all = vaddq_f32(sum_0123, sum_4567);

        output[n] = vaddvq_f32_compat(sum_all);
    }
}

// =============================================================================
// ARM NEON DOTPROD Implementation (ARMv8.2+: Graviton 2+, Apple M-series)
// =============================================================================
// Uses vdotq_s32 instructions to perform dot products on INT8 vectors directly.
//
// OPTIMIZATION: Input quantization hoisted outside N loop for O(K) vs O(N*K).
// =============================================================================

#if defined(__ARM_FEATURE_DOTPROD)

void GemvInt4_NEON_DOTPROD(float* output, const float* input, const uint8_t* weights,
                           const float* scales, const float* zeros, int K, int N, int group_size,
                           int n_start, int n_end) {
    const int num_full_groups = K / group_size;
    const int packed_K = K / 2;
    const float input_scale = 127.0f;

    // Constants for nibble extraction
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
    const int8x16_t sign_bit = vdupq_n_s8(8);
    const int8x16_t offset = vdupq_n_s8(-16);

    // ==========================================================================
    // CRITICAL OPTIMIZATION: Hoist input quantization outside N loop
    // This reduces quantization from O(N*K) to O(K)
    // ==========================================================================

    // Allocate aligned buffer for quantized input (16-byte aligned for NEON)
    int8_t* quantized_input = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&quantized_input), 16,
                       static_cast<size_t>(K > 0 ? K : 1)) != 0) {
        quantized_input = nullptr;
    }
    if (!quantized_input) {
        GemvInt4_NEON(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
        return;
    }

    // Quantize entire input vector ONCE before the N loop using NEON
    const float32x4_t ascale = vdupq_n_f32(input_scale);
    for (int k = 0; k < K; k += 16) {
        const int remaining = K - k;

        if (remaining >= 16) {
            // Load 16 floats
            float32x4_t a0 = vld1q_f32(input + k + 0);
            float32x4_t a1 = vld1q_f32(input + k + 4);
            float32x4_t a2 = vld1q_f32(input + k + 8);
            float32x4_t a3 = vld1q_f32(input + k + 12);

            // Scale and convert to INT8 (saturating narrow)
            int32x4_t i0 = vcvtq_s32_f32(vmulq_f32(a0, ascale));
            int32x4_t i1 = vcvtq_s32_f32(vmulq_f32(a1, ascale));
            int32x4_t i2 = vcvtq_s32_f32(vmulq_f32(a2, ascale));
            int32x4_t i3 = vcvtq_s32_f32(vmulq_f32(a3, ascale));

            // Narrow: i32 -> i16 -> i8
            int16x4_t h0 = vmovn_s32(i0);
            int16x4_t h1 = vmovn_s32(i1);
            int16x4_t h2 = vmovn_s32(i2);
            int16x4_t h3 = vmovn_s32(i3);
            int16x8_t h01 = vcombine_s16(h0, h1);
            int16x8_t h23 = vcombine_s16(h2, h3);
            int8x8_t b01 = vmovn_s16(h01);
            int8x8_t b23 = vmovn_s16(h23);
            int8x16_t result = vcombine_s8(b01, b23);

            vst1q_s8(quantized_input + k, result);
        } else {
            // Scalar fallback for tail
            for (int i = 0; i < remaining; i++) {
                float scaled = input[k + i] * input_scale;
                int32_t clamped = static_cast<int32_t>(scaled);
                clamped = clamped > 127 ? 127 : (clamped < -127 ? -127 : clamped);
                quantized_input[k + i] = static_cast<int8_t>(clamped);
            }
        }
    }

    // ==========================================================================
    // Main GEMV loop: Use pre-quantized input for O(1) per-element access
    // ==========================================================================
    for (int n = n_start; n < n_end; n++) {
        float total_sum = 0.0f;

        for (int g = 0; g < num_full_groups; g++) {
            const int k_offset = g * group_size;
            const int packed_offset = g * (group_size / 2);
            const int8_t* a_ptr = quantized_input + k_offset;  // Pre-quantized!
            const uint8_t* w_ptr = weights + n * packed_K + packed_offset;
            const float scale = scales[n * num_full_groups + g];
            const float zero = zeros[n * num_full_groups + g];

            // Accumulators for dot products (INT32)
            int32x4_t acc0 = vdupq_n_s32(0);
            int32x4_t acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0);
            int32x4_t acc3 = vdupq_n_s32(0);

            // Track sum of inputs for zero-point adjustment
            int32x4_t isum0 = vdupq_n_s32(0);
            int32x4_t isum1 = vdupq_n_s32(0);
            int32x4_t isum2 = vdupq_n_s32(0);
            int32x4_t isum3 = vdupq_n_s32(0);

            int k = 0;
            for (; k + 64 <= group_size; k += 64) {
                // Load 32 packed bytes = 64 nibbles = 64 weights
                uint8x16_t packed0 = vld1q_u8(w_ptr + k / 2);
                uint8x16_t packed1 = vld1q_u8(w_ptr + k / 2 + 16);

                // Extract and interleave nibbles
                uint8x16_t w_low_0 = vandq_u8(packed0, mask_0f);
                uint8x16_t w_high_0 = vshrq_n_u8(packed0, 4);
                uint8x16_t w_low_1 = vandq_u8(packed1, mask_0f);
                uint8x16_t w_high_1 = vshrq_n_u8(packed1, 4);

                uint8x16_t w_ord_0 = vzip1q_u8(w_low_0, w_high_0);
                uint8x16_t w_ord_1 = vzip2q_u8(w_low_0, w_high_0);
                uint8x16_t w_ord_2 = vzip1q_u8(w_low_1, w_high_1);
                uint8x16_t w_ord_3 = vzip2q_u8(w_low_1, w_high_1);

                // Sign extend INT4 to INT8
                int8x16_t w0_s8 = vreinterpretq_s8_u8(w_ord_0);
                int8x16_t w1_s8 = vreinterpretq_s8_u8(w_ord_1);
                int8x16_t w2_s8 = vreinterpretq_s8_u8(w_ord_2);
                int8x16_t w3_s8 = vreinterpretq_s8_u8(w_ord_3);

                uint8x16_t ns0 = vcgeq_s8(w0_s8, sign_bit);
                uint8x16_t ns1 = vcgeq_s8(w1_s8, sign_bit);
                uint8x16_t ns2 = vcgeq_s8(w2_s8, sign_bit);
                uint8x16_t ns3 = vcgeq_s8(w3_s8, sign_bit);

                w0_s8 = vaddq_s8(w0_s8, vandq_s8(vreinterpretq_s8_u8(ns0), offset));
                w1_s8 = vaddq_s8(w1_s8, vandq_s8(vreinterpretq_s8_u8(ns1), offset));
                w2_s8 = vaddq_s8(w2_s8, vandq_s8(vreinterpretq_s8_u8(ns2), offset));
                w3_s8 = vaddq_s8(w3_s8, vandq_s8(vreinterpretq_s8_u8(ns3), offset));

                // Load pre-quantized input (already INT8!)
                int8x16_t a_s8_0 = vld1q_s8(a_ptr + k + 0);
                int8x16_t a_s8_1 = vld1q_s8(a_ptr + k + 16);
                int8x16_t a_s8_2 = vld1q_s8(a_ptr + k + 32);
                int8x16_t a_s8_3 = vld1q_s8(a_ptr + k + 48);

                // DOTPROD: acc += dot(w, a)
                acc0 = vdotq_s32(acc0, w0_s8, a_s8_0);
                acc1 = vdotq_s32(acc1, w1_s8, a_s8_1);
                acc2 = vdotq_s32(acc2, w2_s8, a_s8_2);
                acc3 = vdotq_s32(acc3, w3_s8, a_s8_3);

                // Accumulate weight sums for zero-point correction
                // Accumulate input sums for zero-point correction
                // a_s8 is signed, use ones (s8=1) -> dot(ones, a_s8) = sum(a_s8)
                const int8x16_t ones = vdupq_n_s8(1);
                isum0 = vdotq_s32(isum0, ones, a_s8_0);
                isum1 = vdotq_s32(isum1, ones, a_s8_1);
                isum2 = vdotq_s32(isum2, ones, a_s8_2);
                isum3 = vdotq_s32(isum3, ones, a_s8_3);
            }

            // Horizontal sum of accumulators
            int32x4_t acc_sum = vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3));
            int32_t dot_result = vaddvq_s32(acc_sum);

            int32x4_t isum_sum = vaddq_s32(vaddq_s32(isum0, isum1), vaddq_s32(isum2, isum3));
            int32_t input_sum = vaddvq_s32(isum_sum);

            // Dequantize: scale * (dot/127 - zero*isum/127)
            // = (scale / 127) * (dot - zero*isum)
            float group_sum = (scale / input_scale) * (static_cast<float>(dot_result) -
                                                       zero * static_cast<float>(input_sum));
            total_sum += group_sum;

            // Scalar tail
            for (; k < group_size; k++) {
                const int byte_idx = k / 2;
                const int nibble_idx = k % 2;
                uint8_t packed_byte = w_ptr[byte_idx];
                int8_t q = (nibble_idx == 0) ? static_cast<int8_t>(packed_byte & 0x0F)
                                             : static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                if (q & 0x08)
                    q |= static_cast<int8_t>(0xF0);
                float w_dequant = scale * (static_cast<float>(q) - zero);
                total_sum += input[k_offset + k] * w_dequant;
            }
        }

        output[n] = total_sum;
    }

    free(quantized_input);
}

#else

// Fallback when DOTPROD is not available
void GemvInt4_NEON_DOTPROD(float* output, const float* input, const uint8_t* weights,
                           const float* scales, const float* zeros, int K, int N, int group_size,
                           int n_start, int n_end) {
    GemvInt4_NEON(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
}

#endif  // __ARM_FEATURE_DOTPROD

// =============================================================================
// ARM NEON FP16 Implementation (for Graviton3+, Apple M3/M4, Snapdragon 8 Gen
// 3+)
// =============================================================================
// Uses FP16 compute for 2x throughput on modern ARM CPUs with
// __ARM_FEATURE_FP16_VECTOR_ARITHMETIC support.
// =============================================================================

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

void GemvInt4_NEON_FP16(float* output, const float* input, const uint8_t* weights,
                        const float* scales, const float* zeros, int K, int N, int group_size,
                        int n_start, int n_end) {
    const int num_full_groups = K / group_size;
    const int packed_K = K / 2;

    // Constants for nibble extraction
    const uint8x16_t mask_0f = vdupq_n_u8(0x0F);
    const int8x16_t sign_bit = vdupq_n_s8(8);
    const int8x16_t offset = vdupq_n_s8(-16);

    for (int n = n_start; n < n_end; n++) {
        // Use 4x float16x8_t accumulators
        float16x8_t acc0 = vdupq_n_f16(0.0f);
        float16x8_t acc1 = vdupq_n_f16(0.0f);
        float16x8_t acc2 = vdupq_n_f16(0.0f);
        float16x8_t acc3 = vdupq_n_f16(0.0f);

        for (int g = 0; g < num_full_groups; g++) {
            const int k_offset = g * group_size;
            const int packed_offset = g * (group_size / 2);
            const float* a_ptr = input + k_offset;
            const uint8_t* w_ptr = weights + n * packed_K + packed_offset;
            const float scale_f32 = scales[n * num_full_groups + g];
            const float zero_f32 = zeros[n * num_full_groups + g];

            // Convert scale/zero to FP16
            const float16_t scale_f16 = static_cast<float16_t>(scale_f32);
            const float16_t zero_f16 = static_cast<float16_t>(zero_f32);
            const float16x8_t vscale = vdupq_n_f16(scale_f16);
            const float16x8_t vzero = vdupq_n_f16(zero_f16);

            // Process 32 weights (16 bytes) at a time
            int k = 0;
            for (; k + 32 <= group_size; k += 32) {
                // Load 16 packed bytes = 32 nibbles = 32 weights
                uint8x16_t packed = vld1q_u8(w_ptr + k / 2);

                // Extract nibbles
                uint8x16_t w_low_u8 = vandq_u8(packed, mask_0f);
                uint8x16_t w_high_u8 = vshrq_n_u8(packed, 4);

                // Interleave to correct order using vzip
                uint8x16_t w_ordered_0 = vzip1q_u8(w_low_u8, w_high_u8);
                uint8x16_t w_ordered_1 = vzip2q_u8(w_low_u8, w_high_u8);

                // Sign extension
                int8x16_t w0_s8 = vreinterpretq_s8_u8(w_ordered_0);
                int8x16_t w1_s8 = vreinterpretq_s8_u8(w_ordered_1);
                uint8x16_t needs_sign_0 = vcgeq_s8(w0_s8, sign_bit);
                uint8x16_t needs_sign_1 = vcgeq_s8(w1_s8, sign_bit);
                w0_s8 = vaddq_s8(w0_s8, vandq_s8(vreinterpretq_s8_u8(needs_sign_0), offset));
                w1_s8 = vaddq_s8(w1_s8, vandq_s8(vreinterpretq_s8_u8(needs_sign_1), offset));

                // Widen to INT16
                int16x8_t w0_lo_s16 = vmovl_s8(vget_low_s8(w0_s8));
                int16x8_t w0_hi_s16 = vmovl_s8(vget_high_s8(w0_s8));
                int16x8_t w1_lo_s16 = vmovl_s8(vget_low_s8(w1_s8));
                int16x8_t w1_hi_s16 = vmovl_s8(vget_high_s8(w1_s8));

                // Convert to FP16 and dequantize: scale * (w - zero)
                float16x8_t w_0_7_f16 = vcvtq_f16_s16(w0_lo_s16);
                float16x8_t w_8_15_f16 = vcvtq_f16_s16(w0_hi_s16);
                float16x8_t w_16_23_f16 = vcvtq_f16_s16(w1_lo_s16);
                float16x8_t w_24_31_f16 = vcvtq_f16_s16(w1_hi_s16);

                w_0_7_f16 = vmulq_f16(vscale, vsubq_f16(w_0_7_f16, vzero));
                w_8_15_f16 = vmulq_f16(vscale, vsubq_f16(w_8_15_f16, vzero));
                w_16_23_f16 = vmulq_f16(vscale, vsubq_f16(w_16_23_f16, vzero));
                w_24_31_f16 = vmulq_f16(vscale, vsubq_f16(w_24_31_f16, vzero));

                // Load input activations (FP32) and convert to FP16
                float32x4_t a_0_3 = vld1q_f32(a_ptr + k + 0);
                float32x4_t a_4_7 = vld1q_f32(a_ptr + k + 4);
                float32x4_t a_8_11 = vld1q_f32(a_ptr + k + 8);
                float32x4_t a_12_15 = vld1q_f32(a_ptr + k + 12);
                float32x4_t a_16_19 = vld1q_f32(a_ptr + k + 16);
                float32x4_t a_20_23 = vld1q_f32(a_ptr + k + 20);
                float32x4_t a_24_27 = vld1q_f32(a_ptr + k + 24);
                float32x4_t a_28_31 = vld1q_f32(a_ptr + k + 28);

                // Combine pairs to float16x8
                float16x8_t a_0_7_f16 = vcombine_f16(vcvt_f16_f32(a_0_3), vcvt_f16_f32(a_4_7));
                float16x8_t a_8_15_f16 = vcombine_f16(vcvt_f16_f32(a_8_11), vcvt_f16_f32(a_12_15));
                float16x8_t a_16_23_f16 =
                    vcombine_f16(vcvt_f16_f32(a_16_19), vcvt_f16_f32(a_20_23));
                float16x8_t a_24_31_f16 =
                    vcombine_f16(vcvt_f16_f32(a_24_27), vcvt_f16_f32(a_28_31));

                // FMA: acc += weight * activation (using vfmaq_f16 for 2x throughput)
                acc0 = vfmaq_f16(acc0, w_0_7_f16, a_0_7_f16);
                acc1 = vfmaq_f16(acc1, w_8_15_f16, a_8_15_f16);
                acc2 = vfmaq_f16(acc2, w_16_23_f16, a_16_23_f16);
                acc3 = vfmaq_f16(acc3, w_24_31_f16, a_24_31_f16);
            }

            // Handle remaining elements with scalar
            for (; k < group_size; k++) {
                const int byte_idx = k / 2;
                const int nibble_idx = k % 2;
                uint8_t packed_byte = w_ptr[byte_idx];

                int8_t q;
                if (nibble_idx == 0) {
                    q = static_cast<int8_t>(packed_byte & 0x0F);
                } else {
                    q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                }
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }

                float w_dequant = scale_f32 * (static_cast<float>(q) - zero_f32);
                acc0 = vsetq_lane_f16(vgetq_lane_f16(acc0, 0) +
                                          static_cast<float16_t>(input[k_offset + k] * w_dequant),
                                      acc0, 0);
            }
        }

        // Horizontal reduction
        float16x8_t sum_01 = vaddq_f16(acc0, acc1);
        float16x8_t sum_23 = vaddq_f16(acc2, acc3);
        float16x8_t sum_all = vaddq_f16(sum_01, sum_23);

        float16x4_t sum_low = vadd_f16(vget_low_f16(sum_all), vget_high_f16(sum_all));
        sum_low = vpadd_f16(sum_low, sum_low);
        sum_low = vpadd_f16(sum_low, sum_low);

        output[n] = static_cast<float>(vget_lane_f16(sum_low, 0));
    }
}

#else

// Fallback to FP32 NEON when FP16 vector arithmetic is not available
void GemvInt4_NEON_FP16(float* output, const float* input, const uint8_t* weights,
                        const float* scales, const float* zeros, int K, int N, int group_size,
                        int n_start, int n_end) {
    GemvInt4_NEON(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
}

#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#else

void GemvInt4_NEON(float* output, const float* input, const uint8_t* weights, const float* scales,
                   const float* zeros, int K, int N, int group_size, int n_start, int n_end) {
    // Fallback to scalar on non-NEON platforms
    GemvInt4_Scalar(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
}

void GemvInt4_NEON_FP16(float* output, const float* input, const uint8_t* weights,
                        const float* scales, const float* zeros, int K, int N, int group_size,
                        int n_start, int n_end) {
    // Fallback to scalar on non-NEON platforms
    GemvInt4_Scalar(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
}

#endif  // __aarch64__ || __ARM_NEON

// =============================================================================
// Scalar Implementation
// =============================================================================

void GemvInt4_Scalar(float* output, const float* input, const uint8_t* weights, const float* scales,
                     const float* zeros, int K, int N, int group_size, int n_start, int n_end) {
    // Handle any K value (aligned or not)
    const int num_full_groups = K / group_size;
    const int remainder = K % group_size;
    const int K_aligned = num_full_groups * group_size;
    const int packed_K = K / 2;

    for (int n = n_start; n < n_end; n++) {
        float sum = 0.0f;

        // Process full groups
        for (int g = 0; g < num_full_groups; g++) {
            const float scale = scales[n * num_full_groups + g];
            const float zero = zeros[n * num_full_groups + g];
            const int k_start = g * group_size;
            const uint8_t* w_packed = weights + n * packed_K + g * (group_size / 2);

            for (int k = 0; k < group_size; k++) {
                const int byte_idx = k / 2;
                const int nibble_idx = k % 2;
                uint8_t packed_byte = w_packed[byte_idx];

                int8_t q;
                if (nibble_idx == 0) {
                    q = static_cast<int8_t>(packed_byte & 0x0F);
                } else {
                    q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                }

                // Sign extend from 4-bit
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }

                float w_dequant = scale * (static_cast<float>(q) - zero);
                sum += input[k_start + k] * w_dequant;
            }
        }

        // Handle remainder elements (if K % group_size != 0)
        if (remainder > 0) {
            // Use last group's scale/zero for remainder (best approximation)
            const float scale =
                (num_full_groups > 0) ? scales[n * num_full_groups + num_full_groups - 1] : 1.0f;
            const float zero =
                (num_full_groups > 0) ? zeros[n * num_full_groups + num_full_groups - 1] : 0.0f;

            for (int k = K_aligned; k < K; k++) {
                const int byte_idx = k / 2;
                const int nibble_idx = k % 2;
                uint8_t packed_byte = weights[n * packed_K + byte_idx];

                int8_t q;
                if (nibble_idx == 0) {
                    q = static_cast<int8_t>(packed_byte & 0x0F);
                } else {
                    q = static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                }
                if (q & 0x08) {
                    q |= static_cast<int8_t>(0xF0);
                }

                float w_dequant = scale * (static_cast<float>(q) - zero);
                sum += input[k] * w_dequant;
            }
        }

        output[n] = sum;
    }
}

// =============================================================================
// ARM SVE DotProd Implementation (AWS Graviton 3/4)
// =============================================================================
// Uses ARM Scalable Vector Extension with svdot_s32 for integer dot products.
// SVE provides 256-bit+ scalable vectors on Graviton 3/4 processors.
//
// OPTIMIZATION: Input quantization hoisted outside N loop for O(K) vs O(N*K).
// Uses svwhilelt_b8 predicates for natural tail handling without scalar
// fallback.
// =============================================================================

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

void GemvInt4_SVE_DotProd(float* output, const float* input, const uint8_t* weights,
                          const float* scales, const float* zeros, int K, int N, int group_size,
                          int n_start, int n_end) {
    const int num_full_groups = K / group_size;
    const int packed_K = K / 2;
    const float input_scale = 127.0f;

    // Get SVE vector length (varies by hardware: 256-bit on Graviton 3, etc.)
    const uint64_t vl = svcntb();  // bytes per vector

    // ==========================================================================
    // CRITICAL OPTIMIZATION: Hoist input quantization outside N loop
    // This reduces quantization from O(N*K) to O(K)
    // ==========================================================================

    // Allocate buffer for quantized input (aligned to vector length)
    int8_t* quantized_input = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&quantized_input), svcntb(),
                       static_cast<size_t>(K > 0 ? K : 1)) != 0) {
        quantized_input = nullptr;
    }
    if (!quantized_input) {
        GemvInt4_NEON_DOTPROD(output, input, weights, scales, zeros, K, N, group_size, n_start,
                              n_end);
        return;
    }

    // Quantize entire input vector ONCE before the N loop using SVE
    for (int k = 0; k < K;) {
        // Generate predicate for remaining elements
        svbool_t pg = svwhilelt_b32(static_cast<uint64_t>(k), static_cast<uint64_t>(K));

        // Load floats with predicate
        svfloat32_t a = svld1_f32(pg, input + k);

        // Scale and convert to INT32
        svfloat32_t scaled = svmul_f32_z(pg, a, svdup_f32(input_scale));
        svint32_t a_i32 = svcvt_s32_f32_z(pg, scaled);

        // Saturating narrow: i32 -> i16 -> i8
        svint16_t a_i16 = svqxtnb_s32(a_i32);
        svint8_t a_s8 = svqxtnb_s16(a_i16);

        // Store with predicate
        svbool_t pg8 = svwhilelt_b8(static_cast<uint64_t>(k), static_cast<uint64_t>(K));
        svst1_s8(pg8, quantized_input + k, a_s8);

        k += svcntw();  // Advance by number of 32-bit elements
    }

    // ==========================================================================
    // Main GEMV loop: Use pre-quantized input for O(1) per-element access
    // ==========================================================================
    for (int n = n_start; n < n_end; n++) {
        float total_sum = 0.0f;

        for (int g = 0; g < num_full_groups; g++) {
            const int k_offset = g * group_size;
            const int packed_offset = g * (group_size / 2);
            const int8_t* a_ptr = quantized_input + k_offset;  // Pre-quantized!
            const uint8_t* w_ptr = weights + n * packed_K + packed_offset;
            const float scale = scales[n * num_full_groups + g];
            const float zero = zeros[n * num_full_groups + g];

            // INT32 accumulator for dot product
            svint32_t acc = svdup_s32(0);
            svint32_t isum = svdup_s32(0);

            // Process elements with SVE using predicate for natural tail handling
            int k = 0;
            while (k < group_size) {
                // Generate predicate for remaining packed bytes
                const int remaining_packed = (group_size - k) / 2;
                svbool_t pg = svwhilelt_b8(
                    static_cast<uint64_t>(0),
                    static_cast<uint64_t>(remaining_packed < (int)vl ? remaining_packed : (int)vl));

                // Load packed bytes (predicated)
                svuint8_t packed = svld1_u8(pg, w_ptr + k / 2);

                // Unpack INT4 to INT8
                svuint8_t lo_nibble = svand_u8_z(pg, packed, svdup_u8(0x0F));
                svuint8_t hi_nibble = svlsr_n_u8_z(pg, packed, 4);

                // Interleave nibbles (lo0,hi0,lo1,hi1,...)
                svuint8_t w_interleaved = svzip1_u8(lo_nibble, hi_nibble);

                // Sign extend: if nibble >= 8, subtract 16
                svint8_t w_s8 = svreinterpret_s8_u8(w_interleaved);
                svbool_t needs_sign = svcmpge_s8(pg, w_s8, svdup_s8(8));
                w_s8 = svsub_s8_m(needs_sign, w_s8, svdup_s8(16));

                // Load pre-quantized input (already INT8!)
                svint8_t a_s8 = svld1_s8(pg, a_ptr + k);

                // SVE dot product: acc += w * a
                acc = svdot_s32(acc, w_s8, a_s8);

                // Track input sum for zero correction
                svint8_t ones = svdup_s8(1);
                isum = svdot_s32(isum, ones, a_s8);

                k += vl * 2;  // 2 weights per packed byte
            }

            // Horizontal sum of SVE vector
            int32_t dot_result = svaddv_s32(svptrue_b32(), acc);
            int32_t input_sum = svaddv_s32(svptrue_b32(), isum);

            // Dequantize: scale * (dot/127 - zero*isum/127)
            // = (scale / 127) * (dot - zero*isum)
            float group_sum = (scale / input_scale) * (static_cast<float>(dot_result) -
                                                       zero * static_cast<float>(input_sum));
            total_sum += group_sum;
        }

        output[n] = total_sum;
    }

    free(quantized_input);
}

#else

void GemvInt4_SVE_DotProd(float* output, const float* input, const uint8_t* weights,
                          const float* scales, const float* zeros, int K, int N, int group_size,
                          int n_start, int n_end) {
#if defined(__aarch64__) || defined(__ARM_NEON)
    // Fallback to NEON DOTPROD on ARM without SVE
    GemvInt4_NEON_DOTPROD(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
#else
    // Fallback to Scalar on non-ARM platforms
    GemvInt4_Scalar(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
#endif
}

#endif  // __ARM_FEATURE_SVE

// =============================================================================
// Runtime Feature Detection
// =============================================================================

#if defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif

static bool HasSVE() {
    static bool checked = false;
    static bool has_sve = false;
    if (!checked) {
        has_sve = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
        checked = true;
    }
    return has_sve;
}

static bool HasDotProd() {
    static bool checked = false;
    static bool has_dotprod = false;
    if (!checked) {
        has_dotprod = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
        checked = true;
    }
    return has_dotprod;
}
#endif

// =============================================================================
// Unified Entry Point with Runtime Dispatch
// =============================================================================

void GemvInt4(float* output, const float* input, const uint8_t* weights, const float* scales,
              const float* zeros, int K, int N, int group_size, int n_start, int n_end) {
#if defined(__x86_64__) || defined(_M_X64)
    // x86-64: Runtime dispatch using __builtin_cpu_supports
    // Check from highest to lowest capability

#if defined(__GNUC__) || defined(__clang__)
    // VNNI provides ~4x throughput for integer dot products
    static bool has_vnni = __builtin_cpu_supports("avx512vnni");
    if (has_vnni) {
        GemvInt4_AVX512_VNNI(output, input, weights, scales, zeros, K, N, group_size, n_start,
                             n_end);
        return;
    }

    static bool has_avx512f = __builtin_cpu_supports("avx512f");
    if (has_avx512f) {
        GemvInt4_AVX512(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
        return;
    }

    static bool has_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (has_avx2) {
        GemvInt4_AVX2(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
        return;
    }
#else
    // MSVC or unknown compiler: use compile-time detection
#if defined(__AVX512VNNI__)
    GemvInt4_AVX512_VNNI(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
    return;
#elif defined(__AVX512F__)
    GemvInt4_AVX512(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
    return;
#elif defined(__AVX2__)
    GemvInt4_AVX2(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
    return;
#endif
#endif

#elif defined(__aarch64__)
    // ARM64: Runtime dispatch using getauxval

#if defined(__linux__)
    // SVE provides scalable vectors (256-bit+ on Graviton 3/4)
    if (HasSVE()) {
        GemvInt4_SVE_DotProd(output, input, weights, scales, zeros, K, N, group_size, n_start,
                             n_end);
        return;
    }

    // DotProd provides vdotq_s32 for integer acceleration
    if (HasDotProd()) {
        GemvInt4_NEON_DOTPROD(output, input, weights, scales, zeros, K, N, group_size, n_start,
                              n_end);
        return;
    }
#elif defined(__APPLE__)
    // Apple Silicon: Always has DotProd, use FP16 for M3/M4
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    GemvInt4_NEON_FP16(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
    return;
#else
    GemvInt4_NEON_DOTPROD(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
    return;
#endif
#endif

    // Fallback to base NEON
    GemvInt4_NEON(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
    return;

#elif defined(__ARM_NEON)
    // 32-bit ARM with NEON
    GemvInt4_NEON(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
    return;
#endif

    // Ultimate fallback: scalar
    GemvInt4_Scalar(output, input, weights, scales, zeros, K, N, group_size, n_start, n_end);
}

}  // namespace kernels
}  // namespace densecore
