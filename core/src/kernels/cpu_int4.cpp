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

namespace densecore {
namespace kernels {

// =============================================================================
// AVX-512 Implementation
// =============================================================================

#if defined(__AVX512F__)

void GemvInt4_AVX512(float *output, const float *input, const uint8_t *weights,
                     const float *scales, const float *zeros, int K, int N,
                     int group_size, int n_start, int n_end) {
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
      const float *a_ptr = input + k_offset;

      // Prefetch next group's activations
      if (g + 1 < num_full_groups) {
        _mm_prefetch(
            reinterpret_cast<const char *>(input + (g + 1) * group_size),
            _MM_HINT_T0);
      }

      // Process 32 elements at a time
      for (int k = 0; k < group_size; k += 32) {
        // Load activation vector
        __m512 a0 = _mm512_loadu_ps(a_ptr + k);
        __m512 a1 = _mm512_loadu_ps(a_ptr + k + 16);

// Macro for processing one output row
#define PROCESS_ROW_INT4(idx)                                                  \
  do {                                                                         \
    const int row = n + (idx);                                                 \
    const float scale = scales[row * num_full_groups + g];                     \
    const float zero = zeros[row * num_full_groups + g];                       \
    const __m512 vscale = _mm512_set1_ps(scale);                               \
    const __m512 vzero = _mm512_set1_ps(zero);                                 \
                                                                               \
    const uint8_t *w_ptr = weights + row * packed_K + packed_offset + k / 2;   \
    _mm_prefetch(reinterpret_cast<const char *>(w_ptr + 64), _MM_HINT_T0);     \
                                                                               \
    /* Load 16 bytes = 32 packed weights */                                    \
    __m128i packed =                                                           \
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_ptr));             \
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
        const float scale = (num_full_groups > 0)
                                ? scales[row * num_full_groups + last_group - 1]
                                : 1.0f;
        const float zero = (num_full_groups > 0)
                               ? zeros[row * num_full_groups + last_group - 1]
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
      const float *a_ptr = input + k_offset;
      const float scale = scales[n * num_full_groups + g];
      const float zero = zeros[n * num_full_groups + g];
      const __m512 vscale = _mm512_set1_ps(scale);
      const __m512 vzero = _mm512_set1_ps(zero);
      const uint8_t *w_ptr = weights + n * packed_K + g * (group_size / 2);

      for (int k = 0; k < group_size; k += 32) {
        __m512 a0 = _mm512_loadu_ps(a_ptr + k);
        __m512 a1 = _mm512_loadu_ps(a_ptr + k + 16);

        __m128i packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_ptr + k / 2));
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
          (num_full_groups > 0)
              ? scales[n * num_full_groups + num_full_groups - 1]
              : 1.0f;
      const float zero = (num_full_groups > 0)
                             ? zeros[n * num_full_groups + num_full_groups - 1]
                             : 0.0f;

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

void GemvInt4_AVX512(float *output, const float *input, const uint8_t *weights,
                     const float *scales, const float *zeros, int K, int N,
                     int group_size, int n_start, int n_end) {
  // Fallback to scalar
  GemvInt4_Scalar(output, input, weights, scales, zeros, K, N, group_size,
                  n_start, n_end);
}

#endif // __AVX512F__

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

void GemvInt4_AVX2(float *output, const float *input, const uint8_t *weights,
                   const float *scales, const float *zeros, int K, int N,
                   int group_size, int n_start, int n_end) {
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
      const float *a_ptr = input + k_offset;

      for (int k = 0; k < group_size; k += 16) {
        __m256 a0 = _mm256_loadu_ps(a_ptr + k);
        __m256 a1 = _mm256_loadu_ps(a_ptr + k + 8);

#define PROCESS_ROW_AVX2(idx)                                                  \
  do {                                                                         \
    const int row = n + (idx);                                                 \
    const float scale = scales[row * num_full_groups + g];                     \
    const float zero = zeros[row * num_full_groups + g];                       \
    const __m256 vscale = _mm256_set1_ps(scale);                               \
    const __m256 vzero = _mm256_set1_ps(zero);                                 \
                                                                               \
    const uint8_t *w_ptr = weights + row * packed_K + packed_offset + k / 2;   \
                                                                               \
    /* Load 8 bytes = 16 packed weights */                                     \
    __m128i packed_64 =                                                        \
        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(w_ptr));             \
    __m128i packed_16 = _mm_cvtepu8_epi16(packed_64);                          \
                                                                               \
    __m128i low = _mm_and_si128(packed_16, _mm_set1_epi16(0x0F));              \
    __m128i high = _mm_srli_epi16(packed_16, 4);                               \
    high = _mm_and_si128(high, _mm_set1_epi16(0x0F));                          \
                                                                               \
    low = _mm_slli_epi16(low, 12);                                             \
    low = _mm_srai_epi16(low, 12);                                             \
    high = _mm_slli_epi16(high, 12);                                           \
    high = _mm_srai_epi16(high, 12);                                           \
                                                                               \
    __m128i lo = _mm_unpacklo_epi16(low, high);                                \
    __m128i hi = _mm_unpackhi_epi16(low, high);                                \
                                                                               \
    __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo));                \
    __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi));                \
                                                                               \
    wf0 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf0, vzero));                    \
    wf1 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf1, vzero));                    \
                                                                               \
    acc##idx = _mm256_fmadd_ps(a0, wf0, acc##idx);                             \
    acc##idx = _mm256_fmadd_ps(a1, wf1, acc##idx);                             \
  } while (0)

        PROCESS_ROW_AVX2(0);
        PROCESS_ROW_AVX2(1);
        PROCESS_ROW_AVX2(2);
        PROCESS_ROW_AVX2(3);

#undef PROCESS_ROW_AVX2
      }
    }

    float sums[4] = {hsum_avx2(acc0), hsum_avx2(acc1), hsum_avx2(acc2),
                     hsum_avx2(acc3)};

    // Handle remainder with scalar
    if (remainder > 0) {
      for (int i = 0; i < 4; i++) {
        const int row = n + i;
        const float scale =
            (num_full_groups > 0)
                ? scales[row * num_full_groups + num_full_groups - 1]
                : 1.0f;
        const float zero =
            (num_full_groups > 0)
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
      const uint8_t *w_packed = weights + n * packed_K + g * (group_size / 2);

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
          (num_full_groups > 0)
              ? scales[n * num_full_groups + num_full_groups - 1]
              : 1.0f;
      const float zero = (num_full_groups > 0)
                             ? zeros[n * num_full_groups + num_full_groups - 1]
                             : 0.0f;

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

void GemvInt4_AVX2(float *output, const float *input, const uint8_t *weights,
                   const float *scales, const float *zeros, int K, int N,
                   int group_size, int n_start, int n_end) {
  GemvInt4_Scalar(output, input, weights, scales, zeros, K, N, group_size,
                  n_start, n_end);
}

#endif // __AVX2__

// =============================================================================
// Scalar Implementation
// =============================================================================

void GemvInt4_Scalar(float *output, const float *input, const uint8_t *weights,
                     const float *scales, const float *zeros, int K, int N,
                     int group_size, int n_start, int n_end) {
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
      const uint8_t *w_packed = weights + n * packed_K + g * (group_size / 2);

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
          (num_full_groups > 0)
              ? scales[n * num_full_groups + num_full_groups - 1]
              : 1.0f;
      const float zero = (num_full_groups > 0)
                             ? zeros[n * num_full_groups + num_full_groups - 1]
                             : 0.0f;

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
// Unified Entry Point
// =============================================================================

void GemvInt4(float *output, const float *input, const uint8_t *weights,
              const float *scales, const float *zeros, int K, int N,
              int group_size, int n_start, int n_end) {
#if defined(__AVX512F__)
  GemvInt4_AVX512(output, input, weights, scales, zeros, K, N, group_size,
                  n_start, n_end);
#elif defined(__AVX2__)
  GemvInt4_AVX2(output, input, weights, scales, zeros, K, N, group_size,
                n_start, n_end);
#else
  GemvInt4_Scalar(output, input, weights, scales, zeros, K, N, group_size,
                  n_start, n_end);
#endif
}

} // namespace kernels
} // namespace densecore
