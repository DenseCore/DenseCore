/**
 * @file simd_ops.h
 * @brief SIMD-optimized operations for CPU inference
 *
 * Supports:
 * - AVX-512 (Intel Skylake-X+)
 * - AVX2 (Intel Haswell+, AMD Zen+)
 * - AVX (Intel Sandy Bridge+)
 * - SSE4.1 (Intel Penryn+)
 * - ARM NEON (Apple Silicon, ARM64)
 *
 * Auto-detects best available SIMD and provides unified API.
 */

#ifndef DENSECORE_SIMD_OPS_H
#define DENSECORE_SIMD_OPS_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__linux__)
#include <pthread.h> // For pthread_setaffinity_np
#include <sched.h>   // For cpu_set_t
#include <unistd.h>  // For sysconf, _SC_NPROCESSORS_ONLN
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h> // For SYSTEM_INFO, GetSystemInfo
#endif

// Platform detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#define DENSECORE_X86
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define DENSECORE_ARM
#include <arm_neon.h>
#endif

// FP16 support from ggml
extern "C" {
#include "ggml.h"
}

namespace densecore {
namespace simd {

// =============================================================================
// Runtime SIMD Detection
// =============================================================================

enum class SimdLevel {
  NONE = 0,
  SSE41 = 1,
  AVX = 2,
  AVX2 = 3,
  AVX512 = 4,
  AMX = 5,  // Intel Advanced Matrix Extensions (Sapphire Rapids+)
  NEON = 10 // ARM NEON (separate numbering for clarity)
};

inline SimdLevel DetectSimdLevel() {
#ifdef DENSECORE_X86
  // Check CPUID for x86 - ordered from highest to lowest capability
#if defined(__AMX_TILE__)
  return SimdLevel::AMX;
#elif defined(__AVX512F__)
  return SimdLevel::AVX512;
#elif defined(__AVX2__)
  return SimdLevel::AVX2;
#elif defined(__AVX__)
  return SimdLevel::AVX;
#elif defined(__SSE4_1__)
  return SimdLevel::SSE41;
#else
  return SimdLevel::NONE;
#endif
#elif defined(DENSECORE_ARM)
  return SimdLevel::NEON;
#else
  return SimdLevel::NONE;
#endif
}

inline const char *SimdLevelName(SimdLevel level) {
  switch (level) {
  case SimdLevel::AMX:
    return "Intel AMX";
  case SimdLevel::AVX512:
    return "AVX-512";
  case SimdLevel::AVX2:
    return "AVX2";
  case SimdLevel::AVX:
    return "AVX";
  case SimdLevel::SSE41:
    return "SSE4.1";
  case SimdLevel::NEON:
    return "ARM NEON";
  default:
    return "Scalar";
  }
}

// =============================================================================
// Thread Affinity & NUMA Topology
// =============================================================================

/**
 * Pin the calling thread to a specific CPU core.
 * For multi-socket servers, this prevents costly remote memory access.
 *
 * @param core_id The CPU core ID to pin to (0-indexed)
 * @return true on success, false on failure
 */
inline bool PinThreadToCore(int core_id) {
#if defined(__linux__) && !defined(__ANDROID__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  pthread_t current_thread = pthread_self();
  int result =
      pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
  return result == 0;
#elif defined(_WIN32)
  // Windows: Use SetThreadAffinityMask
  if (core_id < 0 || core_id >= 64)
    return false; // Windows mask limit
  DWORD_PTR mask = 1ULL << core_id;
  DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), mask);
  return prev != 0;
#else
  (void)core_id;
  return false;
#endif
}

/**
 * Get the number of NUMA nodes in the system.
 * Returns 1 for single-socket or non-NUMA systems.
 */
inline int GetNumaNodeCount() {
#if defined(__linux__)
  // Parse /sys/devices/system/node/nodeN directories
  int count = 0;
  for (int i = 0; i < 256; ++i) {
    char path[64];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", i);
    if (access(path, F_OK) == 0) {
      ++count;
    } else {
      break; // Nodes are numbered contiguously
    }
  }
  return count > 0 ? count : 1;
#elif defined(_WIN32)
  ULONG highest_node = 0;
  if (GetNumaHighestNodeNumber(&highest_node)) {
    return static_cast<int>(highest_node) + 1;
  }
  return 1;
#else
  return 1;
#endif
}

/**
 * Get list of CPU core IDs belonging to a NUMA node.
 * @param node_id NUMA node ID (0-indexed)
 * @return Vector of core IDs, empty on failure or unsupported platform
 */
inline std::vector<int> GetCoresInNumaNode(int node_id) {
  std::vector<int> cores;

#if defined(__linux__)
  // Parse /sys/devices/system/node/nodeN/cpulist
  // Format examples: "0-7", "0-7,16-23", "0,1,2,3"
  char path[128];
  snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist",
           node_id);

  FILE *f = fopen(path, "r");
  if (!f)
    return cores;

  char buf[256];
  if (fgets(buf, sizeof(buf), f)) {
    // Remove trailing newline
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n')
      buf[len - 1] = '\0';

    // Parse comma-separated entries
    char *saveptr = nullptr;
    char *token = strtok_r(buf, ",", &saveptr);
    while (token) {
      // Check if it's a range (contains '-')
      char *dash = strchr(token, '-');
      if (dash) {
        int start = atoi(token);
        int end = atoi(dash + 1);
        for (int i = start; i <= end; ++i) {
          cores.push_back(i);
        }
      } else {
        cores.push_back(atoi(token));
      }
      token = strtok_r(nullptr, ",", &saveptr);
    }
  }
  fclose(f);

#elif defined(_WIN32)
  ULONGLONG node_mask = 0;
  if (GetNumaNodeProcessorMask(static_cast<UCHAR>(node_id), &node_mask)) {
    for (int i = 0; i < 64; ++i) {
      if (node_mask & (1ULL << i)) {
        cores.push_back(i);
      }
    }
  }
#else
  (void)node_id;
#endif

  return cores;
}

/**
 * Get the number of available CPU cores.
 */
inline int GetNumCores() {
#if defined(__linux__)
  return static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
#elif defined(_WIN32)
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return static_cast<int>(sysinfo.dwNumberOfProcessors);
#else
  return 1; // Fallback
#endif
}

// =============================================================================
// Memory Prefetch
// =============================================================================

/**
 * Prefetch memory for read (T0 = all cache levels)
 */
inline void Prefetch(const void *ptr) {
#ifdef DENSECORE_X86
  _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
#elif defined(DENSECORE_ARM)
  __builtin_prefetch(ptr, 0, 3); // read, high locality
#else
  (void)ptr;
#endif
}

/**
 * Prefetch a range of memory (cache-line aligned)
 */
inline void PrefetchRange(const void *ptr, size_t bytes) {
  constexpr size_t CACHE_LINE = 64;
  const char *p = reinterpret_cast<const char *>(ptr);
  for (size_t i = 0; i < bytes; i += CACHE_LINE) {
    Prefetch(p + i);
  }
}

/**
 * Prefetch for write (non-temporal if supported)
 */
inline void PrefetchWrite(void *ptr) {
#ifdef DENSECORE_X86
  _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0);
#elif defined(DENSECORE_ARM)
  __builtin_prefetch(ptr, 1, 3); // write, high locality
#else
  (void)ptr;
#endif
}

// =============================================================================
// SIMD Copy Operations
// =============================================================================

/**
 * Fast memory copy using SIMD (aligned or unaligned)
 */
inline void SimdCopy(void *dst, const void *src, size_t bytes) {
#if defined(__AVX512F__)
  // AVX-512: 64 bytes per iteration
  const size_t vec_size = 64;
  size_t i = 0;
  for (; i + vec_size <= bytes; i += vec_size) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(
        reinterpret_cast<const char *>(src) + i));
    _mm512_storeu_si512(
        reinterpret_cast<__m512i *>(reinterpret_cast<char *>(dst) + i), v);
  }
  // Handle remainder
  if (i < bytes) {
    memcpy(reinterpret_cast<char *>(dst) + i,
           reinterpret_cast<const char *>(src) + i, bytes - i);
  }
#elif defined(__AVX2__)
  // AVX2: 32 bytes per iteration
  const size_t vec_size = 32;
  size_t i = 0;
  for (; i + vec_size <= bytes; i += vec_size) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(
        reinterpret_cast<const char *>(src) + i));
    _mm256_storeu_si256(
        reinterpret_cast<__m256i *>(reinterpret_cast<char *>(dst) + i), v);
  }
  if (i < bytes) {
    memcpy(reinterpret_cast<char *>(dst) + i,
           reinterpret_cast<const char *>(src) + i, bytes - i);
  }
#elif defined(__SSE2__)
  // SSE2: 16 bytes per iteration
  const size_t vec_size = 16;
  size_t i = 0;
  for (; i + vec_size <= bytes; i += vec_size) {
    __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
        reinterpret_cast<const char *>(src) + i));
    _mm_storeu_si128(
        reinterpret_cast<__m128i *>(reinterpret_cast<char *>(dst) + i), v);
  }
  if (i < bytes) {
    memcpy(reinterpret_cast<char *>(dst) + i,
           reinterpret_cast<const char *>(src) + i, bytes - i);
  }
#elif defined(DENSECORE_ARM)
  // NEON: 16 bytes per iteration
  const size_t vec_size = 16;
  size_t i = 0;
  const uint8_t *s = reinterpret_cast<const uint8_t *>(src);
  uint8_t *d = reinterpret_cast<uint8_t *>(dst);
  for (; i + vec_size <= bytes; i += vec_size) {
    uint8x16_t v = vld1q_u8(s + i);
    vst1q_u8(d + i, v);
  }
  if (i < bytes) {
    memcpy(d + i, s + i, bytes - i);
  }
#else
  memcpy(dst, src, bytes);
#endif
}

// =============================================================================
// Float32 Operations
// =============================================================================

/**
 * Copy float32 array using SIMD
 */
inline void CopyF32(float *dst, const float *src, size_t n) {
  SimdCopy(dst, src, n * sizeof(float));
}

/**
 * Scale float32 array: dst[i] = src[i] * scale
 */
inline void ScaleF32(float *dst, const float *src, float scale, size_t n) {
#if defined(__AVX2__)
  __m256 vscale = _mm256_set1_ps(scale);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(src + i);
    v = _mm256_mul_ps(v, vscale);
    _mm256_storeu_ps(dst + i, v);
  }
  for (; i < n; i++) {
    dst[i] = src[i] * scale;
  }
#elif defined(DENSECORE_ARM)
  float32x4_t vscale = vdupq_n_f32(scale);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(src + i);
    v = vmulq_f32(v, vscale);
    vst1q_f32(dst + i, v);
  }
  for (; i < n; i++) {
    dst[i] = src[i] * scale;
  }
#else
  for (size_t i = 0; i < n; i++) {
    dst[i] = src[i] * scale;
  }
#endif
}

/**
 * Add float32 arrays: dst[i] = a[i] + b[i]
 */
inline void AddF32(float *dst, const float *a, const float *b, size_t n) {
#if defined(__AVX2__)
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_storeu_ps(dst + i, vc);
  }
  for (; i < n; i++) {
    dst[i] = a[i] + b[i];
  }
#elif defined(DENSECORE_ARM)
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    float32x4_t vc = vaddq_f32(va, vb);
    vst1q_f32(dst + i, vc);
  }
  for (; i < n; i++) {
    dst[i] = a[i] + b[i];
  }
#else
  for (size_t i = 0; i < n; i++) {
    dst[i] = a[i] + b[i];
  }
#endif
}

/**
 * Dot product of float32 arrays
 */
inline float DotF32(const float *a, const float *b, size_t n) {
#if defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum = _mm256_fmadd_ps(va, vb, sum);
  }
  // Horizontal sum
  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  float result = _mm_cvtss_f32(sum128);
  // Remainder
  for (; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
#elif defined(DENSECORE_ARM)
  float32x4_t sum = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    sum = vmlaq_f32(sum, va, vb);
  }
  float result = vaddvq_f32(sum);
  for (; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
#else
  float result = 0.0f;
  for (size_t i = 0; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
#endif
}

/**
 * Find max value in float32 array
 */
inline float MaxF32(const float *a, size_t n) {
  if (n == 0)
    return 0.0f;

#if defined(__AVX2__)
  __m256 vmax = _mm256_set1_ps(-1e30f);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(a + i);
    vmax = _mm256_max_ps(vmax, v);
  }
  // Reduce
  __m128 hi = _mm256_extractf128_ps(vmax, 1);
  __m128 lo = _mm256_castps256_ps128(vmax);
  __m128 max128 = _mm_max_ps(lo, hi);
  max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
  max128 = _mm_max_ss(max128, _mm_shuffle_ps(max128, max128, 1));
  float result = _mm_cvtss_f32(max128);
  for (; i < n; i++) {
    if (a[i] > result)
      result = a[i];
  }
  return result;
#elif defined(DENSECORE_ARM)
  float32x4_t vmax = vdupq_n_f32(-1e30f);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(a + i);
    vmax = vmaxq_f32(vmax, v);
  }
  float result = vmaxvq_f32(vmax);
  for (; i < n; i++) {
    if (a[i] > result)
      result = a[i];
  }
  return result;
#else
  float result = a[0];
  for (size_t i = 1; i < n; i++) {
    if (a[i] > result)
      result = a[i];
  }
  return result;
#endif
}

/**
 * Sum of float32 array
 */
inline float SumF32(const float *a, size_t n) {
#if defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(a + i);
    sum = _mm256_add_ps(sum, v);
  }
  // Reduce
  __m128 hi = _mm256_extractf128_ps(sum, 1);
  __m128 lo = _mm256_castps256_ps128(sum);
  __m128 sum128 = _mm_add_ps(lo, hi);
  sum128 = _mm_hadd_ps(sum128, sum128);
  sum128 = _mm_hadd_ps(sum128, sum128);
  float result = _mm_cvtss_f32(sum128);
  for (; i < n; i++) {
    result += a[i];
  }
  return result;
#elif defined(DENSECORE_ARM)
  float32x4_t sum = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(a + i);
    sum = vaddq_f32(sum, v);
  }
  float result = vaddvq_f32(sum);
  for (; i < n; i++) {
    result += a[i];
  }
  return result;
#else
  float result = 0.0f;
  for (size_t i = 0; i < n; i++) {
    result += a[i];
  }
  return result;
#endif
}

/**
 * Softmax in-place: a[i] = exp(a[i] - max) / sum(exp)
 */
inline void SoftmaxF32(float *a, size_t n) {
  float max_val = MaxF32(a, n);

#if defined(__AVX2__)
  // Note: Using fast exp approximation would be better here
  // For now, use scalar exp
  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    a[i] = expf(a[i] - max_val);
    sum += a[i];
  }
  float inv_sum = 1.0f / sum;
  ScaleF32(a, a, inv_sum, n);
#else
  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    a[i] = expf(a[i] - max_val);
    sum += a[i];
  }
  float inv_sum = 1.0f / sum;
  for (size_t i = 0; i < n; i++) {
    a[i] *= inv_sum;
  }
#endif
}

// =============================================================================
// FP16 Conversion (using ggml)
// =============================================================================

/**
 * Convert FP32 to FP16 using SIMD
 */
inline void ConvertF32ToF16(ggml_fp16_t *dst, const float *src, size_t n) {
  ggml_fp32_to_fp16_row(src, dst, n);
}

/**
 * Convert FP16 to FP32 using SIMD
 */
inline void ConvertF16ToF32(float *dst, const ggml_fp16_t *src, size_t n) {
  ggml_fp16_to_fp32_row(src, dst, n);
}

// =============================================================================
// Matrix Operations (used by Flash Attention)
// =============================================================================

/**
 * Matrix multiply: C = A @ B^T (for attention scores)
 * A: [M, K], B: [N, K] (row-major), C: [M, N]
 */
inline void MatMulTransB(float *C, const float *A, const float *B, int M, int N,
                         int K) {
  // Simple implementation - can be optimized further
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      C[m * N + n] = DotF32(A + m * K, B + n * K, K);
    }
  }
}

// =============================================================================
// Embedding Operations (for RAG/Semantic Search)
// =============================================================================

/**
 * L2 Normalize a vector in-place: v[i] = v[i] / ||v||
 * Essential for cosine similarity in embedding models
 */
inline void NormalizeL2(float *data, size_t n) {
  // Compute L2 norm using SIMD dot product
  float norm_sq = DotF32(data, data, n);
  if (norm_sq < 1e-12f)
    return; // Avoid division by zero

  float inv_norm = 1.0f / sqrtf(norm_sq);
  ScaleF32(data, data, inv_norm, n);
}

/**
 * Mean pooling over sequence dimension
 * Input: [seq_len, hidden_dim] -> Output: [hidden_dim]
 * Standard pooling for sentence-transformers models
 */
inline void MeanPool(const float *input, float *output, int seq_len,
                     int hidden_dim) {
  if (seq_len <= 0)
    return;

  // Initialize output to zero
  std::memset(output, 0, hidden_dim * sizeof(float));

#if defined(__AVX2__)
  // Process 8 floats at a time
  for (int d = 0; d < hidden_dim; d += 8) {
    int remaining = (d + 8 <= hidden_dim) ? 8 : hidden_dim - d;
    if (remaining == 8) {
      __m256 sum = _mm256_setzero_ps();
      for (int s = 0; s < seq_len; s++) {
        __m256 v = _mm256_loadu_ps(input + s * hidden_dim + d);
        sum = _mm256_add_ps(sum, v);
      }
      // Divide by seq_len
      __m256 div = _mm256_set1_ps(1.0f / seq_len);
      sum = _mm256_mul_ps(sum, div);
      _mm256_storeu_ps(output + d, sum);
    } else {
      // Handle remainder
      for (int dd = d; dd < hidden_dim; dd++) {
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
          sum += input[s * hidden_dim + dd];
        }
        output[dd] = sum / seq_len;
      }
      break;
    }
  }
#elif defined(DENSECORE_ARM)
  for (int d = 0; d < hidden_dim; d += 4) {
    int remaining = (d + 4 <= hidden_dim) ? 4 : hidden_dim - d;
    if (remaining == 4) {
      float32x4_t sum = vdupq_n_f32(0.0f);
      for (int s = 0; s < seq_len; s++) {
        float32x4_t v = vld1q_f32(input + s * hidden_dim + d);
        sum = vaddq_f32(sum, v);
      }
      float32x4_t div = vdupq_n_f32(1.0f / seq_len);
      sum = vmulq_f32(sum, div);
      vst1q_f32(output + d, sum);
    } else {
      for (int dd = d; dd < hidden_dim; dd++) {
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
          sum += input[s * hidden_dim + dd];
        }
        output[dd] = sum / seq_len;
      }
      break;
    }
  }
#else
  // Scalar fallback
  for (int s = 0; s < seq_len; s++) {
    for (int d = 0; d < hidden_dim; d++) {
      output[d] += input[s * hidden_dim + d];
    }
  }
  float inv_len = 1.0f / seq_len;
  for (int d = 0; d < hidden_dim; d++) {
    output[d] *= inv_len;
  }
#endif
}

/**
 * Mean pooling with attention mask
 * Only pools over non-masked positions (mask[i] = 1 means valid)
 */
inline void MeanPoolMasked(const float *input, const int *mask, float *output,
                           int seq_len, int hidden_dim) {
  std::memset(output, 0, hidden_dim * sizeof(float));

  int valid_count = 0;
  for (int s = 0; s < seq_len; s++) {
    if (mask[s]) {
      valid_count++;
      for (int d = 0; d < hidden_dim; d++) {
        output[d] += input[s * hidden_dim + d];
      }
    }
  }

  if (valid_count > 0) {
    float inv_count = 1.0f / valid_count;
    ScaleF32(output, output, inv_count, hidden_dim);
  }
}

/**
 * Max pooling over sequence dimension
 * Input: [seq_len, hidden_dim] -> Output: [hidden_dim]
 */
inline void MaxPool(const float *input, float *output, int seq_len,
                    int hidden_dim) {
  if (seq_len <= 0)
    return;

  // Initialize with first row
  std::memcpy(output, input, hidden_dim * sizeof(float));

#if defined(__AVX2__)
  for (int d = 0; d < hidden_dim; d += 8) {
    int remaining = (d + 8 <= hidden_dim) ? 8 : hidden_dim - d;
    if (remaining == 8) {
      __m256 vmax = _mm256_loadu_ps(input + d);
      for (int s = 1; s < seq_len; s++) {
        __m256 v = _mm256_loadu_ps(input + s * hidden_dim + d);
        vmax = _mm256_max_ps(vmax, v);
      }
      _mm256_storeu_ps(output + d, vmax);
    } else {
      for (int dd = d; dd < hidden_dim; dd++) {
        float maxv = input[dd];
        for (int s = 1; s < seq_len; s++) {
          float v = input[s * hidden_dim + dd];
          if (v > maxv)
            maxv = v;
        }
        output[dd] = maxv;
      }
      break;
    }
  }
#elif defined(DENSECORE_ARM)
  for (int d = 0; d < hidden_dim; d += 4) {
    int remaining = (d + 4 <= hidden_dim) ? 4 : hidden_dim - d;
    if (remaining == 4) {
      float32x4_t vmax = vld1q_f32(input + d);
      for (int s = 1; s < seq_len; s++) {
        float32x4_t v = vld1q_f32(input + s * hidden_dim + d);
        vmax = vmaxq_f32(vmax, v);
      }
      vst1q_f32(output + d, vmax);
    } else {
      for (int dd = d; dd < hidden_dim; dd++) {
        float maxv = input[dd];
        for (int s = 1; s < seq_len; s++) {
          float v = input[s * hidden_dim + dd];
          if (v > maxv)
            maxv = v;
        }
        output[dd] = maxv;
      }
      break;
    }
  }
#else
  // Scalar fallback
  for (int s = 1; s < seq_len; s++) {
    for (int d = 0; d < hidden_dim; d++) {
      float v = input[s * hidden_dim + d];
      if (v > output[d])
        output[d] = v;
    }
  }
#endif
}

/**
 * CLS pooling - extract first token
 * Input: [seq_len, hidden_dim] -> Output: [hidden_dim]
 */
inline void ClsPool(const float *input, float *output, int hidden_dim) {
  CopyF32(output, input, hidden_dim);
}

/**
 * Last token pooling
 * Input: [seq_len, hidden_dim] -> Output: [hidden_dim]
 */
inline void LastPool(const float *input, float *output, int seq_len,
                     int hidden_dim) {
  if (seq_len <= 0)
    return;
  CopyF32(output, input + (seq_len - 1) * hidden_dim, hidden_dim);
}

/**
 * Batch L2 normalization
 * Normalize each row of a [batch_size, dim] matrix
 */
inline void BatchNormalizeL2(float *data, int batch_size, int dim) {
  for (int b = 0; b < batch_size; b++) {
    NormalizeL2(data + b * dim, dim);
  }
}

/**
 * Cosine similarity between two vectors
 */
inline float CosineSimilarity(const float *a, const float *b, size_t n) {
  float dot = DotF32(a, b, n);
  float norm_a = sqrtf(DotF32(a, a, n));
  float norm_b = sqrtf(DotF32(b, b, n));
  if (norm_a < 1e-12f || norm_b < 1e-12f)
    return 0.0f;
  return dot / (norm_a * norm_b);
}

// =============================================================================
// INT4 Quantized GEMM (AVX512) - HIGHLY OPTIMIZED
// =============================================================================

#if defined(__AVX512F__)

/**
 * @brief Unpack 64 × 4-bit signed integers using AVX512 (OPTIMIZED)
 *
 * Uses shift trick for sign extension:
 * 1. Shift left 12 bits (positions 4-bit value at top of 16-bit)
 * 2. Arithmetic right shift 12 bits (propagates sign bit)
 *
 * Processes 32 bytes (64 packed 4-bit values) into 64 × int16_t
 *
 * @param packed Input: 32 bytes containing 64 packed 4-bit values
 * @param unpacked_lo Output: first 32 × int16_t values (low 16 bytes input)
 * @param unpacked_hi Output: second 32 × int16_t values (high 16 bytes input)
 */
inline void UnpackInt4x64_AVX512(const uint8_t *packed, __m512i &unpacked_lo,
                                 __m512i &unpacked_hi) {
  // Load 32 bytes = 64 packed 4-bit values into AVX512 register
  __m256i packed_256 =
      _mm256_loadu_si256(reinterpret_cast<const __m256i *>(packed));

  // Expand 8-bit elements to 16-bit
  __m512i packed_16 = _mm512_cvtepu8_epi16(packed_256);

  // Extract lower nibbles (bits 0-3)
  __m512i low_nibbles = _mm512_and_si512(packed_16, _mm512_set1_epi16(0x0F));

  // Extract upper nibbles (bits 4-7), shift down
  __m512i high_nibbles = _mm512_srli_epi16(packed_16, 4);
  high_nibbles = _mm512_and_si512(high_nibbles, _mm512_set1_epi16(0x0F));

  // Sign extension using shift trick:
  // slli by 12 moves 4-bit value to bits 12-15 (sign bit at position 15)
  // srai by 12 propagates sign bit through bits 4-15
  low_nibbles = _mm512_slli_epi16(low_nibbles, 12);
  low_nibbles = _mm512_srai_epi16(low_nibbles, 12);

  high_nibbles = _mm512_slli_epi16(high_nibbles, 12);
  high_nibbles = _mm512_srai_epi16(high_nibbles, 12);

  // Interleave: [a0, b0, a1, b1, ...] where a=low, b=high
  // unpacklo takes low halves of each lane, unpackhi takes high halves
  __m512i interleaved_lo = _mm512_unpacklo_epi16(low_nibbles, high_nibbles);
  __m512i interleaved_hi = _mm512_unpackhi_epi16(low_nibbles, high_nibbles);

  // The 512-bit unpack works on 256-bit lanes, so we need to permute
  // to get correct final order
  unpacked_lo = _mm512_permutex2var_epi64(
      interleaved_lo, _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11),
      interleaved_hi);
  unpacked_hi = _mm512_permutex2var_epi64(
      interleaved_lo, _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15),
      interleaved_hi);
}

/**
 * @brief High-performance GEMM kernel: C = A * W^T (HIGHLY OPTIMIZED)
 *
 * Optimizations:
 * - 16x register blocking along N dimension (maximum ILP)
 * - Shift-based sign extension (no branches, no masks)
 * - Aggressive prefetching (128 bytes ahead)
 * - Minimized loop overhead with unrolling
 *
 * Performance target: >90% of theoretical AVX512 peak
 */
inline void GemmInt4Fp32_AVX512(float *C, const float *A, const uint8_t *W_int4,
                                const float *scales, const float *zero_points,
                                int M, int N, int K, int group_size) {
  if (K % group_size != 0)
    return;

  const int num_groups = K / group_size;
  const int packed_K = K / 2; // Bytes per weight row

  // Constants for prefetch distance
  constexpr int PREFETCH_DIST = 128; // Bytes ahead

  // Outer loop: rows of output
  for (int m = 0; m < M; m++) {
    const float *a_row = A + m * K;

    // Middle loop: columns of output in blocks of 16 (REGISTER BLOCKING)
    int n = 0;
    for (; n + 16 <= N; n += 16) {
      // 16 accumulators for 16 output elements
      __m512 acc00 = _mm512_setzero_ps();
      __m512 acc01 = _mm512_setzero_ps();
      __m512 acc02 = _mm512_setzero_ps();
      __m512 acc03 = _mm512_setzero_ps();
      __m512 acc04 = _mm512_setzero_ps();
      __m512 acc05 = _mm512_setzero_ps();
      __m512 acc06 = _mm512_setzero_ps();
      __m512 acc07 = _mm512_setzero_ps();
      __m512 acc08 = _mm512_setzero_ps();
      __m512 acc09 = _mm512_setzero_ps();
      __m512 acc10 = _mm512_setzero_ps();
      __m512 acc11 = _mm512_setzero_ps();
      __m512 acc12 = _mm512_setzero_ps();
      __m512 acc13 = _mm512_setzero_ps();
      __m512 acc14 = _mm512_setzero_ps();
      __m512 acc15 = _mm512_setzero_ps();

      // Loop over quantization groups
      for (int g = 0; g < num_groups; g++) {
        const int k_offset = g * group_size;
        const int packed_offset = g * (group_size / 2);
        const float *a_ptr = a_row + k_offset;

        // Prefetch activations for next group
        if (g + 1 < num_groups) {
          _mm_prefetch(
              reinterpret_cast<const char *>(a_row + (g + 1) * group_size),
              _MM_HINT_T0);
        }

        // Process 32 weights at a time (64 bits packed = 16 bytes)
        for (int k = 0; k < group_size; k += 32) {
          // Load 2 × 16 floats from activations
          __m512 a0 = _mm512_loadu_ps(a_ptr + k);
          __m512 a1 = _mm512_loadu_ps(a_ptr + k + 16);

// Process all 16 weight rows with 2-way unrolling
#define PROCESS_ROW(idx)                                                       \
  do {                                                                         \
    const int row = n + (idx);                                                 \
    const float scale = scales[row * num_groups + g];                          \
    const float zero = zero_points[row * num_groups + g];                      \
    const __m512 vscale = _mm512_set1_ps(scale);                               \
    const __m512 vzero = _mm512_set1_ps(zero);                                 \
                                                                               \
    const uint8_t *w_ptr = W_int4 + row * packed_K + packed_offset + k / 2;    \
    _mm_prefetch(reinterpret_cast<const char *>(w_ptr + PREFETCH_DIST),        \
                 _MM_HINT_T0);                                                 \
                                                                               \
    /* Load 16 bytes = 32 packed weights */                                    \
    __m128i packed =                                                           \
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_ptr));             \
    __m256i packed_256 = _mm256_cvtepu8_epi16(packed);                         \
                                                                               \
    /* Extract low nibbles */                                                  \
    __m256i low = _mm256_and_si256(packed_256, _mm256_set1_epi16(0x0F));       \
    /* Extract high nibbles */                                                 \
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
    __m256i interleaved_lo = _mm256_unpacklo_epi16(low, high);                 \
    __m256i interleaved_hi = _mm256_unpackhi_epi16(low, high);                 \
                                                                               \
    /* Permute for correct lane order after interleave */                      \
    __m256i w16_0 =                                                            \
        _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x20);       \
    __m256i w16_1 =                                                            \
        _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x31);       \
                                                                               \
    /* Convert to FP32 */                                                      \
    __m512i w32_0 = _mm512_cvtepi16_epi32(w16_0);                              \
    __m512i w32_1 = _mm512_cvtepi16_epi32(w16_1);                              \
    __m512 wf0 = _mm512_cvtepi32_ps(w32_0);                                    \
    __m512 wf1 = _mm512_cvtepi32_ps(w32_1);                                    \
                                                                               \
    /* Dequantize: w_dequant = scale * (q - zero) */                           \
    wf0 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf0, vzero));                    \
    wf1 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf1, vzero));                    \
                                                                               \
    /* FMA */                                                                  \
    acc##idx = _mm512_fmadd_ps(a0, wf0, acc##idx);                             \
    acc##idx = _mm512_fmadd_ps(a1, wf1, acc##idx);                             \
  } while (0)

          PROCESS_ROW(00);
          PROCESS_ROW(01);
          PROCESS_ROW(02);
          PROCESS_ROW(03);
          PROCESS_ROW(04);
          PROCESS_ROW(05);
          PROCESS_ROW(06);
          PROCESS_ROW(07);
          PROCESS_ROW(08);
          PROCESS_ROW(09);
          PROCESS_ROW(10);
          PROCESS_ROW(11);
          PROCESS_ROW(12);
          PROCESS_ROW(13);
          PROCESS_ROW(14);
          PROCESS_ROW(15);

#undef PROCESS_ROW
        }
      }

      // Horizontal reduction and store results
      C[m * N + n + 0] = _mm512_reduce_add_ps(acc00);
      C[m * N + n + 1] = _mm512_reduce_add_ps(acc01);
      C[m * N + n + 2] = _mm512_reduce_add_ps(acc02);
      C[m * N + n + 3] = _mm512_reduce_add_ps(acc03);
      C[m * N + n + 4] = _mm512_reduce_add_ps(acc04);
      C[m * N + n + 5] = _mm512_reduce_add_ps(acc05);
      C[m * N + n + 6] = _mm512_reduce_add_ps(acc06);
      C[m * N + n + 7] = _mm512_reduce_add_ps(acc07);
      C[m * N + n + 8] = _mm512_reduce_add_ps(acc08);
      C[m * N + n + 9] = _mm512_reduce_add_ps(acc09);
      C[m * N + n + 10] = _mm512_reduce_add_ps(acc10);
      C[m * N + n + 11] = _mm512_reduce_add_ps(acc11);
      C[m * N + n + 12] = _mm512_reduce_add_ps(acc12);
      C[m * N + n + 13] = _mm512_reduce_add_ps(acc13);
      C[m * N + n + 14] = _mm512_reduce_add_ps(acc14);
      C[m * N + n + 15] = _mm512_reduce_add_ps(acc15);
    }

    // Handle remaining columns (N % 16)
    for (; n < N; n++) {
      __m512 acc = _mm512_setzero_ps();

      for (int g = 0; g < num_groups; g++) {
        const int k_offset = g * group_size;
        const float *a_ptr = a_row + k_offset;
        const float scale = scales[n * num_groups + g];
        const float zero = zero_points[n * num_groups + g];
        const __m512 vscale = _mm512_set1_ps(scale);
        const __m512 vzero = _mm512_set1_ps(zero);
        const uint8_t *w_ptr = W_int4 + n * packed_K + g * (group_size / 2);

        for (int k = 0; k < group_size; k += 32) {
          // Load 16 bytes = 32 packed weights
          __m128i packed =
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(w_ptr + k / 2));
          __m256i packed_256 = _mm256_cvtepu8_epi16(packed);

          // Extract and sign-extend nibbles
          __m256i low = _mm256_and_si256(packed_256, _mm256_set1_epi16(0x0F));
          __m256i high = _mm256_srli_epi16(packed_256, 4);
          high = _mm256_and_si256(high, _mm256_set1_epi16(0x0F));

          low = _mm256_slli_epi16(low, 12);
          low = _mm256_srai_epi16(low, 12);
          high = _mm256_slli_epi16(high, 12);
          high = _mm256_srai_epi16(high, 12);

          // Interleave and permute
          __m256i lo = _mm256_unpacklo_epi16(low, high);
          __m256i hi = _mm256_unpackhi_epi16(low, high);
          __m256i w16_0 = _mm256_permute2x128_si256(lo, hi, 0x20);
          __m256i w16_1 = _mm256_permute2x128_si256(lo, hi, 0x31);

          // Convert and dequantize
          __m512 wf0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(w16_0));
          __m512 wf1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(w16_1));

          wf0 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf0, vzero));
          wf1 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf1, vzero));

          // Load activations and FMA
          __m512 a0 = _mm512_loadu_ps(a_ptr + k);
          __m512 a1 = _mm512_loadu_ps(a_ptr + k + 16);

          acc = _mm512_fmadd_ps(a0, wf0, acc);
          acc = _mm512_fmadd_ps(a1, wf1, acc);
        }
      }

      C[m * N + n] = _mm512_reduce_add_ps(acc);
    }
  }
}

#else // No AVX512 support

/**
 * Fallback GEMM for INT4 weights (scalar implementation)
 */
inline void GemmInt4Fp32_AVX512(float *C, const float *A, const uint8_t *W_int4,
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

          // Sign extend
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

#endif // __AVX512F__

} // namespace simd
} // namespace densecore

#endif // DENSECORE_SIMD_OPS_H
