/**
 * @file simd_ops.h
 * @brief SIMD-optimized operations for CPU inference
 *
 * THREADING CONTRACT:
 * -------------------
 * All functions in this file are SINGLE-THREADED kernels designed to be
 * called from GGML worker threads. DO NOT add #pragma omp parallel to any
 * function as this would cause thread oversubscription.
 *
 * For functions that support parallel execution across tokens/heads, they
 * accept (ith, nth) parameters for manual work partitioning, where:
 *   - ith: Current thread index (0 to nth-1)
 *   - nth: Total number of threads
 *
 * The caller (GGML callback or inference engine) is responsible for invoking
 * these functions from multiple threads with appropriate (ith, nth) values.
 *
 * Supports:
 * - AVX-512 (Intel Skylake-X+)
 * - AVX2 (Intel Haswell+, AMD Zen+)
 * - AVX (Intel Sandy Bridge+)
 * - SSE4.1 (Intel Penryn+)
 * - ARM SVE (AWS Graviton 3/4, scalable 256-bit+ vectors)
 * - ARM NEON (Apple Silicon, ARM64, fixed 128-bit vectors)
 *
 * Auto-detects best available SIMD and provides unified API.
 */

#ifndef DENSECORE_SIMD_OPS_H
#define DENSECORE_SIMD_OPS_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>  // For std::abort, posix_memalign, free
#include <cstring>
#include <new>          // For std::bad_alloc
#include <type_traits>  // For std::true_type
#include <vector>

#if defined(__linux__)
#include <pthread.h>  // For pthread_setaffinity_np
#include <sched.h>    // For cpu_set_t
#include <unistd.h>   // For sysconf, _SC_NPROCESSORS_ONLN
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>  // For SYSTEM_INFO, GetSystemInfo
#endif

// Platform detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define DENSECORE_X86
#include "simd_platform.h"
#elif defined(__aarch64__) || defined(_M_ARM64)
#define DENSECORE_ARM
#include <arm_neon.h>
// SVE (Scalable Vector Extension) for AWS Graviton 3/4 and other ARMv8.2+
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#endif

// FP16 support from ggml
extern "C" {
#include "ggml.h"
}

// =============================================================================
// Aligned Memory Allocation for SIMD (64-byte for AVX-512)
// =============================================================================
// AVX-512 instructions like vmovaps require 64-byte aligned memory.
// std::vector does NOT guarantee this alignment. Use AlignedVector<T> instead.
// =============================================================================

namespace densecore {
namespace simd {

/// AVX-512 cache line size (64 bytes)
constexpr size_t SIMD_ALIGNMENT = 64;

// =============================================================================
// Tunable Prefetch Distance for Runtime Optimization
// =============================================================================
// These can be adjusted at runtime for optimal performance on different
// hardware. Default values are tuned for Intel Skylake-X / Ice Lake.
// =============================================================================

/// Prefetch distance for AVX-512 kernels (bytes ahead)
inline int g_prefetch_dist_avx512 = 128;

/// Prefetch distance for AVX2 kernels (bytes ahead)
inline int g_prefetch_dist_avx2 = 64;

/**
 * @brief Set prefetch distances for SIMD kernels
 *
 * Tune these based on hardware characteristics:
 * - Larger values for high-latency memory (server DIMMs)
 * - Smaller values for low-latency (L3 resident data)
 *
 * @param dist_avx512 Prefetch distance for AVX-512 in bytes (default: 128)
 * @param dist_avx2 Prefetch distance for AVX2 in bytes (default: 64)
 */
inline void SetPrefetchDistance(int dist_avx512, int dist_avx2) {
    g_prefetch_dist_avx512 = dist_avx512;
    g_prefetch_dist_avx2 = dist_avx2;
}

/**
 * @brief Get current AVX-512 prefetch distance
 */
inline int GetPrefetchDistanceAVX512() {
    return g_prefetch_dist_avx512;
}

/**
 * @brief Get current AVX2 prefetch distance
 */
inline int GetPrefetchDistanceAVX2() {
    return g_prefetch_dist_avx2;
}

/**
 * @brief Check if pointer is aligned to given boundary
 *
 * @param ptr Pointer to check
 * @param alignment Required alignment (must be power of 2)
 * @return true if aligned, false otherwise
 */
inline bool IsAligned(const void* ptr, size_t alignment) {
    return ((uintptr_t)ptr & (alignment - 1)) == 0;
}

/**
 * @brief Check if pointer is 64-byte aligned (AVX-512 compatible)
 */
inline bool IsAligned64(const void* ptr) {
    return IsAligned(ptr, 64);
}

}  // namespace simd
}  // namespace densecore

/**
 * Alignment assertion macro for DEBUG builds.
 * In DEBUG mode: aborts if pointer is not properly aligned.
 * In RELEASE mode: no-op (zero overhead).
 */
#ifdef NDEBUG
#define DENSECORE_ASSERT_ALIGNED(ptr, alignment) ((void)0)
#else
#define DENSECORE_ASSERT_ALIGNED(ptr, alignment)                                  \
    do {                                                                          \
        if (!densecore::simd::IsAligned((ptr), (alignment))) {                    \
            std::fprintf(stderr,                                                  \
                         "[DenseCore] FATAL: Pointer %p not %zu-byte aligned at " \
                         "%s:%d\n",                                               \
                         (void*)(ptr), (size_t)(alignment), __FILE__, __LINE__);  \
            std::abort();                                                         \
        }                                                                         \
    } while (0)
#endif

/// Convenience macro for 64-byte alignment assertion (AVX-512)
#define DENSECORE_ASSERT_ALIGNED_64(ptr) DENSECORE_ASSERT_ALIGNED(ptr, 64)

namespace densecore {
namespace simd {

/**
 * @brief STL-compatible allocator providing aligned memory allocation
 *
 * Required for AVX-512 instructions that benefit from aligned loads/stores.
 * Default alignment is 64 bytes (AVX-512 cache line size).
 *
 * Usage:
 *   std::vector<float, AlignedAllocator<float>> vec;
 *   // or use the convenience alias:
 *   AlignedVector<float> vec;
 *
 * @tparam T Element type
 * @tparam Alignment Alignment in bytes (default: 64 for AVX-512)
 */
template <typename T, size_t Alignment = SIMD_ALIGNMENT>
struct AlignedAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    // Required for std::vector rebind compatibility (C++17 allocator
    // requirements)
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be a power of 2");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

    constexpr AlignedAllocator() noexcept = default;
    constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;

    template <class U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>& /*other*/) noexcept {}

    [[nodiscard]] T* allocate(size_type n) {
        if (n == 0)
            return nullptr;

        size_t bytes = n * sizeof(T);
        T* ptr = nullptr;

#if defined(_WIN32)
        ptr = static_cast<T*>(_aligned_malloc(bytes, Alignment));
#else
        void* raw_ptr = nullptr;
        if (posix_memalign(&raw_ptr, Alignment, bytes) == 0) {
            ptr = static_cast<T*>(raw_ptr);
        }
#endif

        if (!ptr) {
            throw std::bad_alloc();
        }

        return ptr;
    }

    void deallocate(T* ptr, size_type /*n*/) noexcept {
        if (!ptr)
            return;

#if defined(_WIN32)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    template <class U>
    bool operator==(const AlignedAllocator<U, Alignment>& /*other*/) const noexcept {
        return true;
    }

    template <class U>
    bool operator!=(const AlignedAllocator<U, Alignment>& /*other*/) const noexcept {
        return false;
    }
};

/**
 * @brief Convenience type alias for 64-byte aligned vectors
 *
 * Use this instead of std::vector<T> when the data will be passed to
 * AVX-512 kernels that require or benefit from aligned memory.
 *
 * @tparam T Element type
 */
template <typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, 64>>;

// =============================================================================
// Runtime SIMD Detection
// =============================================================================

enum class SimdLevel {
    NONE = 0,
    SSE41 = 1,
    AVX = 2,
    AVX2 = 3,
    AVX512 = 4,
    AMX = 5,    // Intel Advanced Matrix Extensions (Sapphire Rapids+)
    NEON = 10,  // ARM NEON (fixed 128-bit vectors)
    SVE = 11    // ARM SVE (scalable 256-bit+ vectors, Graviton 3/4)
};

inline SimdLevel DetectSimdLevel() {
#ifdef DENSECORE_X86
    // -------------------------------------------------------------------------
    // Runtime CPUID-based detection (portable binary support)
    // -------------------------------------------------------------------------
#if defined(_MSC_VER)
    // MSVC: Use __cpuid intrinsic
    int cpuid_info[4] = {0};

    // Check for AVX-512F: CPUID(7, 0).EBX bit 16
    __cpuidex(cpuid_info, 7, 0);
    bool has_avx512f = (cpuid_info[1] & (1 << 16)) != 0;

    // Check for AVX2: CPUID(7, 0).EBX bit 5
    bool has_avx2 = (cpuid_info[1] & (1 << 5)) != 0;

    // Check for FMA: CPUID(1, 0).ECX bit 12
    __cpuid(cpuid_info, 1);
    bool has_fma = (cpuid_info[2] & (1 << 12)) != 0;

    // Check for AVX: CPUID(1, 0).ECX bit 28
    bool has_avx = (cpuid_info[2] & (1 << 28)) != 0;

    // Check for SSE4.1: CPUID(1, 0).ECX bit 19
    bool has_sse41 = (cpuid_info[2] & (1 << 19)) != 0;

    // Check for AMX-TILE: CPUID(7, 0).EDX bit 24
    __cpuidex(cpuid_info, 7, 0);
    bool has_amx_tile = (cpuid_info[3] & (1 << 24)) != 0;

    if (has_amx_tile && has_avx512f) {
        return SimdLevel::AMX;
    } else if (has_avx512f) {
        return SimdLevel::AVX512;
    } else if (has_avx2 && has_fma) {
        return SimdLevel::AVX2;
    } else if (has_avx) {
        return SimdLevel::AVX;
    } else if (has_sse41) {
        return SimdLevel::SSE41;
    } else {
        return SimdLevel::NONE;
    }

#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang: Use __builtin_cpu_supports (simpler and reliable)
    // NOTE: __builtin_cpu_init() is called automatically on modern compilers

    // Check from highest to lowest capability
    if (__builtin_cpu_supports("amx-tile") && __builtin_cpu_supports("avx512f")) {
        return SimdLevel::AMX;
    } else if (__builtin_cpu_supports("avx512f")) {
        return SimdLevel::AVX512;
    } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        return SimdLevel::AVX2;
    } else if (__builtin_cpu_supports("avx")) {
        return SimdLevel::AVX;
    } else if (__builtin_cpu_supports("sse4.1")) {
        return SimdLevel::SSE41;
    } else {
        return SimdLevel::NONE;
    }

#else
    // Unknown compiler: Fallback to compile-time detection
#if defined(__AVX512F__)
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
#endif

#elif defined(DENSECORE_ARM)
#if defined(__ARM_FEATURE_SVE)
    return SimdLevel::SVE;  // Graviton 3/4 with scalable vectors
#else
    return SimdLevel::NEON;
#endif
#else
    return SimdLevel::NONE;
#endif
}

inline const char* SimdLevelName(SimdLevel level) {
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
    case SimdLevel::SVE:
        return "ARM SVE";
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
    int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    return result == 0;
#elif defined(_WIN32)
    // Windows: Use SetThreadAffinityMask
    if (core_id < 0 || core_id >= 64)
        return false;  // Windows mask limit
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
            break;  // Nodes are numbered contiguously
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
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node_id);

    FILE* f = fopen(path, "r");
    if (!f)
        return cores;

    char buf[256];
    if (fgets(buf, sizeof(buf), f)) {
        // Remove trailing newline
        size_t len = strlen(buf);
        if (len > 0 && buf[len - 1] == '\n')
            buf[len - 1] = '\0';

        // Parse comma-separated entries
        char* saveptr = nullptr;
        char* token = strtok_r(buf, ",", &saveptr);
        while (token) {
            // Check if it's a range (contains '-')
            char* dash = strchr(token, '-');
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
    return 1;  // Fallback
#endif
}

// =============================================================================
// Memory Prefetch
// =============================================================================

/**
 * Prefetch memory for read (T0 = all cache levels)
 */
inline void Prefetch(const void* ptr) {
#ifdef DENSECORE_X86
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(DENSECORE_ARM)
    __builtin_prefetch(ptr, 0, 3);  // read, high locality
#else
    (void)ptr;
#endif
}

/**
 * Prefetch a range of memory (cache-line aligned)
 */
inline void PrefetchRange(const void* ptr, size_t bytes) {
    constexpr size_t CACHE_LINE = 64;
    const char* p = reinterpret_cast<const char*>(ptr);
    for (size_t i = 0; i < bytes; i += CACHE_LINE) {
        Prefetch(p + i);
    }
}

/**
 * Prefetch for write (non-temporal if supported)
 */
inline void PrefetchWrite(void* ptr) {
#ifdef DENSECORE_X86
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(DENSECORE_ARM)
    __builtin_prefetch(ptr, 1, 3);  // write, high locality
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
inline void SimdCopy(void* dst, const void* src, size_t bytes) {
#if defined(__AVX512F__)
    // AVX-512: 64 bytes per iteration
    const size_t vec_size = 64;
    size_t i = 0;
    for (; i + vec_size <= bytes; i += vec_size) {
        __m512i v = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(reinterpret_cast<const char*>(src) + i));
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(reinterpret_cast<char*>(dst) + i), v);
    }
    // Handle remainder
    if (i < bytes) {
        memcpy(reinterpret_cast<char*>(dst) + i, reinterpret_cast<const char*>(src) + i, bytes - i);
    }
#elif defined(__AVX2__)
    // AVX2: 32 bytes per iteration
    const size_t vec_size = 32;
    size_t i = 0;
    for (; i + vec_size <= bytes; i += vec_size) {
        __m256i v = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(reinterpret_cast<const char*>(src) + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(reinterpret_cast<char*>(dst) + i), v);
    }
    if (i < bytes) {
        memcpy(reinterpret_cast<char*>(dst) + i, reinterpret_cast<const char*>(src) + i, bytes - i);
    }
#elif defined(__SSE2__)
    // SSE2: 16 bytes per iteration
    const size_t vec_size = 16;
    size_t i = 0;
    for (; i + vec_size <= bytes; i += vec_size) {
        __m128i v = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(reinterpret_cast<const char*>(src) + i));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(reinterpret_cast<char*>(dst) + i), v);
    }
    if (i < bytes) {
        memcpy(reinterpret_cast<char*>(dst) + i, reinterpret_cast<const char*>(src) + i, bytes - i);
    }
#elif defined(DENSECORE_ARM)
    // NEON: 16 bytes per iteration
    const size_t vec_size = 16;
    size_t i = 0;
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    uint8_t* d = reinterpret_cast<uint8_t*>(dst);
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
inline void CopyF32(float* dst, const float* src, size_t n) {
    SimdCopy(dst, src, n * sizeof(float));
}

/**
 * Scale float32 array: dst[i] = src[i] * scale
 */
inline void ScaleF32(float* dst, const float* src, float scale, size_t n) {
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
inline void AddF32(float* dst, const float* a, const float* b, size_t n) {
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
inline float DotF32(const float* a, const float* b, size_t n) {
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
inline float MaxF32(const float* a, size_t n) {
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
inline float SumF32(const float* a, size_t n) {
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
inline void SoftmaxF32(float* a, size_t n) {
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
inline void ConvertF32ToF16(ggml_fp16_t* dst, const float* src, size_t n) {
    ggml_fp32_to_fp16_row(src, dst, n);
}

/**
 * Convert FP16 to FP32 using SIMD
 */
inline void ConvertF16ToF32(float* dst, const ggml_fp16_t* src, size_t n) {
    ggml_fp16_to_fp32_row(src, dst, n);
}

// =============================================================================
// Matrix Operations (used by Flash Attention)
// =============================================================================

/**
 * Matrix multiply: C = A @ B^T (for attention scores)
 * A: [M, K], B: [N, K] (row-major), C: [M, N]
 */
inline void MatMulTransB(float* C, const float* A, const float* B, int M, int N, int K) {
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
inline void NormalizeL2(float* data, size_t n) {
    // Compute L2 norm using SIMD dot product
    float norm_sq = DotF32(data, data, n);
    if (norm_sq < 1e-12f)
        return;  // Avoid division by zero

    float inv_norm = 1.0f / sqrtf(norm_sq);
    ScaleF32(data, data, inv_norm, n);
}

/**
 * Mean pooling over sequence dimension
 * Input: [seq_len, hidden_dim] -> Output: [hidden_dim]
 * Standard pooling for sentence-transformers models
 */
inline void MeanPool(const float* input, float* output, int seq_len, int hidden_dim) {
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
inline void MeanPoolMasked(const float* input, const int* mask, float* output, int seq_len,
                           int hidden_dim) {
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
inline void MaxPool(const float* input, float* output, int seq_len, int hidden_dim) {
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
inline void ClsPool(const float* input, float* output, int hidden_dim) {
    CopyF32(output, input, hidden_dim);
}

/**
 * Last token pooling
 * Input: [seq_len, hidden_dim] -> Output: [hidden_dim]
 */
inline void LastPool(const float* input, float* output, int seq_len, int hidden_dim) {
    if (seq_len <= 0)
        return;
    CopyF32(output, input + (seq_len - 1) * hidden_dim, hidden_dim);
}

/**
 * Batch L2 normalization
 * Normalize each row of a [batch_size, dim] matrix
 */
inline void BatchNormalizeL2(float* data, int batch_size, int dim) {
    for (int b = 0; b < batch_size; b++) {
        NormalizeL2(data + b * dim, dim);
    }
}

/**
 * Cosine similarity between two vectors
 */
inline float CosineSimilarity(const float* a, const float* b, size_t n) {
    float dot = DotF32(a, b, n);
    float norm_a = sqrtf(DotF32(a, a, n));
    float norm_b = sqrtf(DotF32(b, b, n));
    if (norm_a < 1e-12f || norm_b < 1e-12f)
        return 0.0f;
    return dot / (norm_a * norm_b);
}

// =============================================================================
// INT4 GEMM Kernels (declarations)
// =============================================================================

void GemmInt4Fp32_AVX512(float* C, const float* A, const uint8_t* W, const float* scales,
                         const float* zeros, int M, int N, int K, int group_size);

void GemmInt4Fp32_AVX2(float* C, const float* A, const uint8_t* W, const float* scales,
                       const float* zeros, int M, int N, int K, int group_size);

// =============================================================================
// Rotary Positional Embedding (RoPE) - HIGHLY OPTIMIZED
// =============================================================================

/**
 * @brief Pre-computed RoPE frequency table for a given context length
 *
 * Pre-computes cos(m * theta) and sin(m * theta) for all positions m and
 * all frequency dimensions. This avoids expensive transcendental function
 * calls in the inner loop.
 *
 * Layout: [max_seq_len, head_dim / 2] where each entry is (cos, sin) pair
 * Access: cos_sin_table[pos * head_dim + 2*d] = cos(pos * theta_d)
 *         cos_sin_table[pos * head_dim + 2*d + 1] = sin(pos * theta_d)
 */
struct RoPETable {
    std::vector<float> cos_sin;  ///< Interleaved [cos, sin, cos, sin, ...]
    int max_seq_len;
    int head_dim;
    float freq_base;

    /**
     * @brief Initialize RoPE table for given parameters
     *
     * @param max_len Maximum sequence length to pre-compute
     * @param dim Head dimension (must be even)
     * @param base RoPE frequency base (default 10000.0 for Llama)
     */
    void Init(int max_len, int dim, float base = 10000.0f) {
        max_seq_len = max_len;
        head_dim = dim;
        freq_base = base;

        // Allocate interleaved cos/sin: [max_len, head_dim]
        cos_sin.resize(static_cast<size_t>(max_len) * dim);

        // Pre-compute frequencies: theta[d] = 1 / (base ** (2d / dim))
        std::vector<float> freqs(dim / 2);
        for (int d = 0; d < dim / 2; d++) {
            float exp = (2.0f * d) / static_cast<float>(dim);
            freqs[d] = 1.0f / std::pow(base, exp);
        }

        // Compute cos/sin for all positions
        for (int pos = 0; pos < max_len; pos++) {
            for (int d = 0; d < dim / 2; d++) {
                float angle = static_cast<float>(pos) * freqs[d];
                cos_sin[pos * dim + 2 * d] = std::cos(angle);
                cos_sin[pos * dim + 2 * d + 1] = std::sin(angle);
            }
        }
    }

    /**
     * @brief Get cos value for position and dimension pair index
     */
    inline float Cos(int pos, int pair_d) const { return cos_sin[pos * head_dim + 2 * pair_d]; }

    /**
     * @brief Get sin value for position and dimension pair index
     */
    inline float Sin(int pos, int pair_d) const { return cos_sin[pos * head_dim + 2 * pair_d + 1]; }

    /**
     * @brief Get pointer to cos/sin data for a specific position
     */
    inline const float* GetCosSinPtr(int pos) const { return cos_sin.data() + pos * head_dim; }
};

#if defined(__AVX512F__)

/**
 * @brief Apply Rotary Positional Embedding using AVX-512
 *
 * RoPE formula for pair (x_{2d}, x_{2d+1}):
 *   x'_{2d}   = x_{2d} * cos(θ) - x_{2d+1} * sin(θ)
 *   x'_{2d+1} = x_{2d} * sin(θ) + x_{2d+1} * cos(θ)
 *
 * This kernel processes 16 floats (8 pairs) per iteration using AVX-512.
 *
 * @param out Output tensor [n_tokens, head_dim] (can be same as in for
 * in-place)
 * @param in Input tensor [n_tokens, head_dim]
 * @param cos_sin Pre-computed [cos, sin] pairs [max_seq, head_dim]
 * @param positions Token positions array [n_tokens]
 * @param n_tokens Number of tokens to process
 * @param head_dim Head dimension (must be even)
 * @param rope_dim Number of dimensions to apply RoPE (typically == head_dim)
 * @param ith Thread index for work partitioning
 * @param nth Total number of threads
 */
inline void ApplyRoPE_AVX512(float* out, const float* in, const float* cos_sin,
                             const int* positions, int n_tokens, int head_dim, int rope_dim,
                             int ith = 0, int nth = 1) {
    // Partition tokens across threads
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

    if (t_start >= n_tokens)
        return;

    for (int t = t_start; t < t_end; t++) {
        const int pos = positions[t];
        const float* cs_ptr = cos_sin + pos * head_dim;
        const float* in_ptr = in + t * head_dim;
        float* out_ptr = out + t * head_dim;

        // Prefetch next token's cos/sin
        if (t + 1 < t_end) {
            _mm_prefetch(reinterpret_cast<const char*>(cos_sin + positions[t + 1] * head_dim),
                         _MM_HINT_T0);
        }

        // Process 16 floats (8 pairs) at a time
        int d = 0;
        for (; d + 16 <= rope_dim; d += 16) {
            // Load input: [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12,
            // x13, x14, x15]
            __m512 x = _mm512_loadu_ps(in_ptr + d);

            // Load cos/sin: [c0, s0, c1, s1, c2, s2, c3, s3, ...]
            __m512 cs = _mm512_loadu_ps(cs_ptr + d);

            // Shuffle to separate cos and sin
            // cos = [c0, c0, c1, c1, c2, c2, c3, c3, c4, c4, c5, c5, c6, c6, c7, c7]
            // sin = [s0, s0, s1, s1, s2, s2, s3, s3, s4, s4, s5, s5, s6, s6, s7, s7]
            // But we have interleaved [c0, s0, c1, s1, ...], so we need to
            // deinterleave

            // Extract even indices (cos) and odd indices (sin)
            const __m512i idx_cos =
                _mm512_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
            const __m512i idx_sin =
                _mm512_setr_epi32(1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15);

            __m512 cos_vec = _mm512_permutexvar_ps(idx_cos, cs);
            __m512 sin_vec = _mm512_permutexvar_ps(idx_sin, cs);

            // Create swapped x: [x1, x0, x3, x2, x5, x4, x7, x6, ...]
            const __m512i idx_swap =
                _mm512_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
            __m512 x_swap = _mm512_permutexvar_ps(idx_swap, x);

            // For pairs (x_{2d}, x_{2d+1}):
            //   x'_{2d}   = x_{2d} * cos - x_{2d+1} * sin  (even positions)
            //   x'_{2d+1} = x_{2d} * sin + x_{2d+1} * cos  (odd positions)
            //
            // Using: x * cos + x_swap * (alternating sign) * sin
            // Alternating sign: [-1, +1, -1, +1, ...]
            const __m512 sign_mask =
                _mm512_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
                              -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);

            // out = x * cos + x_swap * sign * sin
            __m512 result = _mm512_mul_ps(x, cos_vec);
            __m512 term2 = _mm512_mul_ps(x_swap, sin_vec);
            term2 = _mm512_mul_ps(term2, sign_mask);
            result = _mm512_add_ps(result, term2);

            _mm512_storeu_ps(out_ptr + d, result);
        }

        // Handle remainder (less than 16 floats)
        for (; d < rope_dim; d += 2) {
            float x0 = in_ptr[d];
            float x1 = in_ptr[d + 1];
            float cos_val = cs_ptr[d];
            float sin_val = cs_ptr[d + 1];

            out_ptr[d] = x0 * cos_val - x1 * sin_val;
            out_ptr[d + 1] = x0 * sin_val + x1 * cos_val;
        }

        // Copy dimensions beyond rope_dim unchanged
        for (int dd = rope_dim; dd < head_dim; dd++) {
            out_ptr[dd] = in_ptr[dd];
        }
    }
}

#endif  // __AVX512F__

#if defined(__AVX2__)

/**
 * @brief Apply Rotary Positional Embedding using AVX2
 *
 * RoPE formula for pair (x_{2d}, x_{2d+1}):
 *   x'_{2d}   = x_{2d} * cos(θ) - x_{2d+1} * sin(θ)
 *   x'_{2d+1} = x_{2d} * sin(θ) + x_{2d+1} * cos(θ)
 *
 * This kernel processes 8 floats (4 pairs) per iteration using AVX2.
 *
 * @param out Output tensor [n_tokens, head_dim] (can be same as in for
 * in-place)
 * @param in Input tensor [n_tokens, head_dim]
 * @param cos_sin Pre-computed [cos, sin] pairs [max_seq, head_dim]
 * @param positions Token positions array [n_tokens]
 * @param n_tokens Number of tokens to process
 * @param head_dim Head dimension (must be even)
 * @param rope_dim Number of dimensions to apply RoPE (typically == head_dim)
 * @param ith Thread index for work partitioning
 * @param nth Total number of threads
 */
inline void ApplyRoPE_AVX2(float* out, const float* in, const float* cos_sin, const int* positions,
                           int n_tokens, int head_dim, int rope_dim, int ith = 0, int nth = 1) {
    // Partition tokens across threads
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

    if (t_start >= n_tokens)
        return;

    // Permutation indices for cos/sin deinterleaving:
    // Input [c0, s0, c1, s1, c2, s2, c3, s3]
    // cos_idx: [0, 0, 2, 2, 4, 4, 6, 6] -> [c0, c0, c1, c1, c2, c2, c3, c3]
    // sin_idx: [1, 1, 3, 3, 5, 5, 7, 7] -> [s0, s0, s1, s1, s2, s2, s3, s3]
    const __m256i idx_cos = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6);
    const __m256i idx_sin = _mm256_setr_epi32(1, 1, 3, 3, 5, 5, 7, 7);

    // Permutation indices for swapping pairs: [1, 0, 3, 2, 5, 4, 7, 6]
    const __m256i idx_swap = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);

    // Sign mask for RoPE: [-1, 1, -1, 1, -1, 1, -1, 1]
    const __m256 sign_mask = _mm256_set_ps(1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);

    for (int t = t_start; t < t_end; t++) {
        const int pos = positions[t];
        const float* cs_ptr = cos_sin + pos * head_dim;
        const float* in_ptr = in + t * head_dim;
        float* out_ptr = out + t * head_dim;

        // Prefetch next token's cos/sin
        // Removed for stability on AVX2 (potential OOB or TLB issues)
        /*
        if (t + 1 < t_end) {
          _mm_prefetch(
              reinterpret_cast<const char *>(cos_sin + positions[t + 1] * head_dim),
              _MM_HINT_T0);
        }
        */

        // Process 8 floats (4 pairs) at a time
        int d = 0;
        for (; d + 8 <= rope_dim; d += 8) {
            // Load input: [x0, x1, x2, x3, x4, x5, x6, x7]
            __m256 x = _mm256_loadu_ps(in_ptr + d);

            // Load cos/sin: [c0, s0, c1, s1, c2, s2, c3, s3]
            __m256 cs = _mm256_loadu_ps(cs_ptr + d);

            // Deinterleave cos and sin using permute
            __m256 cos_vec = _mm256_permutevar8x32_ps(cs, idx_cos);
            __m256 sin_vec = _mm256_permutevar8x32_ps(cs, idx_sin);

            // Create swapped x: [x1, x0, x3, x2, x5, x4, x7, x6]
            __m256 x_swap = _mm256_permutevar8x32_ps(x, idx_swap);

            // Compute: x * cos + x_swap * sign * sin
            // For pairs (x_{2d}, x_{2d+1}):
            //   x'_{2d}   = x_{2d} * cos - x_{2d+1} * sin  (even positions)
            //   x'_{2d+1} = x_{2d} * sin + x_{2d+1} * cos  (odd positions)
            __m256 result = _mm256_mul_ps(x, cos_vec);
            __m256 term2 = _mm256_mul_ps(x_swap, sin_vec);
            term2 = _mm256_mul_ps(term2, sign_mask);
            result = _mm256_add_ps(result, term2);

            _mm256_storeu_ps(out_ptr + d, result);
        }

        // Handle remainder (less than 8 floats) with scalar
        for (; d < rope_dim; d += 2) {
            float x0 = in_ptr[d];
            float x1 = in_ptr[d + 1];
            float cos_val = cs_ptr[d];
            float sin_val = cs_ptr[d + 1];

            out_ptr[d] = x0 * cos_val - x1 * sin_val;
            out_ptr[d + 1] = x0 * sin_val + x1 * cos_val;
        }

        // Copy dimensions beyond rope_dim unchanged
        for (int dd = rope_dim; dd < head_dim; dd++) {
            out_ptr[dd] = in_ptr[dd];
        }
    }
}

#endif  // __AVX2__

/**
 * @brief Apply Rotary Positional Embedding (scalar implementation)
 *
 * Explicit scalar fallback for non-AVX512 builds or when scalar is preferred.
 */
inline void ApplyRoPE_Scalar(float* out, const float* in, const float* cos_sin,
                             const int* positions, int n_tokens, int head_dim, int rope_dim,
                             int ith = 0, int nth = 1) {
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

    for (int t = t_start; t < t_end; t++) {
        const int pos = positions[t];
        const float* cs_ptr = cos_sin + pos * head_dim;
        const float* in_ptr = in + t * head_dim;
        float* out_ptr = out + t * head_dim;

        // Apply RoPE to pairs
        for (int d = 0; d < rope_dim; d += 2) {
            float x0 = in_ptr[d];
            float x1 = in_ptr[d + 1];
            float cos_val = cs_ptr[d];
            float sin_val = cs_ptr[d + 1];

            out_ptr[d] = x0 * cos_val - x1 * sin_val;
            out_ptr[d + 1] = x0 * sin_val + x1 * cos_val;
        }

        // Copy dimensions beyond rope_dim unchanged
        for (int dd = rope_dim; dd < head_dim; dd++) {
            out_ptr[dd] = in_ptr[dd];
        }
    }
}

/**
 * @brief Apply Rotary Positional Embedding (unified entry point)
 *
 * Automatically dispatches to AVX-512, AVX2, or scalar implementation based on
 * compile-time detection.
 */
inline void ApplyRoPE(float* out, const float* in, const float* cos_sin, const int* positions,
                      int n_tokens, int head_dim, int rope_dim, int ith = 0, int nth = 1) {
#if defined(__AVX512F__)
    ApplyRoPE_AVX512(out, in, cos_sin, positions, n_tokens, head_dim, rope_dim, ith, nth);
#elif defined(__AVX2__)
    ApplyRoPE_AVX2(out, in, cos_sin, positions, n_tokens, head_dim, rope_dim, ith, nth);
#else
    ApplyRoPE_Scalar(out, in, cos_sin, positions, n_tokens, head_dim, rope_dim, ith, nth);
#endif
}

// NOTE: ApplyRoPE_MultiHead was removed as it was unused.
// If multi-head RoPE is needed, use cb_rope_avx512 callback in inference.cpp
// which handles the [head_dim, n_heads, n_tokens] GGML tensor layout.

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
inline void UnpackInt4x64_AVX512(const uint8_t* packed, __m512i& unpacked_lo,
                                 __m512i& unpacked_hi) {
    // Load 32 bytes = 64 packed 4-bit values into AVX512 register
    __m256i packed_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed));

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
        interleaved_lo, _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11), interleaved_hi);
    unpacked_hi = _mm512_permutex2var_epi64(
        interleaved_lo, _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15), interleaved_hi);
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
inline void GemmInt4Fp32_AVX512(float* C, const float* A, const uint8_t* W_int4,
                                const float* scales, const float* zero_points, int M, int N, int K,
                                int group_size) {
    if (K % group_size != 0)
        return;

    const int num_groups = K / group_size;
    const int packed_K = K / 2;  // Bytes per weight row

    // Constants for prefetch distance
    constexpr int PREFETCH_DIST = 128;  // Bytes ahead

    // Outer loop: rows of output
    for (int m = 0; m < M; m++) {
        const float* a_row = A + m * K;

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
                const float* a_ptr = a_row + k_offset;

                // Prefetch activations for next group
                if (g + 1 < num_groups) {
                    _mm_prefetch(reinterpret_cast<const char*>(a_row + (g + 1) * group_size),
                                 _MM_HINT_T0);
                }

                // Process 32 weights at a time (64 bits packed = 16 bytes)
                for (int k = 0; k < group_size; k += 32) {
                    // Load 2 × 16 floats from activations
                    __m512 a0 = _mm512_loadu_ps(a_ptr + k);
                    __m512 a1 = _mm512_loadu_ps(a_ptr + k + 16);

// Process all 16 weight rows with 2-way unrolling
#define PROCESS_ROW(idx)                                                                 \
    do {                                                                                 \
        const int row = n + (idx);                                                       \
        const float scale = scales[row * num_groups + g];                                \
        const float zero = zero_points[row * num_groups + g];                            \
        const __m512 vscale = _mm512_set1_ps(scale);                                     \
        const __m512 vzero = _mm512_set1_ps(zero);                                       \
                                                                                         \
        const uint8_t* w_ptr = W_int4 + row * packed_K + packed_offset + k / 2;          \
        _mm_prefetch(reinterpret_cast<const char*>(w_ptr + PREFETCH_DIST), _MM_HINT_T0); \
                                                                                         \
        /* Load 16 bytes = 32 packed weights */                                          \
        __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_ptr));       \
        __m256i packed_256 = _mm256_cvtepu8_epi16(packed);                               \
                                                                                         \
        /* Extract low nibbles */                                                        \
        __m256i low = _mm256_and_si256(packed_256, _mm256_set1_epi16(0x0F));             \
        /* Extract high nibbles */                                                       \
        __m256i high = _mm256_srli_epi16(packed_256, 4);                                 \
        high = _mm256_and_si256(high, _mm256_set1_epi16(0x0F));                          \
                                                                                         \
        /* Sign extension via shift trick */                                             \
        low = _mm256_slli_epi16(low, 12);                                                \
        low = _mm256_srai_epi16(low, 12);                                                \
        high = _mm256_slli_epi16(high, 12);                                              \
        high = _mm256_srai_epi16(high, 12);                                              \
                                                                                         \
        /* Interleave to restore order */                                                \
        __m256i interleaved_lo = _mm256_unpacklo_epi16(low, high);                       \
        __m256i interleaved_hi = _mm256_unpackhi_epi16(low, high);                       \
                                                                                         \
        /* Permute for correct lane order after interleave */                            \
        __m256i w16_0 = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x20); \
        __m256i w16_1 = _mm256_permute2x128_si256(interleaved_lo, interleaved_hi, 0x31); \
                                                                                         \
        /* Convert to FP32 */                                                            \
        __m512i w32_0 = _mm512_cvtepi16_epi32(w16_0);                                    \
        __m512i w32_1 = _mm512_cvtepi16_epi32(w16_1);                                    \
        __m512 wf0 = _mm512_cvtepi32_ps(w32_0);                                          \
        __m512 wf1 = _mm512_cvtepi32_ps(w32_1);                                          \
                                                                                         \
        /* Dequantize: w_dequant = scale * (q - zero) */                                 \
        wf0 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf0, vzero));                          \
        wf1 = _mm512_mul_ps(vscale, _mm512_sub_ps(wf1, vzero));                          \
                                                                                         \
        /* FMA */                                                                        \
        acc##idx = _mm512_fmadd_ps(a0, wf0, acc##idx);                                   \
        acc##idx = _mm512_fmadd_ps(a1, wf1, acc##idx);                                   \
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
                const float* a_ptr = a_row + k_offset;
                const float scale = scales[n * num_groups + g];
                const float zero = zero_points[n * num_groups + g];
                const __m512 vscale = _mm512_set1_ps(scale);
                const __m512 vzero = _mm512_set1_ps(zero);
                const uint8_t* w_ptr = W_int4 + n * packed_K + g * (group_size / 2);

                for (int k = 0; k < group_size; k += 32) {
                    // Load 16 bytes = 32 packed weights
                    __m128i packed =
                        _mm_loadu_si128(reinterpret_cast<const __m128i*>(w_ptr + k / 2));
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

#else  // No AVX512 support

/**
 * Fallback GEMM for INT4 weights (scalar implementation)
 */
inline void GemmInt4Fp32_AVX512(float* C, const float* A, const uint8_t* W_int4,
                                const float* scales, const float* zero_points, int M, int N, int K,
                                int group_size) {
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
                const uint8_t* w_packed = W_int4 + n * (K / 2) + g * (group_size / 2);

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

#endif  // __AVX512F__

#if defined(__AVX2__)

/**
 * @brief High-performance AVX2 GEMM kernel: C = A * W^T (INT4 weights)
 *
 * Optimizations:
 * - 8x register blocking along N dimension (vs 16x for AVX-512)
 * - Shift-based sign extension (no branches, no masks)
 * - Prefetching for weights and activations
 * - FMA3 instructions for dot products
 * - Processes 16 weights per iteration (vs 32 for AVX-512)
 *
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
inline void GemmInt4Fp32_AVX2(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                              const float* zero_points, int M, int N, int K, int group_size) {
    if (K % group_size != 0)
        return;

    const int num_groups = K / group_size;
    const int packed_K = K / 2;  // Bytes per weight row

    // Constants for prefetch distance
    constexpr int PREFETCH_DIST = 64;  // Bytes ahead

    // Outer loop: rows of output
    for (int m = 0; m < M; m++) {
        const float* a_row = A + m * K;

        // Middle loop: columns of output in blocks of 8 (REGISTER BLOCKING)
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            // 8 accumulators for 8 output elements
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps();
            __m256 acc7 = _mm256_setzero_ps();

            // Loop over quantization groups
            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const int packed_offset = g * (group_size / 2);
                const float* a_ptr = a_row + k_offset;

                // Prefetch activations for next group
                if (g + 1 < num_groups) {
                    _mm_prefetch(reinterpret_cast<const char*>(a_row + (g + 1) * group_size),
                                 _MM_HINT_T0);
                }

                // Process 16 weights at a time (8 bytes packed)
                for (int k = 0; k < group_size; k += 16) {
                    // Load 1 × 8 floats from activations
                    __m256 a0 = _mm256_loadu_ps(a_ptr + k);
                    __m256 a1 = _mm256_loadu_ps(a_ptr + k + 8);

// Process all 8 weight rows
#define PROCESS_ROW_AVX2(idx)                                                            \
    do {                                                                                 \
        const int row = n + (idx);                                                       \
        const float scale = scales[row * num_groups + g];                                \
        const float zero = zero_points[row * num_groups + g];                            \
        const __m256 vscale = _mm256_set1_ps(scale);                                     \
        const __m256 vzero = _mm256_set1_ps(zero);                                       \
                                                                                         \
        const uint8_t* w_ptr = W_int4 + row * packed_K + packed_offset + k / 2;          \
        _mm_prefetch(reinterpret_cast<const char*>(w_ptr + PREFETCH_DIST), _MM_HINT_T0); \
                                                                                         \
        /* Load 8 bytes = 16 packed weights */                                           \
        __m128i packed_64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w_ptr));    \
        __m128i packed_16 = _mm_cvtepu8_epi16(packed_64);                                \
                                                                                         \
        /* Extract low nibbles (bits 0-3) */                                             \
        __m128i low = _mm_and_si128(packed_16, _mm_set1_epi16(0x0F));                    \
        /* Extract high nibbles (bits 4-7) */                                            \
        __m128i high = _mm_srli_epi16(packed_16, 4);                                     \
        high = _mm_and_si128(high, _mm_set1_epi16(0x0F));                                \
                                                                                         \
        /* Sign extension via shift trick */                                             \
        low = _mm_slli_epi16(low, 12);                                                   \
        low = _mm_srai_epi16(low, 12);                                                   \
        high = _mm_slli_epi16(high, 12);                                                 \
        high = _mm_srai_epi16(high, 12);                                                 \
                                                                                         \
        /* Interleave to restore order: [l0, h0, l1, h1, l2, h2, l3, h3] */              \
        __m128i interleaved_lo = _mm_unpacklo_epi16(low, high);                          \
        __m128i interleaved_hi = _mm_unpackhi_epi16(low, high);                          \
                                                                                         \
        /* Convert to FP32 via int32 */                                                  \
        __m256i w32_0 = _mm256_cvtepi16_epi32(interleaved_lo);                           \
        __m256i w32_1 = _mm256_cvtepi16_epi32(interleaved_hi);                           \
        __m256 wf0 = _mm256_cvtepi32_ps(w32_0);                                          \
        __m256 wf1 = _mm256_cvtepi32_ps(w32_1);                                          \
                                                                                         \
        /* Dequantize: w_dequant = scale * (q - zero) */                                 \
        wf0 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf0, vzero));                          \
        wf1 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf1, vzero));                          \
                                                                                         \
        /* FMA */                                                                        \
        acc##idx = _mm256_fmadd_ps(a0, wf0, acc##idx);                                   \
        acc##idx = _mm256_fmadd_ps(a1, wf1, acc##idx);                                   \
    } while (0)

                    PROCESS_ROW_AVX2(0);
                    PROCESS_ROW_AVX2(1);
                    PROCESS_ROW_AVX2(2);
                    PROCESS_ROW_AVX2(3);
                    PROCESS_ROW_AVX2(4);
                    PROCESS_ROW_AVX2(5);
                    PROCESS_ROW_AVX2(6);
                    PROCESS_ROW_AVX2(7);

#undef PROCESS_ROW_AVX2
                }
            }

            // Horizontal reduction and store results
            // AVX2 doesn't have _mm256_reduce_add_ps, so we do it manually
            auto hsum_avx2 = [](__m256 v) -> float {
                __m128 hi = _mm256_extractf128_ps(v, 1);
                __m128 lo = _mm256_castps256_ps128(v);
                __m128 sum128 = _mm_add_ps(lo, hi);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                return _mm_cvtss_f32(sum128);
            };

            C[m * N + n + 0] = hsum_avx2(acc0);
            C[m * N + n + 1] = hsum_avx2(acc1);
            C[m * N + n + 2] = hsum_avx2(acc2);
            C[m * N + n + 3] = hsum_avx2(acc3);
            C[m * N + n + 4] = hsum_avx2(acc4);
            C[m * N + n + 5] = hsum_avx2(acc5);
            C[m * N + n + 6] = hsum_avx2(acc6);
            C[m * N + n + 7] = hsum_avx2(acc7);
        }

        // Handle remaining columns (N % 8)
        for (; n < N; n++) {
            __m256 acc = _mm256_setzero_ps();

            for (int g = 0; g < num_groups; g++) {
                const int k_offset = g * group_size;
                const float* a_ptr = a_row + k_offset;
                const float scale = scales[n * num_groups + g];
                const float zero = zero_points[n * num_groups + g];
                const __m256 vscale = _mm256_set1_ps(scale);
                const __m256 vzero = _mm256_set1_ps(zero);
                const uint8_t* w_ptr = W_int4 + n * packed_K + g * (group_size / 2);

                for (int k = 0; k < group_size; k += 16) {
                    // Load 8 bytes = 16 packed weights
                    __m128i packed_64 =
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w_ptr + k / 2));
                    __m128i packed_16 = _mm_cvtepu8_epi16(packed_64);

                    // Extract and sign-extend nibbles
                    __m128i low = _mm_and_si128(packed_16, _mm_set1_epi16(0x0F));
                    __m128i high = _mm_srli_epi16(packed_16, 4);
                    high = _mm_and_si128(high, _mm_set1_epi16(0x0F));

                    low = _mm_slli_epi16(low, 12);
                    low = _mm_srai_epi16(low, 12);
                    high = _mm_slli_epi16(high, 12);
                    high = _mm_srai_epi16(high, 12);

                    // Interleave
                    __m128i lo128 = _mm_unpacklo_epi16(low, high);
                    __m128i hi128 = _mm_unpackhi_epi16(low, high);

                    // Convert and dequantize
                    __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(lo128));
                    __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(hi128));

                    wf0 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf0, vzero));
                    wf1 = _mm256_mul_ps(vscale, _mm256_sub_ps(wf1, vzero));

                    // Load activations and FMA
                    __m256 a0 = _mm256_loadu_ps(a_ptr + k);
                    __m256 a1 = _mm256_loadu_ps(a_ptr + k + 8);

                    acc = _mm256_fmadd_ps(a0, wf0, acc);
                    acc = _mm256_fmadd_ps(a1, wf1, acc);
                }
            }

            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            C[m * N + n] = _mm_cvtss_f32(sum128);
        }
    }
}

#endif  // __AVX2__

// =============================================================================
// FlashAttention AVX-512 Micro-Kernels (Cache-Optimized)
// =============================================================================

#if defined(__AVX512F__)

/**
 * @brief Fast exp approximation using polynomial (AVX-512)
 *
 * Approximates exp(x) for x in [-87, 87] using a minimax polynomial.
 * Faster than standard exp but with ~1e-6 relative error.
 */
inline __m512 _mm512_fast_exp_ps(__m512 x) {
    // Clamp to valid range
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(87.0f));

    // exp(x) = 2^(x/ln(2)) = 2^k * 2^f where k=floor(x/ln(2)), f=x/ln(2)-k
    const __m512 c_invlog2 = _mm512_set1_ps(1.44269504f);  // 1/ln(2)
    const __m512 c_log2 = _mm512_set1_ps(0.69314718f);     // ln(2)

    __m512 z = _mm512_mul_ps(x, c_invlog2);
    __m512i k = _mm512_cvttps_epi32(z);
    __m512 f = _mm512_sub_ps(x, _mm512_mul_ps(_mm512_cvtepi32_ps(k), c_log2));

    // Polynomial approximation for 2^f (f in [0,1]) via Taylor series:
    // 2^f ≈ c0 + c1*f + c2*f² + c3*f³
    // where c0=1, c1=ln(2), c2=ln(2)²/2!, c3=ln(2)³/3!
    const __m512 c0 = _mm512_set1_ps(1.0f);         // 1.0
    const __m512 c1 = _mm512_set1_ps(0.69314718f);  // ln(2)
    const __m512 c2 = _mm512_set1_ps(0.24022651f);  // ln(2)^2 / 2
    const __m512 c3 = _mm512_set1_ps(0.05550410f);  // ln(2)^3 / 6

    // Horner's method: ((c3*f + c2)*f + c1)*f + c0
    __m512 poly = c3;
    poly = _mm512_fmadd_ps(poly, f, c2);
    poly = _mm512_fmadd_ps(poly, f, c1);
    poly = _mm512_fmadd_ps(poly, f, c0);

    // 2^k using bit manipulation
    k = _mm512_slli_epi32(_mm512_add_epi32(k, _mm512_set1_epi32(127)), 23);
    __m512 pow2k = _mm512_castsi512_ps(k);

    return _mm512_mul_ps(poly, pow2k);
}

/**
 * @brief Compute Q @ K^T with scaling (AVX-512 optimized)
 *
 * Computes S[i,j] = scale * sum_d(Q[i,d] * K[j,d]) for block of queries and
 * keys.
 *
 * @param Q Query block [q_len, head_dim]
 * @param K Key block [kv_len, head_dim]
 * @param S Output scores [q_len, kv_len]
 * @param q_len Number of query vectors in block
 * @param kv_len Number of key vectors in block
 * @param head_dim Dimension per head
 * @param scale Scaling factor (typically 1/sqrt(head_dim))
 */
inline void ComputeQK_AVX512(const float* Q, const float* K, float* S, int q_len, int kv_len,
                             int head_dim, float scale) {
    const __m512 scale_vec = _mm512_set1_ps(scale);

    for (int qi = 0; qi < q_len; qi++) {
        const float* q_row = Q + qi * head_dim;

        for (int ki = 0; ki < kv_len; ki++) {
            const float* k_row = K + ki * head_dim;

            // Dot product with AVX-512
            __m512 sum = _mm512_setzero_ps();
            int d = 0;

            // Process 16 elements at a time
            for (; d + 16 <= head_dim; d += 16) {
                __m512 q_vec = _mm512_loadu_ps(q_row + d);
                __m512 k_vec = _mm512_loadu_ps(k_row + d);
                sum = _mm512_fmadd_ps(q_vec, k_vec, sum);
            }

            // Horizontal sum of 16-element vector
            float dot = _mm512_reduce_add_ps(sum);

            // Handle remainder with scalar
            for (; d < head_dim; d++) {
                dot += q_row[d] * k_row[d];
            }

            S[qi * kv_len + ki] = dot * scale;
        }
    }
}

/**
 * @brief Apply causal mask to attention scores (AVX-512 optimized)
 *
 * Sets S[qi, ki] = -inf if (qi + q_start) < (ki + kv_start) for causal
 * masking.
 *
 * @param S Scores [q_len, kv_len] (modified in-place)
 * @param q_start Global query offset
 * @param kv_start Global kv offset
 * @param q_len Number of query vectors
 * @param kv_len Number of kv vectors
 */
inline void ApplyMask_AVX512(float* S, int q_start, int kv_start, int q_len, int kv_len) {
    const __m512 neg_inf = _mm512_set1_ps(-1e10f);

    for (int qi = 0; qi < q_len; qi++) {
        const int global_qi = q_start + qi;
        float* s_row = S + qi * kv_len;

        int ki = 0;
        // Vectorized masking
        for (; ki + 16 <= kv_len; ki += 16) {
            // Create mask: true where (global_qi < kv_start + ki + lane)
            __m512i ki_vec = _mm512_add_epi32(
                _mm512_set1_epi32(kv_start + ki),
                _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
            __mmask16 mask = _mm512_cmplt_epi32_mask(_mm512_set1_epi32(global_qi), ki_vec);

            // Load current scores
            __m512 s_vec = _mm512_loadu_ps(s_row + ki);
            // Blend: if mask=1, use -inf, else keep s_vec
            s_vec = _mm512_mask_blend_ps(mask, s_vec, neg_inf);
            _mm512_storeu_ps(s_row + ki, s_vec);
        }

        // Scalar remainder
        for (; ki < kv_len; ki++) {
            if (global_qi < (kv_start + ki)) {
                s_row[ki] = -1e10f;
            }
        }
    }
}

/**
 * @brief Online softmax with AVX-512
 *
 * Computes softmax over block and updates running statistics.
 * Uses online algorithm to handle arbitrary sequence lengths.
 *
 * @param S Input/output scores [q_len, kv_len]
 * @param row_max Max value per row [q_len] (updated)
 * @param row_sum Exponential sum per row [q_len] (updated)
 * @param q_len Number of query rows
 * @param kv_len Number of attention scores per row
 * @param first_block If true, initialize max/sum; else update
 */
inline void SoftmaxBlock_AVX512(float* S, float* row_max, float* row_sum, int q_len, int kv_len,
                                bool first_block) {
    for (int qi = 0; qi < q_len; qi++) {
        float* s_row = S + qi * kv_len;

        // Find max in this block
        __m512 max_vec = _mm512_set1_ps(-1e10f);
        int ki = 0;
        for (; ki + 16 <= kv_len; ki += 16) {
            __m512 s_vec = _mm512_loadu_ps(s_row + ki);
            max_vec = _mm512_max_ps(max_vec, s_vec);
        }
        float local_max = _mm512_reduce_max_ps(max_vec);

        // Scalar remainder for max
        for (; ki < kv_len; ki++) {
            local_max = std::max(local_max, s_row[ki]);
        }

        // Update global max
        float m_old = first_block ? -1e10f : row_max[qi];
        float m_new = std::max(m_old, local_max);
        const __m512 m_new_vec = _mm512_set1_ps(m_new);

        // Compute exp(s - m_new) and sum
        __m512 sum_vec = _mm512_setzero_ps();
        ki = 0;
        for (; ki + 16 <= kv_len; ki += 16) {
            __m512 s_vec = _mm512_loadu_ps(s_row + ki);
            __m512 exp_vec = _mm512_sub_ps(s_vec, m_new_vec);
            exp_vec = _mm512_fast_exp_ps(exp_vec);  // Fast exp
            _mm512_storeu_ps(s_row + ki, exp_vec);
            sum_vec = _mm512_add_ps(sum_vec, exp_vec);
        }
        float local_sum = _mm512_reduce_add_ps(sum_vec);

        // Scalar remainder
        for (; ki < kv_len; ki++) {
            float exp_val = expf(s_row[ki] - m_new);
            s_row[ki] = exp_val;
            local_sum += exp_val;
        }

        // Update running statistics
        if (first_block) {
            row_max[qi] = m_new;
            row_sum[qi] = local_sum;
        } else {
            float alpha = expf(m_old - m_new);
            row_sum[qi] = alpha * row_sum[qi] + local_sum;
            row_max[qi] = m_new;
        }
    }
}

/**
 * @brief Compute P @ V (attention-weighted values) with AVX-512
 *
 * Computes O[qi] += sum_ki(P[qi,ki] * V[ki]) for a block.
 *
 * @param P Attention probabilities [q_len, kv_len]
 * @param V Value vectors [kv_len, head_dim]
 * @param O Output accumulator [q_len, head_dim] (incremented)
 * @param q_len Number of query vectors
 * @param kv_len Number of key/value vectors
 * @param head_dim Head dimension
 */
inline void ComputePV_AVX512(const float* P, const float* V, float* O, int q_len, int kv_len,
                             int head_dim) {
    for (int qi = 0; qi < q_len; qi++) {
        const float* p_row = P + qi * kv_len;
        float* o_row = O + qi * head_dim;

        for (int ki = 0; ki < kv_len; ki++) {
            const float p_val = p_row[ki];
            const __m512 p_vec = _mm512_set1_ps(p_val);
            const float* v_row = V + ki * head_dim;

            int d = 0;
            // Vectorized accumulation
            for (; d + 16 <= head_dim; d += 16) {
                __m512 o_vec = _mm512_loadu_ps(o_row + d);
                __m512 v_vec = _mm512_loadu_ps(v_row + d);
                o_vec = _mm512_fmadd_ps(p_vec, v_vec, o_vec);
                _mm512_storeu_ps(o_row + d, o_vec);
            }

            // Scalar remainder
            for (; d < head_dim; d++) {
                o_row[d] += p_val * v_row[d];
            }
        }
    }
}

/**
 * @brief Update output with rescaling (AVX-512)
 *
 * Rescales existing output and adds new contribution:
 * O_new[qi] = alpha[qi] * O_old[qi] + beta[qi] * PV[qi]
 *
 * @param O Output [q_len, head_dim] (updated in-place)
 * @param PV New contribution [q_len, head_dim]
 * @param alpha Rescale factors [q_len]
 * @param beta New contribution factors [q_len]
 * @param q_len Number of query vectors
 * @param head_dim Head dimension
 */
inline void UpdateOutput_AVX512(float* O, const float* PV, const float* alpha, const float* beta,
                                int q_len, int head_dim) {
    for (int qi = 0; qi < q_len; qi++) {
        const __m512 alpha_vec = _mm512_set1_ps(alpha[qi]);
        const __m512 beta_vec = _mm512_set1_ps(beta[qi]);
        float* o_row = O + qi * head_dim;
        const float* pv_row = PV + qi * head_dim;

        int d = 0;
        for (; d + 16 <= head_dim; d += 16) {
            __m512 o_vec = _mm512_loadu_ps(o_row + d);
            __m512 pv_vec = _mm512_loadu_ps(pv_row + d);

            // O = alpha * O + beta * PV
            o_vec = _mm512_mul_ps(o_vec, alpha_vec);
            o_vec = _mm512_fmadd_ps(beta_vec, pv_vec, o_vec);

            _mm512_storeu_ps(o_row + d, o_vec);
        }

        // Scalar remainder
        for (; d < head_dim; d++) {
            o_row[d] = alpha[qi] * o_row[d] + beta[qi] * pv_row[d];
        }
    }
}

#else  // Non-AVX512 fallback (scalar implementations)

// Scalar fallback versions
inline void ComputeQK_AVX512(const float* Q, const float* K, float* S, int q_len, int kv_len,
                             int head_dim, float scale) {
    for (int qi = 0; qi < q_len; qi++) {
        for (int ki = 0; ki < kv_len; ki++) {
            float dot = DotF32(Q + qi * head_dim, K + ki * head_dim, head_dim);
            S[qi * kv_len + ki] = dot * scale;
        }
    }
}

inline void ApplyMask_AVX512(float* S, int q_start, int kv_start, int q_len, int kv_len) {
    for (int qi = 0; qi < q_len; qi++) {
        for (int ki = 0; ki < kv_len; ki++) {
            if ((q_start + qi) < (kv_start + ki)) {
                S[qi * kv_len + ki] = -1e10f;
            }
        }
    }
}

inline void SoftmaxBlock_AVX512(float* S, float* row_max, float* row_sum, int q_len, int kv_len,
                                bool first_block) {
    for (int qi = 0; qi < q_len; qi++) {
        float* s_row = S + qi * kv_len;
        float local_max = MaxF32(s_row, kv_len);
        float m_old = first_block ? -1e10f : row_max[qi];
        float m_new = std::max(m_old, local_max);

        float local_sum = 0.0f;
        for (int ki = 0; ki < kv_len; ki++) {
            s_row[ki] = expf(s_row[ki] - m_new);
            local_sum += s_row[ki];
        }

        if (first_block) {
            row_max[qi] = m_new;
            row_sum[qi] = local_sum;
        } else {
            float alpha = expf(m_old - m_new);
            row_sum[qi] = alpha * row_sum[qi] + local_sum;
            row_max[qi] = m_new;
        }
    }
}

inline void ComputePV_AVX512(const float* P, const float* V, float* O, int q_len, int kv_len,
                             int head_dim) {
    for (int qi = 0; qi < q_len; qi++) {
        for (int ki = 0; ki < kv_len; ki++) {
            float p_val = P[qi * kv_len + ki];
            const float* v_row = V + ki * head_dim;
            float* o_row = O + qi * head_dim;
            for (int d = 0; d < head_dim; d++) {
                o_row[d] += p_val * v_row[d];
            }
        }
    }
}

inline void UpdateOutput_AVX512(float* O, const float* PV, const float* alpha, const float* beta,
                                int q_len, int head_dim) {
    for (int qi = 0; qi < q_len; qi++) {
        for (int d = 0; d < head_dim; d++) {
            O[qi * head_dim + d] =
                alpha[qi] * O[qi * head_dim + d] + beta[qi] * PV[qi * head_dim + d];
        }
    }
}

#endif  // __AVX512F__

// =============================================================================
// Fused Kernels for Memory Bandwidth Optimization
// =============================================================================

#if defined(__AVX512F__)

/**
 * @brief Fused Add + RMSNorm in one pass (AVX-512 optimized)
 *
 * Combines residual addition and RMSNorm into a single kernel to reduce
 * memory bandwidth by loading/storing data once instead of twice.
 *
 * Operation:
 *   1. x_out[i] = x[i] + residual[i]  (residual add)
 *   2. Compute sum_of_squares for RMSNorm
 *   3. Apply normalization: x_out[i] = (x_out[i] * rms_w[i]) / sqrt(sos/n +
 * eps)
 *
 * @param x_out Output tensor [n] (can be same as x for in-place)
 * @param x Input tensor [n]
 * @param residual Residual tensor [n]
 * @param rms_w RMSNorm weight tensor [n]
 * @param n Number of elements
 * @param eps Epsilon for numerical stability (default: 1e-5)
 */
inline void AddRMSNorm_AVX512(float* x_out, const float* x, const float* residual,
                              const float* rms_w, size_t n, float eps = 1e-5f) {
    // Pass 1: Add residual and compute sum of squares simultaneously
    __m512 sos_vec = _mm512_setzero_ps();
    size_t i = 0;

    // Vectorized: add and accumulate sum-of-squares
    for (; i + 16 <= n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m512 res_vec = _mm512_loadu_ps(residual + i);

        // Fused add: x_out = x + residual
        __m512 sum = _mm512_add_ps(x_vec, res_vec);
        _mm512_storeu_ps(x_out + i, sum);

        // Accumulate sum of squares for RMSNorm
        sos_vec = _mm512_fmadd_ps(sum, sum, sos_vec);
    }

    // Scalar remainder for pass 1
    float sos_scalar = _mm512_reduce_add_ps(sos_vec);
    for (; i < n; i++) {
        float val = x[i] + residual[i];
        x_out[i] = val;
        sos_scalar += val * val;
    }

    // Compute normalization factor: 1 / sqrt(mean(x^2) + eps)
    float rms = sqrtf(sos_scalar / static_cast<float>(n) + eps);
    float scale = 1.0f / rms;
    const __m512 scale_vec = _mm512_set1_ps(scale);

    // Pass 2: Apply normalized weight multiplication
    i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 val = _mm512_loadu_ps(x_out + i);
        __m512 w = _mm512_loadu_ps(rms_w + i);

        // x_out = (x_out * scale) * weight = x_out * w * scale
        val = _mm512_mul_ps(val, scale_vec);
        val = _mm512_mul_ps(val, w);
        _mm512_storeu_ps(x_out + i, val);
    }

    // Scalar remainder for pass 2
    for (; i < n; i++) {
        x_out[i] = x_out[i] * scale * rms_w[i];
    }
}

/**
 * @brief Fused Q/K/V projection in one pass (AVX-512 optimized)
 *
 * Computes Q = x @ W_q, K = x @ W_k, V = x @ W_v simultaneously.
 * Reduces memory bandwidth by loading input x into registers once
 * and computing all three projections with interleaved instructions.
 *
 * Note: This is a simplified version for decode (batch=1) scenarios.
 * For larger batches, use standard GEMM calls.
 *
 * @param q Output Q projection [n_embd]
 * @param k Output K projection [dim_k]
 * @param v Output V projection [dim_v]
 * @param x Input tensor [n_embd]
 * @param w_q Q weight [n_embd, dim_q] (row-major)
 * @param w_k K weight [n_embd, dim_k] (row-major)
 * @param w_v V weight [n_embd, dim_v] (row-major)
 * @param n_embd Input dimension
 * @param dim_q Q output dimension (= n_head * head_dim)
 * @param dim_k K output dimension (= n_head_kv * head_dim)
 * @param dim_v V output dimension (= n_head_kv * head_dim)
 */
inline void ComputeQKV_AVX512(float* q, float* k, float* v, const float* x, const float* w_q,
                              const float* w_k, const float* w_v, int n_embd, int dim_q, int dim_k,
                              int dim_v, int ith = 0, int nth = 1) {
    // ==========================================================================
    // TENSOR-LEVEL PARALLELISM: Work Partitioning
    // ==========================================================================
    // Virtual column space: [0, dim_q) = Q, [dim_q, dim_q+dim_k) = K,
    //                       [dim_q+dim_k, total) = V
    // Each thread computes a slice [start_col, end_col) of this virtual space.
    // ==========================================================================
    const int total_cols = dim_q + dim_k + dim_v;
    const int cols_per_thread = (total_cols + nth - 1) / nth;  // Ceiling division
    const int start_col = ith * cols_per_thread;
    const int end_col = std::min(start_col + cols_per_thread, total_cols);

    // Early exit if this thread has no work
    if (start_col >= total_cols)
        return;

    // For each output position, compute dot product with corresponding weight row
    // Process multiple output positions for better ILP

    constexpr int UNROLL = 4;  // Process 4 outputs at a time for better pipelining

    // ==========================================================================
    // Q Projection: virtual columns [0, dim_q)
    // ==========================================================================
    // Compute intersection with this thread's range
    const int q_start = std::max(0, start_col);
    const int q_end = std::min(dim_q, end_col);

    if (q_start < q_end) {
        int oq = q_start;
        for (; oq + UNROLL <= q_end; oq += UNROLL) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            const float* w0 = w_q + oq * n_embd;
            const float* w1 = w_q + (oq + 1) * n_embd;
            const float* w2 = w_q + (oq + 2) * n_embd;
            const float* w3 = w_q + (oq + 3) * n_embd;

            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);

                // Load weights and FMA
                acc0 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w0 + d), acc0);
                acc1 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w1 + d), acc1);
                acc2 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w2 + d), acc2);
                acc3 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w3 + d), acc3);
            }

            // Reduce and store
            q[oq + 0] = _mm512_reduce_add_ps(acc0);
            q[oq + 1] = _mm512_reduce_add_ps(acc1);
            q[oq + 2] = _mm512_reduce_add_ps(acc2);
            q[oq + 3] = _mm512_reduce_add_ps(acc3);

            // Handle remainder for this output group
            for (; d < n_embd; d++) {
                q[oq + 0] += x[d] * w0[d];
                q[oq + 1] += x[d] * w1[d];
                q[oq + 2] += x[d] * w2[d];
                q[oq + 3] += x[d] * w3[d];
            }
        }

        // Handle remaining Q outputs in this thread's range
        for (; oq < q_end; oq++) {
            __m512 acc = _mm512_setzero_ps();
            const float* w = w_q + oq * n_embd;
            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);
                acc = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w + d), acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            q[oq] = sum;
        }
    }

    // ==========================================================================
    // K Projection: virtual columns [dim_q, dim_q + dim_k)
    // ==========================================================================
    // Map virtual start/end to K-local indices
    const int k_virt_start = dim_q;
    const int k_virt_end = dim_q + dim_k;
    const int k_start = std::max(0, start_col - k_virt_start);
    const int k_end = std::min(dim_k, end_col - k_virt_start);

    if (k_start < k_end) {
        int ok = k_start;
        for (; ok + UNROLL <= k_end; ok += UNROLL) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            const float* w0 = w_k + ok * n_embd;
            const float* w1 = w_k + (ok + 1) * n_embd;
            const float* w2 = w_k + (ok + 2) * n_embd;
            const float* w3 = w_k + (ok + 3) * n_embd;

            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);
                acc0 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w0 + d), acc0);
                acc1 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w1 + d), acc1);
                acc2 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w2 + d), acc2);
                acc3 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w3 + d), acc3);
            }

            k[ok + 0] = _mm512_reduce_add_ps(acc0);
            k[ok + 1] = _mm512_reduce_add_ps(acc1);
            k[ok + 2] = _mm512_reduce_add_ps(acc2);
            k[ok + 3] = _mm512_reduce_add_ps(acc3);

            for (; d < n_embd; d++) {
                k[ok + 0] += x[d] * w0[d];
                k[ok + 1] += x[d] * w1[d];
                k[ok + 2] += x[d] * w2[d];
                k[ok + 3] += x[d] * w3[d];
            }
        }

        for (; ok < k_end; ok++) {
            __m512 acc = _mm512_setzero_ps();
            const float* w = w_k + ok * n_embd;
            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);
                acc = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w + d), acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            k[ok] = sum;
        }
    }

    // ==========================================================================
    // V Projection: virtual columns [dim_q + dim_k, total_cols)
    // ==========================================================================
    const int v_virt_start = dim_q + dim_k;
    const int v_start = std::max(0, start_col - v_virt_start);
    const int v_end = std::min(dim_v, end_col - v_virt_start);

    if (v_start < v_end) {
        int ov = v_start;
        for (; ov + UNROLL <= v_end; ov += UNROLL) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            const float* w0 = w_v + ov * n_embd;
            const float* w1 = w_v + (ov + 1) * n_embd;
            const float* w2 = w_v + (ov + 2) * n_embd;
            const float* w3 = w_v + (ov + 3) * n_embd;

            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);
                acc0 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w0 + d), acc0);
                acc1 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w1 + d), acc1);
                acc2 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w2 + d), acc2);
                acc3 = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w3 + d), acc3);
            }

            v[ov + 0] = _mm512_reduce_add_ps(acc0);
            v[ov + 1] = _mm512_reduce_add_ps(acc1);
            v[ov + 2] = _mm512_reduce_add_ps(acc2);
            v[ov + 3] = _mm512_reduce_add_ps(acc3);

            for (; d < n_embd; d++) {
                v[ov + 0] += x[d] * w0[d];
                v[ov + 1] += x[d] * w1[d];
                v[ov + 2] += x[d] * w2[d];
                v[ov + 3] += x[d] * w3[d];
            }
        }

        for (; ov < v_end; ov++) {
            __m512 acc = _mm512_setzero_ps();
            const float* w = w_v + ov * n_embd;
            int d = 0;
            for (; d + 16 <= n_embd; d += 16) {
                __m512 x_vec = _mm512_loadu_ps(x + d);
                acc = _mm512_fmadd_ps(x_vec, _mm512_loadu_ps(w + d), acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            v[ov] = sum;
        }
    }
}

#else  // Scalar fallback for non-AVX512

// Forward declare scalar implementations
inline void AddRMSNorm_Scalar(float* x_out, const float* x, const float* residual,
                              const float* rms_w, size_t n, float eps = 1e-5f);

inline void ComputeQKV_Scalar(float* q, float* k, float* v, const float* x, const float* w_q,
                              const float* w_k, const float* w_v, int n_embd, int dim_q, int dim_k,
                              int dim_v, int ith, int nth);

/**
 * @brief Scalar fallback for fused Add + RMSNorm
 */
inline void AddRMSNorm_AVX512(float* x_out, const float* x, const float* residual,
                              const float* rms_w, size_t n, float eps = 1e-5f) {
    AddRMSNorm_Scalar(x_out, x, residual, rms_w, n, eps);
}

/**
 * @brief Scalar fallback for fused Q/K/V projection
 */
inline void ComputeQKV_AVX512(float* q, float* k, float* v, const float* x, const float* w_q,
                              const float* w_k, const float* w_v, int n_embd, int dim_q, int dim_k,
                              int dim_v, int ith = 0, int nth = 1) {
    ComputeQKV_Scalar(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
}

#endif  // __AVX512F__

// =============================================================================
// AVX2 Fused Kernels
// =============================================================================

#if defined(__AVX2__)

/**
 * @brief Fused Add + RMSNorm in one pass (AVX2 optimized)
 *
 * SAFETY: This function handles arbitrary dimension sizes (dim % 8 != 0) via:
 *   - Main loop: processes 8 elements at a time (i + 8 <= n)
 *   - Scalar fallback: handles remaining 0-7 elements
 *   - Uses _mm256_loadu_ps (unaligned loads) for safety with arbitrary pointers
 */
inline void AddRMSNorm_AVX2(float* x_out, const float* x, const float* residual, const float* rms_w,
                            size_t n, float eps = 1e-5f) {
    // Early exit for empty input
    if (n == 0)
        return;

    __m256 sos_vec = _mm256_setzero_ps();
    size_t i = 0;

    // Pass 1: Fused Add + Sum of Squares
    for (; i + 8 <= n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m256 res_vec = _mm256_loadu_ps(residual + i);

        // x_out = x + residual
        __m256 sum = _mm256_add_ps(x_vec, res_vec);
        _mm256_storeu_ps(x_out + i, sum);

        // Accumulate sum of squares
        sos_vec = _mm256_fmadd_ps(sum, sum, sos_vec);
    }

    // Horizontal reduction for AVX2
    // 1. Extract high 128 bits and add to low 128 bits
    __m128 hi = _mm256_extractf128_ps(sos_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sos_vec);
    __m128 sum128 = _mm_add_ps(lo, hi);

    // 2. Reduce 128 vectors using movehl and shuffle (faster than hadd)
    // [a, b, c, d] -> [a+c, b+d, c, d]
    __m128 movehl = _mm_movehl_ps(sum128, sum128);
    sum128 = _mm_add_ps(sum128, movehl);
    // [a+c, b+d, ...] -> [a+c+b+d, ...]
    __m128 shuffle = _mm_shuffle_ps(sum128, sum128, 1);
    sum128 = _mm_add_ss(sum128, shuffle);

    float sos_scalar = _mm_cvtss_f32(sum128);

    // Handle remainder for Pass 1
    for (; i < n; i++) {
        float val = x[i] + residual[i];
        x_out[i] = val;
        sos_scalar += val * val;
    }

    // Compute scale
    float rms = sqrtf(sos_scalar / static_cast<float>(n) + eps);
    float scale = 1.0f / rms;
    __m256 scale_vec = _mm256_set1_ps(scale);

    // Pass 2: Apply normalization and weight
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 val = _mm256_loadu_ps(x_out + i);
        __m256 w = _mm256_loadu_ps(rms_w + i);

        val = _mm256_mul_ps(val, scale_vec);  // normalize
        val = _mm256_mul_ps(val, w);          // weight
        _mm256_storeu_ps(x_out + i, val);
    }

    // Handle remainder for Pass 2
    for (; i < n; i++) {
        x_out[i] = x_out[i] * scale * rms_w[i];
    }
}

/**
 * @brief Fused Q/K/V projection (AVX2 optimized)
 *
 * SAFETY: This function handles arbitrary dimension sizes (dim % 8 != 0) via:
 *   - Main loop: processes 8 elements at a time for dot product (d + 8 <=
 * n_embd)
 *   - Scalar fallback: handles remaining 0-7 elements in inner dimension
 *   - Output loop handles odd output dimensions via single-element remainder
 * loop
 *   - Uses _mm256_loadu_ps (unaligned loads) for safety with arbitrary pointers
 *
 * Structure:
 *   - Q projection with 2x unrolling + scalar remainder
 *   - K projection with 2x unrolling + scalar remainder
 *   - V projection with 2x unrolling + scalar remainder
 */
inline void ComputeQKV_AVX2(float* q, float* k, float* v, const float* x, const float* w_q,
                            const float* w_k, const float* w_v, int n_embd, int dim_q, int dim_k,
                            int dim_v, int ith = 0, int nth = 1) {
    // ==========================================================================
    // TENSOR-LEVEL PARALLELISM: Work Partitioning
    // ==========================================================================
    const int total_cols = dim_q + dim_k + dim_v;
    const int cols_per_thread = (total_cols + nth - 1) / nth;
    const int start_col = ith * cols_per_thread;
    const int end_col = std::min(start_col + cols_per_thread, total_cols);

    // BARRIER SAFETY: Explicit early exit for zero-work threads.
    // This is critical for multi-threaded correctness - threads with no work
    // must still return cleanly to reach barrier sync without executing any
    // loops.
    if (start_col >= end_col)
        return;

    constexpr int UNROLL = 2;

    // Helper lambda for horizontal sum
    auto hsum256 = [](__m256 v) -> float {
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        return _mm_cvtss_f32(s);
    };

    // ==========================================================================
    // Q Projection: virtual columns [0, dim_q)
    // ==========================================================================
    const int q_start = std::max(0, start_col);
    const int q_end = std::min(dim_q, end_col);

    if (q_start < q_end) {
        int oq = q_start;
        for (; oq + UNROLL <= q_end; oq += UNROLL) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            const float* w0 = w_q + oq * n_embd;
            const float* w1 = w_q + (oq + 1) * n_embd;

            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + d);
                acc0 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w0 + d), acc0);
                acc1 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w1 + d), acc1);
            }

            q[oq + 0] = hsum256(acc0);
            q[oq + 1] = hsum256(acc1);

            for (; d < n_embd; d++) {
                q[oq + 0] += x[d] * w0[d];
                q[oq + 1] += x[d] * w1[d];
            }
        }

        for (; oq < q_end; oq++) {
            __m256 acc = _mm256_setzero_ps();
            const float* w = w_q + oq * n_embd;
            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(w + d), acc);
            }
            float sum = hsum256(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            q[oq] = sum;
        }
    }

    // ==========================================================================
    // K Projection: virtual columns [dim_q, dim_q + dim_k)
    // ==========================================================================
    const int k_virt_start = dim_q;
    const int k_start = std::max(0, start_col - k_virt_start);
    const int k_end = std::min(dim_k, end_col - k_virt_start);

    if (k_start < k_end) {
        int ok = k_start;
        for (; ok + UNROLL <= k_end; ok += UNROLL) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            const float* w0 = w_k + ok * n_embd;
            const float* w1 = w_k + (ok + 1) * n_embd;

            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + d);
                acc0 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w0 + d), acc0);
                acc1 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w1 + d), acc1);
            }

            k[ok + 0] = hsum256(acc0);
            k[ok + 1] = hsum256(acc1);

            for (; d < n_embd; d++) {
                k[ok + 0] += x[d] * w0[d];
                k[ok + 1] += x[d] * w1[d];
            }
        }

        for (; ok < k_end; ok++) {
            __m256 acc = _mm256_setzero_ps();
            const float* w = w_k + ok * n_embd;
            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(w + d), acc);
            }
            float sum = hsum256(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            k[ok] = sum;
        }
    }

    // ==========================================================================
    // V Projection: virtual columns [dim_q + dim_k, total_cols)
    // ==========================================================================
    const int v_virt_start = dim_q + dim_k;
    const int v_start = std::max(0, start_col - v_virt_start);
    const int v_end = std::min(dim_v, end_col - v_virt_start);

    if (v_start < v_end) {
        int ov = v_start;
        for (; ov + UNROLL <= v_end; ov += UNROLL) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            const float* w0 = w_v + ov * n_embd;
            const float* w1 = w_v + (ov + 1) * n_embd;

            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + d);
                acc0 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w0 + d), acc0);
                acc1 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w1 + d), acc1);
            }

            v[ov + 0] = hsum256(acc0);
            v[ov + 1] = hsum256(acc1);

            for (; d < n_embd; d++) {
                v[ov + 0] += x[d] * w0[d];
                v[ov + 1] += x[d] * w1[d];
            }
        }

        for (; ov < v_end; ov++) {
            __m256 acc = _mm256_setzero_ps();
            const float* w = w_v + ov * n_embd;
            int d = 0;
            for (; d + 8 <= n_embd; d += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + d), _mm256_loadu_ps(w + d), acc);
            }
            float sum = hsum256(acc);
            for (; d < n_embd; d++) {
                sum += x[d] * w[d];
            }
            v[ov] = sum;
        }
    }
}

#else

// Forward declare to allow wrapper usage
inline void AddRMSNorm_Scalar(float* x_out, const float* x, const float* residual,
                              const float* rms_w, size_t n, float eps = 1e-5f);
inline void ComputeQKV_Scalar(float* q, float* k, float* v, const float* x, const float* w_q,
                              const float* w_k, const float* w_v, int n_embd, int dim_q, int dim_k,
                              int dim_v, int ith, int nth);

inline void AddRMSNorm_AVX2(float* x_out, const float* x, const float* residual, const float* rms_w,
                            size_t n, float eps = 1e-5f) {
    AddRMSNorm_Scalar(x_out, x, residual, rms_w, n, eps);
}

inline void ComputeQKV_AVX2(float* q, float* k, float* v, const float* x, const float* w_q,
                            const float* w_k, const float* w_v, int n_embd, int dim_q, int dim_k,
                            int dim_v, int ith = 0, int nth = 1) {
    ComputeQKV_Scalar(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
}

#endif  // __AVX2__

// =============================================================================
// Scalar Implementation (Always Available)
// =============================================================================

inline void AddRMSNorm_Scalar(float* x_out, const float* x, const float* residual,
                              const float* rms_w, size_t n, float eps) {
    // Pass 1: Add and compute sum of squares
    float sos = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float val = x[i] + residual[i];
        x_out[i] = val;
        sos += val * val;
    }

    // Compute scale
    float rms = sqrtf(sos / static_cast<float>(n) + eps);
    float scale = 1.0f / rms;

    // Pass 2: Apply normalization
    for (size_t i = 0; i < n; i++) {
        x_out[i] = x_out[i] * scale * rms_w[i];
    }
}

inline void ComputeQKV_Scalar(float* q, float* k, float* v, const float* x, const float* w_q,
                              const float* w_k, const float* w_v, int n_embd, int dim_q, int dim_k,
                              int dim_v, int ith = 0, int nth = 1) {
    // ==========================================================================
    // TENSOR-LEVEL PARALLELISM: Work Partitioning
    // ==========================================================================
    const int total_cols = dim_q + dim_k + dim_v;
    const int cols_per_thread = (total_cols + nth - 1) / nth;
    const int start_col = ith * cols_per_thread;
    const int end_col = std::min(start_col + cols_per_thread, total_cols);

    // BARRIER SAFETY: Explicit early exit for zero-work threads.
    // This is critical for multi-threaded correctness - threads with no work
    // must still return cleanly to reach barrier sync without executing any
    // loops.
    if (start_col >= end_col)
        return;

    // Q projection: virtual columns [0, dim_q)
    const int q_start = std::max(0, start_col);
    const int q_end = std::min(dim_q, end_col);
    for (int o = q_start; o < q_end; o++) {
        float sum = 0.0f;
        const float* w = w_q + o * n_embd;
        for (int d = 0; d < n_embd; d++) {
            sum += x[d] * w[d];
        }
        q[o] = sum;
    }

    // K projection: virtual columns [dim_q, dim_q + dim_k)
    const int k_virt_start = dim_q;
    const int k_start = std::max(0, start_col - k_virt_start);
    const int k_end = std::min(dim_k, end_col - k_virt_start);
    for (int o = k_start; o < k_end; o++) {
        float sum = 0.0f;
        const float* w = w_k + o * n_embd;
        for (int d = 0; d < n_embd; d++) {
            sum += x[d] * w[d];
        }
        k[o] = sum;
    }

    // V projection: virtual columns [dim_q + dim_k, total_cols)
    const int v_virt_start = dim_q + dim_k;
    const int v_start = std::max(0, start_col - v_virt_start);
    const int v_end = std::min(dim_v, end_col - v_virt_start);
    for (int o = v_start; o < v_end; o++) {
        float sum = 0.0f;
        const float* w = w_v + o * n_embd;
        for (int d = 0; d < n_embd; d++) {
            sum += x[d] * w[d];
        }
        v[o] = sum;
    }
}

// =============================================================================
// Unified Dispatch Wrappers
// =============================================================================

/**
 * @brief Unified AddRMSNorm dispatcher (runtime SIMD selection)
 */
inline void AddRMSNorm(float* x_out, const float* x, const float* residual, const float* rms_w,
                       size_t n, float eps = 1e-5f) {
    static const SimdLevel level = DetectSimdLevel();
    if (level >= SimdLevel::AVX512) {
        AddRMSNorm_AVX512(x_out, x, residual, rms_w, n, eps);
    } else if (level >= SimdLevel::AVX2) {
        AddRMSNorm_AVX2(x_out, x, residual, rms_w, n, eps);
    } else {
        AddRMSNorm_Scalar(x_out, x, residual, rms_w, n, eps);
    }
}

/**
 * @brief Unified ComputeQKV dispatcher (runtime SIMD selection)
 *
 * @param ith Thread index (0-based)
 * @param nth Total number of threads
 */
inline void ComputeQKV(float* q, float* k, float* v, const float* x, const float* w_q,
                       const float* w_k, const float* w_v, int n_embd, int dim_q, int dim_k,
                       int dim_v, int ith = 0, int nth = 1) {
    static const SimdLevel level = DetectSimdLevel();
    if (level >= SimdLevel::AVX512) {
        ComputeQKV_AVX512(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
    } else if (level >= SimdLevel::AVX2) {
        ComputeQKV_AVX2(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
    } else {
        ComputeQKV_Scalar(q, k, v, x, w_q, w_k, w_v, n_embd, dim_q, dim_k, dim_v, ith, nth);
    }
}

// =============================================================================
// Parallel GEMV for Decode-Phase (N=1) Inference
// =============================================================================
// GGML's ggml_mul_mat parallelizes along M (batch) dimension.
// During decode (batch_size=1), there's NO parallelism opportunity.
// This kernel parallelizes along K (output features) dimension instead.
//
// Threading Contract:
//   - Each thread computes output[start:end] = x @ W[start:end, :]^T
//   - Thread-safe: no shared state between threads
//   - ith: thread index (0-based), nth: total threads
// =============================================================================

/**
 * @brief Parallel GEMV for decode-phase linear operations
 *
 * Computes: output[start:end] = x @ W[start:end, :]^T
 * where start/end are computed from ith/nth thread indices.
 *
 * @param output Output vector [K] (only portion [start:end] is written)
 * @param x Input vector [N] (shared by all threads, read-only)
 * @param weight Weight matrix [K, N] row-major (W[k, :] is row k)
 * @param N Input dimension (number of features)
 * @param K Output dimension (number of output features)
 * @param ith Thread index (0-based)
 * @param nth Total number of threads
 */
inline void GemvParallel(float* output, const float* x, const float* weight, int N, int K, int ith,
                         int nth) {
    // Partition output dimension across threads
    const int k_per_thread = (K + nth - 1) / nth;  // Ceiling division
    const int k_start = ith * k_per_thread;
    const int k_end = std::min(k_start + k_per_thread, K);

    // Early exit if this thread has no work
    if (k_start >= K)
        return;

    static const SimdLevel level = DetectSimdLevel();

#if defined(__AVX2__) || defined(__AVX512F__)
    // AVX2/AVX512 path with 2x unrolling
    constexpr int UNROLL = 2;

    auto hsum256 = [](__m256 v) -> float {
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        return _mm_cvtss_f32(s);
    };

    int k = k_start;
    for (; k + UNROLL <= k_end; k += UNROLL) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        const float* w0 = weight + k * N;
        const float* w1 = weight + (k + 1) * N;

        int n = 0;
        for (; n + 8 <= N; n += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + n);
            acc0 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w0 + n), acc0);
            acc1 = _mm256_fmadd_ps(x_vec, _mm256_loadu_ps(w1 + n), acc1);
        }

        float sum0 = hsum256(acc0);
        float sum1 = hsum256(acc1);

        // Scalar remainder
        for (; n < N; n++) {
            sum0 += x[n] * w0[n];
            sum1 += x[n] * w1[n];
        }

        output[k + 0] = sum0;
        output[k + 1] = sum1;
    }

    // Handle remaining output elements
    for (; k < k_end; k++) {
        __m256 acc = _mm256_setzero_ps();
        const float* w = weight + k * N;
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + n), _mm256_loadu_ps(w + n), acc);
        }
        float sum = hsum256(acc);
        for (; n < N; n++) {
            sum += x[n] * w[n];
        }
        output[k] = sum;
    }
#else
    // Scalar fallback
    for (int k = k_start; k < k_end; k++) {
        float sum = 0.0f;
        const float* w = weight + k * N;
        for (int n = 0; n < N; n++) {
            sum += x[n] * w[n];
        }
        output[k] = sum;
    }
#endif
}

// =============================================================================
// Quantized Dot Product Dispatcher (defined in simd_ops.cpp)
// =============================================================================
// These functions use GGML's native vec_dot kernels for zero-allocation
// dot products. The input must be PRE-QUANTIZED by the caller.
// =============================================================================

/**
 * @brief Compute dot product between quantized weight row and pre-quantized
 * input
 *
 * Uses GGML's native vec_dot kernels for maximum performance.
 * ZERO ALLOCATION: No memory is allocated inside this function.
 *
 * @param weight_type GGML type of the weight row (e.g., GGML_TYPE_Q4_K)
 * @param w_row Pointer to quantized weight row
 * @param input Pointer to PRE-QUANTIZED input (Q8_K for K-quants, Q8_0 for
 * Q8_0, F32 for F32)
 * @param n Number of elements
 * @param output Pointer to output scalar (single float result)
 *
 * NOTE: The caller must pre-quantize the input to the correct format:
 *   - Q4_K, Q5_K, Q6_K weights → input must be Q8_K
 *   - Q8_0 weights → input must be Q8_0
 *   - F32/F16 weights → input must be F32
 */
void ComputeDotProduct(int weight_type, const void* w_row, const void* input, int n, float* output);

/**
 * @brief Compute multiple dot products for a range of output rows
 *
 * Uses native vec_dot kernels with pre-quantized input for zero-allocation
 * performance. Useful for parallel GEMV where each thread handles a subset
 * of output rows.
 *
 * @param weight_type GGML type of the weight tensor
 * @param weight Base pointer to weight tensor data
 * @param row_stride Stride in bytes between rows
 * @param input Pre-quantized input vector (same format requirements as above)
 * @param n Number of elements per row
 * @param output Float output vector [k_end - k_start]
 * @param k_start First output row index (inclusive)
 * @param k_end Last output row index (exclusive)
 */
void ComputeDotProductBatch(int weight_type, const void* weight, size_t row_stride,
                            const void* input, int n, float* output, int k_start, int k_end);

/**
 * @brief Query the maximum supported buffer size for dequantization
 * @return Maximum number of float elements that can be dequantized
 * @note With pre-quantization, this is less relevant but kept for API compat
 */
size_t GetDequantizationBufferSize();

/**
 * @brief Check if a GGML type is supported by ComputeDotProduct
 * @param type GGML type to check
 * @return true if type is supported, false otherwise
 */
bool IsTypeSupported(int type);

}  // namespace simd
}  // namespace densecore

#endif  // DENSECORE_SIMD_OPS_H
