/**
 * @file macros.h
 * @brief DenseCore API export macros for shared library builds
 *
 * This file is part of DenseCore Public API (HAL).
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * Usage:
 * - Define DENSECORE_SHARED when building as a shared library
 * - Define DENSECORE_BUILDING when compiling the library itself
 * - Leave undefined when linking statically
 */

#ifndef DENSECORE_HAL_MACROS_H
#define DENSECORE_HAL_MACROS_H

// =============================================================================
// Platform Detection
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
#define DENSECORE_PLATFORM_WINDOWS 1
#elif defined(__APPLE__)
#define DENSECORE_PLATFORM_APPLE 1
#elif defined(__linux__)
#define DENSECORE_PLATFORM_LINUX 1
#elif defined(__ANDROID__)
#define DENSECORE_PLATFORM_ANDROID 1
#else
#define DENSECORE_PLATFORM_UNKNOWN 1
#endif

// =============================================================================
// API Export/Import Macros
// =============================================================================

/**
 * @def DENSECORE_API
 * @brief Marks symbols for export in shared library builds
 *
 * On Windows: Uses __declspec(dllexport/dllimport)
 * On Unix: Uses visibility("default")
 *
 * For static builds, this expands to nothing.
 */
#ifdef DENSECORE_SHARED
#ifdef DENSECORE_PLATFORM_WINDOWS
#ifdef DENSECORE_BUILDING
// Symbol visibility
#if defined(_WIN32)
#if defined(DENSECORE_BUILD_SHARED)
#define DENSECORE_API __declspec(dllexport)
#else
#define DENSECORE_API __declspec(dllimport)
#endif
#else
// GCC/Clang visibility
#define DENSECORE_API __attribute__((visibility("default")))
#endif
#else
// Static build - no export needed
#define DENSECORE_API
#endif

/**
 * @def DENSECORE_LOCAL
 * @brief Marks symbols as hidden (not exported)
 *
 * Use for internal implementation details that should not be
 * visible to library consumers.
 */
#ifdef DENSECORE_PLATFORM_WINDOWS
#define DENSECORE_LOCAL
#else
#define DENSECORE_LOCAL __attribute__((visibility("hidden")))
#endif

// =============================================================================
// C++ Feature Detection
// =============================================================================

#if __cplusplus >= 201703L
#define DENSECORE_CPP17 1
#endif

#if __cplusplus >= 202002L
#define DENSECORE_CPP20 1
#endif

/**
 * @def DENSECORE_NODISCARD
 * @brief [[nodiscard]] attribute for return values that must be used
 */
#ifdef DENSECORE_CPP17
#define DENSECORE_NODISCARD [[nodiscard]]
#else
#define DENSECORE_NODISCARD
#endif

/**
 * @def DENSECORE_DEPRECATED
 * @brief Marks APIs as deprecated with a migration message
 */
#ifdef DENSECORE_CPP17
#define DENSECORE_DEPRECATED(msg) [[deprecated(msg)]]
#else
#define DENSECORE_DEPRECATED(msg) __attribute__((deprecated(msg)))
#endif
#define DENSECORE_API __attribute__((visibility("default")))
#endif

// Compiler hints
#define DENSECORE_ALWAYS_INLINE inline __attribute__((always_inline))
#define DENSECORE_NOINLINE __attribute__((noinline))
#define DENSECORE_LIKELY(x) __builtin_expect(!!(x), 1)
#define DENSECORE_UNLIKELY(x) __builtin_expect(!!(x), 0)

// Alignment
#define DENSECORE_ALIGN(x) __attribute__((aligned(x)))

// Interface helper
#define DENSECORE_INTERFACE struct

#endif // DENSECORE_HAL_MACROS_H
