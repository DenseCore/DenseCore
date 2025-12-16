cmake_minimum_required(VERSION 3.14)
project(DenseCore VERSION 1.0.0 LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==========================================================================
# Symbol Visibility Control
# ==========================================================================
# Hide all symbols by default, only expose functions marked with DENSECORE_API
# This reduces .so size and improves load time
# ==========================================================================
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
set(CMAKE_C_VISIBILITY_PRESET hidden)

# ==========================================================================
# Version Information Injection
# ==========================================================================
# Generate version header from CMake project version
# ==========================================================================
set(DENSECORE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(DENSECORE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(DENSECORE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(DENSECORE_VERSION_STRING "${PROJECT_VERSION}")

# Get build timestamp
string(TIMESTAMP DENSECORE_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)

# Try to get git commit hash
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE DENSECORE_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT DENSECORE_GIT_COMMIT)
        set(DENSECORE_GIT_COMMIT "unknown")
    endif()
else()
    set(DENSECORE_GIT_COMMIT "unknown")
endif()

# Generate version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/densecore_version.h
    @ONLY
)

# Output directory
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# ggml third party library path
set(GGML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml)

# Check if ggml exists
if(NOT EXISTS ${GGML_DIR})
    message(FATAL_ERROR "ggml not found at ${GGML_DIR}. Please run scripts/setup_ggml.sh first.")
endif()

# Configure ggml options before adding subdirectory
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)

# Add ggml as subdirectory
add_subdirectory(${GGML_DIR})

# Source files (DenseCore specific only)
set(SOURCES
    src/engine.cpp
    src/worker.cpp
    src/inference.cpp
    src/model_types.cpp
    src/model_loader.cpp
    src/tokenizer.cpp
    src/kv_cache.cpp
    src/optimization_bridge.cpp
    src/save_model.cpp
    src/version.cpp
    # Quantization module
    src/quantizer.cpp
    src/quantization/max_quantizer.cpp
    src/quantization/awq_quantizer.cpp
    src/quantization/smoothquant_quantizer.cpp
    src/quantization/int4_quantizer.cpp
    # Pruning module
    src/pruner.cpp
    src/tensor_utils.cpp
    src/pruning/depth_pruner.cpp
    src/pruning/width_pruner.cpp
    src/pruning/attention_pruner.cpp
    src/pruning/combined_pruner.cpp
    # src/gguf.cpp is now provided by ggml library
)

# Create shared library
add_library(densecore SHARED ${SOURCES})

# Include directories - Define BEFORE linking
target_include_directories(densecore PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include  # Generated version header
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${GGML_DIR}/src/ggml-cpu
)

# Link against ggml
target_link_libraries(densecore PRIVATE ggml)

# Define DENSECORE_BUILD_SHARED for symbol export (Windows dllexport)
target_compile_definitions(densecore PRIVATE DENSECORE_BUILD_SHARED)

# AddressSanitizer option for memory debugging
option(ASAN_BUILD "Enable AddressSanitizer for memory error detection" OFF)

if(ASAN_BUILD)
    message(STATUS "AddressSanitizer ENABLED - Building with -fsanitize=address")
    set(ASAN_FLAGS -fsanitize=address -fno-omit-frame-pointer -g -O1)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
endif()

# Compiler flags and optimizations
target_compile_options(densecore PRIVATE
    -Wall
    -Wextra
    $<IF:$<BOOL:${ASAN_BUILD}>,-O1,-O3>
    $<$<BOOL:${ASAN_BUILD}>:-fsanitize=address>
    $<$<BOOL:${ASAN_BUILD}>:-fno-omit-frame-pointer>
    $<$<BOOL:${ASAN_BUILD}>:-g>
)

# Platform-specific optimizations
if(APPLE)
    message(STATUS "Configuring for Apple platform - Using Accelerate framework")

    # Use Apple's Accelerate framework
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(densecore PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(densecore PRIVATE USE_ACCELERATE)
        message(STATUS "Accelerate framework found: ${ACCELERATE_FRAMEWORK}")
    else()
        message(WARNING "Accelerate framework not found")
    endif()

    # Apple Silicon optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        message(STATUS "Apple Silicon detected - enabling ARM NEON optimizations")
        target_compile_options(densecore PRIVATE -mcpu=apple-m1)
        target_compile_definitions(densecore PRIVATE GGML_USE_CPU GGML_USE_ACCELERATE)
    else()
        # Intel Mac
        target_compile_definitions(densecore PRIVATE GGML_USE_CPU GGML_USE_ACCELERATE)
    endif()

else()
    message(STATUS "Configuring for non-Apple platform")

    # Check for AVX support (compiler flags)
    include(CheckCXXCompilerFlag)

    if(MSVC)
        # MSVC uses different flags
        check_cxx_compiler_flag("/arch:AVX" COMPILER_SUPPORTS_AVX)
        check_cxx_compiler_flag("/arch:AVX2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
        set(AVX_FLAG "/arch:AVX")
        set(AVX2_FLAG "/arch:AVX2")
        set(AVX512_FLAG "/arch:AVX512")
    else()
        # GCC/Clang flags
        check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
        set(AVX_FLAG "-mavx")
        set(AVX2_FLAG "-mavx2" "-mfma")
        set(AVX512_FLAG "-mavx512f" "-mavx512dq" "-mavx512vl" "-mavx512bw")
    endif()

    # Check CPU capabilities for AVX-512 (cross-platform approach)
    # We use a compile-and-run test to detect actual CPU support
    set(CPU_HAS_AVX512 FALSE)
    
    if(NOT MSVC)
        # For GCC/Clang: Try to compile and run an AVX-512 test program
        # This detects actual runtime CPU support, not just compiler support
        include(CheckCXXSourceRuns)
        set(CMAKE_REQUIRED_FLAGS "-mavx512f")
        check_cxx_source_runs("
            #include <immintrin.h>
            int main() {
                __m512i a = _mm512_set1_epi32(1);
                __m512i b = _mm512_add_epi32(a, a);
                return _mm512_cvtsi512_si32(b) != 2 ? 1 : 0;
            }
        " AVX512_RUNTIME_SUPPORTED)
        unset(CMAKE_REQUIRED_FLAGS)
        
        if(AVX512_RUNTIME_SUPPORTED)
            set(CPU_HAS_AVX512 TRUE)
            message(STATUS "AVX-512 runtime support detected")
        else()
            # Fallback: check /proc/cpuinfo on Linux
            if(EXISTS "/proc/cpuinfo")
                execute_process(
                    COMMAND grep -q avx512f /proc/cpuinfo
                    RESULT_VARIABLE GREP_RESULT
                    ERROR_QUIET OUTPUT_QUIET
                )
                if(GREP_RESULT EQUAL 0)
                    set(CPU_HAS_AVX512 TRUE)
                    message(STATUS "AVX-512 detected via /proc/cpuinfo")
                endif()
            endif()
        endif()
    endif()

    # Apply optimizations based on detection
    if(CPU_HAS_AVX512 AND COMPILER_SUPPORTS_AVX512)
        message(STATUS "Enabling AVX-512 optimizations (compiler and CPU)")
        target_compile_options(densecore PRIVATE ${AVX512_FLAG})
        target_compile_definitions(densecore PRIVATE USE_AVX512 GGML_USE_CPU GGML_USE_AVX512)
        set(USING_AVX512 TRUE)
    elseif(COMPILER_SUPPORTS_AVX2)
        message(STATUS "Enabling AVX2 optimizations")
        target_compile_options(densecore PRIVATE ${AVX2_FLAG})
        target_compile_definitions(densecore PRIVATE USE_AVX2 GGML_USE_CPU GGML_USE_AVX2 GGML_USE_FMA)
        set(USING_AVX2 TRUE)
    elseif(COMPILER_SUPPORTS_AVX)
        message(STATUS "Enabling AVX optimizations")
        target_compile_options(densecore PRIVATE ${AVX_FLAG})
        target_compile_definitions(densecore PRIVATE USE_AVX GGML_USE_CPU GGML_USE_AVX)
        set(USING_AVX TRUE)
    else()
        message(STATUS "No AVX support detected - using standard optimizations")
        target_compile_definitions(densecore PRIVATE GGML_USE_CPU)
    endif()

    # Additional x86 tuning
    if(NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        # Optimization: Enable -march=native to use all available instructions on the host CPU
        # This breaks portability (binary won't run on older CPUs) but maximizes performance
        message(STATUS "Enabling -march=native for maximum performance on this host")
        target_compile_options(densecore PRIVATE -march=native)
    endif()
endif()

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(densecore PRIVATE Threads::Threads)

# Position Independent Code (required for shared libraries)
set_target_properties(densecore PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# Installation rules
install(TARGETS densecore
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Quantize tool
add_executable(quantize src/quantize.cpp)

target_include_directories(quantize PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GGML_DIR}/include
    ${GGML_DIR}/src
    ${GGML_DIR}/src/ggml-cpu
)

target_compile_definitions(quantize PRIVATE
    GGML_VERSION="${GGML_VERSION}"
    GGML_COMMIT="${GGML_COMMIT}"
    _GNU_SOURCE
)

# Reuse the same compile options/definitions as densecore for AVX/Accelerate
if(USE_ACCELERATE)
    target_link_libraries(quantize PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(quantize PRIVATE USE_ACCELERATE GGML_USE_CPU GGML_USE_ACCELERATE)
elseif(USING_AVX512)
    target_compile_options(quantize PRIVATE -mavx512f -mavx512dq -mavx512vl -mavx512bw)
    target_compile_definitions(quantize PRIVATE USE_AVX512 GGML_USE_CPU GGML_USE_AVX512)
elseif(USE_AVX2)
    target_compile_options(quantize PRIVATE -mavx2 -mfma)
    target_compile_definitions(quantize PRIVATE USE_AVX2 GGML_USE_CPU GGML_USE_AVX2 GGML_USE_FMA)
elseif(USE_AVX)
    target_compile_options(quantize PRIVATE -mavx)
    target_compile_definitions(quantize PRIVATE USE_AVX GGML_USE_CPU GGML_USE_AVX)
else()
    target_compile_definitions(quantize PRIVATE GGML_USE_CPU)
endif()

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    target_compile_options(quantize PRIVATE -mcpu=apple-m1)
endif()

# Link against ggml and threads
target_link_libraries(quantize PRIVATE ggml Threads::Threads)
set_target_properties(quantize PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

install(TARGETS quantize
    RUNTIME DESTINATION bin
)

install(FILES include/densecore.h
    DESTINATION include
)

# ==========================================
# Testing Infrastructure (GoogleTest)
# ==========================================
option(DENSECORE_BUILD_TESTS "Build DenseCore tests" ON)

if(DENSECORE_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # Prevent gtest from overriding compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # Test sources
    set(TEST_SOURCES
        tests/test_simd_ops.cpp
        tests/test_memory_pool.cpp
        tests/test_kv_cache.cpp
    )

    # Create test executable
    # NOTE: We include SOURCES here to compile core files directly into the test executable.
    # This allows tests to access "internal" (hidden visibility) symbols that are not exported
    # by the shared library.
    add_executable(densecore_tests ${TEST_SOURCES} ${SOURCES})

    target_include_directories(densecore_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${GGML_DIR}/include
        ${GGML_DIR}/src
        ${GGML_DIR}/src/ggml-cpu
    )

    # Link dependencies directly (do NOT link densecore shared lib to avoid symbol conflicts/hiding)
    target_link_libraries(densecore_tests PRIVATE
        ggml
        GTest::gtest_main
        Threads::Threads
    )
    
    if(USE_ACCELERATE)
        target_link_libraries(densecore_tests PRIVATE ${ACCELERATE_FRAMEWORK})
    endif()

    # Apply same compile flags as densecore
    if(CPU_HAS_AVX512 AND COMPILER_SUPPORTS_AVX512)
        target_compile_options(densecore_tests PRIVATE ${AVX512_FLAG})
        target_compile_definitions(densecore_tests PRIVATE USE_AVX512 GGML_USE_CPU GGML_USE_AVX512)
    elseif(COMPILER_SUPPORTS_AVX2)
        target_compile_options(densecore_tests PRIVATE ${AVX2_FLAG})
        target_compile_definitions(densecore_tests PRIVATE USE_AVX2 GGML_USE_CPU GGML_USE_AVX2 GGML_USE_FMA)
    elseif(COMPILER_SUPPORTS_AVX)
        target_compile_options(densecore_tests PRIVATE ${AVX_FLAG})
        target_compile_definitions(densecore_tests PRIVATE USE_AVX GGML_USE_CPU GGML_USE_AVX)
    else()
        target_compile_definitions(densecore_tests PRIVATE GGML_USE_CPU)
    endif()
    
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        target_compile_options(densecore_tests PRIVATE -mcpu=apple-m1)
    endif()

    include(GoogleTest)
    gtest_discover_tests(densecore_tests)
endif()

# Print configuration summary
message(STATUS "==========================================")
message(STATUS "DenseCore Configuration Summary")
message(STATUS "==========================================")
message(STATUS "CMake Version: ${CMAKE_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "==========================================")

