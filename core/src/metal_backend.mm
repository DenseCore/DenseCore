/**
 * @file metal_backend.mm
 * @brief Apple Metal GPU backend implementation
 *
 * Objective-C++ implementation of the Metal compute backend.
 * Integrates with GGML's Metal backend while providing custom
 * kernels for performance-critical paths (GEMV, FlashAttention).
 *
 * Architecture:
 * - Uses GGML Metal for graph execution (leverages proven codebase)
 * - Custom Metal shaders for decode-phase GEMV (parallel reduction)
 * - MTLStorageModeShared for zero-copy unified memory
 * - Per-thread command encoders for concurrent graph building
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/metal_backend.h"

#include "../include/apple_silicon.h"

#ifdef __APPLE__

#import <Accelerate/Accelerate.h>  // For cblas_sgemm
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

// GGML Metal backend integration
extern "C" {
#include "ggml-backend.h"
#include "ggml-metal.h"
}

#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace densecore {

// ============================================================================
// Private Implementation (Pimpl)
// ============================================================================

struct MetalBackend::Impl {
    // Core Metal objects
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> shaderLibrary = nil;

    // Custom compute pipeline states
    id<MTLComputePipelineState> gemvPipeline = nil;
    id<MTLComputePipelineState> softmaxPipeline = nil;
    id<MTLComputePipelineState> rmsNormPipeline = nil;
    id<MTLComputePipelineState> flashAttentionDecodePipeline = nil;
    id<MTLComputePipelineState> flashAttentionPrefillPipeline = nil;

    // Quantized GEMV pipeline states
    id<MTLComputePipelineState> gemvQ4_0Pipeline = nil;
    id<MTLComputePipelineState> gemvQ4_1Pipeline = nil;
    id<MTLComputePipelineState> gemvQ8_0Pipeline = nil;

    // Fused QKV pipeline state
    id<MTLComputePipelineState> fusedQKVPipeline = nil;

    // RoPE pipeline state
    id<MTLComputePipelineState> ropePipeline = nil;

    // Dequantization pipeline for M>1 GEMM
    id<MTLComputePipelineState> dequantizeQ4_0Pipeline = nil;

    // GGML Metal backend (for graph execution)
    ggml_backend_t ggmlMetalBackend = nullptr;

    // Memory tracking
    std::atomic<size_t> currentMemoryUsage{0};
    std::atomic<size_t> peakMemoryUsage{0};

    // Buffer registry: maps contents pointer -> MTLBuffer for proper deallocation
    std::mutex bufferRegistryMutex;
    std::unordered_map<void*, id<MTLBuffer>> bufferRegistry;

    // Buffer pool for small allocations
    std::mutex bufferPoolMutex;
    std::vector<id<MTLBuffer>> bufferPool;

    // Chip information (cached)
    AppleSiliconChipInfo chipInfo;

    // GPU capture state
    bool captureEnabled = false;

    // Helper: Get MTLBuffer for a tracked pointer (for MPS zero-copy)
    id<MTLBuffer> GetBufferForPointer(void* ptr) {
        std::lock_guard<std::mutex> lock(bufferRegistryMutex);
        auto it = bufferRegistry.find(ptr);
        return (it != bufferRegistry.end()) ? it->second : nil;
    }

    // Helper: Wrap an untracked pointer with zero-copy MTLBuffer
    // Uses newBufferWithBytesNoCopy for UMA zero-copy access
    id<MTLBuffer> WrapPointerNoCopy(void* ptr, size_t size) {
        if (!ptr || size == 0)
            return nil;
        // newBufferWithBytesNoCopy requires page-aligned memory on some systems
        // For non-aligned, fall back to newBufferWithBytes
        return [device newBufferWithBytesNoCopy:ptr
                                         length:size
                                        options:MTLResourceStorageModeShared
                                    deallocator:nil];
    }

    // Helper: Get or wrap buffer (prefers tracked, falls back to zero-copy wrap)
    id<MTLBuffer> GetOrWrapBuffer(void* ptr, size_t size) {
        id<MTLBuffer> buffer = GetBufferForPointer(ptr);
        if (buffer)
            return buffer;
        // Try zero-copy wrap first
        buffer = WrapPointerNoCopy(ptr, size);
        if (buffer)
            return buffer;
        // Last resort: copy data (shouldn't happen on Apple Silicon)
        return [device newBufferWithBytes:ptr length:size options:MTLResourceStorageModeShared];
    }

    // Scratch buffer cache for temporary allocations (reduces alloc overhead)
    std::mutex scratchBufferMutex;
    id<MTLBuffer> scratchBuffer = nil;
    size_t scratchBufferSize = 0;

    // Helper: Get a scratch buffer of at least the requested size
    // Reuses existing buffer if large enough, otherwise reallocates
    id<MTLBuffer> GetScratchBuffer(size_t size) {
        std::lock_guard<std::mutex> lock(scratchBufferMutex);
        if (scratchBuffer && scratchBufferSize >= size) {
            return scratchBuffer;
        }
        // Allocate with some headroom to reduce reallocs
        size_t allocSize = size + (size / 4);  // 25% headroom
        scratchBuffer = [device newBufferWithLength:allocSize options:MTLResourceStorageModeShared];
        if (scratchBuffer) {
            scratchBufferSize = allocSize;
        }
        return scratchBuffer;
    }

    ~Impl() {
        // Release GGML Metal backend
        if (ggmlMetalBackend) {
            ggml_backend_free(ggmlMetalBackend);
            ggmlMetalBackend = nullptr;
        }

        // Release pipeline states
        gemvPipeline = nil;
        softmaxPipeline = nil;
        rmsNormPipeline = nil;
        flashAttentionDecodePipeline = nil;
        flashAttentionPrefillPipeline = nil;
        gemvQ4_0Pipeline = nil;
        gemvQ4_1Pipeline = nil;
        gemvQ8_0Pipeline = nil;
        fusedQKVPipeline = nil;
        ropePipeline = nil;
        dequantizeQ4_0Pipeline = nil;

        // Release scratch buffer
        scratchBuffer = nil;
        scratchBufferSize = 0;

        // Release shader library
        shaderLibrary = nil;

        // Release all tracked buffers
        {
            std::lock_guard<std::mutex> lock(bufferRegistryMutex);
            for (auto& [ptr, buffer] : bufferRegistry) {
                if (buffer) {
                    CFRelease((__bridge CFTypeRef)buffer);
                }
            }
            bufferRegistry.clear();
        }

        // Clear buffer pool
        {
            std::lock_guard<std::mutex> lock(bufferPoolMutex);
            bufferPool.clear();
        }

        // Release command queue and device
        commandQueue = nil;
        device = nil;
    }
};

// ============================================================================
// Custom Metal Shader Source
// ============================================================================

namespace {

/**
 * @brief Embedded Metal shader source for custom kernels
 *
 * These shaders are compiled at runtime if the .metallib is not found.
 * Production builds should use pre-compiled .metallib for faster startup.
 *
 * Optimizations:
 * - SIMD-group intrinsics for warp-level reductions (32-wide on Apple GPUs)
 * - Minimal threadgroup barriers
 * - FMA instructions for better throughput
 */
const char* kMetalShaderSource = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Apple GPU SIMD width constant
constant uint SIMD_WIDTH = 32;

// =============================================================================
// SIMD-Optimized GEMV Kernel: output = weight @ input ([M,K] @ [K] = [M])
// =============================================================================
// Key optimizations:
// 1. Uses simd_sum() for warp-level reduction (no shared memory needed for first step)
// 2. Only one threadgroup barrier after SIMD reduction
// 3. Each simdgroup handles reduction independently
// 4. Final reduction across simdgroups uses minimal shared memory
// =============================================================================

kernel void gemv_f32(
    device const float* input [[buffer(0)]],      // [K]
    device const float* weight [[buffer(1)]],     // [M, K]
    device float* output [[buffer(2)]],           // [M]
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    // Each threadgroup handles one output row
    uint row = tgid;
    if (row >= M) return;

    device const float* weight_row = weight + row * K;

    // Phase 1: Each thread accumulates its portion of the dot product
    float sum = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        sum = fma(weight_row[k], input[k], sum);
    }

    // Phase 2: SIMD-level reduction using simd_sum (warp-level, no barrier needed)
    sum = simd_sum(sum);

    // Phase 3: First lane of each simdgroup writes to shared memory
    // Only need as many slots as simdgroups (typically 8 for 256 threads)
    threadgroup float simd_results[8];
    uint num_simdgroups = (tg_size + SIMD_WIDTH - 1) / SIMD_WIDTH;

    if (simd_lane == 0 && simd_group < num_simdgroups) {
        simd_results[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: First thread reduces across simdgroups
    if (tid == 0) {
        float final_sum = 0.0f;
        for (uint i = 0; i < num_simdgroups; ++i) {
            final_sum += simd_results[i];
        }
        output[row] = final_sum;
    }
}

// =============================================================================
// SIMD-Optimized Softmax Kernel
// =============================================================================

kernel void softmax_f32(
    device float* data [[buffer(0)]],
    constant uint& N [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_max[8];
    threadgroup float simd_sum_vals[8];
    uint num_simdgroups = (tg_size + SIMD_WIDTH - 1) / SIMD_WIDTH;

    // Phase 1: Find local max
    float local_max = -INFINITY;
    for (uint i = tid; i < N; i += tg_size) {
        local_max = max(local_max, data[i]);
    }

    // SIMD reduction for max
    local_max = simd_max(local_max);
    if (simd_lane == 0) { simd_max[simd_group] = local_max; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global max across simdgroups
    float max_val = simd_max[0];
    for (uint i = 1; i < num_simdgroups; ++i) {
        max_val = max(max_val, simd_max[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Compute exp and local sum
    float local_sum = 0.0f;
    for (uint i = tid; i < N; i += tg_size) {
        float e = exp(data[i] - max_val);
        data[i] = e;
        local_sum += e;
    }

    // SIMD reduction for sum
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) { simd_sum_vals[simd_group] = local_sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global sum across simdgroups
    float sum_val = 0.0f;
    for (uint i = 0; i < num_simdgroups; ++i) {
        sum_val += simd_sum_vals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize
    float inv_sum = 1.0f / sum_val;
    for (uint i = tid; i < N; i += tg_size) {
        data[i] *= inv_sum;
    }
}

// =============================================================================
// SIMD-Optimized RMS Normalization Kernel
// =============================================================================

kernel void rms_norm_f32(
    device const float* input [[buffer(0)]],      // [N, dim]
    device const float* weight [[buffer(1)]],     // [dim]
    device float* output [[buffer(2)]],           // [N, dim]
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float simd_sums[8];
    uint num_simdgroups = (tg_size + SIMD_WIDTH - 1) / SIMD_WIDTH;

    uint row = tgid;
    device const float* input_row = input + row * dim;
    device float* output_row = output + row * dim;

    // Phase 1: Compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = input_row[i];
        sum_sq = fma(val, val, sum_sq);
    }

    // SIMD reduction
    sum_sq = simd_sum(sum_sq);
    if (simd_lane == 0) { simd_sums[simd_group] = sum_sq; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Global sum
    float total_sum_sq = 0.0f;
    for (uint i = 0; i < num_simdgroups; ++i) {
        total_sum_sq += simd_sums[i];
    }

    float rms = rsqrt(total_sum_sq / float(dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Apply normalization and weight
    for (uint i = tid; i < dim; i += tg_size) {
        output_row[i] = input_row[i] * rms * weight[i];
    }
}

// =============================================================================
// RoPE (Rotary Positional Embedding) Kernel
// =============================================================================
// Layout: [n_heads, n_tokens, head_dim] or [batch*n_heads, seq_len, head_dim]
// Each thread handles one pair of elements (2*d, 2*d+1)

kernel void rope_f32(
    device float* data [[buffer(0)]],              // In-place modification
    device const float* cos_sin [[buffer(1)]],    // [max_seq, head_dim]
    device const int* positions [[buffer(2)]],    // [n_tokens]
    constant uint& n_heads [[buffer(3)]],
    constant uint& n_tokens [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& rope_dim [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]])        // (pair_idx, token, head)
{
    uint pair_idx = tid.x;  // Which pair (0 to rope_dim/2 - 1)
    uint token = tid.y;
    uint head = tid.z;

    if (pair_idx >= rope_dim / 2 || token >= n_tokens || head >= n_heads) return;

    int pos = positions[token];
    device const float* pos_cs = cos_sin + pos * head_dim;

    float cos_theta = pos_cs[2 * pair_idx];
    float sin_theta = pos_cs[2 * pair_idx + 1];

    // Index into data: [head, token, dim]
    uint base_idx = (head * n_tokens + token) * head_dim + 2 * pair_idx;

    float x0 = data[base_idx];
    float x1 = data[base_idx + 1];

    data[base_idx] = fma(x0, cos_theta, -x1 * sin_theta);
    data[base_idx + 1] = fma(x0, sin_theta, x1 * cos_theta);
}
)METAL";

/**
 * @brief Quantized GEMV shader source for INT4/INT8 weights
 *
 * Kernels:
 * - gemv_q4_0: Q4_0 format (scale only, 4-bit weights centered at 8)
 * - gemv_q4_1: Q4_1 format (scale + min, unsigned 4-bit weights)
 * - gemv_q8_0: Q8_0 format (scale, 8-bit weights)
 */
const char* kQuantizedGemvShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint QK4_0 = 32;
constant uint QK4_1 = 32;
constant uint QK8_0 = 32;
constant uint THREADGROUP_SIZE = 256;

struct block_q4_0 {
    half scale;
    uint8_t quants[QK4_0 / 2];
};

struct block_q4_1 {
    half scale;
    half min;
    uint8_t quants[QK4_1 / 2];
};

struct block_q8_0 {
    half scale;
    int8_t quants[QK8_0];
};

inline int8_t extract_q4(uint8_t packed, uint idx) {
    int8_t val = (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    return val - 8;
}

inline uint8_t extract_q4_unsigned(uint8_t packed, uint idx) {
    return (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
}

kernel void gemv_q4_0(
    device const float* input [[buffer(0)]],
    device const block_q4_0* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint row = tgid;
    if (row >= M) return;

    threadgroup float shared_sum[THREADGROUP_SIZE];
    uint blocks_per_row = K / QK4_0;
    float sum = 0.0f;

    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        device const block_q4_0* block = &weight[row * blocks_per_row + block_idx];
        float scale = float(block->scale);
        uint k_start = block_idx * QK4_0;

        for (uint i = 0; i < QK4_0; i += 2) {
            uint byte_idx = i / 2;
            uint8_t packed = block->quants[byte_idx];
            float w0 = float(extract_q4(packed, 0)) * scale;
            float w1 = float(extract_q4(packed, 1)) * scale;
            sum = fma(w0, input[k_start + i], sum);
            sum = fma(w1, input[k_start + i + 1], sum);
        }
    }

    shared_sum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row] = shared_sum[0];
    }
}

kernel void gemv_q4_1(
    device const float* input [[buffer(0)]],
    device const block_q4_1* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint row = tgid;
    if (row >= M) return;

    threadgroup float shared_sum[THREADGROUP_SIZE];
    uint blocks_per_row = K / QK4_1;
    float sum = 0.0f;

    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        device const block_q4_1* block = &weight[row * blocks_per_row + block_idx];
        float scale = float(block->scale);
        float min_val = float(block->min);
        uint k_start = block_idx * QK4_1;

        for (uint i = 0; i < QK4_1; i += 2) {
            uint byte_idx = i / 2;
            uint8_t packed = block->quants[byte_idx];
            float w0 = float(extract_q4_unsigned(packed, 0)) * scale + min_val;
            float w1 = float(extract_q4_unsigned(packed, 1)) * scale + min_val;
            sum = fma(w0, input[k_start + i], sum);
            sum = fma(w1, input[k_start + i + 1], sum);
        }
    }

    shared_sum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row] = shared_sum[0];
    }
}

kernel void gemv_q8_0(
    device const float* input [[buffer(0)]],
    device const block_q8_0* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint row = tgid;
    if (row >= M) return;

    threadgroup float shared_sum[THREADGROUP_SIZE];
    uint blocks_per_row = K / QK8_0;
    float sum = 0.0f;

    for (uint block_idx = tid; block_idx < blocks_per_row; block_idx += tg_size) {
        device const block_q8_0* block = &weight[row * blocks_per_row + block_idx];
        float scale = float(block->scale);
        uint k_start = block_idx * QK8_0;

        for (uint i = 0; i < QK8_0; ++i) {
            float w = float(block->quants[i]) * scale;
            sum = fma(w, input[k_start + i], sum);
        }
    }

    shared_sum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[row] = shared_sum[0];
    }
}

// =============================================================================
// Q4_0 Block Dequantization Kernel (for M>1 GEMM)
// =============================================================================
// Converts packed Q4_0 weights to FP32 for MPSMatrixMultiplication
// Grid: (blocks_per_row, N, 1), each thread handles one block

kernel void dequantize_q4_0(
    device const block_q4_0* input [[buffer(0)]],   // [N, blocks_per_row]
    device float* output [[buffer(1)]],              // [N, K]
    constant uint& N [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])           // (block_idx, row_n)
{
    uint block_idx = tid.x;
    uint row = tid.y;
    uint blocks_per_row = (K + QK4_0 - 1) / QK4_0;  // Ceiling division

    if (block_idx >= blocks_per_row || row >= N) return;

    device const block_q4_0* block = &input[row * blocks_per_row + block_idx];
    float scale = float(block->scale);
    uint k_start = block_idx * QK4_0;
    uint k_end = min(k_start + QK4_0, K);  // Handle partial last block

    device float* out_row = output + row * K + k_start;

    for (uint i = 0; i < k_end - k_start; i += 2) {
        uint8_t packed = block->quants[i / 2];
        out_row[i] = float(extract_q4(packed, 0)) * scale;
        if (i + 1 < k_end - k_start) {
            out_row[i + 1] = float(extract_q4(packed, 1)) * scale;
        }
    }
}
)METAL";

/**
 * @brief FlashAttention decode kernel for single-query attention
 *
 * Optimized for decode phase (seq_q = 1). One threadgroup per (batch, head).
 */
const char* kFlashAttentionShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float NEG_INF = -1e9f;
constant uint MAX_HEAD_DIM [[function_constant(0)]];

kernel void flash_attention_decode(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant float& scale [[buffer(4)]],
    constant uint& seq_kv [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& n_heads [[buffer(7)]],
    constant uint& n_kv_heads [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint batch_idx = tgid.z;
    uint head_idx = tgid.x;
    uint kv_head_idx = head_idx / (n_heads / n_kv_heads);

    device const float* Q_ptr = Q + (batch_idx * n_heads + head_idx) * head_dim;
    device const float* K_head = K + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device const float* V_head = V + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device float* O_ptr = output + (batch_idx * n_heads + head_idx) * head_dim;

    float q[MAX_HEAD_DIM];
    for (uint d = 0; d < head_dim; ++d) {
        q[d] = Q_ptr[d];
    }

    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];
    threadgroup float shared_output[256][MAX_HEAD_DIM];

    float local_max = NEG_INF;
    float local_sum = 0.0f;
    float local_output[MAX_HEAD_DIM] = {0.0f};

    for (uint k = tid; k < seq_kv; k += tg_size) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            dot = fma(q[d], K_head[k * head_dim + d], dot);
        }
        float score = dot * scale;

        float old_max = local_max;
        local_max = max(local_max, score);
        float scale_factor = exp(old_max - local_max);
        float exp_score = exp(score - local_max);

        local_sum = local_sum * scale_factor + exp_score;
        for (uint d = 0; d < head_dim; ++d) {
            local_output[d] = local_output[d] * scale_factor +
                              V_head[k * head_dim + d] * exp_score;
        }
    }

    shared_max[tid] = local_max;
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            float m1 = shared_max[tid];
            float m2 = shared_max[tid + stride];
            float new_max = max(m1, m2);
            float s1 = shared_sum[tid] * exp(m1 - new_max);
            float s2 = shared_sum[tid + stride] * exp(m2 - new_max);
            shared_max[tid] = new_max;
            shared_sum[tid] = s1 + s2;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float global_max = shared_max[0];
    float global_sum = shared_sum[0];
    float my_scale = exp(local_max - global_max) / global_sum;

    for (uint d = 0; d < head_dim; ++d) {
        shared_output[tid][d] = local_output[d] * my_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            for (uint d = 0; d < head_dim; ++d) {
                shared_output[tid][d] += shared_output[tid + stride][d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        for (uint d = 0; d < head_dim; ++d) {
            O_ptr[d] = shared_output[0][d];
        }
    }
}
)METAL";

/**
 * @brief FlashAttention prefill kernel for multi-query attention (seq_q > 1)
 *
 * Naive GPU implementation parallelizing over (batch, head, q_pos).
 * Each threadgroup handles one query position, streaming over K/V.
 * Uses online softmax for numerical stability.
 */
const char* kFlashAttentionPrefillSource = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant float NEG_INF = -1e9f;
constant uint SIMD_WIDTH = 32;
constant uint KV_BLOCK_SIZE = 64;  // Process K/V in blocks of 64
constant uint MAX_HEAD_DIM [[function_constant(0)]];  // Configurable via function constant

kernel void flash_attention_prefill(
    device const float* Q [[buffer(0)]],         // [batch, n_head, seq_q, head_dim]
    device const float* K [[buffer(1)]],         // [batch, n_kv_head, seq_kv, head_dim]
    device const float* V [[buffer(2)]],         // [batch, n_kv_head, seq_kv, head_dim]
    device float* output [[buffer(3)]],          // [batch, n_head, seq_q, head_dim]
    constant float& scale [[buffer(4)]],
    constant uint& seq_q [[buffer(5)]],
    constant uint& seq_kv [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& n_heads [[buffer(8)]],
    constant uint& n_kv_heads [[buffer(9)]],
    constant uint& causal [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],    // (head, q_pos, batch)
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    uint head_idx = tgid.x;
    uint q_pos = tgid.y;
    uint batch_idx = tgid.z;

    // Guard: ensure head_dim doesn't exceed MAX_HEAD_DIM
    uint safe_head_dim = min(head_dim, MAX_HEAD_DIM);

    // GQA: map Q head to KV head
    uint n_rep = n_heads / n_kv_heads;
    uint kv_head_idx = head_idx / n_rep;

    // Pointers to current Q row and output row
    device const float* Q_ptr = Q + ((batch_idx * n_heads + head_idx) * seq_q + q_pos) * head_dim;
    device const float* K_head = K + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device const float* V_head = V + (batch_idx * n_kv_heads + kv_head_idx) * seq_kv * head_dim;
    device float* O_ptr = output + ((batch_idx * n_heads + head_idx) * seq_q + q_pos) * head_dim;

    // Causal mask boundary: for causal attention with different seq_q/seq_kv,
    // we mask positions where kv_pos > q_pos + (seq_kv - seq_q)
    uint causal_limit = causal ? (q_pos + (seq_kv - seq_q) + 1) : seq_kv;

    // Threadgroup memory for partial results
    threadgroup float shared_max[8];      // Per-simdgroup max
    threadgroup float shared_sum[8];      // Per-simdgroup sum

    uint num_simdgroups = (tg_size + SIMD_WIDTH - 1) / SIMD_WIDTH;

    // Online softmax state per thread
    float local_max = NEG_INF;
    float local_sum = 0.0f;
    float local_output[MAX_HEAD_DIM] = {0.0f};

    // Load Q into registers (guarded by safe_head_dim)
    float q_reg[MAX_HEAD_DIM];
    for (uint d = 0; d < safe_head_dim; ++d) {
        q_reg[d] = Q_ptr[d];
    }

    // Iterate over K/V positions in blocks
    for (uint kv_start = 0; kv_start < causal_limit; kv_start += KV_BLOCK_SIZE) {
        uint kv_end = min(kv_start + KV_BLOCK_SIZE, causal_limit);

        // Each thread processes a subset of K/V positions in this block
        for (uint kv_pos = kv_start + tid; kv_pos < kv_end; kv_pos += tg_size) {
            // Compute Q @ K^T for this position
            float dot = 0.0f;
            device const float* K_ptr = K_head + kv_pos * head_dim;
            for (uint d = 0; d < safe_head_dim; ++d) {
                dot = fma(q_reg[d], K_ptr[d], dot);
            }
            float score = dot * scale;

            // Online softmax update
            float old_max = local_max;
            local_max = max(local_max, score);
            float scale_factor = exp(old_max - local_max);
            float exp_score = exp(score - local_max);

            // Rescale running sum and output
            local_sum = local_sum * scale_factor + exp_score;

            // Accumulate weighted V
            device const float* V_ptr = V_head + kv_pos * head_dim;
            for (uint d = 0; d < safe_head_dim; ++d) {
                local_output[d] = local_output[d] * scale_factor + V_ptr[d] * exp_score;
            }
        }
    }

    // === Cross-thread reduction ===

    // Step 1: SIMD reduction within simdgroup
    float simd_max = simd_max(local_max);

    // Rescale local state to simdgroup max
    float rescale = exp(local_max - simd_max);
    local_sum *= rescale;
    for (uint d = 0; d < safe_head_dim; ++d) {
        local_output[d] *= rescale;
    }
    local_max = simd_max;

    // Sum within simdgroup
    float simd_sum_val = simd_sum(local_sum);

    // Step 2: Write simdgroup results to shared memory
    if (simd_lane == 0 && simd_group < num_simdgroups) {
        shared_max[simd_group] = local_max;
        shared_sum[simd_group] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: First thread computes global max/sum
    float global_max = shared_max[0];
    for (uint i = 1; i < num_simdgroups; ++i) {
        global_max = max(global_max, shared_max[i]);
    }

    float global_sum = 0.0f;
    for (uint i = 0; i < num_simdgroups; ++i) {
        global_sum += shared_sum[i] * exp(shared_max[i] - global_max);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: Each thread rescales its output contribution
    float final_scale = exp(local_max - global_max) / global_sum;
    for (uint d = 0; d < safe_head_dim; ++d) {
        local_output[d] *= final_scale;
    }

    // Step 5: Reduce output across threads using atomics (simple approach)
    // For better performance, use explicit reduction, but this works for now
    for (uint d = tid; d < safe_head_dim; d += tg_size) {
        float sum = 0.0f;
        // Collect from all threads - use simd shuffle for threads in same simdgroup
        sum = simd_sum(local_output[d]);

        // First lane of each simdgroup writes partial result
        if (simd_lane == 0) {
            shared_max[simd_group] = sum;  // Reuse shared memory
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // First thread aggregates
        if (tid == 0) {
            float final_val = 0.0f;
            for (uint sg = 0; sg < num_simdgroups; ++sg) {
                final_val += shared_max[sg];
            }
            O_ptr[d] = final_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)METAL";

/**
 * @brief Fused Q/K/V projection kernel for decode phase
 *
 * Computes Q, K, V projections in a single dispatch:
 *   Q = input @ Wq^T, K = input @ Wk^T, V = input @ Wv^T
 *
 * For decode phase (batch=1), this reduces kernel launch overhead by 3x
 * compared to three separate MatMulTransB calls.
 *
 * Grid layout: Each threadgroup handles one output row from Q, K, or V.
 * - Rows 0..hidden_q-1: Q output
 * - Rows hidden_q..hidden_q+hidden_kv-1: K output
 * - Rows hidden_q+hidden_kv..total-1: V output
 */
const char* kFusedQKVShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint THREADGROUP_SIZE = 256;

kernel void fused_qkv_gemv(
    device const float* input [[buffer(0)]],     // [K]
    device const float* wq [[buffer(1)]],        // [hidden_q, K]
    device const float* wk [[buffer(2)]],        // [hidden_kv, K]
    device const float* wv [[buffer(3)]],        // [hidden_kv, K]
    device float* q_out [[buffer(4)]],           // [hidden_q]
    device float* k_out [[buffer(5)]],           // [hidden_kv]
    device float* v_out [[buffer(6)]],           // [hidden_kv]
    constant uint& K [[buffer(7)]],              // Input dimension
    constant uint& hidden_q [[buffer(8)]],       // Q output dimension
    constant uint& hidden_kv [[buffer(9)]],      // K/V output dimension
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Determine which output tensor and row this threadgroup handles
    // First hidden_q rows -> Q, next hidden_kv -> K, last hidden_kv -> V
    uint row = tgid;
    device const float* weight;
    device float* output;

    if (row < hidden_q) {
        // Q projection
        weight = wq + row * K;
        output = q_out + row;
    } else if (row < hidden_q + hidden_kv) {
        // K projection
        uint kv_row = row - hidden_q;
        weight = wk + kv_row * K;
        output = k_out + kv_row;
    } else {
        // V projection
        uint kv_row = row - hidden_q - hidden_kv;
        weight = wv + kv_row * K;
        output = v_out + kv_row;
    }

    threadgroup float shared_sum[THREADGROUP_SIZE];

    // Compute dot product: weight_row @ input
    float sum = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        sum = fma(weight[k], input[k], sum);
    }

    shared_sum[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        *output = shared_sum[0];
    }
}
)METAL";

}  // anonymous namespace

// ============================================================================
// Static Methods
// ============================================================================

bool MetalBackend::IsAvailable() {
    @autoreleasepool {
        // Check for Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return false;
        }

        // Check for required features
        // All Apple Silicon supports Metal 2.0+ which has everything we need
        bool supported = [device supportsFamily:MTLGPUFamilyApple7] ||  // M1+
                         [device supportsFamily:MTLGPUFamilyMac2];      // Intel Mac

        return supported;
    }
}

const char* MetalBackend::GetDeviceName() {
    static char deviceName[128] = {0};

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return nullptr;
        }

        NSString* name = [device name];
        if (name) {
            strncpy(deviceName, [name UTF8String], sizeof(deviceName) - 1);
            return deviceName;
        }
    }

    return nullptr;
}

AppleSiliconChipInfo MetalBackend::GetChipInfo() {
    AppleSiliconChipInfo info = {};

    apple::ChipGeneration gen = apple::DetectChipGeneration();
    info.chip_name = apple::ChipGenerationName(gen);
    info.chip_generation = static_cast<int>(gen);
    info.performance_cores = apple::GetPerformanceCoreCount();
    info.efficiency_cores = apple::GetEfficiencyCoreCount();
    info.neural_engine_tops = apple::GetNeuralEngineTOPS();

    apple::MemoryInfo memInfo = apple::GetMemoryInfo();
    info.unified_memory_bytes = memInfo.total_bytes;
    info.memory_bandwidth_gbps = memInfo.bandwidth_gbps;

    // Check GPU features
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            // Count GPU cores (approximation based on registry name)
            info.gpu_cores = 8;  // Default for M1
            if (apple::IsM1Family(gen)) {
                if (gen == apple::ChipGeneration::M1_Pro)
                    info.gpu_cores = 16;
                else if (gen == apple::ChipGeneration::M1_Max)
                    info.gpu_cores = 32;
                else if (gen == apple::ChipGeneration::M1_Ultra)
                    info.gpu_cores = 64;
            } else if (apple::IsM2Family(gen)) {
                info.gpu_cores = 10;
                if (gen == apple::ChipGeneration::M2_Pro)
                    info.gpu_cores = 19;
                else if (gen == apple::ChipGeneration::M2_Max)
                    info.gpu_cores = 38;
                else if (gen == apple::ChipGeneration::M2_Ultra)
                    info.gpu_cores = 76;
            } else if (apple::IsM3Family(gen)) {
                info.gpu_cores = 10;
                if (gen == apple::ChipGeneration::M3_Pro)
                    info.gpu_cores = 18;
                else if (gen == apple::ChipGeneration::M3_Max)
                    info.gpu_cores = 40;
            } else if (apple::IsM4Family(gen)) {
                info.gpu_cores = 10;
                if (gen == apple::ChipGeneration::M4_Pro)
                    info.gpu_cores = 20;
                else if (gen == apple::ChipGeneration::M4_Max)
                    info.gpu_cores = 40;
            }

            // Check feature support
            info.supports_simd_group_reduction = [device supportsFamily:MTLGPUFamilyApple7];
            info.supports_bfloat16 = [device supportsFamily:MTLGPUFamilyApple9];     // M3+
            info.supports_ray_tracing = [device supportsFamily:MTLGPUFamilyApple9];  // M3+
        }
    }

    return info;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        // Get Metal device
        impl_->device = MTLCreateSystemDefaultDevice();
        if (impl_->device == nil) {
            throw std::runtime_error("Failed to create Metal device");
        }

        // Create command queue
        impl_->commandQueue = [impl_->device newCommandQueue];
        if (impl_->commandQueue == nil) {
            throw std::runtime_error("Failed to create Metal command queue");
        }

        // Try to load pre-compiled shader library
        NSError* error = nil;
        NSString* libraryPath = [[NSBundle mainBundle] pathForResource:@"densecore"
                                                                ofType:@"metallib"];
        if (libraryPath) {
            NSURL* libraryURL = [NSURL fileURLWithPath:libraryPath];
            impl_->shaderLibrary = [impl_->device newLibraryWithURL:libraryURL error:&error];
        }

        // Fall back to runtime compilation
        MTLCompileOptions* compileOptions = [[MTLCompileOptions alloc] init];
        compileOptions.fastMathEnabled = YES;

        if (impl_->shaderLibrary == nil) {
            NSString* source = [NSString stringWithUTF8String:kMetalShaderSource];

            impl_->shaderLibrary = [impl_->device newLibraryWithSource:source
                                                               options:compileOptions
                                                                 error:&error];
            if (impl_->shaderLibrary == nil) {
                throw std::runtime_error("Failed to compile Metal shaders: " +
                                         std::string([[error localizedDescription] UTF8String]));
            }
        }

        // Create pipeline states for custom kernels
        id<MTLFunction> gemvFunction = [impl_->shaderLibrary newFunctionWithName:@"gemv_f32"];
        if (gemvFunction) {
            impl_->gemvPipeline = [impl_->device newComputePipelineStateWithFunction:gemvFunction
                                                                               error:&error];
        }

        id<MTLFunction> softmaxFunction = [impl_->shaderLibrary newFunctionWithName:@"softmax_f32"];
        if (softmaxFunction) {
            impl_->softmaxPipeline =
                [impl_->device newComputePipelineStateWithFunction:softmaxFunction error:&error];
        }

        id<MTLFunction> rmsNormFunction =
            [impl_->shaderLibrary newFunctionWithName:@"rms_norm_f32"];
        if (rmsNormFunction) {
            impl_->rmsNormPipeline =
                [impl_->device newComputePipelineStateWithFunction:rmsNormFunction error:&error];
        }

        // RoPE kernel
        id<MTLFunction> ropeFunction = [impl_->shaderLibrary newFunctionWithName:@"rope_f32"];
        if (ropeFunction) {
            impl_->ropePipeline = [impl_->device newComputePipelineStateWithFunction:ropeFunction
                                                                               error:&error];
            if (impl_->ropePipeline) {
                std::cout << "[MetalBackend] RoPE kernel compiled" << std::endl;
            }
        }

        // FlashAttention decode kernel: try external metallib first, then runtime
        // compile
        id<MTLLibrary> externalLibrary = nil;
        NSString* metalLibPath = [[NSBundle mainBundle] pathForResource:@"densecore"
                                                                 ofType:@"metallib"];
        if (metalLibPath) {
            NSURL* metalLibURL = [NSURL fileURLWithPath:metalLibPath];
            externalLibrary = [impl_->device newLibraryWithURL:metalLibURL error:&error];
        }
        if (externalLibrary) {
            id<MTLFunction> flashAttnDecodeFunction =
                [externalLibrary newFunctionWithName:@"flash_attention_decode"];
            if (flashAttnDecodeFunction) {
                impl_->flashAttentionDecodePipeline =
                    [impl_->device newComputePipelineStateWithFunction:flashAttnDecodeFunction
                                                                 error:&error];
                if (impl_->flashAttentionDecodePipeline) {
                    std::cout << "[MetalBackend] FlashAttention decode kernel loaded "
                                 "from metallib"
                              << std::endl;
                }
            }
        }

        // Fallback: compile FlashAttention from embedded source if metallib failed
        if (!impl_->flashAttentionDecodePipeline) {
            std::cout << "[MetalBackend] FlashAttention metallib not found or failed. "
                         "Falling back to runtime compilation."
                      << std::endl;
            NSString* flashAttnSource = [NSString stringWithUTF8String:kFlashAttentionShaderSource];
            id<MTLLibrary> flashAttnLib = [impl_->device newLibraryWithSource:flashAttnSource
                                                                      options:compileOptions
                                                                        error:&error];
            if (flashAttnLib) {
                id<MTLFunction> flashAttnDecodeFunction =
                    [flashAttnLib newFunctionWithName:@"flash_attention_decode"];
                if (flashAttnDecodeFunction) {
                    impl_->flashAttentionDecodePipeline =
                        [impl_->device newComputePipelineStateWithFunction:flashAttnDecodeFunction
                                                                     error:&error];
                    if (impl_->flashAttentionDecodePipeline) {
                        std::cout << "[MetalBackend] FlashAttention decode kernel compiled "
                                     "from source"
                                  << std::endl;
                    }
                }
            } else {
                std::cerr << "[MetalBackend] Warning: Failed to compile FlashAttention "
                             "shader: "
                          << [[error localizedDescription] UTF8String] << std::endl;
            }
        }

        // Compile FlashAttention prefill kernel from embedded source
        {
            NSString* flashAttnPrefillSource =
                [NSString stringWithUTF8String:kFlashAttentionPrefillSource];
            id<MTLLibrary> flashAttnPrefillLib =
                [impl_->device newLibraryWithSource:flashAttnPrefillSource
                                            options:compileOptions
                                              error:&error];
            if (flashAttnPrefillLib) {
                id<MTLFunction> flashAttnPrefillFunction =
                    [flashAttnPrefillLib newFunctionWithName:@"flash_attention_prefill"];
                if (flashAttnPrefillFunction) {
                    impl_->flashAttentionPrefillPipeline =
                        [impl_->device newComputePipelineStateWithFunction:flashAttnPrefillFunction
                                                                     error:&error];
                    if (impl_->flashAttentionPrefillPipeline) {
                        std::cout << "[MetalBackend] FlashAttention prefill kernel compiled "
                                     "from source"
                                  << std::endl;
                    }
                }
            } else {
                std::cerr << "[MetalBackend] Warning: Failed to compile FlashAttention "
                             "prefill shader: "
                          << [[error localizedDescription] UTF8String] << std::endl;
            }
        }

        // Compile quantized GEMV kernels (Q4_0, Q4_1, Q8_0)
        NSString* quantizedGemvSource = [NSString stringWithUTF8String:kQuantizedGemvShaderSource];
        id<MTLLibrary> quantizedGemvLib = [impl_->device newLibraryWithSource:quantizedGemvSource
                                                                      options:compileOptions
                                                                        error:&error];
        if (quantizedGemvLib) {
            id<MTLFunction> gemvQ4_0Function = [quantizedGemvLib newFunctionWithName:@"gemv_q4_0"];
            if (gemvQ4_0Function) {
                impl_->gemvQ4_0Pipeline =
                    [impl_->device newComputePipelineStateWithFunction:gemvQ4_0Function
                                                                 error:&error];
            }

            id<MTLFunction> gemvQ4_1Function = [quantizedGemvLib newFunctionWithName:@"gemv_q4_1"];
            if (gemvQ4_1Function) {
                impl_->gemvQ4_1Pipeline =
                    [impl_->device newComputePipelineStateWithFunction:gemvQ4_1Function
                                                                 error:&error];
            }

            id<MTLFunction> gemvQ8_0Function = [quantizedGemvLib newFunctionWithName:@"gemv_q8_0"];
            if (gemvQ8_0Function) {
                impl_->gemvQ8_0Pipeline =
                    [impl_->device newComputePipelineStateWithFunction:gemvQ8_0Function
                                                                 error:&error];
            }

            if (impl_->gemvQ4_0Pipeline || impl_->gemvQ4_1Pipeline || impl_->gemvQ8_0Pipeline) {
                std::cout << "[MetalBackend] Quantized GEMV kernels compiled: "
                          << (impl_->gemvQ4_0Pipeline ? "Q4_0 " : "")
                          << (impl_->gemvQ4_1Pipeline ? "Q4_1 " : "")
                          << (impl_->gemvQ8_0Pipeline ? "Q8_0 " : "") << std::endl;
            }

            // Compile dequantization kernel for M>1 GEMM path
            id<MTLFunction> dequantQ4_0Function =
                [quantizedGemvLib newFunctionWithName:@"dequantize_q4_0"];
            if (dequantQ4_0Function) {
                impl_->dequantizeQ4_0Pipeline =
                    [impl_->device newComputePipelineStateWithFunction:dequantQ4_0Function
                                                                 error:&error];
                if (impl_->dequantizeQ4_0Pipeline) {
                    std::cout << "[MetalBackend] Dequantize Q4_0 kernel compiled" << std::endl;
                }
            }
        } else {
            std::cerr << "[MetalBackend] Warning: Failed to compile quantized GEMV "
                         "shaders: "
                      << [[error localizedDescription] UTF8String] << std::endl;
        }

        // Compile Fused QKV kernel
        NSString* fusedQKVSource = [NSString stringWithUTF8String:kFusedQKVShaderSource];
        id<MTLLibrary> fusedQKVLib = [impl_->device newLibraryWithSource:fusedQKVSource
                                                                 options:compileOptions
                                                                   error:&error];
        if (fusedQKVLib) {
            id<MTLFunction> fusedQKVFunction = [fusedQKVLib newFunctionWithName:@"fused_qkv_gemv"];
            if (fusedQKVFunction) {
                impl_->fusedQKVPipeline =
                    [impl_->device newComputePipelineStateWithFunction:fusedQKVFunction
                                                                 error:&error];
                if (impl_->fusedQKVPipeline) {
                    std::cout << "[MetalBackend] Fused QKV kernel compiled" << std::endl;
                }
            }
        } else {
            std::cerr << "[MetalBackend] Warning: Failed to compile Fused QKV "
                         "shader: "
                      << [[error localizedDescription] UTF8String] << std::endl;
        }

        // Initialize GGML Metal backend
        impl_->ggmlMetalBackend = ggml_backend_metal_init();
        if (impl_->ggmlMetalBackend == nullptr) {
            std::cerr << "[MetalBackend] Warning: Failed to initialize GGML Metal backend, "
                      << "falling back to custom kernels only" << std::endl;
        }

        // Cache chip info
        impl_->chipInfo = GetChipInfo();

        // Set backend name
        snprintf(name_, sizeof(name_), "Apple-Metal-%s", impl_->chipInfo.chip_name);

        std::cout << "[MetalBackend] Initialized: " << name_ << std::endl;
        std::cout << "  GPU Cores: " << impl_->chipInfo.gpu_cores << std::endl;
        std::cout << "  Unified Memory: " << (impl_->chipInfo.unified_memory_bytes >> 30) << " GB"
                  << std::endl;
        std::cout << "  Memory Bandwidth: " << impl_->chipInfo.memory_bandwidth_gbps << " GB/s"
                  << std::endl;
    }
}

MetalBackend::~MetalBackend() {
    // Wait for all GPU work to complete
    Synchronize();

    // impl_ destructor handles cleanup via RAII
}

// ============================================================================
// ComputeBackend Interface - Identification
// ============================================================================

const char* MetalBackend::Name() const {
    return name_;
}

// ============================================================================
// ComputeBackend Interface - Memory Management
// ============================================================================

void* MetalBackend::AllocateDevice(size_t size_bytes, size_t alignment) {
    if (size_bytes == 0) {
        return nullptr;
    }

    @autoreleasepool {
        // Ensure minimum alignment for Metal
        alignment = std::max(alignment, static_cast<size_t>(64));

        // Round up size to alignment
        size_t aligned_size = ((size_bytes + alignment - 1) / alignment) * alignment;

        // Create Metal buffer with shared storage mode (UMA zero-copy)
        id<MTLBuffer> buffer = [impl_->device newBufferWithLength:aligned_size
                                                          options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            std::cerr << "[MetalBackend] Failed to allocate " << aligned_size << " bytes"
                      << std::endl;
            return nullptr;
        }

        // Track memory usage
        size_t current = impl_->currentMemoryUsage.fetch_add(aligned_size) + aligned_size;
        size_t peak = impl_->peakMemoryUsage.load();
        while (current > peak && !impl_->peakMemoryUsage.compare_exchange_weak(peak, current)) {}

        // Return the buffer's contents pointer
        // Note: The buffer itself is retained by ARC, but we need to track it
        // for deallocation. We use the contents pointer as the key.
        void* ptr = [buffer contents];

        // Retain buffer to prevent ARC from releasing
        CFRetain((__bridge CFTypeRef)buffer);

        // Register pointer -> buffer mapping for deallocation
        {
            std::lock_guard<std::mutex> lock(impl_->bufferRegistryMutex);
            impl_->bufferRegistry[ptr] = buffer;
        }

        return ptr;
    }
}

void MetalBackend::FreeDevice(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    @autoreleasepool {
        id<MTLBuffer> buffer = nil;
        size_t bufferLength = 0;

        // Look up the buffer associated with this pointer
        {
            std::lock_guard<std::mutex> lock(impl_->bufferRegistryMutex);
            auto it = impl_->bufferRegistry.find(ptr);
            if (it != impl_->bufferRegistry.end()) {
                buffer = it->second;
                bufferLength = [buffer length];
                impl_->bufferRegistry.erase(it);
            }
        }

        if (buffer) {
            // Update memory tracking
            impl_->currentMemoryUsage.fetch_sub(bufferLength);

            // Release the CFRetain we did in AllocateDevice
            CFRelease((__bridge CFTypeRef)buffer);
        } else {
            std::cerr << "[MetalBackend] Warning: FreeDevice called with untracked "
                         "pointer: "
                      << ptr << std::endl;
        }
    }
}

void MetalBackend::CopyToDevice(void* dst, const void* src, size_t size_bytes) {
    // On Apple Silicon with UMA, this is just a memcpy
    if (dst && src && size_bytes > 0) {
        std::memcpy(dst, src, size_bytes);
    }
}

void MetalBackend::CopyFromDevice(void* dst, const void* src, size_t size_bytes) {
    // On Apple Silicon with UMA, this is just a memcpy
    if (dst && src && size_bytes > 0) {
        std::memcpy(dst, src, size_bytes);
    }
}

// ============================================================================
// ComputeBackend Interface - Matrix Operations
// ============================================================================

void MetalBackend::MatMul(const Tensor& A, const Tensor& B, Tensor* C) {
    if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) {
        return;
    }

    const int M = static_cast<int>(A.shape[0]);
    const int K = static_cast<int>(A.shape[1]);
    const int N = static_cast<int>(B.shape[1]);

    @autoreleasepool {
        if (M == 1 && impl_->gemvPipeline) {
            // GEMV path: Use custom kernel for decode phase
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:impl_->gemvPipeline];
            [encoder setBytes:A.data length:K * sizeof(float) atIndex:0];
            [encoder setBytes:B.data length:N * K * sizeof(float) atIndex:1];
            [encoder setBytes:C->data length:N * sizeof(float) atIndex:2];
            [encoder setBytes:&N length:sizeof(uint) atIndex:3];
            [encoder setBytes:&K length:sizeof(uint) atIndex:4];

            // Launch one threadgroup per output element
            MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
            MTLSize gridSize = MTLSizeMake(1, 1, N);

            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            [commandBuffer commit];
            // [commandBuffer waitUntilCompleted]; // Removed for pipelining
        } else {
            // GEMM path (M > 1): Use Metal Performance Shaders
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];

            // Get or wrap MTLBuffer handles for the tensor data (zero-copy)
            // Uses newBufferWithBytesNoCopy for untracked pointers (UMA zero-copy)
            id<MTLBuffer> bufferA =
                impl_->GetOrWrapBuffer(const_cast<void*>(A.data), A.SizeBytes());
            id<MTLBuffer> bufferB =
                impl_->GetOrWrapBuffer(const_cast<void*>(B.data), B.SizeBytes());
            id<MTLBuffer> bufferC = impl_->GetOrWrapBuffer(C->data, C->SizeBytes());

            if (!bufferA || !bufferB || !bufferC) {
                // Should never happen on Apple Silicon, but safety fallback
                std::cerr << "[MetalBackend] MatMul: Buffer creation failed, falling "
                             "back to CPU"
                          << std::endl;
                apple::GemmAccelerate(C->DataAs<float>(), A.DataAs<float>(), B.DataAs<float>(), M,
                                      N, K);
                return;
            }

            // Calculate row bytes (must be 4-byte aligned for MPS)
            NSUInteger rowBytesA = static_cast<NSUInteger>(K) * sizeof(float);
            NSUInteger rowBytesB = static_cast<NSUInteger>(N) * sizeof(float);
            NSUInteger rowBytesC = static_cast<NSUInteger>(N) * sizeof(float);

            // Create MPS matrix descriptors
            MPSMatrixDescriptor* descA =
                [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                      columns:static_cast<NSUInteger>(K)
                                                     rowBytes:rowBytesA
                                                     dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descB =
                [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(K)
                                                      columns:static_cast<NSUInteger>(N)
                                                     rowBytes:rowBytesB
                                                     dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor* descC =
                [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                      columns:static_cast<NSUInteger>(N)
                                                     rowBytes:rowBytesC
                                                     dataType:MPSDataTypeFloat32];

            // Wrap buffers as MPSMatrix (zero-copy)
            MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
            MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
            MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

            // Create and encode MPS GEMM: C = A @ B
            MPSMatrixMultiplication* gemm =
                [[MPSMatrixMultiplication alloc] initWithDevice:impl_->device
                                                  transposeLeft:NO
                                                 transposeRight:NO
                                                     resultRows:static_cast<NSUInteger>(M)
                                                  resultColumns:static_cast<NSUInteger>(N)
                                                interiorColumns:static_cast<NSUInteger>(K)
                                                          alpha:1.0
                                                           beta:0.0];

            [gemm encodeToCommandBuffer:commandBuffer
                             leftMatrix:matrixA
                            rightMatrix:matrixB
                           resultMatrix:matrixC];

            [commandBuffer commit];
        }
    }
}

void MetalBackend::MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) {
    // For B transposed, use Accelerate with CblasTrans
    if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) {
        return;
    }

    const int M = static_cast<int>(A.shape[0]);
    const int K = static_cast<int>(A.shape[1]);
    const int N = static_cast<int>(B.shape[0]);

    // Use BLAS with transposed B
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A.DataAs<float>(), K,
                B.DataAs<float>(),
                K,  // B is [N, K] so stride is K
                0.0f, C->DataAs<float>(), N);
}

void MetalBackend::GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales,
                            const Tensor& zero_points, Tensor* C, int group_size) {
    if (!A.IsValid() || !W.IsValid() || !scales.IsValid() || !C || !C->IsValid()) {
        return;
    }

    const int M = static_cast<int>(A.shape[0]);
    const int N = static_cast<int>(C->shape[C->ndim - 1]);
    const int K = static_cast<int>(A.shape[A.ndim - 1]);

    // Decode phase (M == 1): Use custom quantized GEMV kernel on GPU
    if (M == 1) {
        @autoreleasepool {
            // Determine quantization format based on zero_points presence
            // Q4_0: scale only (zero_points empty or all zeros)
            // Q4_1: scale + min (zero_points has actual min values)
            const bool use_q4_1 = zero_points.IsValid() && zero_points.NumElements() > 0;

            id<MTLComputePipelineState> pipeline =
                use_q4_1 ? impl_->gemvQ4_1Pipeline : impl_->gemvQ4_0Pipeline;

            if (pipeline) {
                id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

                [encoder setComputePipelineState:pipeline];

                // Buffer 0: input activations [K]
                [encoder setBytes:A.data length:A.SizeBytes() atIndex:0];
                // Buffer 1: packed weights (block_q4_0 or block_q4_1 structs)
                [encoder setBytes:W.data length:W.SizeBytes() atIndex:1];
                // Buffer 2: output [N]
                [encoder setBytes:C->data length:C->SizeBytes() atIndex:2];
                // Buffer 3: M (output rows)
                uint M_u = static_cast<uint>(N);  // For GEMV, N is the output dim
                [encoder setBytes:&M_u length:sizeof(uint) atIndex:3];
                // Buffer 4: K (input dimension)
                uint K_u = static_cast<uint>(K);
                [encoder setBytes:&K_u length:sizeof(uint) atIndex:4];

                // Dispatch: one threadgroup per output element
                MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
                MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(N), 1, 1);

                [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                // [commandBuffer waitUntilCompleted]; // Removed for pipelining
                return;
            }
        }
    }

    // Prefill (M > 1): Use GPU dequantization + MPS GEMM
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];

        // Step 1: Get scratch buffer for dequantized weights [N, K] (reuses pool)
        const size_t deq_size = static_cast<size_t>(N) * K * sizeof(float);
        id<MTLBuffer> dequantBuffer = impl_->GetScratchBuffer(deq_size);

        if (!dequantBuffer) {
            std::cerr << "[MetalBackend] GemmInt4: Failed to get scratch buffer" << std::endl;
            // Fall through to CPU fallback below
        } else if (impl_->dequantizeQ4_0Pipeline) {
            // Step 2: Dispatch dequantization kernel
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
            [encoder setComputePipelineState:impl_->dequantizeQ4_0Pipeline];

            // Get or wrap buffer for packed weights (zero-copy)
            id<MTLBuffer> weightBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(W.data), W.SizeBytes());

            [encoder setBuffer:weightBuffer offset:0 atIndex:0];
            [encoder setBuffer:dequantBuffer offset:0 atIndex:1];
            uint N_u = static_cast<uint>(N);
            uint K_u = static_cast<uint>(K);
            [encoder setBytes:&N_u length:sizeof(uint) atIndex:2];
            [encoder setBytes:&K_u length:sizeof(uint) atIndex:3];

            const int block_size = 32;
            uint blocks_per_row =
                static_cast<uint>((K + block_size - 1) / block_size);  // Ceiling division
            MTLSize gridSize = MTLSizeMake(blocks_per_row, static_cast<NSUInteger>(N), 1);
            MTLSize threadgroupSize = MTLSizeMake(1, 1, 1);  // One thread per block
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            // Step 3: MPS MatMul on dequantized weights
            // Use GetOrWrapBuffer for zero-copy access
            id<MTLBuffer> bufferA =
                impl_->GetOrWrapBuffer(const_cast<void*>(A.data), A.SizeBytes());
            id<MTLBuffer> bufferC = impl_->GetOrWrapBuffer(C->data, C->SizeBytes());

            if (bufferA && bufferC) {
                // A is [M, K], dequantized B is [N, K], we want C = A @ B^T = [M, N]
                NSUInteger rowBytesA = static_cast<NSUInteger>(K) * sizeof(float);
                NSUInteger rowBytesB = static_cast<NSUInteger>(K) * sizeof(float);
                NSUInteger rowBytesC = static_cast<NSUInteger>(N) * sizeof(float);

                MPSMatrixDescriptor* descA =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                          columns:static_cast<NSUInteger>(K)
                                                         rowBytes:rowBytesA
                                                         dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor* descB =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(N)
                                                          columns:static_cast<NSUInteger>(K)
                                                         rowBytes:rowBytesB
                                                         dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor* descC =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(M)
                                                          columns:static_cast<NSUInteger>(N)
                                                         rowBytes:rowBytesC
                                                         dataType:MPSDataTypeFloat32];

                MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
                MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:dequantBuffer
                                                            descriptor:descB];
                MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

                // C = A @ B^T: [M,K] @ [N,K]^T = [M,N]
                MPSMatrixMultiplication* gemm =
                    [[MPSMatrixMultiplication alloc] initWithDevice:impl_->device
                                                      transposeLeft:NO
                                                     transposeRight:YES
                                                         resultRows:static_cast<NSUInteger>(M)
                                                      resultColumns:static_cast<NSUInteger>(N)
                                                    interiorColumns:static_cast<NSUInteger>(K)
                                                              alpha:1.0
                                                               beta:0.0];

                [gemm encodeToCommandBuffer:commandBuffer
                                 leftMatrix:matrixA
                                rightMatrix:matrixB
                               resultMatrix:matrixC];

                [commandBuffer commit];
                // Note: With zero-copy wrapping via newBufferWithBytesNoCopy,
                // results are written directly to C->data, no copy needed

                return;  // Success - exit early
            }
        }
    }

    // CPU fallback only if GPU path failed
    std::cerr << "[MetalBackend] GemmInt4: GPU path failed, using CPU fallback" << std::endl;

    const float* a_data = A.DataAs<float>();
    const uint8_t* w_data = static_cast<const uint8_t*>(W.data);
    const float* scale_data = scales.DataAs<float>();
    float* c_data = C->DataAs<float>();

    const int block_size = 32;  // QK4_0 = QK4_1 = 32
    const int blocks_per_row = K / block_size;

    for (int m = 0; m < M; ++m) {
        const float* a_row = a_data + m * K;
        float* c_row = c_data + m * N;

        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int blk = 0; blk < blocks_per_row; ++blk) {
                float scale = scale_data[n * blocks_per_row + blk];
                int k_start = blk * block_size;
                const uint8_t* block_quants =
                    w_data + (n * blocks_per_row + blk) * (2 + block_size / 2) + 2;

                for (int i = 0; i < block_size; i += 2) {
                    uint8_t packed = block_quants[i / 2];
                    int8_t q0 = static_cast<int8_t>((packed & 0x0F)) - 8;
                    int8_t q1 = static_cast<int8_t>((packed >> 4) & 0x0F) - 8;
                    sum += (static_cast<float>(q0) * scale) * a_row[k_start + i];
                    sum += (static_cast<float>(q1) * scale) * a_row[k_start + i + 1];
                }
            }
            c_row[n] = sum;
        }
    }
}

// ============================================================================
// ComputeBackend Interface - Normalization
// ============================================================================

void MetalBackend::RMSNorm(const Tensor& input, const Tensor& weight, Tensor* output, float eps) {
    if (!input.IsValid() || !weight.IsValid() || !output || !output->IsValid()) {
        return;
    }

    const int64_t dim = weight.shape[0];
    const int64_t n_tokens = input.NumElements() / dim;

    @autoreleasepool {
        if (impl_->rmsNormPipeline) {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:impl_->rmsNormPipeline];
            [encoder setBytes:input.data length:input.SizeBytes() atIndex:0];
            [encoder setBytes:weight.data length:weight.SizeBytes() atIndex:1];
            [encoder setBytes:output->data length:output->SizeBytes() atIndex:2];

            uint dim_u = static_cast<uint>(dim);
            [encoder setBytes:&dim_u length:sizeof(uint) atIndex:3];
            [encoder setBytes:&eps length:sizeof(float) atIndex:4];

            MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
            MTLSize gridSize = MTLSizeMake(1, 1, static_cast<NSUInteger>(n_tokens));

            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            [commandBuffer commit];
            // [commandBuffer waitUntilCompleted]; // Removed for pipelining
        } else {
            // CPU fallback
            const float* x = input.DataAs<float>();
            const float* w = weight.DataAs<float>();
            float* out = output->DataAs<float>();

            for (int64_t t = 0; t < n_tokens; ++t) {
                const float* x_ptr = x + t * dim;
                float* out_ptr = out + t * dim;

                float sum_sq = 0.0f;
                for (int64_t i = 0; i < dim; ++i) {
                    sum_sq += x_ptr[i] * x_ptr[i];
                }
                float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);

                for (int64_t i = 0; i < dim; ++i) {
                    out_ptr[i] = x_ptr[i] * rms * w[i];
                }
            }
        }
    }
}

void MetalBackend::AddRMSNorm(const Tensor& input, const Tensor& residual, const Tensor& weight,
                              Tensor* output, float eps) {
    // Fused add + RMS norm
    // For now, do separately
    // In production, create a fused Metal kernel

    const int64_t n_elements = input.NumElements();
    float* out = output->DataAs<float>();
    const float* in = input.DataAs<float>();
    const float* res = residual.DataAs<float>();

    // Add residual
    for (int64_t i = 0; i < n_elements; ++i) {
        out[i] = in[i] + res[i];
    }

    // Apply RMS norm
    Tensor temp_input = *output;  // Use output as temp input
    RMSNorm(temp_input, weight, output, eps);
}

// ============================================================================
// ComputeBackend Interface - Activation
// ============================================================================

void MetalBackend::Softmax(const Tensor& input, Tensor* output) {
    CopyToDevice(output->data, input.data, input.SizeBytes());
    SoftmaxInplace(output);
}

void MetalBackend::SoftmaxInplace(Tensor* data) {
    if (!data || !data->IsValid()) {
        return;
    }

    const int64_t n = data->shape[data->ndim - 1];
    int64_t batch_size = 1;
    for (int i = 0; i < data->ndim - 1; ++i) {
        batch_size *= data->shape[i];
    }

    @autoreleasepool {
        if (impl_->softmaxPipeline && batch_size == 1) {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:impl_->softmaxPipeline];
            [encoder setBytes:data->data length:data->SizeBytes() atIndex:0];
            uint n_u = static_cast<uint>(n);
            [encoder setBytes:&n_u length:sizeof(uint) atIndex:1];

            MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
            MTLSize gridSize = MTLSizeMake(1, 1, 1);

            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            [commandBuffer commit];
            // [commandBuffer waitUntilCompleted]; // Removed for pipelining
        } else {
            // CPU fallback for batched softmax
            float* ptr = data->DataAs<float>();
            for (int64_t b = 0; b < batch_size; ++b) {
                float* row = ptr + b * n;

                // Find max
                float max_val = row[0];
                for (int64_t i = 1; i < n; ++i) {
                    if (row[i] > max_val)
                        max_val = row[i];
                }

                // Exp and sum
                float sum = 0.0f;
                for (int64_t i = 0; i < n; ++i) {
                    row[i] = std::exp(row[i] - max_val);
                    sum += row[i];
                }

                // Normalize
                float inv_sum = 1.0f / sum;
                for (int64_t i = 0; i < n; ++i) {
                    row[i] *= inv_sum;
                }
            }
        }
    }
}

// ============================================================================
// ComputeBackend Interface - Position Encoding
// ============================================================================

void MetalBackend::RoPE(const Tensor& input, const Tensor& cos_sin, const int* positions,
                        Tensor* output, int rope_dim) {
    if (!input.IsValid() || !cos_sin.IsValid() || !positions || !output || !output->IsValid()) {
        return;
    }

    // Copy input to output first (RoPE is in-place on output)
    CopyToDevice(output->data, input.data, input.SizeBytes());

    // Parse dimensions
    uint n_tokens, head_dim, n_heads;
    if (input.ndim == 2) {
        n_tokens = static_cast<uint>(input.shape[0]);
        head_dim = static_cast<uint>(input.shape[1]);
        n_heads = 1;
    } else {
        n_heads = static_cast<uint>(input.shape[0]);
        n_tokens = static_cast<uint>(input.shape[1]);
        head_dim = static_cast<uint>(input.shape[2]);
    }

    uint rope_dim_u = (rope_dim < 0) ? head_dim : static_cast<uint>(rope_dim);

    // GPU path
    if (impl_->ropePipeline) {
        @autoreleasepool {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:impl_->ropePipeline];

            // Get or wrap buffers - prefer zero-copy via GetOrWrapBuffer
            id<MTLBuffer> dataBuffer = impl_->GetOrWrapBuffer(output->data, output->SizeBytes());
            id<MTLBuffer> cosSinBuffer =
                impl_->GetOrWrapBuffer(const_cast<void*>(cos_sin.data), cos_sin.SizeBytes());

            // Create buffer for positions
            size_t pos_size = n_tokens * sizeof(int);
            id<MTLBuffer> posBuffer =
                [impl_->device newBufferWithBytes:positions
                                           length:pos_size
                                          options:MTLResourceStorageModeShared];

            [encoder setBuffer:dataBuffer offset:0 atIndex:0];
            [encoder setBuffer:cosSinBuffer offset:0 atIndex:1];
            [encoder setBuffer:posBuffer offset:0 atIndex:2];
            [encoder setBytes:&n_heads length:sizeof(uint) atIndex:3];
            [encoder setBytes:&n_tokens length:sizeof(uint) atIndex:4];
            [encoder setBytes:&head_dim length:sizeof(uint) atIndex:5];
            [encoder setBytes:&rope_dim_u length:sizeof(uint) atIndex:6];

            // Grid: (rope_dim/2, n_tokens, n_heads)
            MTLSize gridSize = MTLSizeMake(rope_dim_u / 2, n_tokens, n_heads);
            MTLSize threadgroupSize = MTLSizeMake(std::min(rope_dim_u / 2, 64u), 1, 1);

            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            [commandBuffer commit];
            // Zero-copy via GetOrWrapBuffer - no copy-back needed

            return;
        }
    }

    // CPU fallback (only if GPU pipeline not available)
    float* out = output->DataAs<float>();
    const float* cs = cos_sin.DataAs<float>();

    for (uint t = 0; t < n_tokens; ++t) {
        int pos = positions[t];
        const float* pos_cs = cs + pos * head_dim;

        for (uint h = 0; h < n_heads; ++h) {
            float* token = out + (h * n_tokens + t) * head_dim;

            for (uint d = 0; d < rope_dim_u / 2; ++d) {
                float cos_theta = pos_cs[2 * d];
                float sin_theta = pos_cs[2 * d + 1];

                float x0 = token[2 * d];
                float x1 = token[2 * d + 1];

                token[2 * d] = x0 * cos_theta - x1 * sin_theta;
                token[2 * d + 1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
}

// ============================================================================
// ComputeBackend Interface - Fused Operations
// ============================================================================

void MetalBackend::FusedQKVProjection(const Tensor& input, const Tensor& wq, const Tensor& wk,
                                      const Tensor& wv, Tensor* q_out, Tensor* k_out,
                                      Tensor* v_out) {
    if (!input.IsValid() || !wq.IsValid() || !wk.IsValid() || !wv.IsValid() || !q_out || !k_out ||
        !v_out) {
        return;
    }

    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(input.shape[input.ndim - 1]);
    const int hidden_q = static_cast<int>(wq.shape[0]);
    const int hidden_kv = static_cast<int>(wk.shape[0]);

    // Decode phase (M == 1): Use fused kernel for single dispatch
    if (M == 1 && impl_->fusedQKVPipeline) {
        @autoreleasepool {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:impl_->fusedQKVPipeline];

            // Set buffers
            [encoder setBytes:input.data length:input.SizeBytes() atIndex:0];
            [encoder setBytes:wq.data length:wq.SizeBytes() atIndex:1];
            [encoder setBytes:wk.data length:wk.SizeBytes() atIndex:2];
            [encoder setBytes:wv.data length:wv.SizeBytes() atIndex:3];
            [encoder setBytes:q_out->data length:q_out->SizeBytes() atIndex:4];
            [encoder setBytes:k_out->data length:k_out->SizeBytes() atIndex:5];
            [encoder setBytes:v_out->data length:v_out->SizeBytes() atIndex:6];

            // Set constants
            uint K_u = static_cast<uint>(K);
            uint hidden_q_u = static_cast<uint>(hidden_q);
            uint hidden_kv_u = static_cast<uint>(hidden_kv);
            [encoder setBytes:&K_u length:sizeof(uint) atIndex:7];
            [encoder setBytes:&hidden_q_u length:sizeof(uint) atIndex:8];
            [encoder setBytes:&hidden_kv_u length:sizeof(uint) atIndex:9];

            // Total rows: hidden_q (Q) + hidden_kv (K) + hidden_kv (V)
            uint total_rows = hidden_q + 2 * hidden_kv;
            MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
            MTLSize gridSize = MTLSizeMake(static_cast<NSUInteger>(total_rows), 1, 1);

            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            [commandBuffer commit];
            // [commandBuffer waitUntilCompleted]; // Removed for pipelining
            return;
        }
    }

    // Prefill (M > 1) or no GPU pipeline: Fallback to three separate MatMuls
    MatMulTransB(input, wq, q_out);
    MatMulTransB(input, wk, k_out);
    MatMulTransB(input, wv, v_out);
}

void MetalBackend::FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output,
                                  float scale, bool causal, int n_head_kv) {
    if (!Q.IsValid() || !K.IsValid() || !V.IsValid() || !output || !output->IsValid()) {
        return;
    }

    const int batch = static_cast<int>(Q.shape[0]);
    const int n_head = static_cast<int>(Q.shape[1]);
    const int seq_q = static_cast<int>(Q.shape[2]);
    const int head_dim = static_cast<int>(Q.shape[3]);
    const int seq_kv = static_cast<int>(K.shape[2]);

    if (n_head_kv <= 0)
        n_head_kv = static_cast<int>(K.shape[1]);

    // ==========================================================================
    // GPU Path: Use Metal FlashAttention for decode (seq_q == 1)
    // ==========================================================================
    if (seq_q == 1 && impl_->flashAttentionDecodePipeline) {
        @autoreleasepool {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:impl_->flashAttentionDecodePipeline];

            // Set buffers - use zero-copy MTLBuffer for UMA efficiency
            id<MTLBuffer> bufferQ =
                impl_->GetOrWrapBuffer(const_cast<void*>(Q.data), Q.SizeBytes());
            id<MTLBuffer> bufferK =
                impl_->GetOrWrapBuffer(const_cast<void*>(K.data), K.SizeBytes());
            id<MTLBuffer> bufferV =
                impl_->GetOrWrapBuffer(const_cast<void*>(V.data), V.SizeBytes());
            id<MTLBuffer> bufferOut = impl_->GetOrWrapBuffer(output->data, output->SizeBytes());

            [encoder setBuffer:bufferQ offset:0 atIndex:0];
            [encoder setBuffer:bufferK offset:0 atIndex:1];
            [encoder setBuffer:bufferV offset:0 atIndex:2];
            [encoder setBuffer:bufferOut offset:0 atIndex:3];

            // Set constants
            [encoder setBytes:&scale length:sizeof(float) atIndex:4];
            uint seq_kv_u = static_cast<uint>(seq_kv);
            uint head_dim_u = static_cast<uint>(head_dim);
            uint n_heads_u = static_cast<uint>(n_head);
            uint n_kv_heads_u = static_cast<uint>(n_head_kv);
            [encoder setBytes:&seq_kv_u length:sizeof(uint) atIndex:5];
            [encoder setBytes:&head_dim_u length:sizeof(uint) atIndex:6];
            [encoder setBytes:&n_heads_u length:sizeof(uint) atIndex:7];
            [encoder setBytes:&n_kv_heads_u length:sizeof(uint) atIndex:8];

            // Dispatch: one threadgroup per (batch, head) pair
            // Threadgroup size: 256 threads (handles K/V sequence parallelism)
            MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
            MTLSize gridSize =
                MTLSizeMake(static_cast<NSUInteger>(n_head), 1, static_cast<NSUInteger>(batch));

            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            [commandBuffer commit];
            // [commandBuffer waitUntilCompleted]; // Removed for pipelining
            return;
        }
    }

    // ==========================================================================
    // GPU Path: Use Metal FlashAttention for prefill (seq_q > 1)
    // ==========================================================================
    if (seq_q > 1 && impl_->flashAttentionPrefillPipeline) {
        @autoreleasepool {
            id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

            [encoder setComputePipelineState:impl_->flashAttentionPrefillPipeline];

            // Set buffers - use zero-copy MTLBuffer for UMA efficiency
            id<MTLBuffer> bufferQ =
                impl_->GetOrWrapBuffer(const_cast<void*>(Q.data), Q.SizeBytes());
            id<MTLBuffer> bufferK =
                impl_->GetOrWrapBuffer(const_cast<void*>(K.data), K.SizeBytes());
            id<MTLBuffer> bufferV =
                impl_->GetOrWrapBuffer(const_cast<void*>(V.data), V.SizeBytes());
            id<MTLBuffer> bufferOut = impl_->GetOrWrapBuffer(output->data, output->SizeBytes());

            [encoder setBuffer:bufferQ offset:0 atIndex:0];
            [encoder setBuffer:bufferK offset:0 atIndex:1];
            [encoder setBuffer:bufferV offset:0 atIndex:2];
            [encoder setBuffer:bufferOut offset:0 atIndex:3];

            // Set constants
            [encoder setBytes:&scale length:sizeof(float) atIndex:4];
            uint seq_q_u = static_cast<uint>(seq_q);
            uint seq_kv_u = static_cast<uint>(seq_kv);
            uint head_dim_u = static_cast<uint>(head_dim);
            uint n_heads_u = static_cast<uint>(n_head);
            uint n_kv_heads_u = static_cast<uint>(n_head_kv);
            uint causal_u = causal ? 1 : 0;
            [encoder setBytes:&seq_q_u length:sizeof(uint) atIndex:5];
            [encoder setBytes:&seq_kv_u length:sizeof(uint) atIndex:6];
            [encoder setBytes:&head_dim_u length:sizeof(uint) atIndex:7];
            [encoder setBytes:&n_heads_u length:sizeof(uint) atIndex:8];
            [encoder setBytes:&n_kv_heads_u length:sizeof(uint) atIndex:9];
            [encoder setBytes:&causal_u length:sizeof(uint) atIndex:10];

            // Dispatch: one threadgroup per (head, q_pos, batch) triple
            // Threadgroup size: 256 threads (handles K/V sequence parallelism)
            MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
            MTLSize gridSize =
                MTLSizeMake(static_cast<NSUInteger>(n_head), static_cast<NSUInteger>(seq_q),
                            static_cast<NSUInteger>(batch));

            [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            [commandBuffer commit];
            // [commandBuffer waitUntilCompleted]; // Removed for pipelining
            return;
        }
    }

    // ==========================================================================
    // CPU Fallback: Naive O(N^2) attention when GPU pipelines unavailable
    // ==========================================================================
    std::cerr << "[MetalBackend] Warning: FlashAttention falling back to CPU. seq_q=" << seq_q
              << std::endl;

    const int n_rep = n_head / n_head_kv;  // GQA repetition factor

    const float* q_data = Q.DataAs<float>();
    const float* k_data = K.DataAs<float>();
    const float* v_data = V.DataAs<float>();
    float* o_data = output->DataAs<float>();

    // Allocate temporary scores
    std::vector<float> scores(seq_q * seq_kv);

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < n_head; ++h) {
            int h_kv = h / n_rep;  // KV head index for GQA

            // Q @ K^T
            for (int i = 0; i < seq_q; ++i) {
                for (int j = 0; j < seq_kv; ++j) {
                    float dot = 0.0f;
                    const float* q_ptr = q_data + ((b * n_head + h) * seq_q + i) * head_dim;
                    const float* k_ptr = k_data + ((b * n_head_kv + h_kv) * seq_kv + j) * head_dim;

                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_ptr[d] * k_ptr[d];
                    }

                    scores[i * seq_kv + j] = dot * scale;

                    // Causal mask
                    if (causal && j > i) {
                        scores[i * seq_kv + j] = -INFINITY;
                    }
                }
            }

            // Softmax per row
            for (int i = 0; i < seq_q; ++i) {
                float* row = scores.data() + i * seq_kv;

                float max_val = row[0];
                for (int j = 1; j < seq_kv; ++j) {
                    if (row[j] > max_val)
                        max_val = row[j];
                }

                float sum = 0.0f;
                for (int j = 0; j < seq_kv; ++j) {
                    row[j] = std::exp(row[j] - max_val);
                    sum += row[j];
                }

                for (int j = 0; j < seq_kv; ++j) {
                    row[j] /= sum;
                }
            }

            // Scores @ V
            for (int i = 0; i < seq_q; ++i) {
                float* o_ptr = o_data + ((b * n_head + h) * seq_q + i) * head_dim;

                for (int d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_kv; ++j) {
                        const float* v_ptr =
                            v_data + ((b * n_head_kv + h_kv) * seq_kv + j) * head_dim;
                        sum += scores[i * seq_kv + j] * v_ptr[d];
                    }
                    o_ptr[d] = sum;
                }
            }
        }
    }
}

// ============================================================================
// ComputeBackend Interface - Synchronization
// ============================================================================

void MetalBackend::Synchronize() {
    @autoreleasepool {
        // Create a completion fence
        id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

// ============================================================================
// Metal-Specific APIs
// ============================================================================

AppleSiliconChipInfo MetalBackend::GetDetailedChipInfo() const {
    return impl_->chipInfo;
}

bool MetalBackend::SupportsGPUFamily(int family) const {
    @autoreleasepool {
        return [impl_->device supportsFamily:static_cast<MTLGPUFamily>(family)];
    }
}

size_t MetalBackend::GetCurrentMemoryUsage() const {
    return impl_->currentMemoryUsage.load();
}

size_t MetalBackend::GetPeakMemoryUsage() const {
    return impl_->peakMemoryUsage.load();
}

void MetalBackend::EnableGPUCapture(const char* capture_path) {
    @autoreleasepool {
        MTLCaptureManager* captureManager = [MTLCaptureManager sharedCaptureManager];
        MTLCaptureDescriptor* descriptor = [[MTLCaptureDescriptor alloc] init];
        descriptor.captureObject = impl_->device;

        if (capture_path) {
            descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
            descriptor.outputURL =
                [NSURL fileURLWithPath:[NSString stringWithUTF8String:capture_path]];
        } else {
            descriptor.destination = MTLCaptureDestinationDeveloperTools;
        }

        NSError* error = nil;
        if ([captureManager startCaptureWithDescriptor:descriptor error:&error]) {
            impl_->captureEnabled = true;
        } else {
            std::cerr << "[MetalBackend] Failed to start GPU capture: " <<
                [[error localizedDescription] UTF8String] << std::endl;
        }
    }
}

void MetalBackend::DisableGPUCapture() {
    @autoreleasepool {
        if (impl_->captureEnabled) {
            [[MTLCaptureManager sharedCaptureManager] stopCapture];
            impl_->captureEnabled = false;
        }
    }
}

}  // namespace densecore

#endif  // __APPLE__
