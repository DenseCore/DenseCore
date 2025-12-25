/**
 * @file gemm_int4_example.cpp
 * @brief Usage example for AVX512 INT4 GEMM kernel
 *
 * Demonstrates how to use the high-performance INT4 GEMM kernel
 * for efficient LLM inference on CPUs.
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "quantization/int4_types.h"
#include "simd_ops.h"

using namespace densecore;
using namespace densecore::simd;

/**
 * Example: Matrix multiplication with INT4 quantized weights
 */
void example_int4_gemm() {
    // Problem dimensions
    const int M = 4;             // Number of output rows (batch size / sequence length)
    const int N = 256;           // Number of output columns (model dim)
    const int K = 1024;          // Inner dimension (input dim)
    const int group_size = 128;  // Quantization block size

    std::cout << "========================================" << std::endl;
    std::cout << "INT4 GEMM Example" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Matrix dimensions: C[" << M << "×" << N << "] = "
              << "A[" << M << "×" << K << "] * W[" << N << "×" << K << "]^T" << std::endl;
    std::cout << "Quantization: INT4 block-wise, group_size=" << group_size << std::endl;
    std::cout << std::endl;

    // Allocate activation matrix (FP32)
    std::vector<float> A(M * K);
    for (int i = 0; i < M * K; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;  // [-0.5, 0.5]
    }

    // Allocate output matrix (FP32)
    std::vector<float> C(M * N, 0.0f);

    // Prepare quantized weights
    const int num_groups = K / group_size;
    std::vector<uint8_t> W_int4(N * K / 2);  // Packed INT4 weights
    std::vector<float> scales(N * num_groups);
    std::vector<float> zero_points(N * num_groups);

    // Simulate quantized weights (in practice, these would come from
    // QuantizeModel)
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < num_groups; g++) {
            // Random scale and zero-point
            scales[n * num_groups + g] = 0.01f + static_cast<float>(rand()) / RAND_MAX * 0.05f;
            zero_points[n * num_groups + g] = 0.0f;

            // Random packed weights
            for (int k = 0; k < group_size / 2; k++) {
                int idx = n * (K / 2) + g * (group_size / 2) + k;
                W_int4[idx] = static_cast<uint8_t>(rand() & 0xFF);
            }
        }
    }

    std::cout << "Running INT4 GEMM kernel..." << std::endl;

    // Call the AVX512 GEMM kernel
    GemmInt4Fp32_AVX512(C.data(),            // Output [M × N]
                        A.data(),            // Activations [M × K]
                        W_int4.data(),       // Packed INT4 weights [N × K/2]
                        scales.data(),       // Per-group scales [N × num_groups]
                        zero_points.data(),  // Per-group zero-points [N × num_groups]
                        M, N, K, group_size);

    std::cout << "✓ GEMM completed successfully!" << std::endl;
    std::cout << std::endl;

    // Display sample output
    std::cout << "Sample output values (first 10 elements):" << std::endl;
    for (int m = 0; m < std::min(2, M); m++) {
        std::cout << "  Row " << m << ": ";
        for (int n = 0; n < std::min(5, N); n++) {
            std::cout << C[m * N + n] << " ";
        }
        std::cout << "..." << std::endl;
    }
    std::cout << std::endl;

    // Performance estimation
    const long ops = 2L * M * N * K;  // Multiply-add operations
    std::cout << "Total operations: " << ops / 1e9 << " GFLOPS" << std::endl;
    std::cout << "Memory saved: " << (N * K * sizeof(float) - W_int4.size()) / 1e6 << " MB vs FP32"
              << std::endl;
}

/**
 * Example: Integration with quantized layer
 */
void example_layer_integration() {
    std::cout << "========================================" << std::endl;
    std::cout << "Layer Integration Example" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Simulate a transformer layer
    const int seq_len = 8;
    const int hidden_dim = 512;
    const int group_size = 64;

    // Input activations [seq_len × hidden_dim]
    std::vector<float> input(seq_len * hidden_dim, 0.1f);

    // Output [seq_len × hidden_dim]
    std::vector<float> output(seq_len * hidden_dim);

    // Weight matrix (quantized to INT4)
    const int num_groups = hidden_dim / group_size;
    std::vector<uint8_t> weight_int4(hidden_dim * hidden_dim / 2);
    std::vector<float> scales(hidden_dim * num_groups, 0.02f);
    std::vector<float> zeros(hidden_dim * num_groups, 0.0f);

    // Initialize with random quantized values
    for (size_t i = 0; i < weight_int4.size(); i++) {
        weight_int4[i] = static_cast<uint8_t>(rand() & 0xFF);
    }

    std::cout << "Computing linear layer: Y = X @ W^T" << std::endl;
    std::cout << "  Input shape: [" << seq_len << ", " << hidden_dim << "]" << std::endl;
    std::cout << "  Weight shape: [" << hidden_dim << ", " << hidden_dim << "] (INT4)" << std::endl;
    std::cout << std::endl;

    // Perform GEMM
    GemmInt4Fp32_AVX512(output.data(), input.data(), weight_int4.data(), scales.data(),
                        zeros.data(),
                        seq_len,     // M
                        hidden_dim,  // N
                        hidden_dim,  // K
                        group_size);

    std::cout << "✓ Layer computation complete!" << std::endl;
    std::cout << "  Output shape: [" << seq_len << ", " << hidden_dim << "]" << std::endl;
}

/**
 * Performance comparison: INT4 vs FP32
 */
void example_performance_comparison() {
    std::cout << "========================================" << std::endl;
    std::cout << "Performance Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Typical LLM inference scenario:" << std::endl;
    std::cout << "  - Model: 7B parameters" << std::endl;
    std::cout << "  - Precision: FP32 → INT4" << std::endl;
    std::cout << "  - Group size: 128" << std::endl;
    std::cout << std::endl;

    const long model_size_fp32 = 7L * 1000 * 1000 * 1000 * 4;        // 7B × 4 bytes
    const long model_size_int4 = 7L * 1000 * 1000 * 1000 / 2;        // 7B × 0.5 bytes
    const long metadata_size = (7L * 1000 * 1000 * 1000 / 128) * 8;  // scales + zeros

    std::cout << "Memory usage:" << std::endl;
    std::cout << "  FP32:  " << model_size_fp32 / 1e9 << " GB" << std::endl;
    std::cout << "  INT4:  " << (model_size_int4 + metadata_size) / 1e9 << " GB" << std::endl;
    std::cout << "  Savings: "
              << ((1.0 - (double)(model_size_int4 + metadata_size) / model_size_fp32) * 100) << "%"
              << std::endl;
    std::cout << std::endl;

    std::cout << "Bandwidth requirements (for 1 token):" << std::endl;
    std::cout << "  FP32:  ~28 GB/s (assuming 7B model @ 1 TFLOPS)" << std::endl;
    std::cout << "  INT4:  ~4 GB/s (8× reduction)" << std::endl;
    std::cout << std::endl;

    std::cout << "Expected speedup on CPU:" << std::endl;
    std::cout << "  Memory-bound workload: 4-6× faster" << std::endl;
    std::cout << "  With AVX512 optimizations: 6-8× faster" << std::endl;
}

int main() {
    // Detect SIMD capabilities
    SimdLevel level = DetectSimdLevel();
    std::cout << "Detected SIMD level: " << SimdLevelName(level) << std::endl;
    std::cout << std::endl;

    if (level < SimdLevel::AVX512) {
        std::cout << "⚠ Warning: AVX512 not available. Using fallback implementation." << std::endl;
        std::cout << std::endl;
    }

    // Run examples
    example_int4_gemm();
    example_layer_integration();
    example_performance_comparison();

    return 0;
}
