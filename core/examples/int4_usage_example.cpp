/**
 * @file int4_usage_example.cpp
 * @brief Usage example for INT4 block-wise quantization
 *
 * This example demonstrates how to use the custom INT4 quantizer
 * to quantize a model's weights for efficient inference.
 */

#include <cstring>
#include <iostream>

#include "densecore.h"

/**
 * Example 1: Quantize a model using C API
 */
void example_quantize_model() {
    const char* model_path = "model_fp16.gguf";
    const char* output_path = "model_int4.gguf";

    // Configuration JSON for INT4 quantization
    const char* config = R"({
        "format": "int4_blockwise",
        "block_size": 128,
        "algorithm": "max"
    })";

    std::cout << "Quantizing model to INT4..." << std::endl;
    int result = QuantizeModel(model_path, output_path, config);

    if (result == 0) {
        std::cout << "✓ Quantization successful!" << std::endl;
        std::cout << "  Output: " << output_path << std::endl;
    } else {
        std::cerr << "✗ Quantization failed with error code: " << result << std::endl;
    }
}

/**
 * Example 2: Different block sizes
 */
void example_different_block_sizes() {
    // Group size 32 - smallest blocks, more overhead but finer granularity
    const char* config_32 = R"({
        "format": "int4_blockwise",
        "block_size": 32
    })";
    QuantizeModel("input.gguf", "output_block32.gguf", config_32);

    // Group size 64 - balanced
    const char* config_64 = R"({
        "format": "int4_blockwise",
        "block_size": 64
    })";
    QuantizeModel("input.gguf", "output_block64.gguf", config_64);

    // Group size 128 - largest blocks, optimal for AVX512 (recommended)
    const char* config_128 = R"({
        "format": "int4_blockwise",
        "block_size": 128
    })";
    QuantizeModel("input.gguf", "output_block128.gguf", config_128);
}

/**
 * Example 3: Programmatic usage with C++ API
 */
#ifdef __cplusplus
#include "quantization/int4_quantizer.h"
#include "quantization_config.h"

void example_cpp_api() {
    using namespace densecore;

    // Create configuration
    QuantConfig config;
    config.format = QuantFormat::INT4_BLOCKWISE;
    config.block_size = 128;
    config.skip_output_layer = true;  // Keep output layer in FP16
    config.skip_embeddings = true;    // Keep embeddings in FP16

    // Create quantizer
    auto quantizer = std::make_unique<INT4Quantizer>(config);

    // Load model (pseudo-code, actual implementation would use LoadGGUFModel)
    // TransformerModel* model = LoadGGUFModel("model.gguf");

    // Quantize all weight tensors
    // for (auto& layer : model->layers) {
    //     if (layer.wq) quantizer->QuantizeWeight(layer.wq);
    //     if (layer.wk) quantizer->QuantizeWeight(layer.wk);
    //     if (layer.wv) quantizer->QuantizeWeight(layer.wv);
    //     if (layer.wo) quantizer->QuantizeWeight(layer.wo);
    //     if (layer.w1) quantizer->QuantizeWeight(layer.w1);
    //     if (layer.w2) quantizer->QuantizeWeight(layer.w2);
    //     if (layer.w3) quantizer->QuantizeWeight(layer.w3);
    // }

    std::cout << "Model quantized using C++ API" << std::endl;
}
#endif

/**
 * Example 4: Python usage (via ctypes)
 */
void example_python_usage() {
    std::cout << R"(
Python Usage Example:
---------------------
import ctypes
import json

# Load DenseCore library
lib = ctypes.CDLL('./libdensecore.so')

# Configure quantization
config = {
    "format": "int4_blockwise",
    "block_size": 128,
    "algorithm": "max"
}
config_json = json.dumps(config)

# Call QuantizeModel
result = lib.QuantizeModel(
    b"model_fp16.gguf",
    b"model_int4.gguf",
    config_json.encode()
)

if result == 0:
    print("✓ Quantization successful")
else:
    print(f"✗ Error: {result}")
    )" << std::endl;
}

/**
 * Main function - run all examples
 */
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "DenseCore INT4 Quantization Examples" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

// Uncomment to run examples
// example_quantize_model();
// example_different_block_sizes();
#ifdef __cplusplus
// example_cpp_api();
#endif
    example_python_usage();

    std::cout << std::endl << "Memory Layout for INT4 Quantization:" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "Block Size 32:  24 bytes per block  (8 + 16)" << std::endl;
    std::cout << "Block Size 64:  40 bytes per block  (8 + 32)" << std::endl;
    std::cout << "Block Size 128: 72 bytes per block  (8 + 64)" << std::endl;
    std::cout << std::endl;
    std::cout << "Each block contains:" << std::endl;
    std::cout << "  - 4 bytes: FP32 scale" << std::endl;
    std::cout << "  - 4 bytes: FP32 zero-point" << std::endl;
    std::cout << "  - N/2 bytes: Packed INT4 weights" << std::endl;
    std::cout << "  - Padding to 64-byte alignment" << std::endl;
    std::cout << std::endl;
    std::cout << "Compression ratio: ~7-8x vs FP32" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
