#ifndef DENSECORE_MODEL_LOADER_H
#define DENSECORE_MODEL_LOADER_H

#include "model_types.h"

/**
 * Load GGUF model (standard mmap-based loading)
 */
TransformerModel *LoadGGUFModel(const char *path);

/**
 * Load GGUF model with NUMA-aware memory placement
 *
 * After initial mmap load, rebinds large tensor data (>1MB) to the specified
 * NUMA node for optimized memory bandwidth on multi-socket systems.
 *
 * @param path Path to GGUF file
 * @param numa_node Target NUMA node for tensor data (-1 for auto/local)
 * @param use_huge_pages Whether to use huge pages for tensor buffers
 * @return Loaded model with NUMA-optimized memory layout, or nullptr on error
 */
TransformerModel *LoadGGUFModelNuma(const char *path, int numa_node = -1,
                                    bool use_huge_pages = false);

/**
 * Save model to GGUF file
 * @param model Pointer to model to save
 * @param path Output path
 * @return 0 on success, negative on error
 */
int SaveModel(const TransformerModel *model, const char *path);

#endif // DENSECORE_MODEL_LOADER_H
