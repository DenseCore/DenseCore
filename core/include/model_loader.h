#ifndef DENSECORE_MODEL_LOADER_H
#define DENSECORE_MODEL_LOADER_H

#include "model_types.h"

TransformerModel *LoadGGUFModel(const char *path);

/**
 * Save model to GGUF file
 * @param model Pointer to model to save
 * @param path Output path
 * @return 0 on success, negative on error
 */
int SaveModel(const TransformerModel *model, const char *path);

#endif // DENSECORE_MODEL_LOADER_H
