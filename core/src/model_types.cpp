#include "model_types.h"

// TransformerModel destructor
TransformerModel::~TransformerModel() {
  if (ctx_w)
    ggml_free(ctx_w);
  if (ctx_gguf)
    gguf_free(ctx_gguf);
  if (backend)
    ggml_backend_free(backend);
}
