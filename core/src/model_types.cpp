#include "model_types.h"
#include "numa_allocator.h"

// TransformerModel destructor
TransformerModel::~TransformerModel() {
  // =========================================================================
  // CRITICAL: Free NUMA-rebound tensor buffers FIRST
  // =========================================================================
  // These buffers were allocated via NumaAllocator for tensor data and are
  // NOT owned by ggml_context. Failing to free them causes memory leaks.
  // =========================================================================
  for (auto &buf : numa_buffers) {
    if (buf.first && buf.second > 0) {
      densecore::NumaAllocator::Free(buf.first, buf.second,
                                     densecore::AllocationType::Aligned);
    }
  }
  numa_buffers.clear();

  // Standard GGML cleanup
  if (ctx_w)
    ggml_free(ctx_w);
  if (ctx_gguf)
    gguf_free(ctx_gguf);
  if (backend)
    ggml_backend_free(backend);
}
