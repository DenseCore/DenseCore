#include "model_loader.h"
#include "hardware_topology.h"
#include "inference.h" // For InitRoPETable
#include "numa_allocator.h"
#include <cstring>
#include <ggml-cpu.h>
#include <iostream>
#include <string>
#include <thread>

// ============================================================================
// Mock Model (Test Build Only)
// ============================================================================

#ifdef DENSECORE_TEST_BUILD
/**
 * Creates a mock model for testing purposes.
 * Only available when DENSECORE_TEST_BUILD is defined.
 */
static TransformerModel *CreateMockModel() {
  std::cout << "[DenseCore] Initializing MOCK model..." << std::endl;
  TransformerModel *model = new TransformerModel();
  model->is_mock = true;
  model->hparams.n_vocab = 32000;
  model->hparams.n_embd = 256;
  model->hparams.n_layer = 2; // Small for mock
  model->hparams.n_head = 8;
  model->hparams.n_head_kv = 8;
  model->hparams.n_rot = 64;

  // Mock vocab
  for (int i = 0; i < 32000; i++) {
    std::string s = "t" + std::to_string(i);
    model->vocab_tokens.push_back(s);
    model->token_to_id[s] = i;
  }

  // Initialize backend
  model->backend = ggml_backend_cpu_init();
  if (!model->backend) {
    std::cerr << "[DenseCore] Error: Failed to initialize CPU backend"
              << std::endl;
    delete model;
    return nullptr;
  }

  // Allocate dummy weights
  struct ggml_init_params params = {
      .mem_size = 1024LL * 1024LL * 1024LL * 2LL, // 2 GB for dummy weights
      .mem_buffer = nullptr,
      .no_alloc = false,
  };
  model->ctx_w = ggml_init(params);

  auto create_tensor = [&](int ne0, int ne1) {
    struct ggml_tensor *t =
        ggml_new_tensor_2d(model->ctx_w, GGML_TYPE_F32, ne0, ne1);
    ggml_set_f32(t, 0.01f); // Set to small value
    return t;
  };

  auto create_tensor_1d = [&](int ne0) {
    struct ggml_tensor *t =
        ggml_new_tensor_1d(model->ctx_w, GGML_TYPE_F32, ne0);
    ggml_set_f32(t, 0.01f);
    return t;
  };

  model->tok_embeddings =
      create_tensor(model->hparams.n_embd, model->hparams.n_vocab);
  model->output_norm = create_tensor_1d(model->hparams.n_embd);
  model->output = create_tensor(model->hparams.n_embd, model->hparams.n_vocab);

  model->layers.resize(model->hparams.n_layer);
  for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
    model->layers[i].attention_norm = create_tensor_1d(model->hparams.n_embd);
    model->layers[i].ffn_norm = create_tensor_1d(model->hparams.n_embd);

    model->layers[i].wq =
        create_tensor(model->hparams.n_embd, model->hparams.n_embd);
    model->layers[i].wk =
        create_tensor(model->hparams.n_embd, model->hparams.n_embd);
    model->layers[i].wv =
        create_tensor(model->hparams.n_embd, model->hparams.n_embd);
    model->layers[i].wo =
        create_tensor(model->hparams.n_embd, model->hparams.n_embd);

    model->layers[i].w1 =
        create_tensor(model->hparams.n_embd, model->hparams.n_embd * 4);
    model->layers[i].w2 =
        create_tensor(model->hparams.n_embd * 4, model->hparams.n_embd);
    model->layers[i].w3 =
        create_tensor(model->hparams.n_embd, model->hparams.n_embd * 4);
  }

  return model;
}
#endif // DENSECORE_TEST_BUILD

// ============================================================================
// LoadGGUFModel
// ============================================================================

TransformerModel *LoadGGUFModel(const char *path) {
  if (!path) {
    std::cerr << "Error: Model path is NULL" << std::endl;
    return nullptr;
  }
  std::cout << "[DenseCore] Loading model from '" << path << "'..."
            << std::endl;

  struct gguf_init_params params = {
      .no_alloc = false,
      .ctx = nullptr,
  };

  struct ggml_context *ctx_w = nullptr;
  params.ctx = &ctx_w;

#ifdef DENSECORE_TEST_BUILD
  if (std::string(path) == "mock") {
    return CreateMockModel();
  }
#else
  if (std::string(path) == "mock") {
    std::cerr << "[DenseCore] Error: Mock model not available in release build"
              << std::endl;
    return nullptr;
  }
#endif

  struct gguf_context *ctx_gguf = gguf_init_from_file(path, params);
  if (!ctx_gguf) {
    std::cerr << "[DenseCore] Error: Failed to load GGUF file" << std::endl;
    return nullptr;
  }

  TransformerModel *model = new TransformerModel();
  model->ctx_gguf = ctx_gguf;
  model->ctx_w = ctx_w;

  // 1. Detect architecture from GGUF metadata
  std::string arch = "llama"; // default fallback
  int idx_arch = gguf_find_key(ctx_gguf, "general.architecture");
  if (idx_arch != -1) {
    arch = gguf_get_val_str(ctx_gguf, idx_arch);
  }

  std::cout << "[DenseCore] Detected architecture: " << arch << std::endl;

  // 2. Generic parameter loader using architecture prefix
  auto get_u32 = [&](const std::string &suffix, uint32_t &val) {
    // Try architecture-specific key first, then fallback to general
    std::string key = arch + "." + suffix;
    int idx = gguf_find_key(ctx_gguf, key.c_str());
    if (idx == -1) {
      // Fallback to "general." prefix
      key = "general." + suffix;
      idx = gguf_find_key(ctx_gguf, key.c_str());
    }
    if (idx != -1) {
      val = gguf_get_val_u32(ctx_gguf, idx);
    }
  };

  auto get_f32 = [&](const std::string &suffix, float &val) {
    std::string key = arch + "." + suffix;
    int idx = gguf_find_key(ctx_gguf, key.c_str());
    if (idx == -1) {
      key = "general." + suffix;
      idx = gguf_find_key(ctx_gguf, key.c_str());
    }
    if (idx != -1) {
      val = gguf_get_val_f32(ctx_gguf, idx);
    }
  };

  // Load hyperparameters using dynamic architecture prefix
  get_u32("vocab_size", model->hparams.n_vocab);
  get_u32("embedding_length", model->hparams.n_embd);
  get_u32("block_count", model->hparams.n_layer);
  get_u32("attention.head_count", model->hparams.n_head);
  get_u32("attention.head_count_kv", model->hparams.n_head_kv);
  get_u32("context_length", model->hparams.n_ctx);

  if (model->hparams.n_head_kv == 0)
    model->hparams.n_head_kv = model->hparams.n_head;

  // Calculate n_rot (rotary embedding dimension)
  model->hparams.n_rot = model->hparams.n_embd / model->hparams.n_head;

  // llama.cpp style: Load head dimensions from GGUF
  get_u32("attention.key_length", model->hparams.n_embd_head_k);
  get_u32("attention.value_length", model->hparams.n_embd_head_v);

  // Fix: Update n_rot if head dimension is explicitly specified and differs
  if (model->hparams.n_embd_head_k > 0 &&
      model->hparams.n_embd_head_k != model->hparams.n_rot) {
    std::cout << "[DenseCore] Updating n_rot (" << model->hparams.n_rot
              << ") to match head_dim (" << model->hparams.n_embd_head_k << ")"
              << std::endl;
    model->hparams.n_rot = model->hparams.n_embd_head_k;
  }

  // Load RMS epsilon
  model->hparams.f_norm_rms_eps = 1e-5f;
  get_f32("attention.layer_norm_rms_epsilon", model->hparams.f_norm_rms_eps);

  std::cout << "[DenseCore] Loaded RMS norm epsilon: "
            << model->hparams.f_norm_rms_eps << std::endl;

  // Load RoPE parameters
  get_f32("rope.freq_base", model->hparams.rope_freq_base);
  get_f32("rope.freq_scale", model->hparams.rope_freq_scale);

  // Load BOS/EOS token IDs from tokenizer metadata
  int idx_bos = gguf_find_key(ctx_gguf, "tokenizer.ggml.bos_token_id");
  if (idx_bos != -1)
    model->bos_token_id = gguf_get_val_u32(ctx_gguf, idx_bos);

  int idx_eos = gguf_find_key(ctx_gguf, "tokenizer.ggml.eos_token_id");
  if (idx_eos != -1)
    model->eos_token_id = gguf_get_val_u32(ctx_gguf, idx_eos);

  // Check if model actually wants BOS added
  int idx_add_bos = gguf_find_key(ctx_gguf, "tokenizer.ggml.add_bos_token");
  bool add_bos = true;
  if (idx_add_bos != -1) {
    add_bos = gguf_get_val_bool(ctx_gguf, idx_add_bos);
  }

  // Disable BOS if BOS equals PAD token
  int idx_pad = gguf_find_key(ctx_gguf, "tokenizer.ggml.padding_token_id");
  if (idx_pad != -1) {
    uint32_t pad_id = gguf_get_val_u32(ctx_gguf, idx_pad);
    if (model->bos_token_id == (int)pad_id) {
      std::cout << "[DenseCore] BOS == PAD, disabling BOS (model likely "
                   "doesn't use BOS)"
                << std::endl;
      add_bos = false;
    }
  }

  if (!add_bos) {
    model->bos_token_id = -1;
  }

  std::cout << "[DenseCore] Model params: "
            << "n_vocab=" << model->hparams.n_vocab << ", "
            << "n_embd=" << model->hparams.n_embd << ", "
            << "n_layer=" << model->hparams.n_layer << ", "
            << "n_head=" << model->hparams.n_head << ", "
            << "n_head_kv=" << model->hparams.n_head_kv << ", "
            << "n_rot=" << model->hparams.n_rot << ", "
            << "n_ctx=" << model->hparams.n_ctx << std::endl;
  std::cout << "[DenseCore] BOS=" << model->bos_token_id
            << ", EOS=" << model->eos_token_id
            << ", rope_freq=" << model->hparams.rope_freq_base
            << ", rope_scale=" << model->hparams.rope_freq_scale << std::endl;

  // 2. Load Vocab
  int token_idx = gguf_find_key(ctx_gguf, "tokenizer.ggml.tokens");
  if (token_idx != -1) {
    int n_tokens = gguf_get_arr_n(ctx_gguf, token_idx);
    model->vocab_tokens.reserve(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
      const char *str = gguf_get_arr_str(ctx_gguf, token_idx, i);
      std::string s(str);
      model->vocab_tokens.push_back(s);
      model->token_to_id[s] = i;
    }

    // Update n_vocab to match actual loaded vocab size
    if (n_tokens != (int)model->hparams.n_vocab) {
      std::cout << "[DenseCore] Vocab size mismatch! Metadata="
                << model->hparams.n_vocab << " but loaded " << n_tokens
                << " tokens. Updating to " << n_tokens << std::endl;
      model->hparams.n_vocab = n_tokens;
    }
  }

  // 3. Initialize backend with error checking
  model->backend = ggml_backend_cpu_init();
  if (!model->backend) {
    std::cerr << "[DenseCore] Error: Failed to initialize CPU backend"
              << std::endl;
    gguf_free(ctx_gguf);
    if (ctx_w)
      ggml_free(ctx_w);
    delete model;
    return nullptr;
  }

  // 4. Map Tensors
  model->layers.resize(model->hparams.n_layer);

  auto get_tensor = [&](const std::string &name) -> struct ggml_tensor * {
    struct ggml_tensor *t = ggml_get_tensor(model->ctx_w, name.c_str());
    return t;
  };

  model->tok_embeddings = get_tensor("token_embd.weight");
  model->output_norm = get_tensor("output_norm.weight");

  // Try multiple possible names for lm_head/output projection
  model->output = get_tensor("output.weight");
  if (!model->output) {
    model->output = get_tensor("lm_head.weight");
  }
  // Tie embeddings fallback
  if (!model->output && model->tok_embeddings) {
    std::cout << "[DenseCore] output.weight not found, using tied embeddings"
              << std::endl;
    model->output = model->tok_embeddings;
    model->tied_embeddings = true;
  }

  if (!model->output) {
    std::cout
        << "[DenseCore] CRITICAL ERROR: Could not find output weight tensor!"
        << std::endl;
    std::cout << "[DenseCore] Available tensors:" << std::endl;
    struct ggml_tensor *t = ggml_get_first_tensor(model->ctx_w);
    while (t) {
      std::cout << "  - " << t->name << " [" << t->ne[0] << ", " << t->ne[1]
                << "]" << std::endl;
      t = ggml_get_next_tensor(model->ctx_w, t);
    }
  }

  // Debug: Print tensor shapes
  if (model->tok_embeddings) {
    std::cout << "[DenseCore] tok_embeddings shape: ["
              << model->tok_embeddings->ne[0] << ", "
              << model->tok_embeddings->ne[1] << "]" << std::endl;
  }
  if (model->output) {
    std::cout << "[DenseCore] output shape: [" << model->output->ne[0] << ", "
              << model->output->ne[1] << "]" << std::endl;

    // Validate n_vocab matches output tensor shape
    uint32_t tensor_vocab_size = model->output->ne[1];
    if (tensor_vocab_size != model->hparams.n_vocab && tensor_vocab_size > 0) {
      std::cout << "[DenseCore] ERROR: Vocab size inconsistency! "
                << "Loaded vocab=" << model->hparams.n_vocab
                << " but output tensor expects " << tensor_vocab_size
                << " tokens." << std::endl;
      std::cout << "[DenseCore] This will cause garbage output. "
                << "Check GGUF file integrity." << std::endl;

      if (tensor_vocab_size > model->hparams.n_vocab) {
        std::cout
            << "[DenseCore] WARNING: Tensor vocab larger than loaded vocab. "
            << "Some tokens may not decode properly." << std::endl;
      }
    }
  }

  for (uint32_t i = 0; i < model->hparams.n_layer; ++i) {
    std::string layer_prefix = "blk." + std::to_string(i) + ".";

    model->layers[i].attention_norm =
        get_tensor(layer_prefix + "attn_norm.weight");
    model->layers[i].ffn_norm = get_tensor(layer_prefix + "ffn_norm.weight");

    model->layers[i].wq = get_tensor(layer_prefix + "attn_q.weight");
    model->layers[i].wk = get_tensor(layer_prefix + "attn_k.weight");
    model->layers[i].wv = get_tensor(layer_prefix + "attn_v.weight");
    model->layers[i].wo = get_tensor(layer_prefix + "attn_output.weight");

    model->layers[i].bq = get_tensor(layer_prefix + "attn_q.bias");
    model->layers[i].bk = get_tensor(layer_prefix + "attn_k.bias");
    model->layers[i].bv = get_tensor(layer_prefix + "attn_v.bias");
    model->layers[i].bo = get_tensor(layer_prefix + "attn_output.bias");

    // QK-Norm (Qwen3)
    model->layers[i].attn_q_norm =
        get_tensor(layer_prefix + "attn_q_norm.weight");
    model->layers[i].attn_k_norm =
        get_tensor(layer_prefix + "attn_k_norm.weight");

    if (i == 0) {
      if (model->layers[i].attn_q_norm && model->layers[i].attn_k_norm) {
        std::cout << "[DenseCore] QK-Norm tensors found (Qwen3 architecture)"
                  << std::endl;
      }
    }

    model->layers[i].w1 = get_tensor(layer_prefix + "ffn_gate.weight");
    model->layers[i].w2 = get_tensor(layer_prefix + "ffn_down.weight");
    model->layers[i].w3 = get_tensor(layer_prefix + "ffn_up.weight");
  }

  // Auto-compute head dimensions from weight tensor shapes
  if (model->hparams.n_embd_head_k == 0 && model->layers[0].wk) {
    model->hparams.n_embd_head_k =
        model->layers[0].wk->ne[1] / model->hparams.n_head_kv;
  }
  if (model->hparams.n_embd_head_v == 0 && model->layers[0].wv) {
    model->hparams.n_embd_head_v =
        model->layers[0].wv->ne[1] / model->hparams.n_head_kv;
  }
  // Fallback to n_embd/n_head
  if (model->hparams.n_embd_head_k == 0) {
    model->hparams.n_embd_head_k =
        model->hparams.n_embd / model->hparams.n_head;
  }
  if (model->hparams.n_embd_head_v == 0) {
    model->hparams.n_embd_head_v =
        model->hparams.n_embd / model->hparams.n_head;
  }

  std::cout << "[DenseCore] Head dimensions: n_embd_head_k="
            << model->hparams.n_embd_head_k
            << ", n_embd_head_v=" << model->hparams.n_embd_head_v << std::endl;

  // Initialize pre-computed RoPE table for optimized inference
  std::cout << "[DenseCore] Initializing RoPE table..." << std::endl;
  InitRoPETable(model);
  std::cout << "[DenseCore] RoPE table initialized: "
            << model->rope_cos_sin.size() << " values (" << model->hparams.n_ctx
            << " positions × " << model->rope_head_dim << " dims)" << std::endl;

  std::cout << "[DenseCore] Model loaded successfully" << std::endl;
  return model;
}

// ============================================================================
// SaveModel
// ============================================================================

int SaveModel(const TransformerModel *model, const char *path) {
  if (!model || !path)
    return -1;

  std::cout << "[DenseCore] Saving model to " << path << "..." << std::endl;
  std::cerr << "[DenseCore] WARNING: SaveModel currently only supports "
               "metadata updates. Tensor data writing is experimental."
            << std::endl;

  struct gguf_context *ctx = gguf_init_empty();

  // 1. Write metadata
  gguf_set_val_str(ctx, "general.architecture", "llama");
  gguf_set_val_str(ctx, "general.name", "DenseCore-Optimized");

  // Write hyperparameters
  gguf_set_val_u32(ctx, "llama.vocab_size", model->hparams.n_vocab);
  gguf_set_val_u32(ctx, "llama.embedding_length", model->hparams.n_embd);
  gguf_set_val_u32(ctx, "llama.block_count", model->hparams.n_layer);
  gguf_set_val_u32(ctx, "llama.attention.head_count", model->hparams.n_head);
  gguf_set_val_u32(ctx, "llama.attention.head_count_kv",
                   model->hparams.n_head_kv);
  gguf_set_val_u32(ctx, "llama.context_length", model->hparams.n_ctx);
  gguf_set_val_f32(ctx, "llama.attention.layer_norm_rms_epsilon",
                   model->hparams.f_norm_rms_eps);
  gguf_set_val_f32(ctx, "llama.rope.freq_base", model->hparams.rope_freq_base);
  gguf_set_val_f32(ctx, "llama.rope.freq_scale",
                   model->hparams.rope_freq_scale);

  // Write vocab if available
  if (!model->vocab_tokens.empty()) {
    std::vector<const char *> tokens_cstr;
    tokens_cstr.reserve(model->vocab_tokens.size());
    for (const auto &s : model->vocab_tokens) {
      tokens_cstr.push_back(s.c_str());
    }
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", tokens_cstr.data(),
                     tokens_cstr.size());
  }

  // 2. Write Tensors (experimental)
  // Note: Full tensor serialization requires properly set up tensors in the
  // context. This is a simplified implementation that may not work for all
  // models.

  // Write file (metadata only is safer, but we try full write)
  bool ok = gguf_write_to_file(ctx, path, false); // false = include tensors

  gguf_free(ctx);

  if (!ok) {
    std::cerr << "[DenseCore] Error: Failed to write GGUF file" << std::endl;
    return -2;
  }

  std::cout << "[DenseCore] Model saved (metadata-only mode)" << std::endl;
  return 0;
}

// ============================================================================
// NUMA-Aware Model Loading
// ============================================================================

/**
 * Rebind a single tensor's data to specified NUMA node.
 * Uses a dedicated thread pinned to the target NUMA node for copying,
 * which enforces first-touch memory policy allocation.
 *
 * @param tensor Tensor to rebind (must have valid data pointer)
 * @param numa_node Target NUMA node for allocation
 * @param model Model to track allocation for cleanup
 * @param min_size_bytes Minimum tensor size threshold (skip smaller tensors)
 * @return True if tensor was successfully rebound
 */
static bool RebindTensorNuma(struct ggml_tensor *tensor, int numa_node,
                             TransformerModel *model,
                             size_t min_size_bytes = 1024 * 1024) {
  if (!tensor || !tensor->data) {
    return false;
  }

  const size_t tensor_size = ggml_nbytes(tensor);
  if (tensor_size < min_size_bytes) {
    return false; // Skip small tensors - overhead not worth it
  }

  // Allocate NUMA-aware buffer
  auto result =
      densecore::NumaAllocator::AllocatePreferred(tensor_size, 64, numa_node);

  if (!result.ptr) {
    std::cerr << "[NUMA] Failed to allocate " << (tensor_size / 1024 / 1024)
              << " MB on node " << numa_node << std::endl;
    return false;
  }

  // =========================================================================
  // CRITICAL: Copy data using thread pinned to target NUMA node
  // =========================================================================
  // This enforces first-touch policy: the physical memory pages are allocated
  // on the NUMA node where the first write occurs. By pinning our copy thread
  // to the target node, we ensure pages are allocated there.
  // =========================================================================
  const void *src_data = tensor->data;
  void *dst_data = result.ptr;

  std::thread copy_thread([dst_data, src_data, tensor_size, numa_node]() {
    // Pin this thread to the target NUMA node
    densecore::HardwareTopology::GetInstance().PinCurrentThreadToNumaNode(
        numa_node, densecore::PinningPolicy::SCATTER);

    // Perform the copy - this triggers first-touch allocation
    std::memcpy(dst_data, src_data, tensor_size);
  });

  // MUST join to ensure copy completes before we use the tensor
  copy_thread.join();

  // Replace tensor data pointer
  // NOTE: Original mmap data will be freed when ctx_gguf is freed
  tensor->data = result.ptr;

  // Track allocation for cleanup in model destructor
  model->numa_buffers.emplace_back(result.ptr, result.size);

  return true;
}

/**
 * Load GGUF model with NUMA-aware memory placement.
 *
 * After initial mmap load, identifies large tensors (>1MB) and rebinds
 * their data to the specified NUMA node for optimized memory bandwidth.
 *
 * @param path Path to GGUF file
 * @param numa_node Target NUMA node (-1 for local/auto)
 * @param use_huge_pages Whether to request huge pages (TBD)
 * @return Loaded model with NUMA-optimized memory layout
 */
TransformerModel *LoadGGUFModelNuma(const char *path, int numa_node,
                                    bool use_huge_pages) {
  // Step 1: Load model normally (mmap-based)
  TransformerModel *model = LoadGGUFModel(path);
  if (!model) {
    return nullptr;
  }

  // Step 2: Check if NUMA rebinding is needed
  if (numa_node < 0 && !densecore::NumaAllocator::IsNumaAvailable()) {
    std::cout << "[DenseCore] NUMA not available, using standard memory layout"
              << std::endl;
    return model;
  }

  // If numa_node is -1 (auto), default to node 0
  // (for more sophisticated scheduling, the caller should specify explicitly)
  if (numa_node < 0) {
    numa_node = 0;
    std::cout << "[DenseCore] Auto-selected NUMA node " << numa_node
              << " for tensor rebinding" << std::endl;
  }

  (void)use_huge_pages; // TODO: implement huge page allocation path

  std::cout << "[DenseCore] Starting NUMA rebinding to node " << numa_node
            << "..." << std::endl;

  // Step 3: Rebind large tensors to target NUMA node
  size_t rebound_bytes = 0;
  int rebound_count = 0;

  // Helper lambda to rebind and track
  auto rebind = [&](struct ggml_tensor *t, const char *name) {
    if (RebindTensorNuma(t, numa_node, model)) {
      rebound_bytes += ggml_nbytes(t);
      rebound_count++;
      std::cout << "  [NUMA] Rebound " << name << " ("
                << (ggml_nbytes(t) / 1024 / 1024) << " MB)" << std::endl;
    }
  };

  // Rebind critical weight tensors
  rebind(model->tok_embeddings, "tok_embeddings");
  rebind(model->output, "output");
  rebind(model->output_norm, "output_norm");

  // Rebind layer weights (these are the bulk of model memory)
  for (size_t i = 0; i < model->layers.size(); ++i) {
    auto &layer = model->layers[i];
    std::string prefix = "blk." + std::to_string(i) + ".";

    // Attention weights (largest tensors per layer)
    rebind(layer.wq, (prefix + "wq").c_str());
    rebind(layer.wk, (prefix + "wk").c_str());
    rebind(layer.wv, (prefix + "wv").c_str());
    rebind(layer.wo, (prefix + "wo").c_str());

    // FFN weights (also large)
    rebind(layer.w1, (prefix + "w1").c_str());
    rebind(layer.w2, (prefix + "w2").c_str());
    rebind(layer.w3, (prefix + "w3").c_str());

    // Norms (small, typically skipped by size threshold)
    rebind(layer.attention_norm, (prefix + "attn_norm").c_str());
    rebind(layer.ffn_norm, (prefix + "ffn_norm").c_str());
  }

  // Step 4: Print summary
  std::cout << "[DenseCore] NUMA rebinding complete: " << rebound_count
            << " tensors (" << (rebound_bytes / 1024 / 1024) << " MB) to Node "
            << numa_node << std::endl;

  // Optional: Verify placement using MemoryDiagnostics
  if (model->tok_embeddings && rebound_count > 0) {
    densecore::MemoryDiagnostics::PrintSystemTopologyReport(
        model->tok_embeddings->data, ggml_nbytes(model->tok_embeddings),
        numa_node, "tok_embeddings (sample)");
  }

  return model;
}
