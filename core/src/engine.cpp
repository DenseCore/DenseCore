#include "backend_registry.h" // Backend abstraction layer
#include "densecore.h"
#include "embedding.h"
#include "engine_internal.h"
#include "inference.h"
#include "kv_cache.h"
#include "model_loader.h"
#include "model_types.h"
#include "optimization_bridge.h" // Runtime SIMD dispatch
#include "simd_ops.h"
#include "tokenizer.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

// API implementations

int SubmitEmbeddingRequest(DenseCoreHandle handle, const char *prompt,
                           EmbeddingCallback callback, void *user_data) {
  // Default: MEAN pooling with normalization
  return SubmitEmbeddingRequestEx(handle, prompt, 0, 1, callback, user_data);
}

int SubmitEmbeddingRequestEx(DenseCoreHandle handle, const char *prompt,
                             int pooling_type, int normalize,
                             EmbeddingCallback callback, void *user_data) {
  if (!handle)
    return -1;
  EngineState *state = (EngineState *)handle;

  // Get model for tokenization
  ModelEntry *model_entry = state->GetDefaultModel();
  if (!model_entry || !model_entry->model) {
    std::cerr << "[DenseCore] No model loaded for tokenization" << std::endl;
    return -1;
  }

  Request *req = state->request_pool.Acquire();
  req->Reset();
  static std::atomic<int> global_req_id{1};
  req->id = global_req_id.fetch_add(1);

  req->prompt = prompt;
  req->max_tokens = 0; // No generation
  req->is_embedding = true;
  req->embedding_callback = callback;
  req->user_data = user_data;

  // Store pooling config
  req->pooling_type = static_cast<densecore::PoolingStrategy>(pooling_type);
  req->normalize_embedding = (normalize != 0);

  // Tokenize immediately (outside hot path)
  req->tokens = Tokenizer::Tokenize(model_entry->model.get(), prompt, true);

  // Record arrival time
  req->arrival_time = std::chrono::steady_clock::now();
  req->priority = 50;    // High priority for embeddings
  req->tier = "premium"; // Embeddings get premium tier

  // Lock-free enqueue: no mutex needed
  state->pending_requests.Push(req, req->tier);
  {
    std::lock_guard<std::mutex> lock(state->cv_mu);
    state->queue_cv.notify_one();
  }

  return req->id;
}

int SubmitBatchEmbeddingRequest(DenseCoreHandle handle, const char **prompts,
                                int num_prompts, int pooling_type,
                                int normalize, EmbeddingCallback callback,
                                void *user_data) {
  if (!handle || !prompts || num_prompts <= 0)
    return -1;

  // Submit each prompt as a separate request (batching happens in worker)
  int first_id = -1;
  for (int i = 0; i < num_prompts; i++) {
    int id = SubmitEmbeddingRequestEx(handle, prompts[i], pooling_type,
                                      normalize, callback, user_data);
    if (i == 0)
      first_id = id;
  }

  return first_id;
}

int GetEmbeddingDimension(DenseCoreHandle handle) {
  if (!handle)
    return -1;
  EngineState *state = (EngineState *)handle;

  ModelEntry *entry = state->GetDefaultModel();
  if (entry && entry->model) {
    return entry->model->hparams.n_embd;
  }
  return -1;
}

DENSECORE_API DenseCoreHandle InitEngine(const char *model_path,
                                         const char * /*reserved*/,
                                         int threads) {
  try {
    // =========================================================================
    // INITIALIZATION DELAY CONFIGURATION (Large Model Support)
    // =========================================================================
    // Qwen3-4B and larger models have large graphs that take longer to build.
    // In WSL2 environments, I/O latency can be significant.
    // Set DENSECORE_INIT_DELAY_MS to add tolerance (default: 0ms).
    // =========================================================================
    int init_delay_ms = 0;
    if (const char *env_val = std::getenv("DENSECORE_INIT_DELAY_MS")) {
      init_delay_ms = std::atoi(env_val);
      if (init_delay_ms > 0) {
        std::cout << "[DenseCore] Init delay: " << init_delay_ms
                  << "ms (large model tolerance)" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(init_delay_ms));
      }
    }

    // =========================================================================
    // THREAD CONFIGURATION (Pure std::thread + GGML thread pool)
    // =========================================================================
    // Auto-detect optimal thread count if not specified
    if (threads <= 0) {
      threads = densecore::simd::GetNumCores();
    }
    std::cout << "[DenseCore] Threading configured: " << threads << " threads"
              << " (GGML thread pool + std::thread, no OpenMP)" << std::endl;

    // =========================================================================
    // SIMD DISPATCH TABLE INITIALIZATION (Must happen before any inference!)
    // =========================================================================
    // OpsRegistry selects optimal kernel implementations based on CPU caps.
    // Initializing here ensures the dispatch table is ready before generate()
    // is called, even if the worker thread hasn't started yet.
    // =========================================================================
    if (!densecore::OpsRegistry::IsInitialized()) {
      densecore::OpsRegistry::Init();
    }

    // =========================================================================
    // BACKEND REGISTRY INITIALIZATION
    // =========================================================================
    // Register the CPU backend (AVX2/AVX-512 kernels wrapped in ComputeBackend
    // interface). Future ASIC backends can be registered similarly.
    // =========================================================================
    densecore::BackendRegistry::Instance().RegisterCpuBackend();

    // Update global inference config for Flash Attention and other parallel ops
    InferenceConfig::Instance().num_threads = threads;

    TransformerModel *model = LoadGGUFModel(model_path);
    if (!model)
      return nullptr;

    EngineState *state = new EngineState();
    // Default values
    state->numa_node_id = -1;
    state->pinning_policy = 0;
    state->n_threads = threads; // Store for worker thread

    // Calculate optimal KV cache size based on model dimensions
    const int head_dim = model->hparams.n_embd / model->hparams.n_head;
    const int n_head_kv = model->hparams.n_head_kv;
    const int n_layer = model->hparams.n_layer;

    // Bytes per token in KV cache (K + V, FP16)
    size_t bytes_per_token =
        (size_t)head_dim * n_head_kv * n_layer * 2 * sizeof(ggml_fp16_t);

    // Target KV cache memory (512MB default)
    size_t target_kv_memory = 512 * 1024 * 1024;

    // Calculate optimal sequence length
    int optimal_seq_len = target_kv_memory / bytes_per_token;
    optimal_seq_len = std::max(256, std::min(optimal_seq_len, 4096));

    // For very small models, allow larger context
    if (bytes_per_token < 1024) {
      optimal_seq_len = std::min(optimal_seq_len * 2, 8192);
    }

    std::cout << "[DenseCore] Auto-configured max_seq_len: " << optimal_seq_len
              << " (KV cache: ~"
              << (bytes_per_token * optimal_seq_len / 1024 / 1024) << " MB, "
              << bytes_per_token << " bytes/token)" << std::endl;

    // Log Flash Attention status based on CPU capabilities
    densecore::simd::SimdLevel simd_level = densecore::simd::DetectSimdLevel();
    if (simd_level >= densecore::simd::SimdLevel::AVX512) {
      std::cout << "[DenseCore] Flash Attention Enabled ("
                << densecore::simd::SimdLevelName(simd_level) << " detected)"
                << std::endl;
    } else {
      std::cout << "[DenseCore] Flash Attention Disabled (requires AVX-512, "
                << "detected: " << densecore::simd::SimdLevelName(simd_level)
                << ")" << std::endl;
    }

    // Log NUMA configuration
    if (state->numa_node_id >= 0) {
      std::cout << "[DenseCore] NUMA binding: node " << state->numa_node_id
                << std::endl;
    }

    // Initialize Paged KV Cache with NUMA-aware allocation
    PagedKVCache *cache =
        InitPagedKVCache(model, 1, optimal_seq_len, GGML_TYPE_F16, -1);
    if (!cache) {
      delete model;
      delete state;
      return nullptr;
    }

    // Wrap model and cache into ModelEntry and add to pool
    auto entry = std::make_unique<ModelEntry>();
    entry->model_id = "default";
    entry->model_path = model_path;
    entry->model = std::unique_ptr<TransformerModel>(model);
    entry->kv_cache = std::unique_ptr<PagedKVCache>(cache);
    entry->last_used = std::chrono::steady_clock::now();
    entry->is_loaded = true;

    state->models["default"] = std::move(entry);
    state->default_model_id = "default";

    // Initialize Scheduler with the default model's block manager
    // Note: The scheduler takes a raw pointer to BlockManager, ownership
    // remains with PagedKVCache
    if (state->models["default"]->kv_cache &&
        state->models["default"]->kv_cache->block_manager) {
      state->scheduler = std::make_unique<densecore::Scheduler>(
          state->models["default"]->kv_cache->block_manager);
      std::cout << "[DenseCore] Scheduler initialized." << std::endl;
    } else {
      std::cerr << "[DenseCore] FATAL: Failed to initialize Scheduler. "
                   "BlockManager is missing."
                << std::endl;
      delete state;
      return nullptr;
    }

    // Initialize compute buffer (Persistent, Aligned)
    state->InitComputeBuffer();

    // Start background threads
    state->status = EngineStatus::RUNNING;
    state->worker_thread = std::thread(EngineLoop, state);
    state->callback_thread = std::thread(CallbackLoop, state);

    std::cout << "[DenseCore] Started worker thread and callback thread"
              << std::endl;

    return (DenseCoreHandle)state;
  } catch (const std::exception &e) {
    std::cerr << "[DenseCore] Exception in InitEngine: " << e.what()
              << std::endl;
    return nullptr;
  } catch (...) {
    std::cerr << "[DenseCore] Unknown exception in InitEngine" << std::endl;
    return nullptr;
  }
}

/**
 * CallbackLoop - Dedicated thread for callback execution.
 *
 * This function runs on a dedicated thread to execute callbacks without
 * blocking the worker thread. It waits on result_cv for new ResultEvents,
 * pops events from the result_queue, and executes the callbacks.
 *
 * This decoupling prevents the Python GIL from blocking the inference
 * hot path, resolving the streaming deadlock.
 */
void CallbackLoop(EngineState *state) {
  try {
    while (true) {
      ResultEvent event;
      {
        std::unique_lock<std::mutex> lock(state->result_mu);
        state->result_cv.wait(lock, [state]() {
          return !state->result_queue.empty() ||
                 state->status == EngineStatus::STOPPED;
        });

        // Check exit condition: STOPPED and queue empty
        if (state->result_queue.empty() &&
            state->status == EngineStatus::STOPPED) {
          break; // Exit: shutdown complete
        }

        if (state->result_queue.empty())
          continue;

        event = std::move(state->result_queue.front());
        state->result_queue.pop_front();
      }

      std::cerr << "[TRACE] CallbackLoop popped event for request "
                << event.request_id << " (finished=" << event.finished << ")"
                << std::endl;

      // Execute callback OUTSIDE the lock (GIL acquisition happens here)
      // This is the ONLY place callbacks should be invoked!
      try {
        if (event.callback) {
          event.callback(event.token_str.c_str(),
                         event.finished ? 1 : (event.error ? 1 : 0),
                         event.user_data);
        } else if (event.emb_callback && !event.embedding_data.empty()) {
          event.emb_callback(event.embedding_data.data(),
                             static_cast<int>(event.embedding_data.size()),
                             event.user_data);
          // Note: embedding_data is RAII-managed by std::vector, no manual free
        }
      } catch (const std::exception &e) {
        std::cerr << "[DenseCore] Exception in callback (request "
                  << event.request_id << "): " << e.what() << std::endl;
      } catch (...) {
        std::cerr << "[DenseCore] Unknown exception in callback (request "
                  << event.request_id << ")" << std::endl;
      }
    }

    std::cerr << "[DenseCore] CallbackLoop exiting" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "[DenseCore] CallbackLoop thread exception: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "[DenseCore] CallbackLoop thread unknown exception"
              << std::endl;
  }
}

int SubmitRequest(DenseCoreHandle handle, const char *prompt, int max_tokens,
                  TokenCallback callback, void *user_data) {
  if (!handle)
    return -1;
  EngineState *state = (EngineState *)handle;

  // Get model for tokenization
  ModelEntry *model_entry = state->GetDefaultModel();
  if (!model_entry || !model_entry->model) {
    std::cerr << "[DenseCore] No model loaded for tokenization" << std::endl;
    return -1;
  }

  Request *req = state->request_pool.Acquire();
  req->Reset();
  static std::atomic<int> global_req_id{1};
  req->id = global_req_id.fetch_add(1);

  req->prompt = prompt;
  req->max_tokens = max_tokens;
  req->callback = callback;
  req->user_data = user_data;

  // Tokenize immediately (outside hot path)
  req->tokens = Tokenizer::Tokenize(model_entry->model.get(), prompt, true);

  // Record arrival time for metrics
  req->arrival_time = std::chrono::steady_clock::now();

  // Priority based on actual token count (shorter = higher priority)
  int n_tokens = req->tokens.size();
  // Set tier based on length (SJF-style prioritization)
  if (n_tokens < 100) {
    req->tier = "premium"; // Short requests get highest priority
  } else if (n_tokens < 500) {
    req->tier = "standard";
  } else {
    req->tier = "batch"; // Long requests get lowest priority
  }

  // Lock-free enqueue: no mutex needed
  state->pending_requests.Push(req, req->tier);
  {
    std::lock_guard<std::mutex> lock(state->cv_mu);
    state->queue_cv.notify_one();
  }

  return req->id;
}

int SubmitRequestIds(DenseCoreHandle handle, const int *tokens, int n_tokens,
                     int max_tokens, TokenCallback callback, void *user_data) {
  if (!handle || !tokens || n_tokens <= 0) {
    return -1;
  }
  EngineState *state = (EngineState *)handle;

  Request *req = state->request_pool.Acquire();
  req->Reset();
  static std::atomic<int> global_req_id{1};
  req->id = global_req_id.fetch_add(1);

  // Copy tokens
  req->tokens.assign(tokens, tokens + n_tokens);
  req->prompt = ""; // No text prompt
  req->max_tokens = max_tokens;
  req->callback = callback;
  req->user_data = user_data;

  // Record arrival time
  req->arrival_time = std::chrono::steady_clock::now();

  // Set tier based on length
  if (n_tokens < 100) {
    req->tier = "premium";
  } else if (n_tokens < 500) {
    req->tier = "standard";
  } else {
    req->tier = "batch";
  }

  // Lock-free enqueue
  state->pending_requests.Push(req, req->tier);

  // Note: The mutex lock provides sufficient memory ordering.
  // notify_one() under cv_mu ensures the worker sees the pushed request.
  {
    std::lock_guard<std::mutex> lock(state->cv_mu);
    state->queue_cv.notify_one();
  }

  return req->id;
}

int SubmitRequestWithFormat(DenseCoreHandle handle, const char *prompt,
                            int max_tokens, int json_mode,
                            TokenCallback callback, void *user_data) {
  if (!handle)
    return -1;
  EngineState *state = (EngineState *)handle;

  // Get model for tokenization
  ModelEntry *model_entry = state->GetDefaultModel();
  if (!model_entry || !model_entry->model) {
    std::cerr << "[DenseCore] No model loaded for tokenization" << std::endl;
    return -1;
  }

  Request *req = state->request_pool.Acquire();
  req->Reset();
  static std::atomic<int> global_req_id{1};
  req->id = global_req_id.fetch_add(1);

  req->prompt = prompt;
  req->max_tokens = max_tokens;
  req->callback = callback;
  req->user_data = user_data;
  req->json_mode = (json_mode != 0);

  // Initialize grammar constraint if JSON mode is enabled
  if (req->json_mode) {
    req->grammar.enabled = true;
    req->grammar.is_json_mode = true;
    req->grammar.state = JSONState::EXPECT_OBJECT_START;
    if (!model_entry->model->vocab_tokens.empty()) {
      InitGrammarConstraint(&req->grammar, model_entry->model->vocab_tokens);
    }
  }

  // Tokenize immediately (outside hot path)
  req->tokens = Tokenizer::Tokenize(model_entry->model.get(), prompt, true);

  // Record arrival time for metrics
  req->arrival_time = std::chrono::steady_clock::now();
  // Set tier based on length
  int n_tokens = req->tokens.size();
  if (n_tokens < 100) {
    req->tier = "premium";
  } else if (n_tokens < 500) {
    req->tier = "standard";
  } else {
    req->tier = "batch";
  }

  // Lock-free enqueue
  state->pending_requests.Push(req, req->tier);
  {
    std::lock_guard<std::mutex> lock(state->cv_mu);
    state->queue_cv.notify_one();
  }

  return req->id;
}

int CancelRequest(DenseCoreHandle handle, int request_id) {
  if (!handle)
    return -1;
  EngineState *state = (EngineState *)handle;

  // ==========================================================================
  // LOCK-FREE QUEUE LIMITATION: Cannot iterate/remove from pending queue
  // ==========================================================================
  // With lock-free queues, we can only cancel active requests.
  // Pending requests will be checked for cancellation when dequeued.
  // ==========================================================================
  std::lock_guard<std::mutex> lock(state->active_mu);

  for (Request *req : state->active_requests) {
    if (req->id == request_id) {
      req->cancelled = true;
      LOG_INFO("Marked active request ", request_id, " for cancellation.");
      return 0; // Success
    }
  }

  // Request not found in active list - it might be pending
  // Mark cancellation will happen lazily when request is dequeued
  LOG_INFO("Request ", request_id,
           " not active - cancellation may be delayed.");
  return -2; // Not found in active
}

DenseCoreMetrics GetMetrics(DenseCoreHandle handle) {
  DenseCoreMetrics m = {};
  if (handle) {
    EngineState *state = (EngineState *)handle;
    m.active_requests = state->metrics.active_requests;
    m.total_tokens_generated = state->metrics.total_tokens_generated;
    // m.requests_per_second = ... calculation
  }
  return m;
}

DetailedMetrics GetDetailedMetrics(DenseCoreHandle handle) {
  DetailedMetrics m = {};
  if (!handle)
    return m;

  EngineState *state = (EngineState *)handle;

  // Request metrics
  m.active_requests = state->metrics.active_requests;
  m.total_requests = state->metrics.total_requests;
  m.completed_requests = state->metrics.completed_requests;
  m.failed_requests = state->metrics.failed_requests;

  // Count pending requests (lock-free approximate count)
  m.pending_requests = state->pending_requests.Size();
  {
    std::lock_guard<std::mutex> lock(state->active_mu);
    m.current_batch_size = state->active_requests.size();
  }

  // Token metrics
  m.total_tokens_generated = state->metrics.total_tokens_generated;
  m.total_prompt_tokens = state->metrics.total_prompt_tokens;

  // Calculate TPS (tokens per second) - simple estimation
  if (m.completed_requests > 0) {
    m.tokens_per_second =
        (float)m.total_tokens_generated / std::max(1L, m.completed_requests);
  }

  // Latency metrics from samples
  {
    std::lock_guard<std::mutex> lock(state->metrics.metrics_mu);

    // TTFT metrics
    m.avg_time_to_first_token =
        state->metrics.CalculateAverage(state->metrics.ttft_samples);
    m.p50_time_to_first_token =
        state->metrics.CalculatePercentile(state->metrics.ttft_samples, 0.5f);
    m.p90_time_to_first_token =
        state->metrics.CalculatePercentile(state->metrics.ttft_samples, 0.9f);
    m.p99_time_to_first_token =
        state->metrics.CalculatePercentile(state->metrics.ttft_samples, 0.99f);

    // ITL metrics
    m.avg_inter_token_latency =
        state->metrics.CalculateAverage(state->metrics.itl_samples);
    m.p50_inter_token_latency =
        state->metrics.CalculatePercentile(state->metrics.itl_samples, 0.5f);
    m.p90_inter_token_latency =
        state->metrics.CalculatePercentile(state->metrics.itl_samples, 0.9f);
    m.p99_inter_token_latency =
        state->metrics.CalculatePercentile(state->metrics.itl_samples, 0.99f);

    // Queue wait time
    m.avg_queue_wait_time =
        state->metrics.CalculateAverage(state->metrics.queue_wait_samples);
    m.p99_queue_wait_time = state->metrics.CalculatePercentile(
        state->metrics.queue_wait_samples, 0.99f);
  }

  // KV Cache metrics
  ModelEntry *entry = state->GetDefaultModel();
  if (entry && entry->kv_cache && entry->kv_cache->block_manager) {
    m.kv_cache_total_blocks = entry->kv_cache->block_manager->num_blocks;
    int free_blocks = entry->kv_cache->block_manager->GetFreeBlockCount();
    m.kv_cache_usage_blocks = m.kv_cache_total_blocks - free_blocks;
    m.kv_cache_usage_percent =
        (float)m.kv_cache_usage_blocks / m.kv_cache_total_blocks * 100.0f;
  }

  // Batch metrics
  if (m.completed_requests > 0) {
    m.avg_batch_size = (float)m.total_requests / m.completed_requests;
  }

  // Error metrics
  m.oom_errors = state->metrics.oom_errors;
  m.timeout_errors = state->metrics.timeout_errors;

  return m;
}

void FreeEngine(DenseCoreHandle handle) {
  if (handle) {
    EngineState *state = (EngineState *)handle;
    state->Shutdown(); // Graceful shutdown with draining
    delete state;
  }
}

// Multi-Model API implementations

int LoadModel(DenseCoreHandle handle, const char *model_id,
              const char *model_path, int /*threads*/) {
  if (!handle || !model_id || !model_path)
    return -1;

  EngineState *state = (EngineState *)handle;

  // Check if model already exists
  {
    std::lock_guard<std::mutex> lock(state->models_mu);
    if (state->models.find(model_id) != state->models.end()) {
      std::cerr << "[DenseCore] Model " << model_id << " already loaded"
                << std::endl;
      return -2; // Already exists
    }
  }

// Load main model
// Note: The user's provided snippet had a syntax error and type mismatch for
// 'main_model' and 'return nullptr'. I've corrected it to use 'model' as in the
// original code and return -3. Assuming LOG_INFO and LOG_ERROR are defined
// elsewhere. If not, std::cerr will be used as a fallback for the error
// message.
#ifdef LOG_INFO
  LOG_INFO("Loading model from: ", model_path);
#endif
  TransformerModel *model = LoadGGUFModel(model_path);
  if (!model) {
#ifdef LOG_ERROR
    LOG_ERROR("Failed to load main model");
#else
    std::cerr << "[DenseCore] Failed to load model from " << model_path
              << std::endl;
#endif
    return -3;
  }

  // Initialize KV cache for this model
  // Note: max_num_seqs=4 to save memory (consumer laptop). 4 * 32k context is
  // plenty for testing.
  PagedKVCache *cache = InitPagedKVCache(model, 4, 8192, GGML_TYPE_F16);
  if (!cache) {
    delete model;
    std::cerr << "[DenseCore] Failed to initialize KV cache for " << model_id
              << std::endl;
    return -4;
  }

  // Create model entry with unique_ptr ownership
  auto entry = std::make_unique<ModelEntry>();
  entry->model_id = model_id;
  entry->model_path = model_path;
  entry->model = std::unique_ptr<TransformerModel>(model);
  entry->kv_cache = std::unique_ptr<PagedKVCache>(cache);
  entry->last_used = std::chrono::steady_clock::now();
  entry->is_loaded = true;

  // Add to pool
  {
    std::lock_guard<std::mutex> lock(state->models_mu);
    state->models[model_id] = std::move(entry);

    // Set as default if no default exists
    if (state->default_model_id.empty()) {
      state->default_model_id = model_id;
    }
  }

  return 0;
}

int UnloadModel(DenseCoreHandle handle, const char *model_id) {
  if (!handle || !model_id)
    return -1;

  EngineState *state = (EngineState *)handle;

  std::lock_guard<std::mutex> lock(state->models_mu);
  auto it = state->models.find(model_id);
  if (it == state->models.end()) {
    return -2; // Not found
  }

  // Cannot unload default model if it's the only one
  if (state->default_model_id == model_id && state->models.size() == 1) {
    return -3; // Cannot unload last model
  }

  // Smart pointers automatically cleanup when erased
  state->models.erase(it);

  // Update default if needed
  if (state->default_model_id == model_id) {
    if (!state->models.empty()) {
      state->default_model_id = state->models.begin()->first;
    } else {
      state->default_model_id = "";
    }
  }

  return 0;
}

int ListModels(DenseCoreHandle handle, char *out_models, int buffer_size) {
  if (!handle || !out_models || buffer_size <= 0)
    return -1;

  EngineState *state = (EngineState *)handle;
  std::lock_guard<std::mutex> lock(state->models_mu);

  std::stringstream ss;
  for (auto it = state->models.begin(); it != state->models.end(); ++it) {
    if (it != state->models.begin())
      ss << ",";
    ss << it->first;
  }

  std::string s = ss.str();
  if (s.length() >= (size_t)buffer_size) {
    return -2; // Buffer too small
  }

  strcpy(out_models, s.c_str());
  return state->models.size();
}

int SetDefaultModel(DenseCoreHandle handle, const char *model_id) {
  if (!handle || !model_id)
    return -1;

  EngineState *state = (EngineState *)handle;
  std::lock_guard<std::mutex> lock(state->models_mu);

  if (state->models.find(model_id) == state->models.end()) {
    return -2; // Not found
  }

  state->default_model_id = model_id;
  return 0;
}
