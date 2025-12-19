#ifndef DENSECORE_ENGINE_INTERNAL_H
#define DENSECORE_ENGINE_INTERNAL_H

#include "densecore.h"
#include "embedding.h"
#include "inference.h"
#include "kv_cache.h"
#include "lockfree_queue.h" // Lock-free sharded priority queue
#include "model_loader.h"
#include "model_types.h"
#include "scheduler.h"
#include "tokenizer.h"
#include "utils/error.h"
#include "utils/logging.h"
#include <climits>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <stack>
#include <string>
#include <thread>
#include <vector>

// Engine lifecycle states for graceful shutdown
enum class EngineStatus {
  RUNNING,  // Normal operation, accepting requests
  DRAINING, // Shutdown initiated, waiting for active requests to complete
  STOPPED   // Fully stopped, ready for cleanup
};

// Metrics tracking with thread-safe operations
struct InternalMetrics {
  std::atomic<long> total_tokens_generated{0};
  std::atomic<int> active_requests{0};
  std::atomic<long> total_requests{0};
  std::atomic<long> completed_requests{0};
  std::atomic<long> failed_requests{0};
  std::atomic<long> total_prompt_tokens{0};
  std::atomic<int> oom_errors{0};
  std::atomic<int> timeout_errors{0};

  // Latency tracking (in microseconds for precision)
  std::vector<long> ttft_samples;       // Time to first token
  std::vector<long> itl_samples;        // Inter-token latency
  std::vector<long> queue_wait_samples; // Queue wait time
  std::mutex metrics_mu;

  void RecordTTFT(long us) {
    std::lock_guard<std::mutex> lock(metrics_mu);
    ttft_samples.push_back(us);
    if (ttft_samples.size() > 10000)
      ttft_samples.erase(ttft_samples.begin(),
                         ttft_samples.begin() + 5000); // Keep last 5000
  }

  void RecordITL(long us) {
    std::lock_guard<std::mutex> lock(metrics_mu);
    itl_samples.push_back(us);
    if (itl_samples.size() > 10000)
      itl_samples.erase(itl_samples.begin(), itl_samples.begin() + 5000);
  }

  void RecordQueueWait(long us) {
    std::lock_guard<std::mutex> lock(metrics_mu);
    queue_wait_samples.push_back(us);
    if (queue_wait_samples.size() > 10000)
      queue_wait_samples.erase(queue_wait_samples.begin(),
                               queue_wait_samples.begin() + 5000);
  }

  float CalculatePercentile(const std::vector<long> &samples,
                            float percentile) {
    if (samples.empty())
      return 0.0f;
    std::vector<long> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    size_t idx = (size_t)(percentile * sorted.size());
    if (idx >= sorted.size())
      idx = sorted.size() - 1;
    return sorted[idx] / 1000.0f; // Convert to ms
  }

  float CalculateAverage(const std::vector<long> &samples) {
    if (samples.empty())
      return 0.0f;
    long sum = 0;
    for (long s : samples)
      sum += s;
    return (sum / samples.size()) / 1000.0f; // Convert to ms
  }
};

/**
 * Request structure representing a single inference request.
 *
 * Lifecycle:
 * 1. Created and submitted to pending queue
 * 2. Moved to active queue when scheduled
 * 3. Processed in batches
 * 4. Marked as finished and cleaned up
 */
struct Request {
  int id;
  std::string prompt;
  int max_tokens;
  TokenCallback callback;
  void *user_data;

  // Generation state
  std::vector<int> tokens;
  int n_past = 0;
  bool is_prefill = true;
  int generated_count = 0;
  bool finished = false;

  // Request lifecycle control
  std::atomic<bool> cancelled{false}; // Cancellation flag
  std::string tier =
      "standard"; // Priority tier: "premium", "standard", "batch"

  // Synchronization for blocking API
  std::mutex mu;
  std::condition_variable cv;

  // PagedAttention block tables
  BlockTable block_table;

  // Embedding mode
  bool is_embedding = false;
  EmbeddingCallback embedding_callback = nullptr;
  densecore::PoolingStrategy pooling_type = densecore::PoolingStrategy::MEAN;
  bool normalize_embedding = true;

  // Grammar-based sampling (JSON mode)
  bool json_mode = false;
  GrammarConstraint grammar;

  // Timing and metrics
  std::chrono::steady_clock::time_point arrival_time;
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point first_token_time;
  std::chrono::steady_clock::time_point last_token_time;

  // Scheduling priority (lower value = higher priority)
  int priority = 100;
  bool is_high_priority = false;
  int estimated_length = 0;

  // Destructor for cleanup
  ~Request() {
    // Block tables are cleaned up by caller before deletion
  }
};

/**
 * Fair queue comparator with tier-based priority and aging.
 * Prevents starvation of long requests using SJF + aging mechanism.
 */
struct FairQueueComparator {
  // Tier priority mapping (lower = higher priority)
  static int GetTierPriority(const std::string &tier) {
    if (tier == "premium")
      return 0;
    if (tier == "standard")
      return 1;
    if (tier == "batch")
      return 2;
    return 1; // Default to standard
  }

  bool operator()(const Request *a, const Request *b) const {
    // 1. Check tier priority first
    int tier_a = GetTierPriority(a->tier);
    int tier_b = GetTierPriority(b->tier);
    if (tier_a != tier_b) {
      return tier_a > tier_b; // Lower tier value = higher priority (inverted
                              // for priority_queue)
    }

    // 2. Calculate effective priority with aging
    // Requests waiting > 500ms get priority boost
    auto now = std::chrono::steady_clock::now();
    constexpr auto kAgingThreshold = std::chrono::milliseconds(500);

    auto wait_a = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - a->arrival_time);
    auto wait_b = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - b->arrival_time);

    int effective_priority_a = a->priority;
    int effective_priority_b = b->priority;

    // Apply aging boost: reduce priority value (increase priority) for
    // long-waiting requests
    if (wait_a > kAgingThreshold) {
      effective_priority_a -= 50; // Boost priority
    }
    if (wait_b > kAgingThreshold) {
      effective_priority_b -= 50;
    }

    // 3. Compare effective priorities (SJF-style: lower priority value = higher
    // priority)
    if (effective_priority_a != effective_priority_b) {
      return effective_priority_a > effective_priority_b;
    }

    // 4. Tie-breaker: FCFS (earlier arrival first)
    return a->arrival_time > b->arrival_time;
  }
};

// Backward compatibility alias
using RequestPriorityComparator = FairQueueComparator;

/**
 * Model entry for multi-model support.
 * Manages a single loaded model with its KV cache and metadata.
 */
struct ModelEntry {
  std::string model_id;
  std::string model_path;

  // Owned resources with automatic cleanup via RAII
  std::unique_ptr<TransformerModel> model;
  std::unique_ptr<PagedKVCache> kv_cache;

  std::chrono::steady_clock::time_point last_used;
  int usage_count = 0;
  bool is_loaded = true;

  // No need for custom destructor - smart pointers handle cleanup automatically
};

// Object Pool for efficient resource reuse
template <typename T> class ObjectPool {
public:
  T *Acquire() {
    std::lock_guard<std::mutex> lock(mu_);
    if (pool_.empty()) {
      return new T();
    }
    T *obj = pool_.top();
    pool_.pop();
    return obj;
  }

  void Release(T *obj) {
    if (!obj)
      return;
    // Reset basic state if possible, though destructor/constructor pattern
    // common
    std::lock_guard<std::mutex> lock(mu_);
    pool_.push(obj);
  }

  ~ObjectPool() {
    while (!pool_.empty()) {
      delete pool_.top();
      pool_.pop();
    }
  }

private:
  std::stack<T *> pool_;
  std::mutex mu_;
};

/**
 * Global engine state.
 *
 * Manages:
 * - Multi-model pool with LRU eviction
 * - Request queues (pending and active)
 * - Worker thread lifecycle
 * - Metrics collection
 */
struct EngineState {
  // Multi-model pool
  std::map<std::string, std::unique_ptr<ModelEntry>>
      models; // Smart pointer ownership
  std::string default_model_id;
  int max_models = 5;
  std::mutex models_mu;

  // NUMA binding configuration (-1 = auto, >= 0 = specific node)
  int numa_node_id = -1;

  // Number of threads for compute (0 = auto-detect)
  int n_threads = 0;

  // Thread pinning policy for compute threads
  // 0 = SCATTER (maximize L3/bandwidth, best for latency-sensitive single-user)
  // 1 = COMPACT (share L2, leave room for other processes, best for throughput)
  int pinning_policy = 0; // Default: SCATTER

  // Advanced scheduler (vLLM-style)
  std::unique_ptr<densecore::Scheduler> scheduler; // Smart pointer ownership

  // ===========================================================================
  // Lock-Free Request Queue (replaces mutex-protected priority_queue)
  // ===========================================================================
  // Uses sharded FIFO queues per priority tier with tagged pointer ABA
  // protection. Much lower contention than mutex under high concurrency.
  // ===========================================================================
  densecore::ShardedPriorityQueue<Request> pending_requests;

  std::vector<Request *> active_requests; // Non-owning pointers

  // Object Pool for requests (thread-safe)
  ObjectPool<Request> request_pool;

  // Mutex only for active_requests (rarely contested, not on hot path)
  std::mutex active_mu;
  // Condition variable for blocking wait when queue is empty
  std::condition_variable queue_cv;
  std::mutex cv_mu; // Mutex for condition variable (required by
                    // std::condition_variable)

  // Worker thread
  std::thread worker_thread;
  std::atomic<EngineStatus> status{EngineStatus::RUNNING};

  // Metrics
  InternalMetrics metrics;

  // Persistent compute buffer for GGML context (eliminates malloc overhead)
  // Allocated once at startup, reused across iterations
  static constexpr size_t COMPUTE_BUFFER_SIZE =
      1024ULL * 1024ULL * 4096ULL; // 4GB
  std::unique_ptr<char[]> compute_buffer;
  bool compute_buffer_initialized = false;

  // Graph Caching (shared across threads because worker is single-threaded
  // usually, but we add a mutex just in case or for future proofing)
  struct ggml_context *graph_ctx = nullptr;

  struct GraphMetadata {
    struct ggml_cgraph *gf;
    struct ggml_tensor *embd_inp;
    struct ggml_tensor *pos;
    struct ggml_tensor *output;
  };
  std::map<int, GraphMetadata> graph_cache;
  std::mutex graph_mu;

  void InitComputeBuffer() {
    if (!compute_buffer_initialized) {
      // Use 64-byte alignment for AVX-512 compatibility
      void *ptr = nullptr;
      if (posix_memalign(&ptr, 64, COMPUTE_BUFFER_SIZE) != 0) {
        throw std::bad_alloc();
      }
      // Custom deleter for unique_ptr to handle free() instead of delete[]
      compute_buffer.reset(static_cast<char *>(ptr));
      compute_buffer_initialized = true;
    }
  }

  void InitGraphCache() {
    // Initialize persistent graph context (holds node definitions)
    // 16MB should be enough for graph topology metadata
    struct ggml_init_params params = {
        .mem_size = 512 * 1024 * 1024, // 512 MB for complex graph caching
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    graph_ctx = ggml_init(params);
    std::cerr << "[DenseCore] InitGraphCache: Context initialized with 512 MB"
              << std::endl;
  }

  void FreeGraphCache() {
    // ggml_free handles all graphs allocated within the context
    if (graph_ctx) {
      ggml_free(graph_ctx);
      graph_ctx = nullptr;
    }
    graph_cache.clear();
  }

  /**
   * Get model by ID with usage tracking.
   */
  ModelEntry *GetModel(const std::string &model_id) {
    std::lock_guard<std::mutex> lock(models_mu);
    auto it = models.find(model_id);
    if (it != models.end()) {
      it->second->last_used = std::chrono::steady_clock::now();
      it->second->usage_count++;
      return it->second.get();
    }
    return nullptr;
  }

  /**
   * Get default model or fall back to legacy model.
   */
  ModelEntry *GetDefaultModel() {
    if (!default_model_id.empty()) {
      return GetModel(default_model_id);
    }

    return nullptr;
  }

  /**
   * Evict least recently used model to free memory.
   */
  void EvictLRUModel() {
    std::lock_guard<std::mutex> lock(models_mu);
    if (models.size() <= (size_t)max_models)
      return;

    // Find LRU model
    std::string lru_id;
    auto oldest_time = std::chrono::steady_clock::now();

    for (auto &kv : models) {
      if (kv.second->last_used < oldest_time) {
        oldest_time = kv.second->last_used;
        lru_id = kv.first;
      }
    }

    if (!lru_id.empty()) {
      LOG_INFO("Evicting LRU model: ", lru_id);
      // Smart pointers automatically cleanup when erased
      models.erase(lru_id);
    }
  }

  /**
   * Shutdown engine and cleanup resources with graceful draining.
   * Waits up to 5 seconds for active requests to complete before force-killing.
   */
  void Shutdown() {
    EngineStatus expected = EngineStatus::RUNNING;
    if (!status.compare_exchange_strong(expected, EngineStatus::DRAINING)) {
      // Already shutting down or stopped
      if (worker_thread.joinable()) {
        worker_thread.join();
      }
      return;
    }

    LOG_INFO("Initiating graceful shutdown (DRAINING)...");
    queue_cv.notify_all();

    // Wait for active requests to drain with 5-second timeout
    constexpr auto kShutdownTimeout = std::chrono::seconds(5);
    const auto start_time = std::chrono::steady_clock::now();

    while (true) {
      {
        std::lock_guard<std::mutex> lock(active_mu);
        if (active_requests.empty()) {
          LOG_INFO("All active requests drained successfully.");
          break;
        }
      }

      auto elapsed = std::chrono::steady_clock::now() - start_time;
      if (elapsed >= kShutdownTimeout) {
        std::lock_guard<std::mutex> lock(active_mu);
        LOG_WARNING("Shutdown timeout after 5 seconds. Force-killing ",
                    active_requests.size(), " active requests.");
        // Mark remaining requests as finished to allow cleanup
        for (Request *req : active_requests) {
          req->finished = true;
          if (req->callback) {
            req->callback("Error: Engine shutdown", 1, req->user_data);
          }
          // We just mark them finished; the loop or pool will handle release,
          // or we rely on pool destructor
        }
        break;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Transition to STOPPED and notify worker
    status = EngineStatus::STOPPED;
    queue_cv.notify_all();

    if (worker_thread.joinable()) {
      worker_thread.join();
    }

    // Cleanup all models - smart pointers handle automatic cleanup
    {
      std::lock_guard<std::mutex> lock(models_mu);
      models.clear();
    }

    // Free graph cache resources
    FreeGraphCache();

    LOG_INFO("Engine shutdown complete.");
  }

  ~EngineState() {
    // Ensure shutdown is called if not already stopped
    if (status != EngineStatus::STOPPED) {
      Shutdown();
    }
  }
};

// Worker function declaration
void EngineLoop(EngineState *state);

#endif // DENSECORE_ENGINE_INTERNAL_H
