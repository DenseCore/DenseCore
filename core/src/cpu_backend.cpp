/**
 * @file cpu_backend.cpp
 * @brief CPU backend implementation using AVX2/AVX-512 kernels
 *
 * This file implements the CpuBackend class, wrapping the existing
 * SIMD-optimized kernels from simd_ops.h into the ComputeBackend interface.
 *
 * Includes a resizable static thread pool for parallel MatMul execution during
 * both decode (M=1) and prefill (M>1) phases.
 */

#include "cpu_backend.h"
#include "flash_attention.h"
#include "hardware_topology.h"
#include "inference.h" // For InferenceConfig
#include "kernels/cpu_int4.h"
#include "optimization_bridge.h"
#include "simd_platform.h"
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// For _mm_pause() / yield intrinsics
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#include <malloc.h> // For _aligned_malloc/_aligned_free
#else
#include <cstdlib> // For aligned_alloc/free
#endif

namespace densecore {

// =============================================================================
// RESIZABLE STATIC THREAD POOL IMPLEMENTATION
// =============================================================================
// A lightweight thread pool that supports dynamic resizing for performance
// tuning on different hardware (AVX2, AVX-512, etc.).
// =============================================================================

namespace {

/**
 * @brief Resizable static thread pool for CpuBackend parallelism
 *
 * Features:
 * - Lazy initialization (threads created on first use)
 * - Dynamic resizing via Configure() for hardware tuning
 * - Barrier-based synchronization for parallel_for
 * - Thread-safe reconfiguration
 */
class StaticThreadPool {
public:
  static StaticThreadPool &Instance() {
    static StaticThreadPool instance;
    return instance;
  }

  /**
   * @brief Configure/resize the thread pool
   *
   * Thread-safe method to change the number of worker threads.
   * If the pool is running, it will be shut down and restarted with
   * the new thread count on the next compute call.
   *
   * @param n_threads Desired thread count (<=0 means auto-detect)
   */
  void Configure(int n_threads) {
    // Resolve auto-detect
    if (n_threads <= 0) {
      n_threads = static_cast<int>(std::thread::hardware_concurrency());
      if (n_threads <= 0)
        n_threads = 4;
    }

    // Cap to reasonable limit
    constexpr int kMaxThreads = 64;
    n_threads = std::min(n_threads, kMaxThreads);

    // Lock for thread-safe reconfiguration
    std::lock_guard<std::mutex> config_lock(config_mutex_);

    // No change needed
    if (n_threads == num_threads_ && initialized_) {
      return;
    }

    // Shutdown existing pool if running
    if (initialized_) {
      ShutdownInternal();
    }

    // Update config
    num_threads_ = n_threads;
    // Pool will be lazily reinitialized on next compute call
  }

  /**
   * @brief Get the number of worker threads (including main thread)
   */
  int GetNumThreads() const { return num_threads_; }

  /**
   * @brief Execute a parallel_for-style loop
   *
   * Distributes work across all threads. The main thread also participates.
   * Blocks until all threads complete their work.
   *
   * @param total_work Total number of work items
   * @param work_fn Function called with (start, end, thread_id) for each thread
   */
  void ParallelFor(int total_work,
                   const std::function<void(int, int, int)> &work_fn) {
    if (total_work <= 0)
      return;

    // Single-threaded fast path
    if (num_threads_ <= 1 || total_work == 1) {
      work_fn(0, total_work, 0);
      return;
    }

    // Ensure pool is initialized (thread-safe)
    EnsureInitialized();

    // Store work function and range
    current_work_fn_ = &work_fn;
    total_work_ = total_work;
    completed_count_.store(0, std::memory_order_relaxed);

    // Wake up worker threads
    {
      std::lock_guard<std::mutex> lock(mutex_);
      work_ready_ = true;
      generation_++;
    }
    cv_work_.notify_all();

    // Main thread (thread 0) does its share
    const int work_per_thread = (total_work + num_threads_ - 1) / num_threads_;
    const int start = 0;
    const int end = std::min(work_per_thread, total_work);
    work_fn(start, end, 0);

    // Mark main thread as done
    completed_count_.fetch_add(1, std::memory_order_release);

    // Wait for all workers to complete
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_done_.wait(lock, [this] {
        return completed_count_.load(std::memory_order_acquire) == num_threads_;
      });
      work_ready_ = false;
    }
  }

  /**
   * @brief Parallel GEMV dispatch - directly calls GemvParallel with thread
   * partitioning
   */
  void ParallelGemv(float *output, const float *input, const float *weight,
                    int K, int N) {
    if (num_threads_ <= 1) {
      simd::GemvParallel(output, input, weight, K, N, 0, 1);
      return;
    }

    EnsureInitialized();

    // Store GEMV parameters
    gemv_output_ = output;
    gemv_input_ = input;
    gemv_weight_ = weight;
    gemv_K_ = K;
    gemv_N_ = N;
    is_gemv_work_ = true;
    completed_count_.store(0, std::memory_order_relaxed);

    // Wake up worker threads
    {
      std::lock_guard<std::mutex> lock(mutex_);
      work_ready_ = true;
      generation_++;
    }
    cv_work_.notify_all();

    // Main thread (thread 0) does its share
    simd::GemvParallel(output, input, weight, K, N, 0, num_threads_);
    completed_count_.fetch_add(1, std::memory_order_release);

    // Wait for all workers to complete
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_done_.wait(lock, [this] {
        return completed_count_.load(std::memory_order_acquire) == num_threads_;
      });
      work_ready_ = false;
      is_gemv_work_ = false;
    }
  }

private:
  StaticThreadPool() {
    // Determine thread count from InferenceConfig or hardware
    num_threads_ = InferenceConfig::Instance().num_threads;
    if (num_threads_ <= 0) {
      num_threads_ = static_cast<int>(std::thread::hardware_concurrency());
      if (num_threads_ <= 0)
        num_threads_ = 4;
    }

    // Cap to reasonable limit for decode performance
    constexpr int kMaxThreads = 64;
    num_threads_ = std::min(num_threads_, kMaxThreads);
  }

  ~StaticThreadPool() { ShutdownInternal(); }

  // Non-copyable
  StaticThreadPool(const StaticThreadPool &) = delete;
  StaticThreadPool &operator=(const StaticThreadPool &) = delete;

  void EnsureInitialized() {
    if (initialized_)
      return;

    std::lock_guard<std::mutex> lock(config_mutex_);
    if (initialized_)
      return; // Double-check

    std::cerr << "[DEBUG] StaticThreadPool::EnsureInitialized: Creating "
              << num_threads_ << " threads..." << std::endl;

    // Create worker threads (thread 0 is main thread, workers are 1..N-1)
    shutdown_ = false;
    workers_.reserve(num_threads_ - 1);

    for (int i = 1; i < num_threads_; ++i) {
      std::cerr << "[DEBUG] Creating worker " << i << std::endl;
      workers_.emplace_back([this, i]() {
        std::cerr << "[DEBUG] Worker " << i << " started" << std::endl;
        WorkerLoop(i);
      });
    }

    std::cerr << "[DEBUG] StaticThreadPool::EnsureInitialized: Done."
              << std::endl;

    initialized_ = true;

    // =========================================================================
    // THREAD PINNING: Pin workers to physical cores for cache locality
    // =========================================================================
    // This prevents OS scheduler from migrating threads between cores,
    // keeping L1/L2 caches hot and maintaining NUMA locality on multi-socket
    // systems. Performance improvement: 30-50% on server-grade CPUs.
    // =========================================================================
    // Default to NUMA node 0. Override via DENSECORE_NUMA_NODE env var.
    int numa_node = 0;
    if (const char *env_numa = std::getenv("DENSECORE_NUMA_NODE")) {
      numa_node = std::atoi(env_numa);
    }
    HardwareTopology::GetInstance().PinThreadPool(workers_, numa_node,
                                                  PinningPolicy::SCATTER);
    std::cerr << "[DEBUG] StaticThreadPool: Pinned " << workers_.size()
              << " workers to NUMA node " << numa_node << std::endl;
  }

  void ShutdownInternal() {
    if (!initialized_)
      return;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      shutdown_ = true;
      generation_++;
    }
    cv_work_.notify_all();

    for (auto &t : workers_) {
      if (t.joinable()) {
        t.join();
      }
    }
    workers_.clear();
    initialized_ = false;
  }

  void WorkerLoop(int thread_id) {
    // Pin this worker thread to assigned core on first entry
    HardwareTopology::GetInstance().PinComputeThread(thread_id);

    uint64_t my_generation = 0;

    while (true) {
      // =========================================================================
      // HYBRID SPINNING: Busy-spin before falling back to cv.wait()
      // =========================================================================
      // During rapid token generation (decode phase), threads stay "hot" by
      // spinning for a short duration before sleeping. This reduces wake-up
      // latency from ~10-50µs to sub-microsecond.
      //
      // The spin loop checks generation_ without taking the lock, using
      // acquire semantics for proper synchronization.
      // =========================================================================
      // During rapid token generation (decode phase), threads stay "hot" by
      // spinning for a short duration before sleeping. This reduces wake-up
      // latency from ~10-50µs to sub-microsecond.
      // =========================================================================
      constexpr int kSpinIterations = 5000; // ~100-500µs on modern CPUs
      bool got_work = false;

      for (int spin = 0; spin < kSpinIterations; ++spin) {
        // Check for new work without lock
        if (generation_.load(std::memory_order_acquire) > my_generation) {
          got_work = true;
          break;
        }
        // CPU pause instruction - reduces power while keeping core active
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
        _mm_pause();
#elif defined(__aarch64__)
        __asm__ volatile("yield");
#endif
      }

      if (got_work) {
        // Found work during spin
      } else {
        // Fallback to wait
      }

      // If spinning didn't find work, fall back to condition variable
      if (!got_work) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_work_.wait(lock, [this, &my_generation] {
          return shutdown_ || (work_ready_ && generation_ > my_generation);
        });

        if (shutdown_)
          return;
      }

      // Important: Always update my_generation to the current generation
      // before executing work, to ensure we don't re-execute the same work.
      my_generation = generation_.load(std::memory_order_acquire);

      // Execute work
      if (is_gemv_work_) {
        // GEMV path
        simd::GemvParallel(gemv_output_, gemv_input_, gemv_weight_, gemv_K_,
                           gemv_N_, thread_id, num_threads_);
      } else if (current_work_fn_) {
        // Generic parallel_for path
        const int work_per_thread =
            (total_work_ + num_threads_ - 1) / num_threads_;
        const int start = thread_id * work_per_thread;
        const int end = std::min(start + work_per_thread, total_work_);

        if (start < total_work_) {
          (*current_work_fn_)(start, end, thread_id);
        }
      }

      // Signal completion
      int count = completed_count_.fetch_add(1, std::memory_order_release) + 1;
      if (count == num_threads_) {
        cv_done_.notify_one();
      }
    }
  }

  // Configuration state (protected by config_mutex_)
  std::mutex config_mutex_;
  int num_threads_;
  std::vector<std::thread> workers_;
  bool initialized_ = false;

  // Synchronization
  std::mutex mutex_;
  std::condition_variable cv_work_;
  std::condition_variable cv_done_;
  std::atomic<int> completed_count_{0};
  std::atomic<bool> work_ready_{false}; // Atomic for lock-free spin check
  std::atomic<bool> shutdown_{false};   // Atomic for lock-free spin check
  std::atomic<uint64_t> generation_{0}; // Atomic for lock-free spin check

  // Work specification (generic parallel_for)
  const std::function<void(int, int, int)> *current_work_fn_ = nullptr;
  int total_work_ = 0;

  // Work specification (GEMV fast path)
  bool is_gemv_work_ = false;
  float *gemv_output_ = nullptr;
  const float *gemv_input_ = nullptr;
  const float *gemv_weight_ = nullptr;
  int gemv_K_ = 0;
  int gemv_N_ = 0;
};

} // anonymous namespace

// =============================================================================
// Public API for Thread Pool Configuration
// =============================================================================

/**
 * @brief Update the number of threads used by the CPU backend
 *
 * This allows runtime tuning of parallelism for different hardware:
 * - AVX2 systems may benefit from more threads to compensate for lower IPC
 * - AVX-512 systems may use fewer threads for better cache locality
 *
 * @param n_threads Number of threads to use (<=0 for auto-detect)
 */
void UpdateBackendThreads(int n_threads) {
  StaticThreadPool::Instance().Configure(n_threads);
}

/**
 * @brief Get the current number of threads used by the CPU backend
 */
int GetBackendThreadCount() {
  return StaticThreadPool::Instance().GetNumThreads();
}

// =============================================================================
// Constructor / Destructor
// =============================================================================

CpuBackend::CpuBackend() {
  // Detect SIMD level at runtime
  simd_level_ = simd::DetectSimdLevel();
  selected_isa_ = simd::SimdLevelName(simd_level_);

  std::cout << "[CpuBackend] Initialized with SIMD level: " << selected_isa_
            << std::endl;
}

CpuBackend::~CpuBackend() {
  // Nothing to clean up - all memory is externally managed
}

// =============================================================================
// Memory Management
// =============================================================================

void *CpuBackend::AllocateDevice(size_t size_bytes, size_t alignment) {
  if (size_bytes == 0) {
    return nullptr;
  }

  void *ptr = nullptr;

#if defined(_WIN32)
  ptr = _aligned_malloc(size_bytes, alignment);
#else
  // C11 aligned_alloc requires size to be multiple of alignment
  size_t aligned_size = ((size_bytes + alignment - 1) / alignment) * alignment;
  ptr = std::aligned_alloc(alignment, aligned_size);
#endif

  return ptr;
}

void CpuBackend::FreeDevice(void *ptr) {
  if (ptr == nullptr) {
    return;
  }

#if defined(_WIN32)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

void CpuBackend::CopyToDevice(void *dst, const void *src, size_t size_bytes) {
  if (dst && src && size_bytes > 0) {
    std::memcpy(dst, src, size_bytes);
  }
}

void CpuBackend::CopyFromDevice(void *dst, const void *src, size_t size_bytes) {
  if (dst && src && size_bytes > 0) {
    std::memcpy(dst, src, size_bytes);
  }
}

// =============================================================================
// Graph Capture (Immediate Mode Fallback)
// =============================================================================

void CpuBackend::BeginCapture() {
  if (capturing_) {
    std::cerr << "[CpuBackend] Warning: Already capturing, resetting graph"
              << std::endl;
  }
  capturing_ = true;
  captured_graph_ = std::make_unique<ImmediateModeGraph>();
}

std::unique_ptr<OperationGraph> CpuBackend::EndCapture() {
  if (!capturing_) {
    std::cerr << "[CpuBackend] Warning: EndCapture called without BeginCapture"
              << std::endl;
    return std::make_unique<ImmediateModeGraph>();
  }
  capturing_ = false;
  return std::move(captured_graph_);
}

void CpuBackend::ExecuteGraph(const OperationGraph &graph) {
  //
  // This file is part of DenseCore Reference Implementation.
  // Licensed under Apache 2.0 (Open Source) or Commercial License.
  //
  // This graph replay demonstrates the Graph API without exposing
  // proprietary graph compilation logic (used in Metal/ANE backends).
  //

  // Fast path: ImmediateModeGraph stores lambdas and replays directly
  if (const auto *imm = dynamic_cast<const ImmediateModeGraph *>(&graph)) {
    const_cast<ImmediateModeGraph *>(imm)->Replay();
    return;
  }

  // Generic graph replay: iterate nodes and dispatch based on OpType
  // This is the reference implementation that proves the Graph API works.
  for (size_t i = 0; i < graph.NodeCount(); ++i) {
    const GraphNode &node = graph.GetNode(i);

    switch (node.op) {
    // =========================================================================
    // Linear Algebra Operations
    // =========================================================================
    case OpType::MatMul: {
      if (node.inputs.size() >= 2 && node.outputs.size() >= 1) {
        const Tensor &A = graph.GetTensor(node.inputs[0]);
        const Tensor &B = graph.GetTensor(node.inputs[1]);
        Tensor output = graph.GetTensor(node.outputs[0]);
        MatMul(A, B, &output);
      }
      break;
    }

    case OpType::MatMulTransB: {
      if (node.inputs.size() >= 2 && node.outputs.size() >= 1) {
        const Tensor &A = graph.GetTensor(node.inputs[0]);
        const Tensor &B = graph.GetTensor(node.inputs[1]);
        Tensor output = graph.GetTensor(node.outputs[0]);
        MatMulTransB(A, B, &output);
      }
      break;
    }

    // =========================================================================
    // Normalization Operations
    // =========================================================================
    case OpType::RMSNorm: {
      if (node.inputs.size() >= 2 && node.outputs.size() >= 1) {
        const Tensor &input = graph.GetTensor(node.inputs[0]);
        const Tensor &weight = graph.GetTensor(node.inputs[1]);
        Tensor output = graph.GetTensor(node.outputs[0]);

        float eps = 1e-5f;
        if (auto *params = std::get_if<RMSNormParams>(&node.params)) {
          eps = params->eps;
        }
        RMSNorm(input, weight, &output, eps);
      }
      break;
    }

    case OpType::AddRMSNorm: {
      if (node.inputs.size() >= 3 && node.outputs.size() >= 1) {
        const Tensor &input = graph.GetTensor(node.inputs[0]);
        const Tensor &residual = graph.GetTensor(node.inputs[1]);
        const Tensor &weight = graph.GetTensor(node.inputs[2]);
        Tensor output = graph.GetTensor(node.outputs[0]);

        float eps = 1e-5f;
        if (auto *params = std::get_if<RMSNormParams>(&node.params)) {
          eps = params->eps;
        }
        AddRMSNorm(input, residual, weight, &output, eps);
      }
      break;
    }

    // =========================================================================
    // Activation Operations
    // =========================================================================
    case OpType::Softmax: {
      if (node.inputs.size() >= 1 && node.outputs.size() >= 1) {
        const Tensor &input = graph.GetTensor(node.inputs[0]);
        Tensor output = graph.GetTensor(node.outputs[0]);
        Softmax(input, &output);
      }
      break;
    }

    // =========================================================================
    // Attention Operations
    // =========================================================================
    case OpType::FlashAttention: {
      if (node.inputs.size() >= 3 && node.outputs.size() >= 1) {
        const Tensor &Q = graph.GetTensor(node.inputs[0]);
        const Tensor &K = graph.GetTensor(node.inputs[1]);
        const Tensor &V = graph.GetTensor(node.inputs[2]);
        Tensor output = graph.GetTensor(node.outputs[0]);

        float scale = 1.0f;
        bool causal = true;
        int n_head_kv = -1;
        if (auto *params = std::get_if<FlashAttentionParams>(&node.params)) {
          scale = params->scale;
          causal = params->causal;
          n_head_kv = params->n_head_kv;
        }
        FlashAttention(Q, K, V, &output, scale, causal, n_head_kv);
      }
      break;
    }

    // =========================================================================
    // Unsupported Operations
    // =========================================================================
    default:
      std::cerr << "[CpuBackend] ExecuteGraph: Unsupported op type '"
                << OpTypeName(node.op) << "' at node " << i << std::endl;
      break;
    }
  }
}

// =============================================================================
// Quantization Operations
// =============================================================================

void CpuBackend::Quantize(const Tensor &src, QuantizedTensorView *dst,
                          QuantType type) {
  if (!src.IsValid() || !dst) {
    return;
  }

  const float *src_data = src.DataAs<float>();
  const int64_t n_elements = src.NumElements();

  switch (type) {
  case QuantType::INT8: {
    // Symmetric INT8 quantization
    // Find max absolute value
    float max_abs = 0.0f;
    for (int64_t i = 0; i < n_elements; ++i) {
      float abs_val = std::fabs(src_data[i]);
      if (abs_val > max_abs) {
        max_abs = abs_val;
      }
    }

    // Compute scale
    float scale = max_abs / 127.0f;
    if (scale == 0.0f) {
      scale = 1.0f; // Avoid division by zero
    }

    // Quantize
    int8_t *dst_data = static_cast<int8_t *>(dst->data);
    float inv_scale = 1.0f / scale;

#if defined(__AVX2__)
    // AVX2 vectorized quantization
    const __m256 v_scale = _mm256_set1_ps(inv_scale);
    const __m256 v_min = _mm256_set1_ps(-127.0f);
    const __m256 v_max = _mm256_set1_ps(127.0f);

    int64_t i = 0;
    for (; i + 8 <= n_elements; i += 8) {
      __m256 v_src = _mm256_loadu_ps(src_data + i);
      __m256 v_scaled = _mm256_mul_ps(v_src, v_scale);
      v_scaled = _mm256_max_ps(_mm256_min_ps(v_scaled, v_max), v_min);
      __m256i v_int = _mm256_cvtps_epi32(v_scaled);

      // Pack to int8 (AVX2 doesn't have direct pack to i8, so we do it
      // manually)
      alignas(32) int32_t temp[8];
      _mm256_store_si256(reinterpret_cast<__m256i *>(temp), v_int);
      for (int j = 0; j < 8; ++j) {
        dst_data[i + j] = static_cast<int8_t>(temp[j]);
      }
    }

    // Scalar tail
    for (; i < n_elements; ++i) {
      float val = src_data[i] * inv_scale;
      val = std::max(-127.0f, std::min(127.0f, val));
      dst_data[i] = static_cast<int8_t>(std::round(val));
    }
#else
    // Scalar fallback
    for (int64_t i = 0; i < n_elements; ++i) {
      float val = src_data[i] * inv_scale;
      val = std::max(-127.0f, std::min(127.0f, val));
      dst_data[i] = static_cast<int8_t>(std::round(val));
    }
#endif

    // Update quantization params
    dst->type = QuantType::INT8;
    dst->quant_params = QuantizationParams::PerTensor(scale, 0);
    break;
  }

  case QuantType::Q4_0:
  case QuantType::Q4_K: {
    // Block-wise 4-bit quantization (simplified)
    // For production, use GGML's optimized quantization routines
    std::cerr
        << "[CpuBackend] Quantize: Q4 quantization not fully implemented, "
        << "use GGML model loader for pre-quantized models" << std::endl;
    break;
  }

  default:
    std::cerr << "[CpuBackend] Quantize: Unsupported type "
              << QuantTypeName(type) << std::endl;
    break;
  }
}

void CpuBackend::Dequantize(const QuantizedTensorView &src, Tensor *dst) {
  if (!src.IsValid() || !dst || !dst->IsValid()) {
    return;
  }

  const int64_t n_elements = src.NumElements();
  float *dst_data = dst->DataAs<float>();

  switch (src.type) {
  case QuantType::INT8: {
    const int8_t *src_data = static_cast<const int8_t *>(src.data);
    float scale = 1.0f;

    if (src.quant_params.granularity ==
        QuantizationParams::Granularity::PerTensor) {
      scale = src.quant_params.tensor.scale;
    }

#if defined(__AVX2__)
    const __m256 v_scale = _mm256_set1_ps(scale);
    int64_t i = 0;
    for (; i + 8 <= n_elements; i += 8) {
      // Load 8 int8 values and convert to float
      alignas(32) float temp[8];
      for (int j = 0; j < 8; ++j) {
        temp[j] = static_cast<float>(src_data[i + j]);
      }
      __m256 v_src = _mm256_load_ps(temp);
      __m256 v_out = _mm256_mul_ps(v_src, v_scale);
      _mm256_storeu_ps(dst_data + i, v_out);
    }

    // Scalar tail
    for (; i < n_elements; ++i) {
      dst_data[i] = static_cast<float>(src_data[i]) * scale;
    }
#else
    for (int64_t i = 0; i < n_elements; ++i) {
      dst_data[i] = static_cast<float>(src_data[i]) * scale;
    }
#endif
    break;
  }

  case QuantType::FP16: {
    // FP16 to FP32 conversion
    // Note: Requires FP16 support headers
    std::cerr << "[CpuBackend] Dequantize: FP16 not implemented" << std::endl;
    break;
  }

  default:
    std::cerr << "[CpuBackend] Dequantize: Unsupported type "
              << QuantTypeName(src.type) << std::endl;
    break;
  }
}

// =============================================================================
// Matrix Operations
// =============================================================================

void CpuBackend::MatMul(const Tensor &A, const Tensor &B, Tensor *C) {
  // Basic validation
  if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) {
    return;
  }

  // A: [M, K], B: [K, N], C: [M, N]
  const int M = static_cast<int>(A.shape[0]);
  const int K = static_cast<int>(A.shape[1]);
  const int N = static_cast<int>(B.shape[1]);

  const float *a_data = A.DataAs<float>();
  const float *b_data = B.DataAs<float>();
  float *c_data = C->DataAs<float>();

  // ===========================================================================
  // OPTIMIZED PARALLEL DISPATCH
  // ===========================================================================

  if (M == 1) {
    // -------------------------------------------------------------------------
    // GEMV Path: Vector-Matrix multiplication (decode phase)
    // Use thread pool for parallel execution across output dimension
    // -------------------------------------------------------------------------
    StaticThreadPool::Instance().ParallelGemv(c_data, a_data, b_data, K, N);
  } else {
    // -------------------------------------------------------------------------
    // GEMM Path: Matrix-Matrix multiplication (prefill phase)
    // Use cache-blocking with parallel distribution across M dimension
    // -------------------------------------------------------------------------
    constexpr int BLOCK_M = 32; // Tile size for M dimension
    constexpr int BLOCK_N = 32; // Tile size for N dimension
    constexpr int BLOCK_K =
        64; // Tile size for K dimension (larger for cache line efficiency)

    // Zero output first
    std::memset(c_data, 0, M * N * sizeof(float));

    // Get thread pool
    auto &pool = StaticThreadPool::Instance();

    // Parallelize over M dimension blocks
    const int num_m_blocks = (M + BLOCK_M - 1) / BLOCK_M;

    pool.ParallelFor(num_m_blocks, [&](int block_start, int block_end,
                                       int /*tid*/) {
      for (int block_idx = block_start; block_idx < block_end; ++block_idx) {
        const int m0 = block_idx * BLOCK_M;
        const int m_end = std::min(m0 + BLOCK_M, M);

        for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
          const int n_end = std::min(n0 + BLOCK_N, N);

          for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
            const int k_end = std::min(k0 + BLOCK_K, K);

            // Inner kernel: compute small block
            for (int m = m0; m < m_end; ++m) {
              const float *a_row = a_data + m * K;
              float *c_row = c_data + m * N;

              for (int k = k0; k < k_end; ++k) {
                const float a_mk = a_row[k];
                const float *b_row = b_data + k * N;

// Vectorizable inner loop
#pragma GCC ivdep
                for (int n = n0; n < n_end; ++n) {
                  c_row[n] += a_mk * b_row[n];
                }
              }
            }
          }
        }
      }
    });
  }
}

void CpuBackend::MatMulTransB(const Tensor &A, const Tensor &B, Tensor *C) {
  if (!A.IsValid() || !B.IsValid() || !C || !C->IsValid()) {
    return;
  }

  // A: [M, K], B: [N, K] (stored row-major, will be transposed), C: [M, N]
  const int M = static_cast<int>(A.shape[0]);
  const int K = static_cast<int>(A.shape[1]);
  const int N = static_cast<int>(B.shape[0]);

  const float *a_data = A.DataAs<float>();
  const float *b_data = B.DataAs<float>();
  float *c_data = C->DataAs<float>();

  // Use simd::MatMulTransB which is optimized for this layout
  simd::MatMulTransB(c_data, a_data, b_data, M, N, K);
}

void CpuBackend::GemmInt4(const Tensor &A, const Tensor &W,
                          const Tensor &scales, const Tensor &zero_points,
                          Tensor *C, int group_size) {
  if (!A.IsValid() || !W.IsValid() || !C || !C->IsValid()) {
    return;
  }

  const int M = static_cast<int>(A.shape[0]);
  const int K = static_cast<int>(A.shape[1]);
  const int N = static_cast<int>(W.shape[0]);

  const float *a_data = A.DataAs<float>();
  const uint8_t *w_data = W.DataAs<uint8_t>();
  const float *scales_data = scales.DataAs<float>();
  const float *zeros_data = zero_points.DataAs<float>();
  float *c_data = C->DataAs<float>();

  // ===========================================================================
  // DECODE OPTIMIZATION: Use GEMV kernel for M=1 (token generation)
  // ===========================================================================
  // During decode, we generate one token at a time (M=1), making this a
  // matrix-vector multiplication rather than matrix-matrix. The GEMV kernel
  // parallelizes across the N dimension for better utilization.
  // ===========================================================================
  if (M == 1) {
    const int n_threads = StaticThreadPool::Instance().GetNumThreads();
    if (n_threads <= 1) {
      // Single-threaded: process all N at once
      kernels::GemvInt4(c_data, a_data, w_data, scales_data, zeros_data, K, N,
                        group_size, 0, N);
    } else {
      // Multi-threaded: partition N across threads using ParallelFor
      StaticThreadPool::Instance().ParallelFor(
          N, [&](int n_start, int n_end, int thread_id) {
            (void)thread_id;
            kernels::GemvInt4(c_data, a_data, w_data, scales_data, zeros_data,
                              K, N, group_size, n_start, n_end);
          });
    }
    return;
  }

  // ===========================================================================
  // PREFILL: Use GEMM kernel for M>1 (batch processing)
  // ===========================================================================
  if (OpsRegistry::IsInitialized()) {
    OpsRegistry::Instance().GemmInt4(c_data, a_data, w_data, scales_data,
                                     zeros_data, M, N, K, group_size);
  } else {
#if defined(__AVX512F__)
    simd::GemmInt4Fp32_AVX512(c_data, a_data, w_data, scales_data, zeros_data,
                              M, N, K, group_size);
#elif defined(__AVX2__)
    simd::GemmInt4Fp32_AVX2(c_data, a_data, w_data, scales_data, zeros_data, M,
                            N, K, group_size);
#else
    std::cerr << "[CpuBackend] GemmInt4: No SIMD support, operation skipped"
              << std::endl;
#endif
  }
}

// =============================================================================
// Normalization Operations
// =============================================================================

void CpuBackend::RMSNorm(const Tensor &input, const Tensor &weight,
                         Tensor *output, float eps) {
  if (!input.IsValid() || !weight.IsValid() || !output || !output->IsValid()) {
    return;
  }

  const int64_t n_elements = input.NumElements();
  const int64_t hidden_dim = weight.shape[0];

  if (hidden_dim == 0) {
    return;
  }

  const int64_t n_tokens = n_elements / hidden_dim;
  const float *x = input.DataAs<float>();
  const float *w = weight.DataAs<float>();
  float *out = output->DataAs<float>();

  // Process each token
  for (int64_t t = 0; t < n_tokens; ++t) {
    const float *x_ptr = x + t * hidden_dim;
    float *out_ptr = out + t * hidden_dim;

    // Compute RMS
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < hidden_dim; ++i) {
      sum_sq += x_ptr[i] * x_ptr[i];
    }
    float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_dim) + eps);

    // Normalize and apply weight
    for (int64_t i = 0; i < hidden_dim; ++i) {
      out_ptr[i] = x_ptr[i] * rms * w[i];
    }
  }
}

void CpuBackend::AddRMSNorm(const Tensor &input, const Tensor &residual,
                            const Tensor &weight, Tensor *output, float eps) {
  if (!input.IsValid() || !residual.IsValid() || !weight.IsValid() || !output ||
      !output->IsValid()) {
    return;
  }

  const int64_t hidden_dim = weight.shape[0];
  const int64_t n_elements = input.NumElements();

  if (hidden_dim == 0) {
    return;
  }

  const int64_t n_tokens = n_elements / hidden_dim;
  const float *x = input.DataAs<float>();
  const float *res = residual.DataAs<float>();
  const float *w = weight.DataAs<float>();
  float *out = output->DataAs<float>();

  // Use simd::AddRMSNorm for each token (it handles one vector at a time)
  for (int64_t t = 0; t < n_tokens; ++t) {
    const float *x_ptr = x + t * hidden_dim;
    const float *res_ptr = res + t * hidden_dim;
    float *out_ptr = out + t * hidden_dim;

    simd::AddRMSNorm(out_ptr, x_ptr, res_ptr, w,
                     static_cast<size_t>(hidden_dim), eps);
  }
}

// =============================================================================
// Activation Operations
// =============================================================================

void CpuBackend::Softmax(const Tensor &input, Tensor *output) {
  if (!input.IsValid() || !output || !output->IsValid()) {
    return;
  }

  // Copy input to output first, then do in-place softmax
  const size_t size = input.SizeBytes();
  std::memcpy(output->data, input.data, size);

  SoftmaxInplace(output);
}

void CpuBackend::SoftmaxInplace(Tensor *data) {
  if (!data || !data->IsValid()) {
    return;
  }

  // Apply softmax along last dimension
  const int64_t n = data->shape[data->ndim - 1];
  int64_t batch_size = 1;
  for (int i = 0; i < data->ndim - 1; ++i) {
    batch_size *= data->shape[i];
  }

  float *ptr = data->DataAs<float>();
  for (int64_t b = 0; b < batch_size; ++b) {
    simd::SoftmaxF32(ptr + b * n, static_cast<size_t>(n));
  }
}

// =============================================================================
// Position Encoding Operations
// =============================================================================

void CpuBackend::RoPE(const Tensor &input, const Tensor &cos_sin,
                      const int *positions, Tensor *output, int rope_dim) {
  if (!input.IsValid() || !cos_sin.IsValid() || !positions || !output ||
      !output->IsValid()) {
    return;
  }

  // Determine dimensions from input shape
  // Supported layouts: [seq_len, head_dim] or [n_heads, seq_len, head_dim]
  int n_tokens, head_dim, n_heads;

  if (input.ndim == 2) {
    n_tokens = static_cast<int>(input.shape[0]);
    head_dim = static_cast<int>(input.shape[1]);
    n_heads = 1;
  } else if (input.ndim == 3) {
    n_heads = static_cast<int>(input.shape[0]);
    n_tokens = static_cast<int>(input.shape[1]);
    head_dim = static_cast<int>(input.shape[2]);
  } else {
    return; // Unsupported layout
  }

  if (rope_dim < 0) {
    rope_dim = head_dim;
  }

  const float *in = input.DataAs<float>();
  const float *cs = cos_sin.DataAs<float>();
  float *out = output->DataAs<float>();

  // Use simd::ApplyRoPE which handles the actual rotation
  simd::ApplyRoPE(out, in, cs, positions, n_tokens, head_dim, rope_dim);
}

// =============================================================================
// Fused Operations
// =============================================================================

void CpuBackend::FusedQKVProjection(const Tensor &input, const Tensor &wq,
                                    const Tensor &wk, const Tensor &wv,
                                    Tensor *q_out, Tensor *k_out,
                                    Tensor *v_out) {
  if (!input.IsValid() || !wq.IsValid() || !wk.IsValid() || !wv.IsValid() ||
      !q_out || !k_out || !v_out) {
    return;
  }

  const int n_tokens = static_cast<int>(input.shape[0]);
  const int n_embd = static_cast<int>(input.shape[1]);
  const int dim_q = static_cast<int>(wq.shape[0]);
  const int dim_k = static_cast<int>(wk.shape[0]);
  const int dim_v = static_cast<int>(wv.shape[0]);

  const float *x = input.DataAs<float>();
  const float *w_q = wq.DataAs<float>();
  const float *w_k = wk.DataAs<float>();
  const float *w_v = wv.DataAs<float>();
  float *q = q_out->DataAs<float>();
  float *k = k_out->DataAs<float>();
  float *v = v_out->DataAs<float>();

  // Process each token
  for (int t = 0; t < n_tokens; ++t) {
    const float *x_t = x + t * n_embd;
    float *q_t = q + t * dim_q;
    float *k_t = k + t * dim_k;
    float *v_t = v + t * dim_v;

    // Use simd::ComputeQKV for the actual computation
    // Single-threaded per token; parallelism handled at higher level
    simd::ComputeQKV(q_t, k_t, v_t, x_t, w_q, w_k, w_v, n_embd, dim_q, dim_k,
                     dim_v, 0, 1);
  }
}

void CpuBackend::FlashAttention(const Tensor &Q, const Tensor &K,
                                const Tensor &V, Tensor *output, float scale,
                                bool causal, int n_head_kv) {
  if (!Q.IsValid() || !K.IsValid() || !V.IsValid() || !output ||
      !output->IsValid()) {
    return;
  }

  // Expected layout: [batch, n_head, seq, head_dim]
  const int batch = static_cast<int>(Q.shape[0]);
  const int n_head = static_cast<int>(Q.shape[1]);
  const int seq_q = static_cast<int>(Q.shape[2]);
  const int head_dim = static_cast<int>(Q.shape[3]);
  const int seq_kv = static_cast<int>(K.shape[2]);

  // GQA Support: If n_head_kv not provided, infer from K tensor or assume MHA
  if (n_head_kv <= 0) {
    // Try to infer from K tensor shape (K.shape[1] is n_head_kv)
    n_head_kv = static_cast<int>(K.shape[1]);
    if (n_head_kv <= 0) {
      n_head_kv = n_head; // Fallback to MHA
    }
  }

  const float *q_data = Q.DataAs<float>();
  const float *k_data = K.DataAs<float>();
  const float *v_data = V.DataAs<float>();
  float *o_data = output->DataAs<float>();

  FlashAttentionConfig config;
  config.scale = scale;
  config.causal = causal;

  // Use FlashAttentionGQA which correctly handles both:
  // - MHA (n_head == n_head_kv, n_rep = 1)
  // - GQA (n_head > n_head_kv, n_rep = n_head / n_head_kv)
  // The GQA function computes h_kv = h_q / n_rep for correct KV head offset
  FlashAttentionGQA(q_data, k_data, v_data, o_data, batch, n_head, n_head_kv,
                    seq_q, seq_kv, head_dim, config);
}

// =============================================================================
// Singleton Accessor
// =============================================================================

CpuBackend &GetCpuBackend() {
  static CpuBackend instance;
  return instance;
}

} // namespace densecore
