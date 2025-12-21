#include "engine_internal.h"
#include "ggml.h"
#include "hardware_topology.h"
#include "optimization_bridge.h" // Runtime SIMD dispatch
#include "simd_ops.h"
#include <cstring>
#include <ggml-cpu.h>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "utils/raii_guards.h"

// Worker Loop (Continuous Batching) - Uses Scheduler for batch formation
void EngineLoop(EngineState *state) {
  try {
    // =========================================================================
    // STEP 0: Initialize SIMD dispatch table (must be first!)
    // =========================================================================
    // OpsRegistry selects optimal kernel implementations based on CPU caps.
    // Must be called before any inference operations that use SIMD kernels.
    // =========================================================================
    if (!densecore::OpsRegistry::IsInitialized()) {
      densecore::OpsRegistry::Init();
    }

    // =========================================================================
    // THREADING MODEL (DenseCore Unified Threading):
    // =========================================================================
    // DenseCore uses a two-level parallelism strategy WITHOUT OpenMP:
    //
    // 1. Request-level parallelism: std::thread workers in worker.cpp
    //    - Each worker processes requests from the scheduler queue
    //
    // 2. Compute-level parallelism: GGML's internal thread pool
    //    - Configured via ggml_backend_cpu_set_n_threads() below
    //    - All SIMD kernels in simd_ops.h are single-threaded
    //    - GGML callbacks invoke kernels with (ith, nth) for work partitioning
    //
    // OpenMP is NOT used to avoid thread oversubscription (nested parallelism).
    // =========================================================================

    // Resolve model once at loop start (for performance)
    ModelEntry *model_entry = state->GetDefaultModel();
    if (!model_entry) {
      std::cerr << "[DenseCore] FATAL: No model loaded. EngineLoop exiting."
                << std::endl;
      return;
    }
    TransformerModel *current_model = model_entry->model.get();
    PagedKVCache *current_kv_cache = model_entry->kv_cache.get();

    // =========================================================================
    // GGML BACKEND THREAD CONFIGURATION (Critical for performance)
    // =========================================================================
    // This is the key call that enables multi-threaded GGML compute!
    // Without this, GGML defaults to single-threaded execution.
    // GGML BACKEND THREAD CONFIGURATION (Critical for performance)
    // =========================================================================
    // This is the key call that enables multi-threaded GGML compute!
    // Without this, GGML defaults to single-threaded execution.
    int n_threads = state->n_threads;
    if (n_threads <= 0) {
      // Use PHYSICAL cores by default to avoid Hyperthreading thrashing
      // on AVX workloads where execution units are shared.
      auto &topo = densecore::HardwareTopology::GetInstance();
      int num_physical = topo.GetPhysicalCoreCount();
      // Fallback if detection fails
      if (num_physical <= 0)
        n_threads = densecore::simd::GetNumCores() / 2;
      else
        n_threads = num_physical;

      if (n_threads < 1)
        n_threads = 1;
    }

    ggml_backend_cpu_set_n_threads(current_model->backend, n_threads);
    std::cout << "[DenseCore] GGML backend configured: " << n_threads
              << " threads (Physical Cores prioritized)" << std::endl;

    // =========================================================================
    // THREAD PINNING POLICY (Critical for avoiding deadlocks)
    // =========================================================================
    // The orchestration/control thread (this thread) should NEVER be pinned!
    // Aggressive pinning was causing deadlocks on consumer hardware (e.g.,
    // i7-10870H) where the main thread and compute workers contended for Core
    // 0.
    //
    // Strategy:
    // - Orchestration thread: Let OS schedule freely (unpinned)
    // - Compute threads: Pin via SetupComputeThreadAffinity (done below)
    // =========================================================================
    {
      auto &topo = densecore::HardwareTopology::GetInstance();
      std::cout << "[DenseCore] Orchestration thread unpinned (OS Scheduled). "
                << "NUMA nodes detected: " << topo.GetNumaNodeCount()
                << std::endl;
      // NOTE: Do NOT call PinCurrentThread() or PinCurrentThreadToNumaNode()
      // for this thread. Only compute worker threads should be pinned.
    }

    // Setup compute thread affinity for GGML workers
    // Use SCATTER policy to spread threads across physical cores
    {
      auto &topo = densecore::HardwareTopology::GetInstance();
      int target_node = (state->numa_node_id >= 0) ? state->numa_node_id : 0;
      int physical_cores = topo.GetPhysicalCoreCount(target_node);

      if (n_threads > 0 && physical_cores > 0) {
        // Use simple SCATTER policy - spread threads across all physical cores
        topo.SetupComputeThreadAffinity(target_node, n_threads,
                                        densecore::PinningPolicy::SCATTER);
      }
    }

    // Initialize persistent compute buffer (eliminates malloc/free per
    // iteration)
    state->InitComputeBuffer();
    state->InitGraphCache(current_model);

    // Mapping: scheduler seq_id -> Request*
    std::unordered_map<int, Request *> seq_to_request;

    while (state->status != EngineStatus::STOPPED) {
      // 1. Fetch new requests and register with scheduler
      // =========================================================================
      // WAIT FOR WORK (Condition Variable - Lost Wakeup Safe)
      // =========================================================================
      // Use CV wait with predicate to atomically check conditions while
      // holding the lock. This prevents the "lost wakeup" race where a
      // producer pushes after our check but before we sleep.
      // =========================================================================
      {
        std::unique_lock<std::mutex> lock(state->cv_mu);
        state->queue_cv.wait(lock, [state]() {
          // Check if there's work to do OR we should exit
          bool pending_empty = state->pending_requests.Empty();
          bool active_empty;
          {
            std::lock_guard<std::mutex> active_lock(state->active_mu);
            active_empty = state->active_requests.empty();
          }
          // Wake up if: there's pending work, OR there's active work,
          // OR we're no longer running (DRAINING/STOPPED)
          return !pending_empty || !active_empty ||
                 state->status != EngineStatus::RUNNING;
        });
      }

      // std::cerr << "[DEBUG] EngineLoop: Woke up" << std::endl;

      // Check if draining and all work is done
      if (state->status == EngineStatus::DRAINING) {
        std::lock_guard<std::mutex> active_lock(state->active_mu);
        if (state->active_requests.empty() && state->pending_requests.Empty()) {
          break; // Exit: draining complete, no more work
        }
      }

      {
        // Note: std::mutex lock/unlock provides sufficient memory ordering.
        // No atomic fence needed - cv.wait() holding cv_mu provides acquire
        // semantics.

        while (state->status == EngineStatus::RUNNING) {
          Request *req = state->pending_requests.Pop();
          if (!req)
            break; // Queue empty

          // std::cerr << "[DEBUG] EngineLoop: Popped request " << req->id
          //           << std::endl;

          // Check if cancelled before even starting
          if (req->cancelled) {
            LOG_INFO("Skipping cancelled request: ", req->id);
            state->request_pool.Release(req);
            continue;
          }

          req->start_time = std::chrono::steady_clock::now();
          auto queue_wait_us =
              std::chrono::duration_cast<std::chrono::microseconds>(
                  req->start_time - req->arrival_time)
                  .count();
          state->metrics.RecordQueueWait(queue_wait_us);

          // Tokens are pre-tokenized in SubmitRequest - assert this
          // invariant
          if (req->tokens.empty()) {
            std::cerr << "[DenseCore] FATAL: Request " << req->id
                      << " has no tokens (should be pre-tokenized)"
                      << std::endl;
            req->finished = true;
            state->metrics.failed_requests++;
            // Push error event to callback queue (instead of direct callback)
            if (req->callback) {
              PushResultEvent(state, req->id, "Error: Missing tokens", true,
                              true, req->callback, req->user_data);
            }
            state->request_pool.Release(req);
            continue;
          }
          state->metrics.total_prompt_tokens += req->tokens.size();

          // Register with scheduler (blocks allocated by scheduler)
          int seq_id = state->scheduler->AddRequest(
              req->id, req->tokens.size(), req->max_tokens, req->priority,
              &req->tokens);

          if (seq_id < 0) {
            // Scheduler rejected (e.g., queue full)
            std::cerr << "[DenseCore] Scheduler rejected request " << req->id
                      << std::endl;
            req->finished = true;
            state->metrics.failed_requests++;
            // Push error event to callback queue (instead of direct callback)
            if (req->callback) {
              PushResultEvent(state, req->id, "Error: Scheduler queue full",
                              true, true, req->callback, req->user_data);
            }
            state->request_pool.Release(req);
            continue;
          }

          // Store seq_id in request for O(1) cleanup lookup
          req->seq_id = seq_id;

          // Store mapping and add to active list
          seq_to_request[seq_id] = req;
          state->active_requests.push_back(req);
          state->metrics.active_requests++;
          state->metrics.total_requests++;
          std::cerr << "[TRACE] Request " << req->id << " moved to active"
                    << std::endl;
        }
      }

      if (state->active_requests.empty())
        continue;

      // Check if we should exit (DRAINING and no more active work)
      {
        std::lock_guard<std::mutex> lock(state->active_mu);
        if (state->status == EngineStatus::DRAINING &&
            state->active_requests.empty()) {
          break;
        }
      }

      // 2. Process cancelled/finished requests first
      {
        std::lock_guard<std::mutex> lock(state->active_mu);
        auto it = state->active_requests.begin();
        while (it != state->active_requests.end()) {
          Request *req = *it;
          if (req->cancelled && !req->finished) {
            LOG_INFO("Cancelling active request: ", req->id);
            req->finished = true;
            state->metrics.failed_requests++;
            // Push cancellation event to callback queue (instead of direct
            // callback)
            if (req->callback) {
              PushResultEvent(state, req->id, "Error: Request cancelled", true,
                              true, req->callback, req->user_data);
            }
            // Remove from scheduler using stored seq_id (O(1) instead of O(n))
            if (req->seq_id >= 0) {
              state->scheduler->RemoveRequest(req->seq_id, false);
              seq_to_request.erase(req->seq_id);
            }
            // Free blocks if allocated
            current_kv_cache->block_manager->Free(req->block_table);
            req->block_table.clear();
            {
              std::lock_guard<std::mutex> lk(req->mu);
              req->cv.notify_all();
            }
          }
          ++it;
        }
      }

      // 3. Query Scheduler for next batch
      // std::cerr << "[DEBUG] EngineLoop: Calling Scheduler->Schedule()"
      //           << std::endl;
      densecore::SchedulerOutput sched_output = state->scheduler->Schedule();
      // std::cerr << "[DEBUG] EngineLoop: Schedule returned "
      //           << sched_output.prefill_seq_ids.size() << " prefill, "
      //           << sched_output.decode_seq_ids.size() << " decode" <<
      //           std::endl;

      // 4. Handle scheduler output: block allocations
      for (const auto &alloc : sched_output.new_block_allocations) {
        int seq_id = alloc.first;
        const std::vector<int> &block_ids = alloc.second;
        auto req_it = seq_to_request.find(seq_id);
        if (req_it != seq_to_request.end()) {
          Request *req = req_it->second;
          // Append new blocks to request's block table
          req->block_table.insert(req->block_table.end(), block_ids.begin(),
                                  block_ids.end());
        }
      }

      // 5. Handle scheduler output: freed blocks
      for (const auto &freed : sched_output.freed_blocks) {
        int seq_id = freed.first;
        const std::vector<int> &block_ids = freed.second;
        (void)seq_id; // Scheduler already handles via BlockManager
        current_kv_cache->block_manager->Free(block_ids);
      }

      // 6. Handle scheduler output: preempted sequences (swap out)
      for (int preempted_seq_id : sched_output.preempted_seq_ids) {
        auto req_it = seq_to_request.find(preempted_seq_id);
        if (req_it != seq_to_request.end()) {
          Request *req = req_it->second;
          // Simulated swapping: free blocks, mark for re-prefill later
          // For now, we just free the blocks and let scheduler handle
          // re-admission
          LOG_INFO("Preempting request ", req->id,
                   " (seq_id=", preempted_seq_id, ")");
          current_kv_cache->block_manager->Free(req->block_table);
          req->block_table.clear();
          req->is_prefill = true; // Will need re-prefill when un-swapped
          req->n_past = 0;
        }
      }

      // =========================================================================
      // 7. Handle Empty Scheduler Output (Latency-Aware Timed Wait)
      // =========================================================================
      // If scheduler returned empty batch but active_requests exist:
      //   - Memory fragmentation may prevent scheduling
      //   - Wait with SHORT timeout (100us) for fast response to freed memory
      //
      // Using wait_for with 100us timeout instead of yield():
      //   - Prevents 100% CPU usage from busy-looping
      //   - Enables sub-millisecond latency when work arrives
      //   - Balances power efficiency with responsiveness
      // =========================================================================
      if (sched_output.IsEmpty()) {
        bool has_active;
        {
          std::lock_guard<std::mutex> lock(state->active_mu);
          has_active = !state->active_requests.empty();
        }

        if (has_active) {
          // Active requests exist but scheduler couldn't schedule them
          // Wait with VERY short timeout for latency-sensitive operation
          std::unique_lock<std::mutex> lock(state->cv_mu);
          state->queue_cv.wait_for(
              lock, std::chrono::microseconds(100), [state]() {
                // Wake up if: new pending requests OR engine stopping
                return !state->pending_requests.Empty() ||
                       state->status != EngineStatus::RUNNING;
              });
        } else {
          // No active requests and scheduler empty - truly nothing to do
          // Use slightly longer wait to save power
          std::unique_lock<std::mutex> lock(state->cv_mu);
          state->queue_cv.wait_for(
              lock, std::chrono::milliseconds(1), [state]() {
                return !state->pending_requests.Empty() ||
                       state->status != EngineStatus::RUNNING;
              });
        }
        continue;
      }

      // 8. Form BatchSpec from SchedulerOutput
      BatchSpec batch;
      std::vector<Request *> batch_requests;
      bool is_embedding_batch = false;
      bool first_request = true;

      // Process prefill sequences
      for (int seq_id : sched_output.prefill_seq_ids) {
        auto req_it = seq_to_request.find(seq_id);
        if (req_it == seq_to_request.end())
          continue;

        Request *req = req_it->second;
        std::cerr << "[TRACE] Prefill loop: Found request " << req->id
                  << " for seq " << seq_id << " (finished=" << req->finished
                  << ")" << std::endl;
        if (req->finished)
          continue;

        // Enforce homogeneous batch type
        if (first_request) {
          is_embedding_batch = req->is_embedding;
          first_request = false;
        } else if (req->is_embedding != is_embedding_batch) {
          continue; // Skip mixed types
        }

        // Full prefill: all prompt tokens
        std::vector<int> tokens = req->tokens;
        std::vector<int> pos;
        for (size_t i = 0; i < tokens.size(); ++i) {
          pos.push_back(req->n_past + i);
        }

        int batch_seq_idx = batch_requests.size();
        batch.tokens.insert(batch.tokens.end(), tokens.begin(), tokens.end());
        batch.pos.insert(batch.pos.end(), pos.begin(), pos.end());
        for (size_t k = 0; k < tokens.size(); ++k) {
          batch.seq_id.push_back(batch_seq_idx);
        }
        batch.block_tables.push_back(req->block_table);
        batch.n_past.push_back(req->n_past);
        batch_requests.push_back(req);
      }

      // Process decode sequences
      for (int seq_id : sched_output.decode_seq_ids) {
        auto req_it = seq_to_request.find(seq_id);
        if (req_it == seq_to_request.end())
          continue;

        Request *req = req_it->second;
        std::cerr << "[TRACE] Decode loop: Found request " << req->id
                  << " for seq " << seq_id << " (finished=" << req->finished
                  << ")" << std::endl;
        if (req->finished)
          continue;

        // Enforce homogeneous batch type
        if (first_request) {
          is_embedding_batch = req->is_embedding;
          first_request = false;
        } else if (req->is_embedding != is_embedding_batch) {
          continue;
        }

        // Decode: single token
        int batch_seq_idx = batch_requests.size();
        batch.tokens.push_back(req->tokens.back());
        batch.pos.push_back(req->n_past);
        batch.seq_id.push_back(batch_seq_idx);
        batch.block_tables.push_back(req->block_table);
        batch.n_past.push_back(req->n_past);
        batch_requests.push_back(req);
      }

      if (batch_requests.empty()) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        continue;
      }

      batch.num_seqs = batch_requests.size();

      // =========================================================================
      // 9. Run Inference ("Rebuild Graph, Reuse Memory" Strategy)
      // =========================================================================
      // Graph caching is DISABLED because n_past changes every token, causing
      // stale RoPE offsets and tensor shapes. Instead, we:
      //   - Rebuild the graph every iteration (correct shapes for current
      //   n_past)
      //   - Reuse the memory pool (Reset() is O(1), no malloc/free syscalls)
      // =========================================================================
      struct ggml_cgraph *gf = nullptr;
      struct ggml_tensor *output = nullptr;
      struct ggml_tensor *embd_inp = nullptr;
      struct ggml_tensor *pos = nullptr;

      // Initialize inference context if not already done
      if (!state->inference_ctx.IsInitialized()) {
        size_t ctx_size = EngineState::CalculateGraphContextSize(current_model);
        state->inference_ctx.Init(ctx_size);
      }

      // Reset persistent context (O(1) - reuses existing memory buffer)
      // This prepares a fresh context for graph building without malloc/free
      state->inference_ctx.Reset();
      struct ggml_context *ctx_nodes = state->inference_ctx.GetContext();

      if (!ctx_nodes) {
        std::cerr << "[DenseCore] FATAL: InferenceContext not initialized!"
                  << std::endl;
        continue;
      }

      // Always build fresh graph with correct n_past/positions
      gf = ggml_new_graph_custom(ctx_nodes, 32768, false);

      // Build nodes in ctx_nodes
      output =
          BuildTransformerGraph(current_model, current_kv_cache, ctx_nodes,
                                batch, is_embedding_batch, gf, &embd_inp, &pos);
      // Note: Graph caching removed - incompatible with dynamic n_past

      if (!output || !embd_inp || !pos) {
        std::cerr << "Fatal: Content creation failed or tensors missing"
                  << std::endl;
        continue; // Recover
      }

      // SETUP INPUTS (Manual Data Assignment)
      // We use state->compute_buffer for input data

      // Use offset 0 of compute buffer for inputs
      char *input_base = state->compute_buffer.get();
      size_t embd_size = ggml_nbytes(embd_inp);

      embd_inp->data = input_base;
      pos->data = input_base + embd_size + 256; // alignment padding

      memcpy(embd_inp->data, batch.tokens.data(),
             batch.tokens.size() * sizeof(int));
      memcpy(pos->data, batch.pos.data(), batch.pos.size() * sizeof(int));

      // =======================================================================
      // MEMORY FENCE: Ensures visibility of input data to GGML worker threads.
      // =======================================================================
      // The memcpy operations above write input data to buffers that will be
      // read by GGML's internal thread pool workers. On modern CPUs with
      // out-of-order execution and store buffers, these writes may not be
      // immediately visible to other CPU cores.
      //
      // We use a release fence as the publishing side of the synchronization:
      // - std::memory_order_release guarantees that all preceding writes
      //   (the memcpy calls) are visible before any subsequent synchronization
      //   operation that has acquire semantics.
      //
      // GGML's thread pool internally uses proper synchronization (typically
      // condition variables or atomics) with acquire semantics when worker
      // threads start execution, completing the acquire-release synchronization
      // pair and ensuring they observe the input data correctly.
      //
      // This replaces the previous fragile sleep_for(100us) workaround which:
      // - Was not guaranteed to work (timing-based, system-dependent)
      // - Added unnecessary latency to every inference call
      // - Could fail under heavy load or on different CPU architectures
      // =======================================================================
      SetCurrentBatch(&batch);

      // Note: Memory ordering for GGML worker threads is handled internally
      // by GGML's thread pool (uses CV/mutex). No explicit fence needed.

      // =========================================================================
      // DYNAMIC THREAD THROTTLING (Decode Latency Optimization)
      // =========================================================================
      // MAXIMUM THREADING: Use all available threads for both Prefill and
      // Decode
      // =========================================================================
      // User requested maximum multi-threading performance.
      // Previous 8-thread cap for decode removed - let hardware saturate.
      // Modern CPUs with high core counts benefit from full parallelism.
      // =========================================================================
      ggml_backend_cpu_set_n_threads(current_model->backend, state->n_threads);

      // std::cerr << "[DEBUG] EngineLoop: Calling ggml_backend_graph_compute"
      //           << std::endl;
      ggml_backend_graph_compute(current_model->backend, gf);
      std::cerr << "[TRACE] Graph compute done for batch size "
                << batch.num_seqs << std::endl;
      // std::cerr << "[DEBUG] EngineLoop: ggml_backend_graph_compute returned"
      //           << std::endl;

      // Note: GGML's thread pool join provides acquire semantics.
      // No explicit fence needed to see results.

      // 10. Process Outputs
      int token_offset = 0;
      int n_embd = current_model->hparams.n_embd;

      for (int i = 0; i < batch.num_seqs; ++i) {
        Request *req = batch_requests[i];
        std::cerr << "[DEBUG] EngineLoop: Processing output for request "
                  << req->id << std::endl;
        int processed_count = 0;
        if (req->is_prefill)
          processed_count = req->tokens.size();
        else
          processed_count = 1;

        int last_token_idx = token_offset + processed_count - 1;

        if (is_embedding_batch) {
          float *data = (float *)output->data;
          int seq_len = processed_count;
          float *hidden_states = data + token_offset * n_embd;
          std::vector<float> embedding(n_embd);

          switch (req->pooling_type) {
          case densecore::PoolingStrategy::MEAN:
            densecore::simd::MeanPool(hidden_states, embedding.data(), seq_len,
                                      n_embd);
            break;
          case densecore::PoolingStrategy::CLS:
            densecore::simd::ClsPool(hidden_states, embedding.data(), n_embd);
            break;
          case densecore::PoolingStrategy::LAST:
            densecore::simd::LastPool(hidden_states, embedding.data(), seq_len,
                                      n_embd);
            break;
          case densecore::PoolingStrategy::MAX:
            densecore::simd::MaxPool(hidden_states, embedding.data(), seq_len,
                                     n_embd);
            break;
          default:
            densecore::simd::MeanPool(hidden_states, embedding.data(), seq_len,
                                      n_embd);
            break;
          }

          if (req->normalize_embedding) {
            densecore::simd::NormalizeL2(embedding.data(), n_embd);
          }

          // Push embedding result to callback queue (RAII-safe via std::move)
          if (req->embedding_callback) {
            // Move the embedding vector directly - no manual allocation needed
            PushEmbeddingResultEvent(state, req->id, std::move(embedding),
                                     req->embedding_callback, req->user_data);
          }
          req->finished = true;

          // Cleanup and remove from scheduler
          current_kv_cache->block_manager->Free(req->block_table);
          req->block_table.clear();
          // Cleanup using stored seq_id (O(1) instead of O(n))
          if (req->seq_id >= 0) {
            state->scheduler->RemoveRequest(req->seq_id, true);
            seq_to_request.erase(req->seq_id);
          }
          {
            std::lock_guard<std::mutex> lk(req->mu);
            req->cv.notify_all();
          }
        } else {
          // Generation
          if (req->is_prefill) {
            req->n_past += processed_count;
            req->is_prefill = false;
            req->first_token_time = std::chrono::steady_clock::now();
            auto ttft_us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    req->first_token_time - req->start_time)
                    .count();
            state->metrics.RecordTTFT(ttft_us);
            req->last_token_time = req->first_token_time;
          }

          // Sample token
          SamplingParams sampling_params;
          if (req->json_mode) {
            sampling_params.grammar = &req->grammar;
            sampling_params.vocab = &current_model->vocab_tokens;
          }

          int best_token = SampleToken(output, last_token_idx, sampling_params);
          std::cerr << "[TRACE] Sampled token " << best_token << " for request "
                    << req->id << std::endl;

          req->tokens.clear();
          req->tokens.push_back(best_token);
          req->generated_count++;

          std::string token_str =
              Tokenizer::Detokenize(current_model, best_token);

          if (req->json_mode) {
            req->grammar.UpdateState(token_str);
          }

          // =========================================================================
          // TOKEN STREAMING VIA CALLBACK QUEUE (GIL-Free Hot Path)
          // =========================================================================
          // Push token to callback queue instead of invoking callback directly.
          // This prevents the worker thread from blocking on Python GIL.
          // =========================================================================
          if (req->callback) {
            std::cerr << "[TRACE] Pushing result for request " << req->id
                      << std::endl;
            PushResultEvent(state, req->id, token_str, false, false,
                            req->callback, req->user_data);
          }

          state->metrics.total_tokens_generated++;

          // Update scheduler progress
          for (auto &kv : seq_to_request) {
            if (kv.second == req) {
              state->scheduler->UpdateProgress(kv.first, 1);
              break;
            }
          }

          // Update n_past for decode step
          if (!req->is_prefill) {
            req->n_past += 1;
            auto now = std::chrono::steady_clock::now();
            auto itl_us = std::chrono::duration_cast<std::chrono::microseconds>(
                              now - req->last_token_time)
                              .count();
            state->metrics.RecordITL(itl_us);
            req->last_token_time = now;
          }

          // Ensure we have enough blocks for the NEXT token
          int blocks_needed = (req->n_past + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
          while ((int)req->block_table.size() < blocks_needed) {
            auto new_blocks = current_kv_cache->block_manager->Allocate(1);
            if (new_blocks.empty()) {
              std::cerr << "[DenseCore] OOM during generation for req "
                        << req->id << std::endl;
              req->finished = true;
              state->metrics.oom_errors++;
              state->metrics.failed_requests++;
              // Push OOM error to callback queue (instead of direct callback)
              if (req->callback) {
                PushResultEvent(state, req->id, "Error: Out of memory", true,
                                true, req->callback, req->user_data);
              }
              break;
            }
            req->block_table.push_back(new_blocks[0]);
          }

          // Check finish conditions
          if (best_token == current_model->eos_token_id ||
              req->generated_count >= req->max_tokens) {
            req->finished = true;
            state->metrics.completed_requests++;
            // Push finished signal to callback queue (instead of direct
            // callback)
            if (req->callback) {
              PushResultEvent(state, req->id, "", true, false, req->callback,
                              req->user_data);
            }
            {
              std::lock_guard<std::mutex> lk(req->mu);
              req->cv.notify_all();
            }
            // Free blocks and remove from scheduler
            current_kv_cache->block_manager->Free(req->block_table);
            req->block_table.clear();
            // Cleanup using stored seq_id (O(1) instead of O(n))
            if (req->seq_id >= 0) {
              state->scheduler->RemoveRequest(req->seq_id, true);
              seq_to_request.erase(req->seq_id);
            }
          }
        }
        token_offset += processed_count;
      }

      // RAII guard (ctx_temp_guard) automatically frees non-cached contexts

      // 11. Sync active_requests: remove finished requests
      // Also signal queue_cv when requests finish (frees memory for scheduler)
      {
        bool any_finished = false;
        std::lock_guard<std::mutex> lock(state->active_mu);
        auto it = state->active_requests.begin();
        while (it != state->active_requests.end()) {
          Request *req = *it;
          if (req->finished) {
            it = state->active_requests.erase(it);
            state->metrics.active_requests--;
            state->request_pool.Release(req);
            any_finished = true;
          } else {
            ++it;
          }
        }

        // Signal queue_cv when requests finish - wakes scheduler if it's
        // waiting due to memory fragmentation (freed blocks now available)
        if (any_finished) {
          std::lock_guard<std::mutex> cv_lock(state->cv_mu);
          state->queue_cv.notify_one();
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[DenseCore] Worker thread exception: " << e.what()
              << std::endl;
  } catch (...) {
    std::cerr << "[DenseCore] Worker thread unknown exception" << std::endl;
  }
}
