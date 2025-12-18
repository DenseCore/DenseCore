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

#ifdef _OPENMP
#include <omp.h>
#endif

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
    // CRITICAL: Disable OpenMP nested parallelism
    // =========================================================================
    // GGML spawns its own thread pool (n_threads workers). If any function
    // called by GGML workers uses #pragma omp parallel (e.g. FlashAttention),
    // OpenMP would spawn ANOTHER thread team per GGML worker, causing:
    //   Total threads = n_threads × OMP_NUM_THREADS (thread explosion)
    // Setting omp_set_num_threads(1) forces OpenMP to run sequentially,
    // allowing GGML's thread pool to be the sole source of parallelism.
    // =========================================================================
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif

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
    int n_threads = state->n_threads;
    if (n_threads <= 0) {
      n_threads = densecore::simd::GetNumCores();
    }
    ggml_backend_cpu_set_n_threads(current_model->backend, n_threads);
    std::cout << "[DenseCore] GGML backend configured: " << n_threads
              << " threads" << std::endl;

    // NUMA-aware thread pinning using HardwareTopology
    {
      auto &topo = densecore::HardwareTopology::GetInstance();
      int target_node = (state->numa_node_id >= 0) ? state->numa_node_id : 0;

      if (topo.GetNumaNodeCount() > 1) {
        // Multi-socket: pin worker to physical cores on target NUMA node
        bool pinned = topo.PinCurrentThreadToNumaNode(
            target_node, densecore::PinningPolicy::SCATTER);
        if (pinned) {
          std::cout << "[DenseCore] Worker thread pinned to NUMA node "
                    << target_node << " (" << topo.GetNumaNodeCount()
                    << " nodes detected, SCATTER policy)" << std::endl;
        } else {
          std::cerr << "[DenseCore] Warning: Thread pinning to NUMA node "
                    << target_node << " failed" << std::endl;
        }
      } else {
        // Single NUMA node: pin to first physical core for cache locality
        auto physical_cores = topo.GetPhysicalCoreIds(-1);
        if (!physical_cores.empty()) {
          densecore::HardwareTopology::PinCurrentThread(physical_cores[0]);
          std::cout << "[DenseCore] Worker pinned to core " << physical_cores[0]
                    << " (single NUMA node)" << std::endl;
        }
      }
    }

    // Setup compute thread affinity mapping for GGML workers
    // This pre-computes core assignments so GGML threads can self-pin on
    // first use
    {
      auto &topo = densecore::HardwareTopology::GetInstance();
      int target_node = (state->numa_node_id >= 0) ? state->numa_node_id : 0;
      // Use configured thread count from engine initialization
      if (n_threads > 0) {
        // Use configured pinning policy (0=SCATTER, 1=COMPACT)
        densecore::PinningPolicy policy =
            (state->pinning_policy == 1) ? densecore::PinningPolicy::COMPACT
                                         : densecore::PinningPolicy::SCATTER;
        topo.SetupComputeThreadAffinity(target_node, n_threads, policy);
      }
    }

    // Initialize persistent compute buffer (eliminates malloc/free per
    // iteration)
    state->InitComputeBuffer();
    state->InitGraphCache();

    // Mapping: scheduler seq_id -> Request*
    std::unordered_map<int, Request *> seq_to_request;

    while (state->status != EngineStatus::STOPPED) {
      // 1. Fetch new requests and register with scheduler
      {
        std::unique_lock<std::mutex> lock(state->queue_mu);

        // Check exit conditions
        if (state->active_requests.empty() && state->pending_requests.empty()) {
          if (state->status == EngineStatus::DRAINING) {
            break; // Exit: draining complete, no more work
          }
          state->queue_cv.wait_for(lock, std::chrono::milliseconds(50));
          continue;
        }

        // Move pending to active and register with scheduler
        // Skip if DRAINING - don't accept new requests, only drain active
        while (state->status == EngineStatus::RUNNING &&
               !state->pending_requests.empty()) {
          Request *req = state->pending_requests.top();
          state->pending_requests.pop();

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
            if (req->callback)
              req->callback("Error: Missing tokens", 1, req->user_data);
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
            if (req->callback)
              req->callback("Error: Scheduler queue full", 1, req->user_data);
            state->request_pool.Release(req);
            continue;
          }

          // Store mapping and add to active list
          seq_to_request[seq_id] = req;
          state->active_requests.push_back(req);
          state->metrics.active_requests++;
          state->metrics.total_requests++;
        }
      }

      if (state->active_requests.empty())
        continue;

      // Check if we should exit (DRAINING and no more active work)
      if (state->status == EngineStatus::DRAINING &&
          state->active_requests.empty()) {
        break;
      }

      // 2. Process cancelled/finished requests first
      {
        std::lock_guard<std::mutex> lock(state->queue_mu);
        auto it = state->active_requests.begin();
        while (it != state->active_requests.end()) {
          Request *req = *it;
          if (req->cancelled && !req->finished) {
            LOG_INFO("Cancelling active request: ", req->id);
            req->finished = true;
            state->metrics.failed_requests++;
            if (req->callback) {
              req->callback("Error: Request cancelled", 1, req->user_data);
            }
            // Remove from scheduler
            for (auto &kv : seq_to_request) {
              if (kv.second == req) {
                state->scheduler->RemoveRequest(kv.first, false);
                seq_to_request.erase(kv.first);
                break;
              }
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
      densecore::SchedulerOutput sched_output = state->scheduler->Schedule();

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

      // 7. If scheduler returned empty batch, continue waiting
      if (sched_output.IsEmpty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

      if (batch_requests.empty())
        continue;

      batch.num_seqs = batch_requests.size();

      // 9. Run Inference (using persistent compute buffer for efficiency)
      // The compute buffer is reused across iterations to avoid malloc
      // overhead Graph Caching Logic
      struct ggml_cgraph *gf = nullptr;
      struct ggml_tensor *output = nullptr;
      struct ggml_tensor *embd_inp = nullptr;
      struct ggml_tensor *pos = nullptr;

      // Determine if cacheable: only strictly consistent batches (e.g.
      // Decode with 1 token/seq) For MVP we cache by num_seqs. We verify if
      // input size matches expected 1 token/seq for decode.
      bool cacheable = (!is_embedding_batch &&
                        batch.tokens.size() == (size_t)batch.num_seqs);

      if (cacheable) {
        std::lock_guard<std::mutex> lock(state->graph_mu);
        if (state->graph_cache.count(batch.num_seqs)) {
          auto &meta = state->graph_cache[batch.num_seqs];
          gf = meta.gf;
          embd_inp = meta.embd_inp;
          pos = meta.pos;
          output = meta.output;
        }
      }

      // RAII guard for temporary context - declared here so it lives
      // through compute For cached graphs: this stays nullptr (no-op) For
      // non-cached graphs: holds the temp context until end of loop
      // iteration
      densecore::GGMLContextGuard ctx_temp_guard(nullptr);

      if (!gf) {
        // Build new graph
        // For cached graphs: use persistent state->graph_ctx
        // For non-cached graphs: allocate temporary context with RAII guard
        struct ggml_context *ctx_nodes = state->graph_ctx;

        if (!cacheable) {
          // Temp context for one-off graphs (Prefill etc)
          struct ggml_init_params params = {
              .mem_size = 16 * 1024 * 1024,
              .mem_buffer = nullptr,
              .no_alloc = false,
          };
          ctx_temp_guard = densecore::GGMLContextGuard(ggml_init(params));
          ctx_nodes = ctx_temp_guard.get();
        }

        gf = ggml_new_graph_custom(ctx_nodes, 32768, false);

        // Build nodes in ctx_nodes
        output = BuildTransformerGraph(current_model, current_kv_cache,
                                       ctx_nodes, batch, is_embedding_batch, gf,
                                       &embd_inp, &pos);

        if (cacheable) {
          std::lock_guard<std::mutex> lock(state->graph_mu);
          state->graph_cache[batch.num_seqs] = {gf, embd_inp, pos, output};
        }
      }
      // ctx_temp_guard destructor will call ggml_free() at end of loop
      // iteration

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

      ggml_backend_graph_compute(current_model->backend, gf);

      // 10. Process Outputs
      int token_offset = 0;
      int n_embd = current_model->hparams.n_embd;

      for (int i = 0; i < batch.num_seqs; ++i) {
        Request *req = batch_requests[i];
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

          if (req->embedding_callback) {
            req->embedding_callback(embedding.data(), n_embd, req->user_data);
          }
          req->finished = true;

          // Cleanup and remove from scheduler
          current_kv_cache->block_manager->Free(req->block_table);
          req->block_table.clear();
          for (auto &kv : seq_to_request) {
            if (kv.second == req) {
              state->scheduler->RemoveRequest(kv.first, true);
              seq_to_request.erase(kv.first);
              break;
            }
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

          req->tokens.clear();
          req->tokens.push_back(best_token);
          req->generated_count++;

          // Decode token for callback (using C++ tokenizer to avoid Python
          // thread safety issues)
          std::string token_str =
              Tokenizer::Detokenize(current_model, best_token);

          if (req->json_mode) {
            req->grammar.UpdateState(token_str);
          }

          if (req->callback) {
            req->callback(token_str.c_str(), 0, req->user_data);
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
              if (req->callback)
                req->callback("Error: Out of memory", 1, req->user_data);
              break;
            }
            req->block_table.push_back(new_blocks[0]);
          }

          // Check finish conditions
          if (best_token == current_model->eos_token_id ||
              req->generated_count >= req->max_tokens) {
            req->finished = true;
            state->metrics.completed_requests++;
            if (req->callback)
              req->callback("", 1, req->user_data);
            {
              std::lock_guard<std::mutex> lk(req->mu);
              req->cv.notify_all();
            }
            // Free blocks and remove from scheduler
            current_kv_cache->block_manager->Free(req->block_table);
            req->block_table.clear();
            for (auto &kv : seq_to_request) {
              if (kv.second == req) {
                state->scheduler->RemoveRequest(kv.first, true);
                seq_to_request.erase(kv.first);
                break;
              }
            }
          }
        }

        token_offset += processed_count;
      }

      // RAII guard (ctx_temp_guard) automatically frees non-cached contexts

      // 11. Sync active_requests: remove finished requests
      {
        std::lock_guard<std::mutex> lock(state->queue_mu);
        auto it = state->active_requests.begin();
        while (it != state->active_requests.end()) {
          Request *req = *it;
          if (req->finished) {
            it = state->active_requests.erase(it);
            state->metrics.active_requests--;
            state->request_pool.Release(req);
          } else {
            ++it;
          }
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
