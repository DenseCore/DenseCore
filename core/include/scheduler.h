/**
 * @file scheduler.h
 * @brief Advanced scheduler for continuous batching
 *
 * Implements vLLM-style iteration-level scheduling:
 * - Separate prefill and decode batches
 * - Preemption support for priority requests
 * - Memory-aware scheduling
 */

#ifndef DENSECORE_SCHEDULER_H
#define DENSECORE_SCHEDULER_H

#include "kv_cache.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <deque>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace densecore {

/**
 * Sequence state in the scheduler
 */
enum class SequenceStatus {
  WAITING,   // In queue, not started
  RUNNING,   // Currently being processed
  SWAPPED,   // Preempted and swapped to CPU
  FINISHED,  // Completed generation
  CANCELLED, // User cancelled
};

/**
 * Sequence group - represents one request that may have multiple sequences
 * (e.g., beam search, parallel sampling)
 */
struct SequenceGroup {
  int request_id;
  std::vector<int> sequence_ids;
  int prompt_len;
  int max_output_len;
  int priority; // Lower = higher priority
  std::chrono::steady_clock::time_point arrival_time;

  // Prefix sharing
  int shared_prefix_len = 0;
  std::vector<int> shared_block_ids;

  // Computed at each iteration
  int num_running_seqs = 0;
  int num_tokens_to_process = 0;

  // Progress tracking (for smart preemption)
  int generated_tokens = 0; // Tokens generated so far

  bool IsPrefill() const { return num_tokens_to_process > 1; }
};

/**
 * Scheduler output for one iteration
 */
struct SchedulerOutput {
  // Sequences to process in this iteration
  std::vector<int> prefill_seq_ids;   // Full prefill
  std::vector<int> decode_seq_ids;    // Single token decode
  std::vector<int> preempted_seq_ids; // Preempted this round

  // Block allocations
  std::vector<std::pair<int, std::vector<int>>>
      new_block_allocations; // (seq_id, block_ids)
  std::vector<std::pair<int, std::vector<int>>>
      freed_blocks; // (seq_id, block_ids)

  // Batching info
  int total_tokens = 0;
  int num_prefill_tokens = 0;
  int num_decode_tokens = 0;

  bool IsEmpty() const {
    return prefill_seq_ids.empty() && decode_seq_ids.empty();
  }
};

/**
 * Scheduler configuration
 */
struct SchedulerConfig {
  int max_num_seqs = 256;            // Max concurrent sequences
  int max_num_batched_tokens = 2048; // Max tokens per iteration
  int max_model_len = 4096;          // Max sequence length

  // Priority scheduling
  bool enable_priority = true;
  int priority_preempt_threshold = 10; // Priority diff to trigger preemption

  // Memory management
  float watermark_high = 0.9f; // Start preemption
  float watermark_low = 0.8f;  // Stop preemption

  // Chunked prefill
  bool enable_chunked_prefill = true;
  int max_prefill_tokens = 512; // Max prefill tokens per iteration
};

/**
 * Advanced scheduler for continuous batching
 */
class Scheduler {
public:
  explicit Scheduler(BlockManager *block_manager,
                     const SchedulerConfig &config = SchedulerConfig())
      : block_manager_(block_manager), config_(config) {}

  /**
   * Add a new request to the scheduler
   */
  int AddRequest(int request_id, int prompt_len, int max_output_len,
                 int priority = 100,
                 const std::vector<int> *prefix_tokens = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);

    SequenceGroup group;
    group.request_id = request_id;
    group.sequence_ids = {next_seq_id_++};
    group.prompt_len = prompt_len;
    group.max_output_len = max_output_len;
    group.priority = priority;
    group.arrival_time = std::chrono::steady_clock::now();
    group.num_tokens_to_process = prompt_len;

    // Check for prefix cache hit
    if (prefix_tokens && !prefix_tokens->empty()) {
      uint64_t hash =
          BlockManager::ComputeTokenHash(prefix_tokens->data(), prompt_len);
      int cached_block = block_manager_->FindCachedBlock(hash);
      if (cached_block >= 0) {
        // Found cached prefix
        group.shared_prefix_len = prompt_len;
        group.shared_block_ids.push_back(cached_block);
      }
    }

    waiting_queue_.push_back(group);
    seq_status_[group.sequence_ids[0]] = SequenceStatus::WAITING;

    // Track for smart preemption
    int seq_id = group.sequence_ids[0];
    seq_priority_[seq_id] = priority;
    seq_arrival_[seq_id] = group.arrival_time;
    seq_generated_tokens_[seq_id] = 0;

    return seq_id;
  }

  /**
   * Remove a request (cancel or complete)
   */
  void RemoveRequest(int seq_id, bool finished = true) {
    std::lock_guard<std::mutex> lock(mutex_);

    seq_status_[seq_id] =
        finished ? SequenceStatus::FINISHED : SequenceStatus::CANCELLED;

    // Remove from waiting queue
    waiting_queue_.erase(
        std::remove_if(waiting_queue_.begin(), waiting_queue_.end(),
                       [seq_id](const SequenceGroup &g) {
                         return std::find(g.sequence_ids.begin(),
                                          g.sequence_ids.end(),
                                          seq_id) != g.sequence_ids.end();
                       }),
        waiting_queue_.end());

    // Remove from running set
    running_seqs_.erase(seq_id);
  }

  /**
   * Schedule next iteration
   * Main scheduling algorithm
   */
  SchedulerOutput Schedule() {
    std::lock_guard<std::mutex> lock(mutex_);

    SchedulerOutput output;

    // 1. Check memory pressure
    float memory_usage = GetMemoryUsage();
    if (memory_usage > config_.watermark_high) {
      // Need to preempt some sequences
      PreemptSequences(output);
    }

    // 2. Schedule running sequences (decode)
    ScheduleRunning(output);

    // 3. Schedule waiting sequences (prefill) if we have capacity
    ScheduleWaiting(output);

    return output;
  }

  /**
   * Fork a sequence for beam search / parallel sampling
   */
  int ForkSequence(int parent_seq_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    int new_seq_id = next_seq_id_++;
    seq_status_[new_seq_id] = seq_status_[parent_seq_id];

    // Fork is handled at block level through BlockManager::Fork
    // The caller needs to fork the block table entries

    return new_seq_id;
  }

  /**
   * Get sequence status
   */
  SequenceStatus GetStatus(int seq_id) const {
    auto it = seq_status_.find(seq_id);
    if (it == seq_status_.end())
      return SequenceStatus::FINISHED;
    return it->second;
  }

  /**
   * Get scheduler stats
   */
  struct Stats {
    int waiting_count;
    int running_count;
    int swapped_count;
    float memory_usage;
  };

  Stats GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats stats;
    stats.waiting_count = waiting_queue_.size();
    stats.running_count = running_seqs_.size();
    stats.swapped_count = swapped_seqs_.size();
    stats.memory_usage = GetMemoryUsage();
    return stats;
  }

private:
  float GetMemoryUsage() const {
    int used = block_manager_->GetUsedBlockCount();
    int total = block_manager_->num_blocks;
    return (float)used / total;
  }

  void PreemptSequences(SchedulerOutput &output) {
    // Smart preemption: Select victim based on priority and progress
    // Goal: Minimize wasted computation by preempting sequences that:
    //   1. Have lowest priority (if priority-based scheduling enabled)
    //   2. Have made least progress (fewer generated tokens = less waste)

    while (GetMemoryUsage() > config_.watermark_low && !running_seqs_.empty()) {
      int victim = SelectPreemptionVictim();
      if (victim < 0)
        break; // No valid victim found

      running_seqs_.erase(victim);
      swapped_seqs_.insert(victim);
      seq_status_[victim] = SequenceStatus::SWAPPED;
      output.preempted_seq_ids.push_back(victim);
    }
  }

  /**
   * Select the best victim for preemption using smart criteria:
   * 1. Lowest priority (if enable_priority is true)
   * 2. Least progress (fewest generated tokens - minimizes wasted work)
   * 3. Most recent arrival (tie-breaker)
   */
  int SelectPreemptionVictim() {
    if (running_seqs_.empty())
      return -1;

    int best_victim = -1;
    int best_priority = -1; // Higher value = lower priority = better victim
    int best_progress = INT_MAX; // Fewer tokens = better victim
    std::chrono::steady_clock::time_point best_arrival;

    for (int seq_id : running_seqs_) {
      // Get sequence info from our tracking
      int priority = GetSequencePriority(seq_id);
      int progress = GetSequenceProgress(seq_id);
      auto arrival = GetSequenceArrival(seq_id);

      bool is_better_victim = false;

      if (config_.enable_priority) {
        // Priority-based selection (higher priority value = lower priority =
        // better victim)
        if (priority > best_priority) {
          is_better_victim = true;
        } else if (priority == best_priority) {
          // Same priority: prefer less progress (waste less computation)
          if (progress < best_progress) {
            is_better_victim = true;
          } else if (progress == best_progress) {
            // Same progress: prefer most recent (LIFO for fairness)
            if (best_victim < 0 || arrival > best_arrival) {
              is_better_victim = true;
            }
          }
        }
      } else {
        // No priority: purely progress-based
        if (progress < best_progress) {
          is_better_victim = true;
        } else if (progress == best_progress) {
          if (best_victim < 0 || arrival > best_arrival) {
            is_better_victim = true;
          }
        }
      }

      if (is_better_victim) {
        best_victim = seq_id;
        best_priority = priority;
        best_progress = progress;
        best_arrival = arrival;
      }
    }

    return best_victim;
  }

  int GetSequencePriority(int seq_id) const {
    auto it = seq_priority_.find(seq_id);
    return (it != seq_priority_.end()) ? it->second : 100; // Default priority
  }

  int GetSequenceProgress(int seq_id) const {
    auto it = seq_generated_tokens_.find(seq_id);
    return (it != seq_generated_tokens_.end()) ? it->second : 0;
  }

  std::chrono::steady_clock::time_point GetSequenceArrival(int seq_id) const {
    auto it = seq_arrival_.find(seq_id);
    return (it != seq_arrival_.end()) ? it->second
                                      : std::chrono::steady_clock::now();
  }

public:
  /**
   * Update sequence progress (call after each token generation)
   */
  void UpdateProgress(int seq_id, int tokens_generated = 1) {
    std::lock_guard<std::mutex> lock(mutex_);
    seq_generated_tokens_[seq_id] += tokens_generated;
  }

private:
  void ScheduleRunning(SchedulerOutput &output) {
    int tokens_budget =
        config_.max_num_batched_tokens - output.num_prefill_tokens;

    for (int seq_id : running_seqs_) {
      if (output.decode_seq_ids.size() >= (size_t)config_.max_num_seqs) {
        break;
      }

      // Check if we can allocate new block if needed
      // (simplified - real implementation would check seq state)

      output.decode_seq_ids.push_back(seq_id);
      output.num_decode_tokens++;
      output.total_tokens++;
      tokens_budget--;

      if (tokens_budget <= 0)
        break;
    }
  }

  void ScheduleWaiting(SchedulerOutput &output) {
    int tokens_budget = config_.max_num_batched_tokens - output.total_tokens;

    if (config_.enable_chunked_prefill) {
      tokens_budget = std::min(tokens_budget, config_.max_prefill_tokens);
    }

    // Sort waiting queue by priority
    std::sort(waiting_queue_.begin(), waiting_queue_.end(),
              [](const SequenceGroup &a, const SequenceGroup &b) {
                if (a.priority != b.priority)
                  return a.priority < b.priority; // Lower = higher priority
                return a.arrival_time < b.arrival_time; // FCFS within priority
              });

    std::deque<SequenceGroup> still_waiting;

    for (auto &group : waiting_queue_) {
      if (output.prefill_seq_ids.size() >= (size_t)config_.max_num_seqs) {
        still_waiting.push_back(group);
        continue;
      }

      int tokens_needed = group.num_tokens_to_process;

      // Check if we have enough tokens budget
      if (tokens_needed > tokens_budget) {
        if (config_.enable_chunked_prefill && tokens_budget > 0) {
          // Partial prefill
          group.num_tokens_to_process -= tokens_budget;
          tokens_needed = tokens_budget;
        } else {
          still_waiting.push_back(group);
          continue;
        }
      }

      // Try to allocate blocks
      int blocks_needed = (tokens_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
      if (block_manager_->GetFreeBlockCount() < blocks_needed) {
        still_waiting.push_back(group);
        continue;
      }

      // Allocate blocks
      std::vector<int> new_blocks = block_manager_->Allocate(blocks_needed);
      if (new_blocks.empty()) {
        still_waiting.push_back(group);
        continue;
      }

      // Schedule this sequence
      int seq_id = group.sequence_ids[0];
      output.prefill_seq_ids.push_back(seq_id);
      output.num_prefill_tokens += tokens_needed;
      output.total_tokens += tokens_needed;
      output.new_block_allocations.push_back({seq_id, new_blocks});

      running_seqs_.insert(seq_id);
      seq_status_[seq_id] = SequenceStatus::RUNNING;

      tokens_budget -= tokens_needed;
    }

    waiting_queue_ = std::move(still_waiting);
  }

  BlockManager *block_manager_;
  SchedulerConfig config_;
  mutable std::mutex mutex_;

  std::deque<SequenceGroup> waiting_queue_;
  std::unordered_set<int> running_seqs_;
  std::unordered_set<int> swapped_seqs_;
  std::unordered_map<int, SequenceStatus> seq_status_;

  int next_seq_id_ = 1;

  // Progress tracking for smart preemption
  std::unordered_map<int, int>
      seq_generated_tokens_;                  // seq_id -> tokens generated
  std::unordered_map<int, int> seq_priority_; // seq_id -> priority
  std::unordered_map<int, std::chrono::steady_clock::time_point>
      seq_arrival_; // seq_id -> arrival time
};

} // namespace densecore

#endif // DENSECORE_SCHEDULER_H
