/**
 * @file scheduler.cpp
 * @brief Scheduler implementation for continuous batching inference
 *
 * Implements the Scheduler class defined in scheduler.h.
 * Responsible for managing request queues, scheduling prefill/decode batches,
 * and handling preemption based on memory pressure.
 */

#include "scheduler.h"
#include <algorithm>
#include <iostream>

namespace densecore {

// ============================================================================
// Scheduler Implementation
// ============================================================================

Scheduler::Scheduler(BlockManager *block_manager, const SchedulerConfig &config)
    : block_manager_(block_manager), config_(config) {}

int Scheduler::AddRequest(int request_id, int prompt_len, int max_output_len,
                          int priority, const std::vector<int> *prefix_tokens) {
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

void Scheduler::RemoveRequest(int seq_id, bool finished) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Update status (for tracking)
  if (seq_status_.find(seq_id) != seq_status_.end()) {
    seq_status_[seq_id] =
        finished ? SequenceStatus::FINISHED : SequenceStatus::CANCELLED;
  }

  // Free resources
  FreeSequence(seq_id);

  // Remove from waiting queue
  waiting_queue_.erase(
      std::remove_if(waiting_queue_.begin(), waiting_queue_.end(),
                     [seq_id](const SequenceGroup &g) {
                       return std::find(g.sequence_ids.begin(),
                                        g.sequence_ids.end(),
                                        seq_id) != g.sequence_ids.end();
                     }),
      waiting_queue_.end());

  // Remove from running/swapped sets
  running_seqs_.erase(seq_id);
  swapped_seqs_.erase(seq_id);

  // Clean up tracking maps
  seq_priority_.erase(seq_id);
  seq_arrival_.erase(seq_id);
  seq_generated_tokens_.erase(seq_id);
}

bool Scheduler::HasRequests() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !waiting_queue_.empty() || !running_seqs_.empty();
}

void Scheduler::FreeSequence(int seq_id) {
  // Placeholder for releasing sequence-specific resources
  // In a full implementation, this might release pre-allocated blocks
  // if not managed by BlockManager globally.
  // Since BlockManager handles global blocks, we just ensure
  // internal bookkeeping is clean.
  (void)seq_id;
}

SchedulerOutput Scheduler::Schedule() {
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

  // std::cerr << "[DEBUG] Scheduler::Schedule exit. Empty? " <<
  // output.IsEmpty() << std::endl;
  return output;
}

int Scheduler::ForkSequence(int parent_seq_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  int new_seq_id = next_seq_id_++;
  seq_status_[new_seq_id] = seq_status_[parent_seq_id];

  // Fork is handled at block level through BlockManager::Fork
  // The caller needs to fork the block table entries

  return new_seq_id;
}

SequenceStatus Scheduler::GetStatus(int seq_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = seq_status_.find(seq_id);
  if (it == seq_status_.end())
    return SequenceStatus::FINISHED;
  return it->second;
}

void Scheduler::UpdateProgress(int seq_id, int tokens_generated) {
  std::lock_guard<std::mutex> lock(mutex_);
  seq_generated_tokens_[seq_id] += tokens_generated;
}

Scheduler::Stats Scheduler::GetStats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  Stats stats;
  stats.waiting_count = waiting_queue_.size();
  stats.running_count = running_seqs_.size();
  stats.swapped_count = swapped_seqs_.size();
  stats.memory_usage = GetMemoryUsage();
  return stats;
}

float Scheduler::GetMemoryUsage() const {
  if (block_manager_->num_blocks == 0)
    return 0.0f;
  int used = block_manager_->GetUsedBlockCount();
  int total = block_manager_->num_blocks;
  return (float)used / total;
}

void Scheduler::PreemptSequences(SchedulerOutput &output) {
  while (GetMemoryUsage() > config_.watermark_low && !running_seqs_.empty()) {
    int victim = SelectPreemptionVictim();
    if (victim < 0)
      break;

    running_seqs_.erase(victim);
    swapped_seqs_.insert(victim);
    seq_status_[victim] = SequenceStatus::SWAPPED;
    output.preempted_seq_ids.push_back(victim);
  }
}

int Scheduler::SelectPreemptionVictim() {
  if (running_seqs_.empty())
    return -1;

  int best_victim = -1;
  int best_priority = -1;
  int best_progress = INT_MAX;
  std::chrono::steady_clock::time_point best_arrival;

  for (int seq_id : running_seqs_) {
    int priority = GetSequencePriority(seq_id);
    int progress = GetSequenceProgress(seq_id);
    auto arrival = GetSequenceArrival(seq_id);

    bool is_better_victim = false;

    if (config_.enable_priority) {
      if (priority > best_priority) {
        is_better_victim = true;
      } else if (priority == best_priority) {
        if (progress < best_progress) {
          is_better_victim = true;
        } else if (progress == best_progress) {
          if (best_victim < 0 || arrival > best_arrival) {
            is_better_victim = true;
          }
        }
      }
    } else {
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

int Scheduler::GetSequencePriority(int seq_id) const {
  auto it = seq_priority_.find(seq_id);
  return (it != seq_priority_.end()) ? it->second : 100;
}

int Scheduler::GetSequenceProgress(int seq_id) const {
  auto it = seq_generated_tokens_.find(seq_id);
  return (it != seq_generated_tokens_.end()) ? it->second : 0;
}

std::chrono::steady_clock::time_point
Scheduler::GetSequenceArrival(int seq_id) const {
  auto it = seq_arrival_.find(seq_id);
  return (it != seq_arrival_.end()) ? it->second
                                    : std::chrono::steady_clock::now();
}

void Scheduler::ScheduleRunning(SchedulerOutput &output) {
  int tokens_budget =
      config_.max_num_batched_tokens - output.num_prefill_tokens;

  for (int seq_id : running_seqs_) {
    if (output.decode_seq_ids.size() >= (size_t)config_.max_num_seqs) {
      break;
    }

    output.decode_seq_ids.push_back(seq_id);
    output.num_decode_tokens++;
    output.total_tokens++;
    tokens_budget--;

    if (tokens_budget <= 0)
      break;
  }
}

void Scheduler::ScheduleWaiting(SchedulerOutput &output) {
  int tokens_budget = config_.max_num_batched_tokens - output.total_tokens;

  if (config_.enable_chunked_prefill) {
    tokens_budget = std::min(tokens_budget, config_.max_prefill_tokens);
  }

  std::vector<SequenceGroup> sorted_queue(waiting_queue_.begin(),
                                          waiting_queue_.end());
  std::sort(sorted_queue.begin(), sorted_queue.end(),
            [](const SequenceGroup &a, const SequenceGroup &b) {
              if (a.priority != b.priority)
                return a.priority < b.priority;
              return a.arrival_time < b.arrival_time;
            });

  std::deque<SequenceGroup> still_waiting;

  for (auto &group : sorted_queue) {
    if (output.prefill_seq_ids.size() >= (size_t)config_.max_num_seqs) {
      still_waiting.push_back(group);
      continue;
    }

    int tokens_needed = group.num_tokens_to_process;

    if (tokens_needed > tokens_budget) {
      if (config_.enable_chunked_prefill && tokens_budget > 0) {
        group.num_tokens_to_process -= tokens_budget;
        tokens_needed = tokens_budget;
      } else {
        still_waiting.push_back(group);
        continue;
      }
    }

    int blocks_needed = (tokens_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int seq_id = group.sequence_ids[0];
    if (block_manager_->GetFreeBlockCount() < blocks_needed) {
      std::cerr << "[TRACE] Scheduler: Not enough blocks for seq " << seq_id
                << " (needed " << blocks_needed << ", free "
                << block_manager_->GetFreeBlockCount() << ")" << std::endl;
      still_waiting.push_back(group);
      continue;
    }

    std::vector<int> new_blocks = block_manager_->Allocate(blocks_needed);
    if (new_blocks.empty()) {
      std::cerr << "[TRACE] Scheduler: Allocate failed for seq " << seq_id
                << std::endl;
      still_waiting.push_back(group);
      continue;
    }

    std::cerr << "[TRACE] Scheduler: Scheduled seq " << seq_id << " (prefill)"
              << std::endl;
    output.prefill_seq_ids.push_back(seq_id);
    output.num_prefill_tokens += tokens_needed;
    output.total_tokens += tokens_needed;
    output.new_block_allocations.push_back({seq_id, new_blocks});

    running_seqs_.insert(seq_id);
    seq_status_[seq_id] = SequenceStatus::RUNNING;

    seq_generated_tokens_[seq_id] = 0;
    seq_priority_[seq_id] = group.priority;
    seq_arrival_[seq_id] = group.arrival_time;

    tokens_budget -= tokens_needed;
  }

  waiting_queue_ = std::move(still_waiting);
}

// ============================================================================
// Factory Function for Default Scheduler Configuration
// ============================================================================

SchedulerConfig CreateInteractiveConfig() {
  SchedulerConfig config;
  config.max_num_seqs = 4;
  config.max_num_batched_tokens = 512;
  config.max_model_len = 4096;
  config.enable_priority = false;
  config.enable_chunked_prefill = false;
  return config;
}

SchedulerConfig CreateThroughputConfig() {
  SchedulerConfig config;
  config.max_num_seqs = 256;
  config.max_num_batched_tokens = 4096;
  config.max_model_len = 8192;
  config.enable_priority = true;
  config.priority_preempt_threshold = 5;
  config.enable_chunked_prefill = true;
  config.max_prefill_tokens = 1024;
  return config;
}

SchedulerConfig CreateMemoryEfficientConfig() {
  SchedulerConfig config;
  config.max_num_seqs = 8;
  config.max_num_batched_tokens = 256;
  config.max_model_len = 2048;
  config.enable_priority = false;
  config.watermark_high = 0.7f;
  config.watermark_low = 0.5f;
  config.enable_chunked_prefill = true;
  config.max_prefill_tokens = 128;
  return config;
}

// ============================================================================
// Helper Functions
// ============================================================================

bool ScheduleStep(Scheduler &scheduler, SchedulerOutput &output) {
  output = scheduler.Schedule();
  return !output.IsEmpty();
}

void PrintSchedulerDebugInfo(const Scheduler &scheduler) {
  auto stats = scheduler.GetStats();
  std::cout << "[Scheduler] Waiting: " << stats.waiting_count
            << " Running: " << stats.running_count
            << " Swapped: " << stats.swapped_count
            << " Mem: " << stats.memory_usage << std::endl;
}

const char *SequenceStatusToString(SequenceStatus status) {
  switch (status) {
  case SequenceStatus::WAITING:
    return "WAITING";
  case SequenceStatus::RUNNING:
    return "RUNNING";
  case SequenceStatus::SWAPPED:
    return "SWAPPED";
  case SequenceStatus::FINISHED:
    return "FINISHED";
  case SequenceStatus::CANCELLED:
    return "CANCELLED";
  default:
    return "UNKNOWN";
  }
}

void PrintSchedulerOutput(const SchedulerOutput &output) {
  std::cout << "[SchedulerOutput] Total: " << output.total_tokens
            << " Prefill: " << output.num_prefill_tokens
            << " Decode: " << output.num_decode_tokens << std::endl;
}

} // namespace densecore
