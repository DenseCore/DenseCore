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

#include <algorithm>
#include <chrono>
#include <climits>
#include <deque>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "kv_cache.h"

namespace densecore {

/**
 * Sequence state in the scheduler
 */
enum class SequenceStatus {
    WAITING,    // In queue, not started
    RUNNING,    // Currently being processed
    SWAPPED,    // Preempted and swapped to CPU
    FINISHED,   // Completed generation
    CANCELLED,  // User cancelled
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
    int priority;  // Lower = higher priority
    std::chrono::steady_clock::time_point arrival_time;

    // Prefix sharing
    int shared_prefix_len = 0;
    std::vector<int> shared_block_ids;

    // Computed at each iteration
    int num_running_seqs = 0;
    int num_tokens_to_process = 0;

    // Progress tracking (for smart preemption)
    int generated_tokens = 0;  // Tokens generated so far

    bool IsPrefill() const { return num_tokens_to_process > 1; }
};

/**
 * Scheduler output for one iteration
 */
struct SchedulerOutput {
    // Sequences to process in this iteration
    std::vector<int> prefill_seq_ids;    // Full prefill
    std::vector<int> decode_seq_ids;     // Single token decode
    std::vector<int> preempted_seq_ids;  // Preempted this round

    // Block allocations
    std::vector<std::pair<int, std::vector<int>>> new_block_allocations;  // (seq_id, block_ids)
    std::vector<std::pair<int, std::vector<int>>> freed_blocks;           // (seq_id, block_ids)

    // Batching info
    int total_tokens = 0;
    int num_prefill_tokens = 0;
    int num_decode_tokens = 0;

    bool IsEmpty() const { return prefill_seq_ids.empty() && decode_seq_ids.empty(); }
};

/**
 * Scheduler configuration
 */
struct SchedulerConfig {
    int max_num_seqs = 256;             // Max concurrent sequences
    int max_num_batched_tokens = 2048;  // Max tokens per iteration
    int max_model_len = 4096;           // Max sequence length

    // Priority scheduling
    bool enable_priority = true;
    int priority_preempt_threshold = 10;  // Priority diff to trigger preemption

    // Memory management
    float watermark_high = 0.9f;  // Start preemption
    float watermark_low = 0.8f;   // Stop preemption

    // Chunked prefill
    bool enable_chunked_prefill = true;
    int max_prefill_tokens = 512;  // Max prefill tokens per iteration
};

/**
 * Advanced scheduler for continuous batching
 */
class Scheduler {
public:
    explicit Scheduler(BlockManager* block_manager,
                       const SchedulerConfig& config = SchedulerConfig());

    /**
     * Add a new request to the scheduler
     */
    int AddRequest(int request_id, int prompt_len, int max_output_len, int priority = 100,
                   const std::vector<int>* prefix_tokens = nullptr);

    /**
     * Remove a request (cancel or complete)
     */
    void RemoveRequest(int seq_id, bool finished = true);

    /**
     * Check if scheduler has any pending or active requests
     */
    bool HasRequests() const;

    /**
     * Schedule next iteration
     * Main scheduling algorithm
     */
    SchedulerOutput Schedule();

    /**
     * Fork a sequence for beam search / parallel sampling
     */
    int ForkSequence(int parent_seq_id);

    /**
     * Get sequence status
     */
    SequenceStatus GetStatus(int seq_id) const;

    /**
     * Update sequence progress (call after each token generation)
     */
    void UpdateProgress(int seq_id, int tokens_generated = 1);

    /**
     * Get scheduler stats
     */
    struct Stats {
        int waiting_count;
        int running_count;
        int swapped_count;
        float memory_usage;
    };

    Stats GetStats() const;

private:
    float GetMemoryUsage() const;

    void PreemptSequences(SchedulerOutput& output);

    /**
     * Free sequence resources (blocks and metadata)
     */
    void FreeSequence(int seq_id);

    /**
     * Select the best victim for preemption using smart criteria
     */
    int SelectPreemptionVictim();

    int GetSequencePriority(int seq_id) const;
    int GetSequenceProgress(int seq_id) const;
    std::chrono::steady_clock::time_point GetSequenceArrival(int seq_id) const;

    void ScheduleRunning(SchedulerOutput& output);
    void ScheduleWaiting(SchedulerOutput& output);

    BlockManager* block_manager_;
    SchedulerConfig config_;
    mutable std::mutex mutex_;

    std::deque<SequenceGroup> waiting_queue_;
    std::unordered_set<int> running_seqs_;
    std::unordered_set<int> swapped_seqs_;
    std::unordered_map<int, SequenceStatus> seq_status_;

    int next_seq_id_ = 1;

    // Progress tracking for smart preemption
    std::unordered_map<int, int> seq_generated_tokens_;  // seq_id -> tokens generated
    std::unordered_map<int, int> seq_priority_;          // seq_id -> priority
    std::unordered_map<int, std::chrono::steady_clock::time_point>
        seq_arrival_;  // seq_id -> arrival time
};

// Configuration factories
SchedulerConfig CreateInteractiveConfig();
SchedulerConfig CreateThroughputConfig();
SchedulerConfig CreateMemoryEfficientConfig();

// Helper functions (exposed for testing/utils)
bool ScheduleStep(Scheduler& scheduler, SchedulerOutput& output);
void PrintSchedulerDebugInfo(const Scheduler& scheduler);
const char* SequenceStatusToString(SequenceStatus status);
void PrintSchedulerOutput(const SchedulerOutput& output);

}  // namespace densecore

#endif  // DENSECORE_SCHEDULER_H
