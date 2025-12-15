#ifndef DENSECORE_KV_CACHE_H
#define DENSECORE_KV_CACHE_H

#include "model_types.h"
#include <unordered_map>

// ============================================================================
// vLLM-style PagedAttention v2 for CPU
// ============================================================================

// Block size: number of tokens per block
// vLLM uses 16 as default, good for CPU cache line alignment
constexpr int BLOCK_SIZE = 16;

// ============================================================================
// Physical Block with Reference Counting (Copy-on-Write Support)
// ============================================================================
struct PhysicalBlock {
  int id = -1;
  int ref_count = 0;         // Reference count for CoW
  int num_filled_slots = 0;  // Tokens currently stored (0 ~ BLOCK_SIZE)
  uint64_t content_hash = 0; // Hash for prefix caching
  bool is_full() const { return num_filled_slots >= BLOCK_SIZE; }
};

// ============================================================================
// Block Table: Logical to Physical Block Mapping for a Sequence
// ============================================================================
struct SequenceBlockTable {
  int seq_id = -1;
  std::vector<int> block_ids; // Logical block index -> Physical block ID
  int num_tokens = 0;         // Total tokens in this sequence

  // Get the physical block ID for a given token position
  int GetPhysicalBlockId(int token_pos) const {
    int logical_block = token_pos / BLOCK_SIZE;
    if (logical_block < 0 || logical_block >= (int)block_ids.size())
      return -1;
    return block_ids[logical_block];
  }

  // Get the slot index within a block for a given token position
  int GetSlotIndex(int token_pos) const { return token_pos % BLOCK_SIZE; }

  // Get the number of blocks allocated
  int GetNumBlocks() const { return block_ids.size(); }

  // Calculate blocks needed for n tokens
  static int BlocksNeeded(int n_tokens) {
    return (n_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
  }
};

// ============================================================================
// Block Manager with Copy-on-Write and Prefix Caching
// ============================================================================
struct BlockManager {
  int num_blocks;
  int block_size;

  // Physical block metadata
  std::vector<PhysicalBlock> blocks;

  // Free block stack (LIFO for cache locality)
  std::vector<int> free_blocks;

  // Prefix caching: hash -> physical block id
  std::unordered_map<uint64_t, int> prefix_cache;

  // Thread safety
  std::mutex mu;

  // Constructor
  BlockManager(int num_blocks, int block_size);
  ~BlockManager() = default;

  // -------------------------------------------------------------------------
  // Basic Allocation
  // -------------------------------------------------------------------------

  // Allocate n blocks, return their IDs (empty if OOM)
  [[nodiscard]] std::vector<int> Allocate(int n);

  // Allocate a single block, return ID (-1 if OOM)
  [[nodiscard]] int AllocateSingle();

  // Free blocks (decrements ref_count, only truly frees when ref_count == 0)
  void Free(const std::vector<int> &block_ids);
  void FreeSingle(int block_id);

  // Get number of free blocks
  int GetFreeBlockCount();

  // Get number of used blocks
  int GetUsedBlockCount();

  // -------------------------------------------------------------------------
  // Copy-on-Write (CoW) Support
  // -------------------------------------------------------------------------

  // Fork a block: increment reference count (for parallel sampling/beam search)
  // Returns the same block_id (now shared)
  int Fork(int block_id);

  // Copy-on-Write: if ref_count > 1, allocate new block and copy data
  // Returns new block_id if copied, same block_id if no copy needed, -1 on OOM
  // NOTE: Caller must copy the actual tensor data after this call
  int CopyOnWrite(int block_id);

  // Check if a block is shared (ref_count > 1)
  bool IsShared(int block_id) const;

  // Get reference count
  int GetRefCount(int block_id) const;

  // -------------------------------------------------------------------------
  // Slot Management
  // -------------------------------------------------------------------------

  // Mark slots as filled in a block
  void SetFilledSlots(int block_id, int num_slots);

  // Get number of filled slots
  int GetFilledSlots(int block_id) const;

  // -------------------------------------------------------------------------
  // Prefix Caching
  // -------------------------------------------------------------------------

  // Find a cached block by hash (returns -1 if not found)
  // If found, increments ref_count (Fork)
  int FindCachedBlock(uint64_t hash);

  // Register a block in prefix cache
  void RegisterPrefixBlock(int block_id, uint64_t hash);

  // Remove a block from prefix cache
  void UnregisterPrefixBlock(int block_id);

  // Compute hash for a sequence of tokens
  static uint64_t ComputeTokenHash(const int *tokens, int n_tokens);
};

// ============================================================================
// Paged KV Cache Structure
// ============================================================================
struct PagedKVCache {
  struct ggml_context *ctx = nullptr;

  // The pre-allocated KV cache tensors
  // Layout: [head_dim, n_head_kv, BLOCK_SIZE, num_blocks * n_layer]
  // This interleaves layers within blocks for better cache locality
  struct ggml_tensor *k_cache = nullptr;
  struct ggml_tensor *v_cache = nullptr;

  BlockManager *block_manager = nullptr;

  // Model dimensions
  int head_dim;
  int n_head_kv;
  int n_layer;
  int max_blocks;

  // Cache type
  ggml_type cache_type;

  ~PagedKVCache();

  // -------------------------------------------------------------------------
  // Block Data Operations (CPU-optimized)
  // -------------------------------------------------------------------------

  // Copy block data from src to dst (all layers)
  void CopyBlockData(int src_block_id, int dst_block_id);

  // Copy block data for a specific layer
  void CopyBlockDataLayer(int src_block_id, int dst_block_id, int layer);

  // Get pointer to K cache for a specific block and layer
  void *GetKBlockPtr(int block_id, int layer);
  const void *GetKBlockPtr(int block_id, int layer) const;

  // Get pointer to V cache for a specific block and layer
  void *GetVBlockPtr(int block_id, int layer);
  const void *GetVBlockPtr(int block_id, int layer) const;

  // Get pointer to a specific slot within a block
  void *GetKSlotPtr(int block_id, int layer, int slot);
  void *GetVSlotPtr(int block_id, int layer, int slot);

  // Get bytes per slot (head_dim * n_head_kv * type_size)
  size_t GetBytesPerSlot() const;

  // Get bytes per block (BLOCK_SIZE * bytes_per_slot)
  size_t GetBytesPerBlock() const;

  // -------------------------------------------------------------------------
  // Quantized KV Cache Operations
  // -------------------------------------------------------------------------

  // Check if cache uses quantized type
  bool IsQuantized() const;

  // Get elements per slot (head_dim * n_head_kv)
  int GetElementsPerSlot() const;

  // Write a single KV slot with automatic quantization
  // Input: fp32 data of size [head_dim * n_head_kv]
  void WriteKSlot(int block_id, int layer, int slot, const float *data);
  void WriteVSlot(int block_id, int layer, int slot, const float *data);

  // Read a single KV slot with automatic dequantization
  // Output: fp32 data of size [head_dim * n_head_kv]
  void ReadKSlot(int block_id, int layer, int slot, float *out) const;
  void ReadVSlot(int block_id, int layer, int slot, float *out) const;

  // Batch write/read for multiple slots (more efficient for Q8)
  void WriteKSlots(int block_id, int layer, int start_slot, int num_slots,
                   const float *data);
  void WriteVSlots(int block_id, int layer, int start_slot, int num_slots,
                   const float *data);
  void ReadKSlots(int block_id, int layer, int start_slot, int num_slots,
                  float *out) const;
  void ReadVSlots(int block_id, int layer, int start_slot, int num_slots,
                  float *out) const;

  // -------------------------------------------------------------------------
  // SIMD Prefetch Operations
  // -------------------------------------------------------------------------

  // Prefetch a block for the given layer (for async access)
  void PrefetchBlock(int block_id, int layer) const;

  // Prefetch blocks for the next layer (overlapped with current computation)
  void PrefetchNextLayer(const std::vector<int> &block_ids,
                         int next_layer) const;
};

// ============================================================================
// Initialization
// ============================================================================

// Initialize Paged KV Cache
PagedKVCache *InitPagedKVCache(TransformerModel *model, int max_num_seqs,
                               int max_seq_len, ggml_type type = GGML_TYPE_F16);

// ============================================================================
// Backward Compatibility
// ============================================================================
using BlockTable = std::vector<int>;

#endif // DENSECORE_KV_CACHE_H
