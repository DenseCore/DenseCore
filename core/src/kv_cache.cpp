#include "kv_cache.h"

#include <cstring>
#include <iostream>

#include "block_allocator.h"
#include "numa_allocator.h"
#include "simd_ops.h"

// ============================================================================
// KVCache (Legacy) Implementation
// ============================================================================

KVCache::~KVCache() {
    if (ctx)
        ggml_free(ctx);
}

void KVCache::Reset() {
    if (k)
        ggml_set_zero(k);
    if (v)
        ggml_set_zero(v);
    n_tokens = 0;
}

void KVCache::RemoveLast(int n) {
    n_tokens = std::max(0, n_tokens - n);
}

KVCache* InitKVCache(TransformerModel* model) {
    KVCache* cache = new KVCache();
    cache->n_ctx = model->hparams.n_ctx;
    cache->n_layer = model->hparams.n_layer;
    cache->head_dim = model->hparams.n_embd / model->hparams.n_head;

    size_t cache_size = (size_t)cache->n_layer * cache->n_ctx * model->hparams.n_head_kv *
                        cache->head_dim * sizeof(float) * 2;

    struct ggml_init_params params = {
        .mem_size = cache_size + 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };

    cache->ctx = ggml_init(params);
    if (!cache->ctx) {
        delete cache;
        return nullptr;
    }

    cache->k = ggml_new_tensor_4d(cache->ctx, GGML_TYPE_F32, cache->head_dim,
                                  model->hparams.n_head_kv, cache->n_ctx, cache->n_layer);
    cache->v = ggml_new_tensor_4d(cache->ctx, GGML_TYPE_F32, cache->head_dim,
                                  model->hparams.n_head_kv, cache->n_ctx, cache->n_layer);

    ggml_set_zero(cache->k);
    ggml_set_zero(cache->v);

    return cache;
}

// ============================================================================
// BlockManager Implementation
// ============================================================================

BlockManager::BlockManager(int num_blocks, int block_size)
    : num_blocks(num_blocks), block_size(block_size) {
    // Initialize physical blocks
    blocks.resize(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        blocks[i].id = i;
        blocks[i].ref_count = 0;
        blocks[i].num_filled_slots = 0;
        blocks[i].content_hash = 0;
    }

    // Initialize free list (reverse order for LIFO)
    free_blocks.reserve(num_blocks);
    for (int i = num_blocks - 1; i >= 0; i--) {
        free_blocks.push_back(i);
    }
}

std::vector<int> BlockManager::Allocate(int n) {
    std::lock_guard<std::mutex> lock(mu);
    std::vector<int> allocated;

    if ((int)free_blocks.size() < n) {
        return allocated;  // OOM
    }

    allocated.reserve(n);
    for (int i = 0; i < n; i++) {
        int block_id = free_blocks.back();
        free_blocks.pop_back();

        // Initialize block metadata
        blocks[block_id].ref_count = 1;
        blocks[block_id].num_filled_slots = 0;
        blocks[block_id].content_hash = 0;

        allocated.push_back(block_id);
    }

    return allocated;
}

int BlockManager::AllocateSingle() {
    std::lock_guard<std::mutex> lock(mu);

    if (free_blocks.empty()) {
        return -1;  // OOM
    }

    int block_id = free_blocks.back();
    free_blocks.pop_back();

    blocks[block_id].ref_count = 1;
    blocks[block_id].num_filled_slots = 0;
    blocks[block_id].content_hash = 0;

    return block_id;
}

void BlockManager::Free(const std::vector<int>& block_ids) {
    std::lock_guard<std::mutex> lock(mu);

    for (int id : block_ids) {
        if (id < 0 || id >= num_blocks)
            continue;

        blocks[id].ref_count--;

        if (blocks[id].ref_count <= 0) {
            // Remove from prefix cache if present
            if (blocks[id].content_hash != 0) {
                prefix_cache.erase(blocks[id].content_hash);
                blocks[id].content_hash = 0;
            }

            blocks[id].ref_count = 0;
            blocks[id].num_filled_slots = 0;
            free_blocks.push_back(id);
        }
    }
}

void BlockManager::FreeSingle(int block_id) {
    if (block_id < 0 || block_id >= num_blocks)
        return;

    std::lock_guard<std::mutex> lock(mu);

    blocks[block_id].ref_count--;

    if (blocks[block_id].ref_count <= 0) {
        if (blocks[block_id].content_hash != 0) {
            prefix_cache.erase(blocks[block_id].content_hash);
            blocks[block_id].content_hash = 0;
        }

        blocks[block_id].ref_count = 0;
        blocks[block_id].num_filled_slots = 0;
        free_blocks.push_back(block_id);
    }
}

int BlockManager::GetFreeBlockCount() {
    std::lock_guard<std::mutex> lock(mu);
    return free_blocks.size();
}

int BlockManager::GetUsedBlockCount() {
    std::lock_guard<std::mutex> lock(mu);
    return num_blocks - free_blocks.size();
}

// ============================================================================
// Copy-on-Write Implementation
// ============================================================================

int BlockManager::Fork(int block_id) {
    if (block_id < 0 || block_id >= num_blocks)
        return -1;

    std::lock_guard<std::mutex> lock(mu);

    if (blocks[block_id].ref_count <= 0) {
        return -1;  // Block not in use
    }

    blocks[block_id].ref_count++;
    return block_id;
}

int BlockManager::CopyOnWrite(int block_id) {
    if (block_id < 0 || block_id >= num_blocks)
        return -1;

    std::lock_guard<std::mutex> lock(mu);

    // If not shared, no copy needed
    if (blocks[block_id].ref_count <= 1) {
        return block_id;
    }

    // Need to allocate new block
    if (free_blocks.empty()) {
        return -1;  // OOM
    }

    int new_block_id = free_blocks.back();
    free_blocks.pop_back();

    // Decrement old block's ref count
    blocks[block_id].ref_count--;

    // Initialize new block with same metadata
    blocks[new_block_id].ref_count = 1;
    blocks[new_block_id].num_filled_slots = blocks[block_id].num_filled_slots;
    blocks[new_block_id].content_hash = 0;  // New block, no prefix cache

    // NOTE: Actual data copy must be done by caller using
    // PagedKVCache::CopyBlockData()

    return new_block_id;
}

bool BlockManager::IsShared(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks)
        return false;
    return blocks[block_id].ref_count > 1;
}

int BlockManager::GetRefCount(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks)
        return 0;
    return blocks[block_id].ref_count;
}

// ============================================================================
// Slot Management
// ============================================================================

void BlockManager::SetFilledSlots(int block_id, int num_slots) {
    if (block_id < 0 || block_id >= num_blocks)
        return;
    blocks[block_id].num_filled_slots = std::min(num_slots, block_size);
}

int BlockManager::GetFilledSlots(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks)
        return 0;
    return blocks[block_id].num_filled_slots;
}

// ============================================================================
// Prefix Caching
// ============================================================================

int BlockManager::FindCachedBlock(uint64_t hash) {
    if (hash == 0)
        return -1;

    std::lock_guard<std::mutex> lock(mu);

    auto it = prefix_cache.find(hash);
    if (it == prefix_cache.end()) {
        return -1;
    }

    int block_id = it->second;

    // Verify block is still valid
    if (blocks[block_id].ref_count <= 0) {
        prefix_cache.erase(it);
        return -1;
    }

    // Fork the block (increment ref count)
    blocks[block_id].ref_count++;
    return block_id;
}

void BlockManager::RegisterPrefixBlock(int block_id, uint64_t hash) {
    if (block_id < 0 || block_id >= num_blocks || hash == 0)
        return;

    std::lock_guard<std::mutex> lock(mu);
    blocks[block_id].content_hash = hash;
    prefix_cache[hash] = block_id;
}

void BlockManager::UnregisterPrefixBlock(int block_id) {
    if (block_id < 0 || block_id >= num_blocks)
        return;

    std::lock_guard<std::mutex> lock(mu);

    if (blocks[block_id].content_hash != 0) {
        prefix_cache.erase(blocks[block_id].content_hash);
        blocks[block_id].content_hash = 0;
    }
}

uint64_t BlockManager::ComputeTokenHash(const int* tokens, int n_tokens) {
    // FNV-1a hash
    uint64_t hash = 14695981039346656037ULL;
    for (int i = 0; i < n_tokens; i++) {
        hash ^= (uint64_t)tokens[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// ============================================================================
// PagedKVCache Implementation
// ============================================================================

PagedKVCache::~PagedKVCache() {
    // Note: ggml_free does not call free() on mem_buffer if it was provided
    // externally, so we must free our NUMA buffer ourselves AFTER ggml_free
    if (ctx)
        ggml_free(ctx);

    // Free NUMA-allocated buffer if we allocated it (only for legacy ggml path)
    if (!use_block_allocator && numa_buffer && numa_buffer_size > 0) {
        densecore::NumaAllocator::Free(numa_buffer, numa_buffer_size);
        numa_buffer = nullptr;
        numa_buffer_size = 0;
    }

    // Block allocators are unique_ptr, will be auto-deleted
    // k_allocator.reset() and v_allocator.reset() called automatically

    if (block_manager)
        delete block_manager;
}

size_t PagedKVCache::GetBytesPerSlot() const {
    size_t type_size = ggml_type_size(cache_type);
    size_t blck_size = ggml_blck_size(cache_type);
    // For non-quantized types, blck_size is 1
    // bytes = head_dim * n_head_kv * type_size / blck_size
    return (size_t)head_dim * n_head_kv * type_size / blck_size;
}

size_t PagedKVCache::GetBytesPerBlock() const {
    return GetBytesPerSlot() * BLOCK_SIZE;
}

void* PagedKVCache::GetKBlockPtr(int block_id, int layer) {
    if (block_id < 0)
        return nullptr;

    // Layout: [head_dim, n_head_kv, BLOCK_SIZE, num_blocks * n_layer]
    // Block index in 4th dimension = block_id * n_layer + layer
    int64_t block_index = (int64_t)block_id * n_layer + layer;
    size_t byte_offset = block_index * GetBytesPerBlock();

    if (use_block_allocator && k_allocator) {
        // Use block allocator arena
        return static_cast<char*>(k_allocator->ArenaBase()) + byte_offset;
    } else if (k_cache) {
        // Legacy: use ggml tensor
        return (char*)k_cache->data + byte_offset;
    }
    return nullptr;
}

const void* PagedKVCache::GetKBlockPtr(int block_id, int layer) const {
    return const_cast<PagedKVCache*>(this)->GetKBlockPtr(block_id, layer);
}

void* PagedKVCache::GetVBlockPtr(int block_id, int layer) {
    if (block_id < 0)
        return nullptr;

    int64_t block_index = (int64_t)block_id * n_layer + layer;
    size_t byte_offset = block_index * GetBytesPerBlock();

    if (use_block_allocator && v_allocator) {
        // Use block allocator arena
        return static_cast<char*>(v_allocator->ArenaBase()) + byte_offset;
    } else if (v_cache) {
        // Legacy: use ggml tensor
        return (char*)v_cache->data + byte_offset;
    }
    return nullptr;
}

const void* PagedKVCache::GetVBlockPtr(int block_id, int layer) const {
    return const_cast<PagedKVCache*>(this)->GetVBlockPtr(block_id, layer);
}

void* PagedKVCache::GetKSlotPtr(int block_id, int layer, int slot) {
    void* block_ptr = GetKBlockPtr(block_id, layer);
    if (!block_ptr || slot < 0 || slot >= BLOCK_SIZE)
        return nullptr;
    return (char*)block_ptr + (size_t)slot * GetBytesPerSlot();
}

void* PagedKVCache::GetVSlotPtr(int block_id, int layer, int slot) {
    void* block_ptr = GetVBlockPtr(block_id, layer);
    if (!block_ptr || slot < 0 || slot >= BLOCK_SIZE)
        return nullptr;
    return (char*)block_ptr + (size_t)slot * GetBytesPerSlot();
}

void PagedKVCache::CopyBlockData(int src_block_id, int dst_block_id) {
    if (src_block_id == dst_block_id)
        return;
    if (src_block_id < 0 || dst_block_id < 0)
        return;

    size_t bytes_per_block = GetBytesPerBlock();

    for (int layer = 0; layer < n_layer; layer++) {
        void* k_src = GetKBlockPtr(src_block_id, layer);
        void* k_dst = GetKBlockPtr(dst_block_id, layer);
        void* v_src = GetVBlockPtr(src_block_id, layer);
        void* v_dst = GetVBlockPtr(dst_block_id, layer);

        if (k_src && k_dst)
            memcpy(k_dst, k_src, bytes_per_block);
        if (v_src && v_dst)
            memcpy(v_dst, v_src, bytes_per_block);
    }
}

void PagedKVCache::CopyBlockDataLayer(int src_block_id, int dst_block_id, int layer) {
    if (src_block_id == dst_block_id)
        return;
    if (src_block_id < 0 || dst_block_id < 0)
        return;
    if (layer < 0 || layer >= n_layer)
        return;

    size_t bytes_per_block = GetBytesPerBlock();

    void* k_src = GetKBlockPtr(src_block_id, layer);
    void* k_dst = GetKBlockPtr(dst_block_id, layer);
    void* v_src = GetVBlockPtr(src_block_id, layer);
    void* v_dst = GetVBlockPtr(dst_block_id, layer);

    if (k_src && k_dst)
        memcpy(k_dst, k_src, bytes_per_block);
    if (v_src && v_dst)
        memcpy(v_dst, v_src, bytes_per_block);
}

// ============================================================================
// Initialization
// ============================================================================

PagedKVCache* InitPagedKVCache(TransformerModel* model, int max_num_seqs, int max_seq_len,
                               ggml_type type, int numa_node_id) {
    PagedKVCache* cache = new PagedKVCache();

    // llama.cpp style: Use pre-computed head dimension from hparams
    cache->head_dim = model->hparams.n_embd_head_k;
    cache->n_head_kv = model->hparams.n_head_kv;
    cache->n_layer = model->hparams.n_layer;
    cache->cache_type = type;

    // Validate supported types for KV cache
    // vLLM style: FP16 is preferred. INT8 (Q8_0) supported for aggressive
    // quantization.
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_Q8_0) {
        std::cerr << "[KVCache] Warning: Unsupported cache type, falling back to F16" << std::endl;
        type = GGML_TYPE_F16;
        cache->cache_type = type;
    }

    // Calculate total blocks needed
    int total_tokens = max_num_seqs * max_seq_len;
    cache->max_blocks = (total_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Initialize BlockManager
    cache->block_manager = new BlockManager(cache->max_blocks, BLOCK_SIZE);

    // Calculate memory size per tensor (K or V)
    // Layout: [head_dim, n_head_kv, BLOCK_SIZE, num_blocks * n_layer]
    int64_t n_layer_blocks = (int64_t)cache->max_blocks * cache->n_layer;
    int64_t nelements = (int64_t)cache->head_dim * cache->n_head_kv * BLOCK_SIZE * n_layer_blocks;

    size_t type_size = ggml_type_size(type);
    size_t blck_size = ggml_blck_size(type);
    size_t tensor_size = nelements * type_size / blck_size;
    size_t total_size = tensor_size * 2;  // K and V

    // ==========================================================================
    // KVBlockAllocator-based memory pool (PRIMARY PATH)
    // Pre-allocates a single contiguous arena per K/V to eliminate fragmentation
    // ==========================================================================

    // Block stride: bytes per "block" in our linear arena
    // Each block_id maps to: block_id * n_layer + layer_idx
    // So total "logical blocks" = max_blocks * n_layer
    size_t block_stride = cache->GetBytesPerBlock();
    size_t total_logical_blocks = static_cast<size_t>(n_layer_blocks);

    // Create K allocator with NUMA awareness
    cache->k_allocator = std::make_unique<densecore::KVBlockAllocator>(
        total_logical_blocks, block_stride, 64, numa_node_id);

    // Create V allocator with NUMA awareness
    cache->v_allocator = std::make_unique<densecore::KVBlockAllocator>(
        total_logical_blocks, block_stride, 64, numa_node_id);

    if (!cache->k_allocator->IsValid() || !cache->v_allocator->IsValid()) {
        std::cerr << "[KVCache] Error: Failed to allocate " << (total_size / 1024 / 1024)
                  << " MB for KV cache via block allocators. Try reducing max_seq_len."
                  << std::endl;
        delete cache;
        return nullptr;
    }

    // Enable block allocator mode
    cache->use_block_allocator = true;

    // ==========================================================================
    // ALIGNMENT CHECK: Verify 64-byte alignment for AVX-512 safety
    // ==========================================================================
    DENSECORE_ASSERT_ALIGNED_64(cache->k_allocator->ArenaBase());
    DENSECORE_ASSERT_ALIGNED_64(cache->v_allocator->ArenaBase());

    // Type name for logging
    const char* type_name = "F32";
    if (type == GGML_TYPE_F16)
        type_name = "F16";
    else if (type == GGML_TYPE_Q8_0)
        type_name = "Q8_0 (INT8)";

    std::cout << "[KVCache] Initialized PagedKVCache (Block Allocator):" << std::endl;
    std::cout << "  - max_blocks: " << cache->max_blocks << std::endl;
    std::cout << "  - block_size: " << BLOCK_SIZE << " tokens" << std::endl;
    std::cout << "  - n_layer: " << cache->n_layer << std::endl;
    std::cout << "  - head_dim: " << cache->head_dim << std::endl;
    std::cout << "  - n_head_kv: " << cache->n_head_kv << std::endl;
    std::cout << "  - cache_type: " << type_name << std::endl;
    std::cout << "  - k_arena: " << (cache->k_allocator->ArenaSize() / 1024 / 1024) << " MB"
              << std::endl;
    std::cout << "  - v_arena: " << (cache->v_allocator->ArenaSize() / 1024 / 1024) << " MB"
              << std::endl;
    std::cout << "  - total_memory: " << (total_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  - allocation_mode: BLOCK_ALLOCATOR (zero-fragmentation)" << std::endl;

    return cache;
}

// ============================================================================
// KV Cache Read/Write Operations (vLLM-style)
// FP16 is the primary format (similar to FP8 on GPU)
// ============================================================================

bool PagedKVCache::IsQuantized() const {
    return cache_type == GGML_TYPE_F16;
}

int PagedKVCache::GetElementsPerSlot() const {
    return head_dim * n_head_kv;
}

void PagedKVCache::WriteKSlot(int block_id, int layer, int slot, const float* data) {
    void* ptr = GetKSlotPtr(block_id, layer, slot);
    if (!ptr)
        return;

    const int n = GetElementsPerSlot();

    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF32ToF16((ggml_fp16_t*)ptr, data, n);
    } else if (cache_type == GGML_TYPE_Q8_0) {
        // Quantize FP32 to Q8_0 (INT8) using ggml's generic quantization
        // Note: Q8_0 block size is 32, so n should be aligned
        ggml_quantize_chunk(GGML_TYPE_Q8_0, data, ptr, 0, 1, n, nullptr);
    } else {
        densecore::simd::CopyF32((float*)ptr, data, n);
    }
}

void PagedKVCache::WriteVSlot(int block_id, int layer, int slot, const float* data) {
    void* ptr = GetVSlotPtr(block_id, layer, slot);
    if (!ptr)
        return;

    const int n = GetElementsPerSlot();

    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF32ToF16((ggml_fp16_t*)ptr, data, n);
    } else if (cache_type == GGML_TYPE_Q8_0) {
        ggml_quantize_chunk(GGML_TYPE_Q8_0, data, ptr, 0, 1, n, nullptr);
    } else {
        densecore::simd::CopyF32((float*)ptr, data, n);
    }
}

void PagedKVCache::ReadKSlot(int block_id, int layer, int slot, float* out) const {
    const void* ptr = const_cast<PagedKVCache*>(this)->GetKSlotPtr(block_id, layer, slot);
    if (!ptr)
        return;

    const int n = GetElementsPerSlot();

    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF16ToF32(out, (const ggml_fp16_t*)ptr, n);
    } else if (cache_type == GGML_TYPE_Q8_0) {
        // Dequantize Q8_0 to FP32 using ggml type traits
        const auto* type_traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        type_traits->to_float(ptr, out, n);
    } else {
        densecore::simd::CopyF32(out, (const float*)ptr, n);
    }
}

void PagedKVCache::ReadVSlot(int block_id, int layer, int slot, float* out) const {
    const void* ptr = const_cast<PagedKVCache*>(this)->GetVSlotPtr(block_id, layer, slot);
    if (!ptr)
        return;

    const int n = GetElementsPerSlot();

    if (cache_type == GGML_TYPE_F16) {
        densecore::simd::ConvertF16ToF32(out, (const ggml_fp16_t*)ptr, n);
    } else if (cache_type == GGML_TYPE_Q8_0) {
        const auto* type_traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
        type_traits->to_float(ptr, out, n);
    } else {
        densecore::simd::CopyF32(out, (const float*)ptr, n);
    }
}

void PagedKVCache::WriteKSlots(int block_id, int layer, int start_slot, int num_slots,
                               const float* data) {
    if (num_slots <= 0)
        return;

    const int n_per_slot = GetElementsPerSlot();

    for (int i = 0; i < num_slots; i++) {
        WriteKSlot(block_id, layer, start_slot + i, data + i * n_per_slot);
    }
}

void PagedKVCache::WriteVSlots(int block_id, int layer, int start_slot, int num_slots,
                               const float* data) {
    if (num_slots <= 0)
        return;

    const int n_per_slot = GetElementsPerSlot();

    for (int i = 0; i < num_slots; i++) {
        WriteVSlot(block_id, layer, start_slot + i, data + i * n_per_slot);
    }
}

void PagedKVCache::ReadKSlots(int block_id, int layer, int start_slot, int num_slots,
                              float* out) const {
    if (num_slots <= 0)
        return;

    const int n_per_slot = GetElementsPerSlot();

    for (int i = 0; i < num_slots; i++) {
        ReadKSlot(block_id, layer, start_slot + i, out + i * n_per_slot);
    }
}

void PagedKVCache::ReadVSlots(int block_id, int layer, int start_slot, int num_slots,
                              float* out) const {
    if (num_slots <= 0)
        return;

    const int n_per_slot = GetElementsPerSlot();

    for (int i = 0; i < num_slots; i++) {
        ReadVSlot(block_id, layer, start_slot + i, out + i * n_per_slot);
    }
}

// ============================================================================
// SIMD Prefetch Operations
// ============================================================================

void PagedKVCache::PrefetchBlock(int block_id, int layer) const {
    const void* k_ptr = const_cast<PagedKVCache*>(this)->GetKBlockPtr(block_id, layer);
    const void* v_ptr = const_cast<PagedKVCache*>(this)->GetVBlockPtr(block_id, layer);

    if (k_ptr) {
        densecore::simd::PrefetchRange(k_ptr, GetBytesPerBlock());
    }
    if (v_ptr) {
        densecore::simd::PrefetchRange(v_ptr, GetBytesPerBlock());
    }
}

void PagedKVCache::PrefetchNextLayer(const std::vector<int>& block_ids, int next_layer) const {
    if (next_layer >= n_layer)
        return;

    for (int block_id : block_ids) {
        PrefetchBlock(block_id, next_layer);
    }
}
