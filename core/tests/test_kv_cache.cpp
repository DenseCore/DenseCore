/**
 * @file test_kv_cache.cpp
 * @brief Unit tests for KV cache components
 */

#include <gtest/gtest.h>
#include <set>

#include "kv_cache.h"

// =============================================================================
// SequenceBlockTable Tests
// =============================================================================

TEST(SequenceBlockTable, GetPhysicalBlockId) {
    SequenceBlockTable table;
    table.block_ids = {10, 20, 30, 40};  // 4 blocks

    // Token 0-15 in block 0 (id=10)
    EXPECT_EQ(table.GetPhysicalBlockId(0), 10);
    EXPECT_EQ(table.GetPhysicalBlockId(15), 10);

    // Token 16-31 in block 1 (id=20)
    EXPECT_EQ(table.GetPhysicalBlockId(16), 20);
    EXPECT_EQ(table.GetPhysicalBlockId(31), 20);

    // Token 32-47 in block 2 (id=30)
    EXPECT_EQ(table.GetPhysicalBlockId(32), 30);
}

TEST(SequenceBlockTable, GetPhysicalBlockId_InvalidPos) {
    SequenceBlockTable table;
    table.block_ids = {10, 20};  // 2 blocks = 32 tokens max

    // Token beyond allocated blocks
    EXPECT_EQ(table.GetPhysicalBlockId(32), -1);
    EXPECT_EQ(table.GetPhysicalBlockId(100), -1);
}

TEST(SequenceBlockTable, GetSlotIndex) {
    SequenceBlockTable table;

    // BLOCK_SIZE is 16
    EXPECT_EQ(table.GetSlotIndex(0), 0);
    EXPECT_EQ(table.GetSlotIndex(1), 1);
    EXPECT_EQ(table.GetSlotIndex(15), 15);
    EXPECT_EQ(table.GetSlotIndex(16), 0);  // Wraps to next block
    EXPECT_EQ(table.GetSlotIndex(17), 1);
    EXPECT_EQ(table.GetSlotIndex(32), 0);
}

TEST(SequenceBlockTable, GetNumBlocks) {
    SequenceBlockTable table;
    EXPECT_EQ(table.GetNumBlocks(), 0);

    table.block_ids = {1, 2, 3};
    EXPECT_EQ(table.GetNumBlocks(), 3);
}

TEST(SequenceBlockTable, BlocksNeeded) {
    // BLOCK_SIZE is 16
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(0), 0);
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(1), 1);
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(16), 1);
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(17), 2);
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(32), 2);
    EXPECT_EQ(SequenceBlockTable::BlocksNeeded(33), 3);
}

// =============================================================================
// PhysicalBlock Tests
// =============================================================================

TEST(PhysicalBlock, IsFull) {
    PhysicalBlock block;
    block.num_filled_slots = 0;

    EXPECT_FALSE(block.is_full());

    block.num_filled_slots = BLOCK_SIZE - 1;
    EXPECT_FALSE(block.is_full());

    block.num_filled_slots = BLOCK_SIZE;
    EXPECT_TRUE(block.is_full());
}

TEST(PhysicalBlock, RefCount) {
    PhysicalBlock block;
    block.ref_count = 1;

    block.ref_count++;
    EXPECT_EQ(block.ref_count, 2);

    block.ref_count--;
    EXPECT_EQ(block.ref_count, 1);
}

// =============================================================================
// BlockManager Tests
// =============================================================================

class BlockManagerTest : public ::testing::Test {
protected:
    void SetUp() override { manager = std::make_unique<BlockManager>(32, BLOCK_SIZE); }

    void TearDown() override { manager.reset(); }

    std::unique_ptr<BlockManager> manager;
};

TEST_F(BlockManagerTest, AllocateSingleBlock) {
    int block_id = manager->AllocateSingle();
    EXPECT_GE(block_id, 0);

    int used = manager->GetUsedBlockCount();
    int free = manager->GetFreeBlockCount();
    EXPECT_EQ(used, 1);
    EXPECT_EQ(free, 31);
}

TEST_F(BlockManagerTest, FreeSingleBlock) {
    int block_id = manager->AllocateSingle();
    EXPECT_GE(block_id, 0);

    manager->FreeSingle(block_id);

    int used = manager->GetUsedBlockCount();
    int free = manager->GetFreeBlockCount();
    EXPECT_EQ(used, 0);
    EXPECT_EQ(free, 32);
}

TEST_F(BlockManagerTest, AllocateMultipleBlocks) {
    std::vector<int> blocks = manager->Allocate(5);
    EXPECT_EQ(blocks.size(), 5);

    // All block IDs should be unique
    std::set<int> unique_ids(blocks.begin(), blocks.end());
    EXPECT_EQ(unique_ids.size(), 5);

    int used = manager->GetUsedBlockCount();
    EXPECT_EQ(used, 5);
}

TEST_F(BlockManagerTest, FreeMultipleBlocks) {
    std::vector<int> blocks = manager->Allocate(5);
    EXPECT_EQ(blocks.size(), 5);

    manager->Free(blocks);

    int used = manager->GetUsedBlockCount();
    int free = manager->GetFreeBlockCount();
    EXPECT_EQ(used, 0);
    EXPECT_EQ(free, 32);
}

TEST_F(BlockManagerTest, AllocateAllBlocks) {
    std::vector<int> blocks = manager->Allocate(32);
    EXPECT_EQ(blocks.size(), 32);

    // All blocks allocated
    int free = manager->GetFreeBlockCount();
    EXPECT_EQ(free, 0);

    // Next single allocation should fail
    int exhausted = manager->AllocateSingle();
    EXPECT_EQ(exhausted, -1);

    // Free one block
    manager->FreeSingle(blocks[0]);

    // Now allocation should succeed
    int reclaimed = manager->AllocateSingle();
    EXPECT_GE(reclaimed, 0);
}

TEST_F(BlockManagerTest, Fork_IncrementRefCount) {
    int block_id = manager->AllocateSingle();
    EXPECT_EQ(manager->GetRefCount(block_id), 1);

    // Fork (increment ref count for CoW)
    int forked_id = manager->Fork(block_id);
    EXPECT_EQ(forked_id, block_id);
    EXPECT_EQ(manager->GetRefCount(block_id), 2);

    // IsShared should return true
    EXPECT_TRUE(manager->IsShared(block_id));
}

TEST_F(BlockManagerTest, CopyOnWrite_SharedBlock) {
    int block_id = manager->AllocateSingle();
    manager->Fork(block_id);  // Now ref_count = 2

    // CoW should allocate a new block
    int cow_id = manager->CopyOnWrite(block_id);
    EXPECT_GE(cow_id, 0);
    EXPECT_NE(cow_id, block_id);  // Different block

    // Original still has ref_count = 1 (decremented from 2)
    // The CoW returns a new block with ref_count = 1
    EXPECT_EQ(manager->GetRefCount(block_id), 1);
}

TEST_F(BlockManagerTest, CopyOnWrite_UniqueBlock) {
    int block_id = manager->AllocateSingle();
    EXPECT_EQ(manager->GetRefCount(block_id), 1);

    // CoW on unique block should return same block
    int cow_id = manager->CopyOnWrite(block_id);
    EXPECT_EQ(cow_id, block_id);
}

TEST_F(BlockManagerTest, SlotManagement) {
    int block_id = manager->AllocateSingle();

    EXPECT_EQ(manager->GetFilledSlots(block_id), 0);

    manager->SetFilledSlots(block_id, 10);
    EXPECT_EQ(manager->GetFilledSlots(block_id), 10);

    manager->SetFilledSlots(block_id, BLOCK_SIZE);
    EXPECT_EQ(manager->GetFilledSlots(block_id), BLOCK_SIZE);
}

TEST_F(BlockManagerTest, ComputeTokenHash) {
    std::vector<int> tokens1 = {1, 2, 3, 4};
    std::vector<int> tokens2 = {1, 2, 3, 4};
    std::vector<int> tokens3 = {1, 2, 3, 5};

    uint64_t hash1 = BlockManager::ComputeTokenHash(tokens1.data(), tokens1.size());
    uint64_t hash2 = BlockManager::ComputeTokenHash(tokens2.data(), tokens2.size());
    uint64_t hash3 = BlockManager::ComputeTokenHash(tokens3.data(), tokens3.size());

    EXPECT_EQ(hash1, hash2);  // Same tokens -> same hash
    EXPECT_NE(hash1, hash3);  // Different tokens -> different hash
}

TEST_F(BlockManagerTest, PrefixCaching) {
    int block_id = manager->AllocateSingle();
    uint64_t hash = 0x12345678;

    // Register block in prefix cache
    manager->RegisterPrefixBlock(block_id, hash);

    // Should find the cached block
    int cached_id = manager->FindCachedBlock(hash);
    EXPECT_EQ(cached_id, block_id);

    // Unregister and verify not found
    manager->UnregisterPrefixBlock(block_id);
    int not_found = manager->FindCachedBlock(hash);
    EXPECT_EQ(not_found, -1);
}

// =============================================================================
// BLOCK_SIZE constant test
// =============================================================================

TEST(KVCache, BlockSizeConstant) {
    EXPECT_EQ(BLOCK_SIZE, 16);  // Verify expected block size
}
