/**
 * @file kernel_context.cpp
 * @brief Implementation of KernelContext
 */

#include "kernel_context.h"
#include <iostream>
#include <utility>

namespace densecore {

KernelContext::KernelContext(size_t scratchpad_bytes)
    : scratchpad_size_(scratchpad_bytes) {
  if (scratchpad_bytes > 0) {
    scratchpad_ = make_aligned<uint8_t>(scratchpad_bytes);
  }
}

KernelContext::~KernelContext() {
  // Release hardware resources if necessary
}

KernelContext::KernelContext(KernelContext &&other) noexcept
    : scratchpad_(std::move(other.scratchpad_)),
      scratchpad_size_(other.scratchpad_size_),
      amx_configured_(other.amx_configured_), cached_rows_(other.cached_rows_),
      cached_cols_(other.cached_cols_) {

  other.scratchpad_size_ = 0;
  other.amx_configured_ = false;
}

KernelContext &KernelContext::operator=(KernelContext &&other) noexcept {
  if (this != &other) {
    scratchpad_ = std::move(other.scratchpad_);
    scratchpad_size_ = other.scratchpad_size_;
    amx_configured_ = other.amx_configured_;
    cached_rows_ = other.cached_rows_;
    cached_cols_ = other.cached_cols_;

    other.scratchpad_size_ = 0;
    other.amx_configured_ = false;
  }
  return *this;
}

void KernelContext::ConfigureAMX(int rows, int cols) {
  if (amx_configured_ && cached_rows_ == rows && cached_cols_ == cols) {
    return; // Already configured
  }

  // TODO: Actual AMX syscall/ldtilecfg implementation goes here
  // For now, we simulate the state tracking

  cached_rows_ = rows;
  cached_cols_ = cols;
  amx_configured_ = true;
}

void KernelContext::ResizeScratchpad(size_t new_size) {
  if (new_size > scratchpad_size_) {
    scratchpad_ = make_aligned<uint8_t>(new_size);
    scratchpad_size_ = new_size;
  }
}

} // namespace densecore
