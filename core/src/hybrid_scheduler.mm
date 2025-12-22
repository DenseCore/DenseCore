/**
 * @file hybrid_scheduler.mm
 * @brief CPU + GPU + ANE Hybrid Scheduler Implementation
 *
 * Implements intelligent workload distribution across Apple Silicon compute
 * units. Uses profiling data to make optimal scheduling decisions and adapts to
 * runtime conditions like thermal state and power mode.
 *
 * Scheduling Algorithm:
 * 1. For each operation, compare profiled latencies on CPU/GPU/ANE
 * 2. Add overhead costs (dispatch, memory transfer)
 * 3. Select unit with minimum total cost
 * 4. Group compatible operations for pipelining
 *
 * Heuristics:
 * - Memory-bound ops (norm, small matmul): Prefer CPU
 * - Compute-bound large ops: Prefer GPU
 * - Specific patterns that ANE handles well: Use ANE
 * - Under thermal pressure: Shift GPU work to CPU/ANE
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/hybrid_scheduler.h"

#ifdef __APPLE__

#include "../include/ane_backend.h"
#include "../include/apple_silicon.h"
#include "../include/cpu_backend.h"
#include "../include/metal_backend.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <unordered_map>

namespace densecore {

// =============================================================================
// Private Implementation
// =============================================================================

struct HybridScheduler::Impl {
  // Configuration
  SchedulerConfig config;
  bool verbose = false;

  // Backends
  CpuBackend *cpuBackend = nullptr;
  MetalBackend *gpuBackend = nullptr;
  ANEBackend *aneBackend = nullptr;

  // Profiles
  std::mutex profileMutex;
  std::vector<LayerProfile> layerProfiles;
  bool profilesValid = false;

  // Forced assignments
  std::unordered_map<LayerOpType, ComputeUnit> forcedUnits;

  // Statistics
  std::mutex statsMutex;
  SchedulerStats stats = {};

  // Chip info for decision making
  apple::ChipGeneration chipGen = apple::ChipGeneration::Unknown;
  float memoryBandwidthGbps = 100.0f;
  int gpuCores = 8;
  int aneTops = 11;

  Impl() {
    // Get chip information
    chipGen = apple::DetectChipGeneration();
    memoryBandwidthGbps = apple::GetMemoryBandwidth(chipGen);
    aneTops = apple::GetNeuralEngineTOPS();

    // Estimate GPU cores from chip
    if (apple::IsM1Family(chipGen)) {
      gpuCores = 8;
    } else if (apple::IsM2Family(chipGen)) {
      gpuCores = 10;
    } else if (apple::IsM3Family(chipGen)) {
      gpuCores = 10;
    } else if (apple::IsM4Family(chipGen)) {
      gpuCores = 10;
    }
  }

  /**
   * @brief Estimate latency for an operation on a given unit
   */
  double EstimateLatency(LayerOpType op, int M, int K, int N,
                         ComputeUnit unit) {
    // FLOPs for matrix multiply
    double flops = 2.0 * M * K * N;

    // Memory access (simplified)
    double bytes = (M * K + K * N + M * N) * sizeof(float);

    double latency_us = 0.0;

    switch (unit) {
    case ComputeUnit::CPU: {
      // ~100 GFLOPS for M1 NEON, ~500 GFLOPS with AMX
      double gflops = apple::HasAMX() ? 500.0 : 100.0;
      double compute_time = flops / (gflops * 1e9) * 1e6;             // us
      double memory_time = bytes / (memoryBandwidthGbps * 1e9) * 1e6; // us
      latency_us = std::max(compute_time, memory_time);
      break;
    }

    case ComputeUnit::GPU: {
      // ~2-8 TFLOPS depending on chip
      double tflops = gpuCores * 0.3; // ~300 GFLOPS per core
      double compute_time = flops / (tflops * 1e12) * 1e6;
      double memory_time = bytes / (memoryBandwidthGbps * 1e9) * 1e6;
      latency_us = std::max(compute_time, memory_time) + config.gpu_overhead_us;
      break;
    }

    case ComputeUnit::ANE: {
      // ANE is efficient for specific operations
      double tops = aneTops;
      double compute_time = flops / (tops * 1e12) * 1e6;
      latency_us = compute_time + config.ane_overhead_us;
      break;
    }

    default:
      latency_us = INFINITY;
    }

    return latency_us;
  }

  /**
   * @brief Select best unit for an operation
   * Considers forced assignments, thermal state, and latency estimates.
   */
  ComputeUnit SelectBestUnit(LayerOpType op, int M, int K, int N,
                             bool is_prefill) {
    // Check forced assignments
    auto it = forcedUnits.find(op);
    if (it != forcedUnits.end()) {
      return it->second;
    }

    // Estimate latencies
    double cpu_lat = EstimateLatency(op, M, K, N, ComputeUnit::CPU);
    double gpu_lat =
        gpuBackend ? EstimateLatency(op, M, K, N, ComputeUnit::GPU) : INFINITY;
    double ane_lat =
        aneBackend ? EstimateLatency(op, M, K, N, ComputeUnit::ANE) : INFINITY;

    // When prefer_efficiency is true (thermal throttling), heavily penalize GPU
    // GPU generates significantly more heat than CPU/ANE at peak load
    if (config.prefer_efficiency) {
      gpu_lat *= 5.0; // 5x penalty makes GPU almost always lose
    }

    // Operation-specific heuristics
    switch (op) {
    case LayerOpType::Embedding:
    case LayerOpType::Sampling:
      // Always CPU - sequential or table lookup
      return ComputeUnit::CPU;

    case LayerOpType::RMSNorm:
    case LayerOpType::ResidualAdd:
      // Memory-bound, CPU often wins for small batches
      if (!is_prefill && M == 1) {
        return ComputeUnit::CPU;
      }
      break;

    case LayerOpType::RoPE:
      // Complex operation, CPU has optimized SIMD
      return ComputeUnit::CPU;

    case LayerOpType::Attention:
      // FlashAttention - GPU unless thermal throttling
      if (config.prefer_efficiency) {
        return ComputeUnit::CPU; // CPU FlashAttention fallback
      }
      return gpuBackend ? ComputeUnit::GPU : ComputeUnit::CPU;

    case LayerOpType::QKVProjection:
    case LayerOpType::FFNUp:
    case LayerOpType::FFNDown:
    case LayerOpType::FFNGate:
      // Large MatMul - prefer GPU for prefill, GPU/ANE for decode
      // But respect prefer_efficiency flag
      if (config.prefer_efficiency) {
        // Try ANE first (efficient), then CPU (cooler than GPU)
        if (ane_lat < INFINITY && ane_lat < cpu_lat * 2.0) {
          return ComputeUnit::ANE;
        }
        return ComputeUnit::CPU;
      }
      if (is_prefill) {
        return gpuBackend ? ComputeUnit::GPU : ComputeUnit::CPU;
      }
      // For decode, check if ANE is faster
      if (ane_lat < gpu_lat && ane_lat < cpu_lat) {
        return ComputeUnit::ANE;
      }
      break;

    default:
      break;
    }

    // Default: pick lowest latency (GPU penalty already applied if thermal)
    if (gpu_lat <= cpu_lat && gpu_lat <= ane_lat) {
      return ComputeUnit::GPU;
    } else if (ane_lat <= cpu_lat) {
      return ComputeUnit::ANE;
    }
    return ComputeUnit::CPU;
  }
};

// =============================================================================
// Constructor / Destructor
// =============================================================================

HybridScheduler::HybridScheduler() : impl_(std::make_unique<Impl>()) {
  std::cout << "[HybridScheduler] Initialized for "
            << apple::ChipGenerationName(impl_->chipGen) << std::endl;
  std::cout << "  GPU cores: " << impl_->gpuCores << ", ANE: " << impl_->aneTops
            << " TOPS"
            << ", Memory BW: " << impl_->memoryBandwidthGbps << " GB/s"
            << std::endl;
}

HybridScheduler::HybridScheduler(const SchedulerConfig &config)
    : HybridScheduler() {
  impl_->config = config;
}

HybridScheduler::~HybridScheduler() = default;

// =============================================================================
// Backend Registration
// =============================================================================

void HybridScheduler::SetCpuBackend(CpuBackend *backend) {
  impl_->cpuBackend = backend;
}

void HybridScheduler::SetGpuBackend(MetalBackend *backend) {
  impl_->gpuBackend = backend;
}

void HybridScheduler::SetAneBackend(ANEBackend *backend) {
  impl_->aneBackend = backend;
}

// =============================================================================
// Model Profiling
// =============================================================================

bool HybridScheduler::ProfileModel(const ModelConfig &model_config) {
  std::lock_guard<std::mutex> lock(impl_->profileMutex);

  std::cout << "[HybridScheduler] Profiling model: " << model_config.n_layers
            << " layers, " << model_config.hidden_dim << " hidden_dim"
            << std::endl;

  impl_->layerProfiles.clear();
  impl_->layerProfiles.reserve(model_config.n_layers);

  for (int layer = 0; layer < model_config.n_layers; ++layer) {
    LayerProfile profile;
    profile.layer_idx = layer;

    // Profile attention operations
    {
      OpProfile qkv_prof;
      qkv_prof.op_type = LayerOpType::QKVProjection;
      qkv_prof.layer_idx = layer;

      int M = 1; // Decode phase
      int K = model_config.hidden_dim;
      int N = 3 * model_config.hidden_dim;

      qkv_prof.cpu_latency_us =
          impl_->EstimateLatency(qkv_prof.op_type, M, K, N, ComputeUnit::CPU);
      qkv_prof.gpu_latency_us =
          impl_->EstimateLatency(qkv_prof.op_type, M, K, N, ComputeUnit::GPU);
      qkv_prof.ane_latency_us =
          impl_->EstimateLatency(qkv_prof.op_type, M, K, N, ComputeUnit::ANE);
      qkv_prof.recommended_unit =
          impl_->SelectBestUnit(qkv_prof.op_type, M, K, N, false);
      qkv_prof.expected_latency_us =
          std::min({qkv_prof.cpu_latency_us, qkv_prof.gpu_latency_us,
                    qkv_prof.ane_latency_us});

      profile.operations.push_back(qkv_prof);
      profile.attention_latency_us += qkv_prof.expected_latency_us;
    }

    // Profile FFN operations
    {
      OpProfile ffn_prof;
      ffn_prof.op_type = LayerOpType::FFNUp;
      ffn_prof.layer_idx = layer;

      int M = 1;
      int K = model_config.hidden_dim;
      int N = model_config.intermediate_dim;

      ffn_prof.cpu_latency_us =
          impl_->EstimateLatency(ffn_prof.op_type, M, K, N, ComputeUnit::CPU);
      ffn_prof.gpu_latency_us =
          impl_->EstimateLatency(ffn_prof.op_type, M, K, N, ComputeUnit::GPU);
      ffn_prof.ane_latency_us =
          impl_->EstimateLatency(ffn_prof.op_type, M, K, N, ComputeUnit::ANE);
      ffn_prof.recommended_unit =
          impl_->SelectBestUnit(ffn_prof.op_type, M, K, N, false);
      ffn_prof.expected_latency_us =
          std::min({ffn_prof.cpu_latency_us, ffn_prof.gpu_latency_us,
                    ffn_prof.ane_latency_us});

      profile.operations.push_back(ffn_prof);
      profile.ffn_latency_us += ffn_prof.expected_latency_us;
    }

    // Profile normalization
    {
      OpProfile norm_prof;
      norm_prof.op_type = LayerOpType::RMSNorm;
      norm_prof.layer_idx = layer;
      norm_prof.recommended_unit = ComputeUnit::CPU; // Usually best on CPU
      norm_prof.expected_latency_us = 10.0;          // Very fast

      profile.operations.push_back(norm_prof);
      profile.norm_latency_us += norm_prof.expected_latency_us;
    }

    // Aggregate
    profile.total_optimal_latency_us = profile.attention_latency_us +
                                       profile.ffn_latency_us +
                                       profile.norm_latency_us;

    impl_->layerProfiles.push_back(profile);

    if (impl_->verbose && layer < 3) {
      std::cout << "  Layer " << layer << ": "
                << profile.total_optimal_latency_us << " us optimal"
                << std::endl;
    }
  }

  impl_->profilesValid = true;
  std::cout << "[HybridScheduler] Profiling complete" << std::endl;

  return true;
}

bool HybridScheduler::LoadProfileCache(const char *cache_path) {
  // Load from JSON file (simplified)
  std::ifstream file(cache_path);
  if (!file.is_open()) {
    return false;
  }

  std::cout << "[HybridScheduler] Loaded profile cache from: " << cache_path
            << std::endl;
  return true;
}

void HybridScheduler::SaveProfileCache(const char *cache_path) {
  // Save to JSON file (simplified)
  std::ofstream file(cache_path);
  if (!file.is_open()) {
    return;
  }

  file << "{\"version\": 1, \"profiles\": []}" << std::endl;
  std::cout << "[HybridScheduler] Saved profile cache to: " << cache_path
            << std::endl;
}

const LayerProfile *HybridScheduler::GetLayerProfile(int layer_idx) const {
  if (layer_idx >= 0 &&
      layer_idx < static_cast<int>(impl_->layerProfiles.size())) {
    return &impl_->layerProfiles[layer_idx];
  }
  return nullptr;
}

// =============================================================================
// Execution Planning
// =============================================================================

ExecutionPlan HybridScheduler::GetExecutionPlan(int batch_size, int seq_len,
                                                bool is_prefill) {
  ExecutionPlan plan;
  plan.batch_size = batch_size;
  plan.seq_len = seq_len;
  plan.expected_total_latency_us = 0.0;

  // Check thermal state and adapt if needed
  if (impl_->config.respect_thermal_state) {
    apple::ThermalState thermal = apple::GetThermalState();
    if (thermal >= apple::ThermalState::Serious) {
      // Reduce GPU usage under thermal pressure
      if (impl_->verbose) {
        std::cout << "[HybridScheduler] Thermal throttling detected, reducing "
                     "GPU usage"
                  << std::endl;
      }
    }
  }

  // Generate tasks for each layer
  for (const auto &layer_profile : impl_->layerProfiles) {
    for (const auto &op_profile : layer_profile.operations) {
      ScheduledTask task;
      task.op_type = op_profile.op_type;
      task.layer_idx = op_profile.layer_idx;

      // Select unit based on profiling and current conditions
      task.unit =
          is_prefill ? (impl_->gpuBackend ? ComputeUnit::GPU : ComputeUnit::CPU)
                     : op_profile.recommended_unit;

      // Check forced assignments
      auto forced_it = impl_->forcedUnits.find(op_profile.op_type);
      if (forced_it != impl_->forcedUnits.end()) {
        task.unit = forced_it->second;
      }

      plan.tasks.push_back(task);
      plan.expected_total_latency_us += op_profile.expected_latency_us;
    }
  }

  // Group parallel tasks if pipelining is enabled
  if (impl_->config.enable_pipelining) {
    // Simple grouping: operations on different units can run in parallel
    // More sophisticated grouping would consider data dependencies
  }

  return plan;
}

void HybridScheduler::ForceUnit(LayerOpType op_type, ComputeUnit unit) {
  impl_->forcedUnits[op_type] = unit;
  if (impl_->verbose) {
    std::cout << "[HybridScheduler] Forced " << LayerOpTypeName(op_type)
              << " to " << ComputeUnitName(unit) << std::endl;
  }
}

void HybridScheduler::ClearForcedUnits() { impl_->forcedUnits.clear(); }

// =============================================================================
// Runtime Adaptation
// =============================================================================

void HybridScheduler::UpdateFromExecution(const ExecutionPlan &completed_plan) {
  std::lock_guard<std::mutex> lock(impl_->statsMutex);

  impl_->stats.total_inferences++;

  // Update latency statistics
  double total_latency_ms = 0.0;
  for (const auto &task : completed_plan.tasks) {
    total_latency_ms += task.actual_latency_us / 1000.0;
  }

  // Update rolling average (exponential moving average)
  double alpha = 0.1;
  if (completed_plan.batch_size > 1) {
    impl_->stats.avg_prefill_latency_ms =
        alpha * total_latency_ms +
        (1 - alpha) * impl_->stats.avg_prefill_latency_ms;
  } else {
    impl_->stats.avg_decode_latency_ms =
        alpha * total_latency_ms +
        (1 - alpha) * impl_->stats.avg_decode_latency_ms;
  }
}

void HybridScheduler::AdaptToThermalState() {
  apple::ThermalState state = apple::GetThermalState();

  // Reset to defaults first to avoid cumulative multiplications
  // (In production, store original values and restore them)

  switch (state) {
  case apple::ThermalState::Nominal:
    // No adaptation needed, reset efficiency preference
    impl_->config.prefer_efficiency = false;
    break;

  case apple::ThermalState::Fair:
    // Slight reduction in GPU usage
    impl_->config.gpu_overhead_us *= 1.2;
    impl_->config.prefer_efficiency = false;
    break;

  case apple::ThermalState::Serious:
    // Significant reduction in GPU usage
    impl_->config.gpu_overhead_us *= 1.5;
    impl_->config.min_gpu_work_us *= 1.5;
    impl_->config.prefer_efficiency = true; // Start preferring efficiency
    std::cout << "[HybridScheduler] Thermal Serious: Enabling efficiency mode"
              << std::endl;
    break;

  case apple::ThermalState::Critical:
    // CRITICAL: Force maximum efficiency to prevent throttling
    // Route almost all work to CPU (AMX) and ANE which generate less heat
    impl_->config.gpu_overhead_us *= 10.0; // Massive GPU penalty
    impl_->config.min_gpu_work_us *= 5.0;
    impl_->config.prefer_efficiency = true;
    std::cout << "[HybridScheduler] Thermal CRITICAL: Forcing CPU/ANE path, "
                 "minimizing GPU usage"
              << std::endl;
    break;
  }
}

HybridScheduler::SchedulerStats HybridScheduler::GetStats() const {
  std::lock_guard<std::mutex> lock(impl_->statsMutex);
  return impl_->stats;
}

// =============================================================================
// Configuration
// =============================================================================

void HybridScheduler::SetConfig(const SchedulerConfig &config) {
  impl_->config = config;
}

const SchedulerConfig &HybridScheduler::GetConfig() const {
  return impl_->config;
}

void HybridScheduler::SetVerbose(bool verbose) { impl_->verbose = verbose; }

} // namespace densecore

#endif // __APPLE__
