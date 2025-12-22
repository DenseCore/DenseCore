/**
 * @file operation_graph.h
 * @brief Graph container for deferred/batched execution on NPUs
 *
 * NPUs like Apple ANE and Qualcomm Hexagon DSP achieve peak efficiency when
 * executing a pre-compiled graph of operations rather than individual kernels.
 * This header provides the infrastructure to capture, store, and execute
 * operation graphs.
 *
 * **Design Philosophy:**
 * - Lightweight capture: Minimal overhead when recording operations
 * - Backend-agnostic: Same graph can run on different accelerators
 * - Compilation hook: NPU backends can optimize the graph representation
 *
 * **Usage Pattern:**
 * @code
 * backend.BeginCapture();
 * // These operations are recorded, not executed
 * backend.MatMul(A, B, &C);
 * backend.RMSNorm(C, weight, &D);
 * auto graph = backend.EndCapture();
 *
 * // Optional: NPU-specific compilation
 * if (traits.requires_graph_compilation) {
 *     graph.Compile();
 * }
 *
 * // Execute the graph (potentially many times)
 * backend.ExecuteGraph(graph);
 * @endcode
 */

#ifndef DENSECORE_OPERATION_GRAPH_H
#define DENSECORE_OPERATION_GRAPH_H

#include "accelerator_traits.h"
#include "tensor.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace densecore {

// ============================================================================
// Operation Types and Parameters
// ============================================================================

/**
 * @brief Enumeration of supported graph operations
 */
enum class OpType : uint8_t {
  // Linear Algebra
  MatMul = 0,
  MatMulTransB,
  GemmInt4,

  // Normalization
  RMSNorm,
  AddRMSNorm,

  // Activation
  Softmax,
  SiLU,
  GELU,

  // Position Encoding
  RoPE,

  // Attention
  FlashAttention,
  FusedQKVProjection,

  // Memory
  Copy,
  Quantize,
  Dequantize,

  // Custom
  Custom
};

/**
 * @brief Get string name for operation type
 */
inline const char *OpTypeName(OpType op) {
  switch (op) {
  case OpType::MatMul:
    return "MatMul";
  case OpType::MatMulTransB:
    return "MatMulTransB";
  case OpType::GemmInt4:
    return "GemmInt4";
  case OpType::RMSNorm:
    return "RMSNorm";
  case OpType::AddRMSNorm:
    return "AddRMSNorm";
  case OpType::Softmax:
    return "Softmax";
  case OpType::SiLU:
    return "SiLU";
  case OpType::GELU:
    return "GELU";
  case OpType::RoPE:
    return "RoPE";
  case OpType::FlashAttention:
    return "FlashAttention";
  case OpType::FusedQKVProjection:
    return "FusedQKVProjection";
  case OpType::Copy:
    return "Copy";
  case OpType::Quantize:
    return "Quantize";
  case OpType::Dequantize:
    return "Dequantize";
  case OpType::Custom:
    return "Custom";
  default:
    return "Unknown";
  }
}

// ============================================================================
// Operation-Specific Parameter Structs
// ============================================================================

/**
 * @brief Parameters for MatMul operations
 */
struct MatMulParams {
  bool transpose_a = false;
  bool transpose_b = false;
  float alpha = 1.0f; ///< Scale factor for C = alpha * A @ B
  float beta = 0.0f;  ///< Scale factor for C += beta * C_orig
};

/**
 * @brief Parameters for RMSNorm operations
 */
struct RMSNormParams {
  float eps = 1e-5f;
  bool fused_add = false; ///< True for AddRMSNorm
};

/**
 * @brief Parameters for RoPE operations
 */
struct RoPEParams {
  int rope_dim = -1;       ///< Dimensions to rotate (-1 = all)
  bool neox_style = false; ///< GPT-NeoX interleaving pattern
};

/**
 * @brief Parameters for FlashAttention operations
 */
struct FlashAttentionParams {
  float scale = 1.0f;
  bool causal = true;
  int n_head_kv = -1; ///< For GQA, -1 = MHA mode
};

/**
 * @brief Parameters for quantization operations
 */
struct QuantizeParams {
  QuantType target_type = QuantType::INT8;
  int group_size = 128;
};

/**
 * @brief Variant holding all possible operation parameters
 */
using OpParams = std::variant<std::monostate, // No params (e.g., Softmax)
                              MatMulParams, RMSNormParams, RoPEParams,
                              FlashAttentionParams, QuantizeParams>;

// ============================================================================
// Graph Node
// ============================================================================

/**
 * @brief Single node in the operation graph
 *
 * Each node represents one operation with its inputs, outputs, and parameters.
 * Tensors are referenced by index into the graph's tensor registry.
 */
struct GraphNode {
  OpType op = OpType::MatMul;

  /**
   * @brief Indices of input tensors in the graph's tensor table
   */
  std::vector<size_t> inputs;

  /**
   * @brief Indices of output tensors in the graph's tensor table
   */
  std::vector<size_t> outputs;

  /**
   * @brief Operation-specific parameters
   */
  OpParams params;

  /**
   * @brief Human-readable name for debugging
   */
  std::string name;
};

// ============================================================================
// Operation Graph
// ============================================================================

/**
 * @brief Container for a sequence of operations to be executed together
 *
 * The OperationGraph class stores a DAG of operations that can be:
 * 1. Executed immediately by replaying on any backend (CPU fallback)
 * 2. Compiled to an optimized representation for NPU execution
 *
 * **Thread Safety:** Not thread-safe. Create one graph per thread or
 * synchronize externally.
 */
class OperationGraph {
public:
  OperationGraph() = default;
  virtual ~OperationGraph() = default;

  // =========================================================================
  // Graph Construction
  // =========================================================================

  /**
   * @brief Add a tensor to the graph's tensor table
   * @param tensor Tensor descriptor (pointer + shape + dtype)
   * @return Index of the tensor in the table
   */
  size_t RegisterTensor(const Tensor &tensor) {
    size_t idx = tensors_.size();
    tensors_.push_back(tensor);
    return idx;
  }

  /**
   * @brief Add a node to the operation graph
   * @param node Operation node with inputs/outputs as tensor indices
   */
  void AddNode(GraphNode node) { nodes_.push_back(std::move(node)); }

  // =========================================================================
  // Graph Accessors
  // =========================================================================

  /**
   * @brief Number of operations in the graph
   */
  size_t NodeCount() const { return nodes_.size(); }

  /**
   * @brief Number of tensors registered in the graph
   */
  size_t TensorCount() const { return tensors_.size(); }

  /**
   * @brief Get node at index
   */
  const GraphNode &GetNode(size_t idx) const { return nodes_[idx]; }

  /**
   * @brief Get mutable node at index (for optimization passes)
   */
  GraphNode &GetMutableNode(size_t idx) { return nodes_[idx]; }

  /**
   * @brief Get tensor at index
   */
  const Tensor &GetTensor(size_t idx) const { return tensors_[idx]; }

  /**
   * @brief Get all nodes (for iteration)
   */
  const std::vector<GraphNode> &Nodes() const { return nodes_; }

  /**
   * @brief Get all tensors (for iteration)
   */
  const std::vector<Tensor> &Tensors() const { return tensors_; }

  // =========================================================================
  // Compilation (Override in NPU backends)
  // =========================================================================

  /**
   * @brief Compile the graph for optimized execution
   *
   * Default implementation is a no-op. NPU backends override this to:
   * - Convert to CoreML model (Apple)
   * - Generate QNN graph (Qualcomm)
   * - Build ONNX runtime session
   *
   * This may be slow (100ms+) and should be called during initialization.
   */
  virtual void Compile() { compiled_ = true; }

  /**
   * @brief Check if graph has been compiled
   */
  bool IsCompiled() const { return compiled_; }

  /**
   * @brief Clear all nodes and tensors
   */
  void Clear() {
    nodes_.clear();
    tensors_.clear();
    compiled_ = false;
  }

protected:
  std::vector<GraphNode> nodes_;
  std::vector<Tensor> tensors_;
  bool compiled_ = false;
};

/**
 * @brief Immediate-mode graph for CPU backend
 *
 * This graph stores operation callbacks and replays them synchronously
 * on ExecuteGraph. Used as a fallback when no NPU is available.
 */
class ImmediateModeGraph : public OperationGraph {
public:
  using OperationCallback = std::function<void()>;

  /**
   * @brief Record an operation for later replay
   */
  void RecordOperation(OperationCallback op) {
    recorded_ops_.push_back(std::move(op));
  }

  /**
   * @brief Replay all recorded operations
   */
  void Replay() {
    for (const auto &op : recorded_ops_) {
      op();
    }
  }

  /**
   * @brief Compilation for immediate mode is a no-op
   */
  void Compile() override { compiled_ = true; }

  /**
   * @brief Clear recorded operations
   */
  void ClearRecorded() { recorded_ops_.clear(); }

private:
  std::vector<OperationCallback> recorded_ops_;
};

} // namespace densecore

#endif // DENSECORE_OPERATION_GRAPH_H
