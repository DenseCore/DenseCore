/**
 * @file tensor.h
 * @brief Lightweight tensor descriptor for backend-agnostic operations
 *
 * Design Goals:
 * - Zero-copy interop with GGML tensors
 * - Minimal overhead (no virtual functions, no heap allocation)
 * - Thread-safe read access, mutable data pointer
 *
 * Memory Model:
 * - Non-owning: `data` points to externally managed memory
 * - Backends allocate memory via `ComputeBackend::AllocateDevice`
 */

#ifndef DENSECORE_TENSOR_H
#define DENSECORE_TENSOR_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace densecore {

/// Supported data types
enum class DType : uint8_t {
  F32 = 0,  // 32-bit float
  F16 = 1,  // 16-bit float
  BF16 = 2, // Brain float 16
  INT8 = 3, // 8-bit integer
  INT4 = 4, // 4-bit integer (packed)
  UNKNOWN = 255
};

/// Device types for memory placement
enum class DeviceType : uint8_t {
  CPU = 0,
  METAL = 1, // Apple Metal GPU
  CUDA = 2,  // NVIDIA CUDA
  ASIC = 3,  // Custom ASIC (future)
  UNKNOWN = 255
};

/**
 * @brief Size of data type in bytes
 * @param dtype Data type
 * @return Size in bytes (for INT4, returns 0 as it's packed)
 */
inline size_t DTypeSizeBytes(DType dtype) {
  switch (dtype) {
  case DType::F32:
    return 4;
  case DType::F16:
    return 2;
  case DType::BF16:
    return 2;
  case DType::INT8:
    return 1;
  case DType::INT4:
    return 0; // Packed, use NumElements/2
  default:
    return 0;
  }
}

/**
 * @brief Get human-readable name for data type
 */
inline const char *DTypeName(DType dtype) {
  switch (dtype) {
  case DType::F32:
    return "F32";
  case DType::F16:
    return "F16";
  case DType::BF16:
    return "BF16";
  case DType::INT8:
    return "INT8";
  case DType::INT4:
    return "INT4";
  default:
    return "UNKNOWN";
  }
}

/**
 * @brief Get human-readable name for device type
 */
inline const char *DeviceTypeName(DeviceType device) {
  switch (device) {
  case DeviceType::CPU:
    return "CPU";
  case DeviceType::METAL:
    return "METAL";
  case DeviceType::CUDA:
    return "CUDA";
  case DeviceType::ASIC:
    return "ASIC";
  default:
    return "UNKNOWN";
  }
}

/**
 * @brief Lightweight tensor descriptor for backend-agnostic operations
 *
 * This is a POD-like struct that describes tensor metadata without owning
 * the underlying data. Designed for efficient passing to backend kernels.
 *
 * Shape Convention:
 * - shape[0] is the outermost dimension (batch)
 * - shape[ndim-1] is the innermost dimension (features)
 * - Unused dimensions are set to 0
 *
 * Stride Convention:
 * - stride[i] is the number of elements to skip for dimension i
 * - Row-major by default: stride[ndim-1] = 1, stride[i] = product of
 * shape[i+1:]
 */
struct Tensor {
  void *data = nullptr;                ///< Non-owning pointer to tensor data
  std::array<int64_t, 4> shape;        ///< Dimensions [dim0, dim1, dim2, dim3]
  std::array<int64_t, 4> stride;       ///< Strides in elements (not bytes)
  int ndim = 0;                        ///< Number of dimensions used (1-4)
  DType dtype = DType::F32;            ///< Data type
  DeviceType device = DeviceType::CPU; ///< Device placement

  /// Default constructor (creates empty tensor)
  Tensor() : shape{0, 0, 0, 0}, stride{0, 0, 0, 0} {}

  // ===========================================================================
  // Factory Methods
  // ===========================================================================

  /**
   * @brief Construct 1D tensor
   * @param data Pointer to tensor data
   * @param n Number of elements
   * @param dtype Data type (default: F32)
   * @param device Device type (default: CPU)
   */
  static Tensor Make1D(void *data, int64_t n, DType dtype = DType::F32,
                       DeviceType device = DeviceType::CPU) {
    Tensor t;
    t.data = data;
    t.shape = {n, 0, 0, 0};
    t.stride = {1, 0, 0, 0};
    t.ndim = 1;
    t.dtype = dtype;
    t.device = device;
    return t;
  }

  /**
   * @brief Construct 2D tensor (row-major)
   * @param data Pointer to tensor data
   * @param rows Number of rows
   * @param cols Number of columns
   * @param dtype Data type (default: F32)
   * @param device Device type (default: CPU)
   */
  static Tensor Make2D(void *data, int64_t rows, int64_t cols,
                       DType dtype = DType::F32,
                       DeviceType device = DeviceType::CPU) {
    Tensor t;
    t.data = data;
    t.shape = {rows, cols, 0, 0};
    t.stride = {cols, 1, 0, 0}; // Row-major: stride[0] = cols
    t.ndim = 2;
    t.dtype = dtype;
    t.device = device;
    return t;
  }

  /**
   * @brief Construct 3D tensor (row-major)
   * @param data Pointer to tensor data
   * @param d0 First dimension
   * @param d1 Second dimension
   * @param d2 Third dimension
   * @param dtype Data type (default: F32)
   * @param device Device type (default: CPU)
   */
  static Tensor Make3D(void *data, int64_t d0, int64_t d1, int64_t d2,
                       DType dtype = DType::F32,
                       DeviceType device = DeviceType::CPU) {
    Tensor t;
    t.data = data;
    t.shape = {d0, d1, d2, 0};
    t.stride = {d1 * d2, d2, 1, 0};
    t.ndim = 3;
    t.dtype = dtype;
    t.device = device;
    return t;
  }

  /**
   * @brief Construct 4D tensor (row-major)
   * @param data Pointer to tensor data
   * @param d0 First dimension (batch)
   * @param d1 Second dimension
   * @param d2 Third dimension
   * @param d3 Fourth dimension (features)
   * @param dtype Data type (default: F32)
   * @param device Device type (default: CPU)
   */
  static Tensor Make4D(void *data, int64_t d0, int64_t d1, int64_t d2,
                       int64_t d3, DType dtype = DType::F32,
                       DeviceType device = DeviceType::CPU) {
    Tensor t;
    t.data = data;
    t.shape = {d0, d1, d2, d3};
    t.stride = {d1 * d2 * d3, d2 * d3, d3, 1};
    t.ndim = 4;
    t.dtype = dtype;
    t.device = device;
    return t;
  }

  // ===========================================================================
  // Accessors
  // ===========================================================================

  /**
   * @brief Total number of elements in tensor
   */
  int64_t NumElements() const {
    int64_t n = 1;
    for (int i = 0; i < ndim; ++i) {
      n *= shape[i];
    }
    return n;
  }

  /**
   * @brief Size in bytes (accounting for packed types)
   */
  size_t SizeBytes() const {
    if (dtype == DType::INT4) {
      return (NumElements() + 1) / 2; // 2 elements per byte
    }
    return static_cast<size_t>(NumElements()) * DTypeSizeBytes(dtype);
  }

  /**
   * @brief Type-safe data access
   */
  template <typename T> T *DataAs() { return static_cast<T *>(data); }

  template <typename T> const T *DataAs() const {
    return static_cast<const T *>(data);
  }

  /**
   * @brief Check if tensor is contiguous in memory (row-major order)
   */
  bool IsContiguous() const {
    if (ndim == 0)
      return true;
    int64_t expected_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
      if (stride[i] != expected_stride)
        return false;
      expected_stride *= shape[i];
    }
    return true;
  }

  /**
   * @brief Check if tensor is valid (has data and dimensions)
   */
  bool IsValid() const { return data != nullptr && ndim > 0 && shape[0] > 0; }

  /**
   * @brief Get dimension at index (with bounds checking)
   */
  int64_t Dim(int index) const {
    if (index < 0 || index >= ndim)
      return 0;
    return shape[index];
  }
};

} // namespace densecore

#endif // DENSECORE_TENSOR_H
