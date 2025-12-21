/**
 * @file backend_registry.h
 * @brief Global registry for compute backends
 *
 * Provides a central point for backend management:
 * - Automatic CPU backend registration
 * - Support for custom backend registration (ASIC, GPU)
 * - Default backend selection
 */

#ifndef DENSECORE_BACKEND_REGISTRY_H
#define DENSECORE_BACKEND_REGISTRY_H

#include "compute_backend.h"
#include <memory>
#include <mutex>
#include <unordered_map>

namespace densecore {

/**
 * @brief Global registry for compute backends
 *
 * Usage:
 *   // At startup (usually automatic)
 *   BackendRegistry::Instance().RegisterCpuBackend();
 *
 *   // Get default backend
 *   ComputeBackend* backend = BackendRegistry::Instance().GetDefault();
 *
 *   // Get specific backend
 *   ComputeBackend* cpu = BackendRegistry::Instance().Get(DeviceType::CPU);
 *
 *   // Register custom ASIC backend
 *   BackendRegistry::Instance().Register(
 *       DeviceType::ASIC,
 *       std::make_unique<MyAsicBackend>()
 *   );
 *   BackendRegistry::Instance().SetDefault(DeviceType::ASIC);
 */
class BackendRegistry {
public:
  /**
   * @brief Get singleton instance
   */
  static BackendRegistry &Instance();

  /**
   * @brief Register CPU backend
   *
   * Automatically called on first use. Safe to call multiple times.
   */
  void RegisterCpuBackend();

  /**
   * @brief Register custom backend
   *
   * For ASIC or GPU backends implemented in separate modules.
   * Ownership is transferred to the registry.
   *
   * @param device Device type for this backend
   * @param backend Backend implementation
   */
  void Register(DeviceType device, std::unique_ptr<ComputeBackend> backend);

  /**
   * @brief Get backend by device type
   *
   * @param device Device type to look up
   * @return Pointer to backend, nullptr if not registered
   */
  ComputeBackend *Get(DeviceType device);

  /**
   * @brief Get default backend
   *
   * Returns the currently selected default backend.
   * If no backends are registered, returns nullptr.
   *
   * @return Pointer to default backend
   */
  ComputeBackend *GetDefault();

  /**
   * @brief Set default backend device type
   *
   * Future calls to GetDefault() will return this backend.
   *
   * @param device Device type to set as default
   * @return true if device was registered, false otherwise
   */
  bool SetDefault(DeviceType device);

  /**
   * @brief Check if a backend is registered
   *
   * @param device Device type to check
   * @return true if registered
   */
  bool IsRegistered(DeviceType device);

  /**
   * @brief Check if registry has been initialized
   */
  bool IsInitialized() const { return initialized_; }

private:
  BackendRegistry() = default;
  BackendRegistry(const BackendRegistry &) = delete;
  BackendRegistry &operator=(const BackendRegistry &) = delete;

  std::mutex mutex_;
  std::unordered_map<DeviceType, std::unique_ptr<ComputeBackend>> backends_;
  DeviceType default_device_ = DeviceType::CPU;
  bool initialized_ = false;
};

/**
 * @brief Convenience function to get default backend
 *
 * Equivalent to BackendRegistry::Instance().GetDefault()
 */
inline ComputeBackend *GetDefaultBackend() {
  return BackendRegistry::Instance().GetDefault();
}

} // namespace densecore

#endif // DENSECORE_BACKEND_REGISTRY_H
