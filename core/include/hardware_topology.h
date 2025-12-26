/**
 * @file hardware_topology.h
 * @brief Hardware topology detection and NUMA-aware thread affinity using hwloc
 *
 * Provides:
 * - NUMA node and core topology detection
 * - Physical vs hyper-thread core identification
 * - Thread pinning with Scatter/Compact policies
 * - Thread pool affinity management for GEMM workers
 */

#ifndef DENSECORE_HARDWARE_TOPOLOGY_H
#define DENSECORE_HARDWARE_TOPOLOGY_H

#include <atomic>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#ifdef DENSECORE_USE_HWLOC
#include <hwloc.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace densecore {

/**
 * Information about a single CPU core
 */
struct CoreInfo {
    int logical_id;       ///< OS-visible core ID (0-indexed)
    int physical_id;      ///< Physical core ID (same for SMT siblings)
    int numa_node;        ///< NUMA node this core belongs to
    bool is_hyperthread;  ///< true if this is an SMT sibling (not primary thread)
};

/**
 * Thread pinning policy for multi-core systems
 */
enum class PinningPolicy {
    COMPACT,  ///< Pack threads on adjacent cores (share L2 cache)
    SCATTER   ///< Spread across physical cores (maximize L3 utilization)
};

/**
 * Hardware topology singleton using hwloc for precise core/NUMA detection
 *
 * Falls back to sysfs-based detection when hwloc is not available.
 */
class HardwareTopology {
public:
    /**
     * Get singleton instance (thread-safe, lazy initialization)
     */
    static HardwareTopology& GetInstance() {
        static HardwareTopology instance;
        return instance;
    }

    // Non-copyable
    HardwareTopology(const HardwareTopology&) = delete;
    HardwareTopology& operator=(const HardwareTopology&) = delete;

    // =========================================================================
    // Topology Queries
    // =========================================================================

    /**
     * Get number of NUMA nodes in the system
     * @return Number of NUMA nodes (minimum 1)
     */
    int GetNumaNodeCount() const { return numa_node_count_; }

    /**
     * Get total number of logical CPUs
     */
    int GetLogicalCoreCount() const { return static_cast<int>(cores_.size()); }

    /**
     * Get number of physical cores (excluding hyper-threads)
     * @param numa_node Specific NUMA node (-1 for all nodes)
     */
    int GetPhysicalCoreCount(int numa_node = -1) const {
        int count = 0;
        for (const auto& core : cores_) {
            if (!core.is_hyperthread) {
                if (numa_node < 0 || core.numa_node == numa_node) {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * Get all core info for a NUMA node
     * @param numa_node NUMA node ID (0-indexed)
     * @return Vector of CoreInfo for cores in that node
     */
    std::vector<CoreInfo> GetCoresInNumaNode(int numa_node) const {
        std::vector<CoreInfo> result;
        for (const auto& core : cores_) {
            if (core.numa_node == numa_node) {
                result.push_back(core);
            }
        }
        return result;
    }

    /**
     * Get physical core IDs only (excludes hyper-threads)
     * @param numa_node Specific NUMA node (-1 for all)
     * @return Vector of logical core IDs for physical cores
     */
    std::vector<int> GetPhysicalCoreIds(int numa_node = -1) const {
        std::vector<int> result;
        for (const auto& core : cores_) {
            if (!core.is_hyperthread) {
                if (numa_node < 0 || core.numa_node == numa_node) {
                    result.push_back(core.logical_id);
                }
            }
        }
        return result;
    }

    // =========================================================================
    // Thread Pinning
    // =========================================================================

    /**
     * Pin the calling thread to a specific CPU core
     * @param core_id Logical core ID (0-indexed)
     * @return true on success
     */
    static bool PinCurrentThread(int core_id) {
#if defined(__linux__) && !defined(__ANDROID__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
#elif defined(_WIN32)
        if (core_id < 0 || core_id >= 64)
            return false;
        DWORD_PTR mask = 1ULL << core_id;
        return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#else
        (void)core_id;
        return false;
#endif
    }

    /**
     * Pin the calling thread to a NUMA node using specified policy
     * @param numa_node Target NUMA node
     * @param policy SCATTER spreads across physical cores, COMPACT packs densely
     * @return true if pinned successfully
     */
    bool PinCurrentThreadToNumaNode(int numa_node, PinningPolicy policy = PinningPolicy::SCATTER) {
        auto cores = GetPhysicalCoreIds(numa_node);
        if (cores.empty())
            return false;

        // For single-thread pinning, use first available physical core
        int target_core = cores[0];
        if (policy == PinningPolicy::SCATTER && cores.size() > 1) {
            // Use a simple round-robin for multiple callers
            static std::atomic<int> scatter_idx{0};
            int idx = scatter_idx.fetch_add(1) % cores.size();
            target_core = cores[idx];
        }

        return PinCurrentThread(target_core);
    }

    /**
     * Pin a thread pool to cores within a NUMA node
     *
     * SCATTER: Distributes threads across physical cores to maximize L3 sharing
     * COMPACT: Packs threads on adjacent cores for L2 sharing
     *
     * @param threads Vector of threads to pin (must be joinable)
     * @param numa_node Target NUMA node (-1 for node 0)
     * @param policy Pinning strategy
     */
    void PinThreadPool(std::vector<std::thread>& threads, int numa_node,
                       PinningPolicy policy = PinningPolicy::SCATTER) {
        int target_node = (numa_node >= 0) ? numa_node : 0;
        auto physical_cores = GetPhysicalCoreIds(target_node);
        if (physical_cores.empty())
            return;

        size_t num_threads = threads.size();
        size_t num_cores = physical_cores.size();

        for (size_t i = 0; i < num_threads; ++i) {
            if (!threads[i].joinable())
                continue;

            int target_core;
            if (policy == PinningPolicy::SCATTER) {
                // Spread: thread i -> core i % num_cores
                target_core = physical_cores[i % num_cores];
            } else {
                // Compact: pack threads on same cores if more threads than cores
                target_core = physical_cores[std::min(i, num_cores - 1)];
            }

            // Pin using native handle
            PinThreadByHandle(threads[i].native_handle(), target_core);
        }
    }

    /**
     * Pin a thread by its native handle
     * Used for GGML thread pool integration
     */
    static bool PinThreadByHandle(std::thread::native_handle_type handle, int core_id) {
#if defined(__linux__) && !defined(__ANDROID__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        return pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset) == 0;
#elif defined(_WIN32)
        if (core_id < 0 || core_id >= 64)
            return false;
        DWORD_PTR mask = 1ULL << core_id;
        return SetThreadAffinityMask((HANDLE)handle, mask) != 0;
#else
        (void)handle;
        (void)core_id;
        return false;
#endif
    }

    /**
     * Execute a function on each thread in a pool with NUMA-aware pinning
     * Useful for late binding when threads are created externally (e.g., GGML)
     *
     * @param thread_init_fn Function called on each thread with (thread_idx,
     * core_id)
     * @param num_threads Number of threads to configure
     * @param numa_node Target NUMA node
     * @param policy Pinning strategy
     */
    void ConfigureThreadPoolAffinity(std::function<void(int, int)> thread_init_fn, int num_threads,
                                     int numa_node, PinningPolicy policy = PinningPolicy::SCATTER) {
        int target_node = (numa_node >= 0) ? numa_node : 0;
        auto physical_cores = GetPhysicalCoreIds(target_node);
        if (physical_cores.empty())
            return;

        size_t num_cores = physical_cores.size();

        for (int i = 0; i < num_threads; ++i) {
            int target_core;
            if (policy == PinningPolicy::SCATTER) {
                target_core = physical_cores[i % num_cores];
            } else {
                target_core = physical_cores[std::min(static_cast<size_t>(i), num_cores - 1)];
            }
            thread_init_fn(i, target_core);
        }
    }

    // =========================================================================
    // Compute Thread Affinity (for GGML thread pool integration)
    // =========================================================================

    /**
     * Setup compute thread affinity mapping for GGML workers
     *
     * Call this before compute operations. It pre-computes the core assignment
     * for each thread index so workers can pin themselves on first use.
     *
     * @param numa_node Target NUMA node
     * @param n_threads Number of compute threads (from ggml n_threads setting)
     * @param policy SCATTER spreads threads across physical cores
     */
    void SetupComputeThreadAffinity(int numa_node, int n_threads,
                                    PinningPolicy policy = PinningPolicy::SCATTER) {
        std::lock_guard<std::mutex> lock(compute_affinity_mu_);

        int target_node = (numa_node >= 0) ? numa_node : 0;
        auto physical_cores = GetPhysicalCoreIds(target_node);
        if (physical_cores.empty())
            return;

        compute_thread_cores_.resize(n_threads);
        size_t num_cores = physical_cores.size();

        for (int i = 0; i < n_threads; ++i) {
            if (policy == PinningPolicy::SCATTER) {
                compute_thread_cores_[i] = physical_cores[i % num_cores];
            } else {
                compute_thread_cores_[i] =
                    physical_cores[std::min(static_cast<size_t>(i), num_cores - 1)];
            }
        }

        compute_affinity_configured_ = true;
        compute_affinity_numa_node_ = target_node;

        fprintf(stderr,
                "[HardwareTopology] Compute thread affinity configured: "
                "%d threads on NUMA node %d (%s policy)\n",
                n_threads, target_node, (policy == PinningPolicy::SCATTER) ? "SCATTER" : "COMPACT");
    }

    /**
     * Get the assigned core for a compute thread index
     *
     * @param thread_idx Thread index (ith from GGML callback)
     * @return Core ID to pin to, or -1 if not configured
     */
    int GetAssignedCore(int thread_idx) const {
        std::lock_guard<std::mutex> lock(compute_affinity_mu_);
        if (!compute_affinity_configured_ || thread_idx < 0 ||
            thread_idx >= static_cast<int>(compute_thread_cores_.size())) {
            return -1;
        }
        return compute_thread_cores_[thread_idx];
    }

    /**
     * Pin the current thread based on its GGML thread index
     *
     * Call this from within GGML callbacks (cb_int4_gemm, etc.) on first
     * invocation. Uses thread-local flag to avoid re-pinning on every call.
     *
     * @param thread_idx Thread index (ith from GGML callback)
     * @return true if pinned successfully (or already pinned)
     */
    bool PinComputeThread(int thread_idx) {
        // Thread-local to track if we've already pinned this thread
        thread_local bool already_pinned = false;
        thread_local int pinned_to_core = -1;

        if (already_pinned) {
            return true;  // Already pinned, skip syscall overhead
        }

        int core_id = GetAssignedCore(thread_idx);
        if (core_id < 0) {
            return false;  // Affinity not configured
        }

        if (PinCurrentThread(core_id)) {
            already_pinned = true;
            pinned_to_core = core_id;
            return true;
        }
        return false;
    }

    /**
     * Check if compute thread affinity is configured
     */
    bool IsComputeAffinityConfigured() const {
        std::lock_guard<std::mutex> lock(compute_affinity_mu_);
        return compute_affinity_configured_;
    }

private:
    HardwareTopology() { Initialize(); }

    ~HardwareTopology() {
#ifdef DENSECORE_USE_HWLOC
        if (topology_initialized_) {
            hwloc_topology_destroy(topology_);
        }
#endif
    }

    void Initialize() {
#ifdef DENSECORE_USE_HWLOC
        InitializeWithHwloc();
#else
        InitializeWithSysfs();
#endif
    }

#ifdef DENSECORE_USE_HWLOC
    void InitializeWithHwloc() {
        if (hwloc_topology_init(&topology_) < 0) {
            InitializeWithSysfs();
            return;
        }

        if (hwloc_topology_load(topology_) < 0) {
            hwloc_topology_destroy(topology_);
            InitializeWithSysfs();
            return;
        }

        topology_initialized_ = true;

        // Count NUMA nodes
        int depth = hwloc_get_type_depth(topology_, HWLOC_OBJ_NUMANODE);
        if (depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
            numa_node_count_ = hwloc_get_nbobjs_by_depth(topology_, depth);
        }
        if (numa_node_count_ < 1)
            numa_node_count_ = 1;

        // Enumerate PUs (processing units = logical cores)
        int num_pus = hwloc_get_nbobjs_by_type(topology_, HWLOC_OBJ_PU);
        cores_.reserve(num_pus);

        for (int i = 0; i < num_pus; ++i) {
            hwloc_obj_t pu = hwloc_get_obj_by_type(topology_, HWLOC_OBJ_PU, i);
            if (!pu)
                continue;

            CoreInfo info;
            info.logical_id = pu->os_index;

            // Find parent core to get physical_id and hyper-thread status
            hwloc_obj_t core = hwloc_get_ancestor_obj_by_type(topology_, HWLOC_OBJ_CORE, pu);
            if (core) {
                info.physical_id = core->logical_index;
                // First PU in core is primary, others are hyper-threads
                info.is_hyperthread = (pu != core->first_child);
            } else {
                info.physical_id = info.logical_id;
                info.is_hyperthread = false;
            }

            // Find NUMA node
            hwloc_obj_t numa = hwloc_get_ancestor_obj_by_type(topology_, HWLOC_OBJ_NUMANODE, pu);
            if (numa) {
                info.numa_node = numa->logical_index;
            } else {
                // Some systems have memory at package level instead of NUMA node
                info.numa_node = 0;
            }

            cores_.push_back(info);
        }

        // Sort by logical_id for consistent ordering
        std::sort(cores_.begin(), cores_.end(),
                  [](const CoreInfo& a, const CoreInfo& b) { return a.logical_id < b.logical_id; });
    }
#endif  // DENSECORE_USE_HWLOC

    void InitializeWithSysfs() {
        // Fallback: basic detection without hwloc
        numa_node_count_ = 1;

#if defined(__linux__)
        // Count NUMA nodes
        for (int i = 0; i < 256; ++i) {
            char path[64];
            snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", i);
            if (access(path, F_OK) == 0) {
                numa_node_count_ = i + 1;
            } else {
                break;
            }
        }

        // Get number of CPUs
        int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        cores_.reserve(num_cpus);

        for (int i = 0; i < num_cpus; ++i) {
            CoreInfo info;
            info.logical_id = i;
            info.physical_id = i;  // Assume 1:1 without hwloc
            info.numa_node = 0;
            info.is_hyperthread = false;  // Cannot detect without hwloc

            // Try to detect NUMA node from sysfs
            for (int n = 0; n < numa_node_count_; ++n) {
                char path[128];
                snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpu%d", n, i);
                if (access(path, F_OK) == 0) {
                    info.numa_node = n;
                    break;
                }
            }

            cores_.push_back(info);
        }
#elif defined(_WIN32)
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        int num_cpus = sysinfo.dwNumberOfProcessors;

        cores_.reserve(num_cpus);
        for (int i = 0; i < num_cpus; ++i) {
            CoreInfo info;
            info.logical_id = i;
            info.physical_id = i;
            info.numa_node = 0;
            info.is_hyperthread = false;
            cores_.push_back(info);
        }

        // Try to get NUMA node count on Windows
        ULONG highest_node = 0;
        if (GetNumaHighestNodeNumber(&highest_node)) {
            numa_node_count_ = static_cast<int>(highest_node) + 1;
        }
#endif
    }

#ifdef DENSECORE_USE_HWLOC
    hwloc_topology_t topology_;
    bool topology_initialized_ = false;
#endif

    std::vector<CoreInfo> cores_;
    int numa_node_count_ = 1;

    // Compute thread affinity state
    mutable std::mutex compute_affinity_mu_;
    std::vector<int> compute_thread_cores_;
    bool compute_affinity_configured_ = false;
    int compute_affinity_numa_node_ = -1;
};

}  // namespace densecore

#endif  // DENSECORE_HARDWARE_TOPOLOGY_H
