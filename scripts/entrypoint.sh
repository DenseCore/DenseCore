#!/bin/bash
# =============================================================================
# DenseCore Entrypoint Script
# =============================================================================
# Detects Cgroup CPU limit and configures OMP thread count for optimal
# performance, reserving 1 core for the Go runtime.
# =============================================================================
set -e

# -----------------------------------------------------------------------------
# detect_cpu_limit: Read CPU quota from Cgroup v2 or v1
# Returns the number of CPUs allocated (or nproc as fallback)
# -----------------------------------------------------------------------------
detect_cpu_limit() {
    local limit

    # Cgroup v2 (modern kernels, default in K8s 1.25+)
    if [[ -f /sys/fs/cgroup/cpu.max ]]; then
        local quota period
        read -r quota period < /sys/fs/cgroup/cpu.max
        if [[ "$quota" != "max" && "$period" -gt 0 ]]; then
            limit=$((quota / period))
        fi
    # Cgroup v1 fallback
    elif [[ -f /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
        local quota period
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        if [[ "$quota" -gt 0 && "$period" -gt 0 ]]; then
            limit=$((quota / period))
        fi
    fi

    # Clamp to minimum of 1, default to nproc if no limit detected
    if [[ -z "$limit" || "$limit" -lt 1 ]]; then
        limit=$(nproc)
    fi

    echo "$limit"
}

# -----------------------------------------------------------------------------
# Main Entrypoint Logic
# -----------------------------------------------------------------------------
CPU_LIMIT=$(detect_cpu_limit)

# Reserve 1 core for Go runtime scheduler, minimum 1 OMP thread
if [[ "$CPU_LIMIT" -gt 1 ]]; then
    OMP_THREADS=$((CPU_LIMIT - 1))
else
    OMP_THREADS=1
fi

# Allow environment override, otherwise use computed value
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$OMP_THREADS}"
export GOMAXPROCS="${GOMAXPROCS:-$CPU_LIMIT}"

# OpenMP thread binding for NUMA awareness (optional, can be overridden)
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

echo "[entrypoint] CPU limit detected: ${CPU_LIMIT} cores"
echo "[entrypoint] OMP_NUM_THREADS=${OMP_NUM_THREADS}, GOMAXPROCS=${GOMAXPROCS}"
echo "[entrypoint] OMP_PROC_BIND=${OMP_PROC_BIND}, OMP_PLACES=${OMP_PLACES}"

# Execute the main binary with any passed arguments
exec /app/densecore-server "$@"
