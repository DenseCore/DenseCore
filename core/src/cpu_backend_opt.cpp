/**
 * @file cpu_backend_opt.cpp
 * @brief Specialized HPC optimized kernels (AMX, SVE) for CpuBackend
 */

#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sys/syscall.h>
#include <vector>

#include "cpu_backend.h"
#include "simd_platform.h"

// =========================================================================
// Intel AMX Implementation (x86_64)
// =========================================================================
#if defined(__x86_64__) || defined(_M_X64)

#define XFEATURE_XTILEDATA 18
#define ARCH_REQ_XCOMP_PERM 0x1023

// Define tile config struct if not ready in headers
struct tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

static bool RequestAMXPermission() {
#if defined(__linux__)
    // Request permission to use AMX (XTILEDATA)
    long ret = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    return (ret == 0);
#else
    return false;
#endif
}

static void ConfigTiles(int rows, int col_bytes, int K_bytes)
    __attribute__((target("amx-tile,amx-int8,amx-bf16")));

static void ConfigTiles(int rows, int col_bytes, int K_bytes) {
    // 16 tiles configuration
    // We normally use 3 tiles: 1 for dest (C), 2 for sources (A, B)
    // TMM0: Accumulator (M x N)
    // TMM1: Src1 (M x K)
    // TMM2: Src2 (K x N)

    tile_config cfg = {0};
    cfg.palette_id = 1;
    cfg.start_row = 0;

    // TMM0: C (M rows, N*4 bytes) - float32 output (4 bytes)
    cfg.rows[0] = rows;
    cfg.colsb[0] = col_bytes;

    // TMM1: A (M rows, K_bytes)
    cfg.rows[1] = rows;
    cfg.colsb[1] = K_bytes;

    // TMM2: B (K rows, N*4 bytes? or packed?)
    // This depends on the specific kernel layout (matrix B usually needs
    // VNNI-style packing) For simplicity of this task, we set generic config.
    cfg.rows[2] = rows;  // Placeholder
    cfg.colsb[2] = col_bytes;

    _tile_loadconfig(&cfg);
}

void densecore::CpuBackend::MatMulAMX(const TensorView& A, const TensorView& B, TensorView& C) {
#if defined(__AMX_TILE__) && defined(__AMX_BF16__)
    // 1. Request OS Permission (once)
    static bool amx_ready = RequestAMXPermission();
    if (!amx_ready)
        return;

    // 2. Configure Tiles
    // Define a 16x16 block size
    const int M = 16;
    const int N = 16;
    const int K_bf16_pairs = 32;  // 32 pairs = 64 bytes

    ConfigTiles(M, N * 4, K_bf16_pairs * 2);

    _tile_zero(0);  // TMM0 = 0

    // Load A (M x K) to TMM1
    // Use TensorView stride (bytes) directly
    // This assumes stride[0] is the row stride in bytes
    _tile_loadd(1, A.data, A.strides[0]);

    // Load B (K x N) into TMM2
    _tile_loadd(2, B.data, B.strides[0]);

    // Compute: Dot Product BF16 -> FP32
    _tile_dpbf16ps(0, 1, 2);

    // Store C
    _tile_stored(0, C.data, C.strides[0]);

    _tile_release();
#else
    (void)A;
    (void)B;
    (void)C;
#endif
}

#else
// Non-x86 Stub
void densecore::CpuBackend::MatMulAMX(const TensorView& A, const TensorView& B, TensorView& C) {
    (void)A;
    (void)B;
    (void)C;
}
#endif

// =========================================================================
// ARM SVE Implementation (ARM64)
// =========================================================================
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>

void densecore::CpuBackend::MatMulSVE(const TensorView& A, const TensorView& B, TensorView& C) {
    int M = A.shape[0];
    int K = A.shape[1];
    int N = B.shape[1];

    float* ptr_A = static_cast<float*>(A.data);
    float* ptr_B = static_cast<float*>(B.data);
    float* ptr_C = static_cast<float*>(C.data);

    // Stride in ELEMENTS (SVE intrinsics often work with element pointers,
    // unless using gather/scatter with byte offsets)
    // svld1 expects a pointer.
    // If memory is contiguous, we advance pointers.
    // If strided, we need to be careful.
    // Assuming row-major contiguous within rows for now, or using stride to
    // advance rows. TensorView strides are in bytes.
    long stride_A_bytes = A.strides[0];
    long stride_B_bytes = B.strides[0];
    long stride_C_bytes = C.strides[0];

    // VLA: Get vector length for float32
    uint64_t vl = svcntw();

    for (int i = 0; i < M; ++i) {
        char* row_A_ptr = reinterpret_cast<char*>(ptr_A) + i * stride_A_bytes;
        char* row_C_ptr = reinterpret_cast<char*>(ptr_C) + i * stride_C_bytes;

        for (int k = 0; k < K; ++k) {
            // A[i, k]
            float val_a = *(reinterpret_cast<float*>(row_A_ptr) + k);  // Assuming dense row?
            // If A is row-major dense, then A.strides[1] (element stride) should be
            // sizeof(float) Ideally: val_a = *reinterpret_cast<float*>(row_A_ptr + k
            // * A.strides[1]);

            svfloat32_t vec_a = svdup_f32(val_a);

            // Vectorized loop over N
            int j = 0;
            svbool_t pg = svwhilelt_b32(j, N);

            char* row_B_ptr = reinterpret_cast<char*>(ptr_B) + k * stride_B_bytes;

            while (svptest_any(svptrue_b32(), pg)) {
                // Load B[k, j]
                // B is row-major?
                // ptr_B[k * B_stride + j]
                // Using byte ptrs:
                float* pB = reinterpret_cast<float*>(row_B_ptr + j * sizeof(float));
                svfloat32_t vec_b = svld1_f32(pg, pB);

                // Load C[i, j]
                float* pC = reinterpret_cast<float*>(row_C_ptr + j * sizeof(float));
                svfloat32_t vec_c = svld1_f32(pg, pC);

                // FMA: C = C + A * B
                vec_c = svmla_f32_m(pg, vec_c, vec_a, vec_b);

                // Store C back
                svst1_f32(pg, pC, vec_c);

                j += vl;
                pg = svwhilelt_b32(j, N);
            }
        }
    }
}
#else
// Non-SVE Stub
void densecore::CpuBackend::MatMulSVE(const TensorView& A, const TensorView& B, TensorView& C) {
    (void)A;
    (void)B;
    (void)C;
}
#endif
