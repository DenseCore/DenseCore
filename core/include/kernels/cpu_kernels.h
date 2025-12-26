#ifndef DENSECORE_CPU_KERNELS_H
#define DENSECORE_CPU_KERNELS_H

namespace densecore {

// AMX Matrix Multiplication (BF16)
// A: [M, K], B: [K, N], C: [M, N]
void MatMulAMX_BF16(const float* A, const float* B, float* C, int M, int K, int N);

}  // namespace densecore

#endif  // DENSECORE_CPU_KERNELS_H
