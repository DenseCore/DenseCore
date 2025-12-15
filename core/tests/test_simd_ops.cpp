/**
 * @file test_simd_ops.cpp
 * @brief Unit tests for SIMD operations
 */

#include "simd_ops.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace densecore::simd;

// =============================================================================
// SIMD Detection Tests
// =============================================================================

TEST(SimdOps, DetectSimdLevel) {
  SimdLevel level = DetectSimdLevel();
  // Should detect at least NONE or a valid SIMD level
  EXPECT_GE(static_cast<int>(level), static_cast<int>(SimdLevel::NONE));

  // Name should not be null
  const char *name = SimdLevelName(level);
  EXPECT_NE(name, nullptr);
  EXPECT_GT(strlen(name), 0);
}

TEST(SimdOps, GetNumCores) {
  int cores = GetNumCores();
  EXPECT_GT(cores, 0);
}

// =============================================================================
// Float32 Operations Tests
// =============================================================================

TEST(SimdOps, ScaleF32_Basic) {
  std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> dst(src.size());

  ScaleF32(dst.data(), src.data(), 2.0f, src.size());

  for (size_t i = 0; i < src.size(); i++) {
    EXPECT_FLOAT_EQ(dst[i], src[i] * 2.0f);
  }
}

TEST(SimdOps, ScaleF32_LargeArray) {
  constexpr size_t N = 1024;
  std::vector<float> src(N);
  std::vector<float> dst(N);

  for (size_t i = 0; i < N; i++) {
    src[i] = static_cast<float>(i);
  }

  ScaleF32(dst.data(), src.data(), 0.5f, N);

  for (size_t i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(dst[i], src[i] * 0.5f);
  }
}

TEST(SimdOps, AddF32_Basic) {
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> b = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  std::vector<float> c(a.size());

  AddF32(c.data(), a.data(), b.data(), a.size());

  for (size_t i = 0; i < a.size(); i++) {
    EXPECT_FLOAT_EQ(c[i], 9.0f); // All sums should be 9
  }
}

TEST(SimdOps, DotF32_Basic) {
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> b = {1.0f, 1.0f, 1.0f, 1.0f};

  float result = DotF32(a.data(), b.data(), a.size());

  EXPECT_FLOAT_EQ(result, 10.0f); // 1+2+3+4 = 10
}

TEST(SimdOps, DotF32_LargeArray) {
  constexpr size_t N = 1024;
  std::vector<float> a(N, 1.0f);
  std::vector<float> b(N, 2.0f);

  float result = DotF32(a.data(), b.data(), N);

  EXPECT_FLOAT_EQ(result, static_cast<float>(N * 2)); // 1024 * 2 = 2048
}

TEST(SimdOps, MaxF32_Basic) {
  std::vector<float> a = {1.0f, 5.0f, 3.0f, 9.0f, 2.0f, 8.0f, 4.0f, 7.0f};

  float result = MaxF32(a.data(), a.size());

  EXPECT_FLOAT_EQ(result, 9.0f);
}

TEST(SimdOps, MaxF32_NegativeValues) {
  std::vector<float> a = {-5.0f, -2.0f, -8.0f, -1.0f, -10.0f};

  float result = MaxF32(a.data(), a.size());

  EXPECT_FLOAT_EQ(result, -1.0f);
}

TEST(SimdOps, SumF32_Basic) {
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  float result = SumF32(a.data(), a.size());

  EXPECT_FLOAT_EQ(result, 36.0f); // 1+2+...+8 = 36
}

TEST(SimdOps, SoftmaxF32_Basic) {
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};

  SoftmaxF32(a.data(), a.size());

  // Sum should be 1.0
  float sum = SumF32(a.data(), a.size());
  EXPECT_NEAR(sum, 1.0f, 1e-5);

  // All values should be positive
  for (float v : a) {
    EXPECT_GT(v, 0.0f);
  }

  // Values should be in increasing order (since input was increasing)
  for (size_t i = 1; i < a.size(); i++) {
    EXPECT_GT(a[i], a[i - 1]);
  }
}

// =============================================================================
// Embedding Operations Tests
// =============================================================================

TEST(SimdOps, NormalizeL2_Basic) {
  std::vector<float> v = {3.0f, 4.0f}; // 3-4-5 triangle

  NormalizeL2(v.data(), v.size());

  // Check unit length
  float norm = std::sqrt(DotF32(v.data(), v.data(), v.size()));
  EXPECT_NEAR(norm, 1.0f, 1e-5);

  // Check values
  EXPECT_NEAR(v[0], 0.6f, 1e-5);
  EXPECT_NEAR(v[1], 0.8f, 1e-5);
}

TEST(SimdOps, NormalizeL2_ZeroVector) {
  std::vector<float> v = {0.0f, 0.0f, 0.0f, 0.0f};

  NormalizeL2(v.data(), v.size());

  // Should remain zero (avoid NaN)
  for (float val : v) {
    EXPECT_FLOAT_EQ(val, 0.0f);
  }
}

TEST(SimdOps, MeanPool_Basic) {
  constexpr int seq_len = 4;
  constexpr int hidden_dim = 8;

  std::vector<float> input(seq_len * hidden_dim, 1.0f);
  std::vector<float> output(hidden_dim);

  MeanPool(input.data(), output.data(), seq_len, hidden_dim);

  for (int d = 0; d < hidden_dim; d++) {
    EXPECT_FLOAT_EQ(output[d], 1.0f);
  }
}

TEST(SimdOps, MeanPool_Varying) {
  constexpr int seq_len = 2;
  constexpr int hidden_dim = 4;

  // Row 0: [1,2,3,4], Row 1: [3,4,5,6]
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> output(hidden_dim);

  MeanPool(input.data(), output.data(), seq_len, hidden_dim);

  EXPECT_FLOAT_EQ(output[0], 2.0f); // (1+3)/2
  EXPECT_FLOAT_EQ(output[1], 3.0f); // (2+4)/2
  EXPECT_FLOAT_EQ(output[2], 4.0f); // (3+5)/2
  EXPECT_FLOAT_EQ(output[3], 5.0f); // (4+6)/2
}

TEST(SimdOps, MaxPool_Basic) {
  constexpr int seq_len = 3;
  constexpr int hidden_dim = 4;

  std::vector<float> input = {1.0f, 5.0f, 2.0f, 8.0f, 3.0f, 2.0f,
                              7.0f, 4.0f, 9.0f, 1.0f, 3.0f, 6.0f};
  std::vector<float> output(hidden_dim);

  MaxPool(input.data(), output.data(), seq_len, hidden_dim);

  EXPECT_FLOAT_EQ(output[0], 9.0f);
  EXPECT_FLOAT_EQ(output[1], 5.0f);
  EXPECT_FLOAT_EQ(output[2], 7.0f);
  EXPECT_FLOAT_EQ(output[3], 8.0f);
}

TEST(SimdOps, ClsPool_Basic) {
  constexpr int hidden_dim = 4;

  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, // CLS token
                              5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> output(hidden_dim);

  ClsPool(input.data(), output.data(), hidden_dim);

  EXPECT_FLOAT_EQ(output[0], 1.0f);
  EXPECT_FLOAT_EQ(output[1], 2.0f);
  EXPECT_FLOAT_EQ(output[2], 3.0f);
  EXPECT_FLOAT_EQ(output[3], 4.0f);
}

TEST(SimdOps, LastPool_Basic) {
  constexpr int seq_len = 3;
  constexpr int hidden_dim = 4;

  std::vector<float> input = {1.0f, 2.0f, 3.0f,  4.0f,  5.0f, 6.0f, 7.0f,
                              8.0f, 9.0f, 10.0f, 11.0f, 12.0f}; // Last token
  std::vector<float> output(hidden_dim);

  LastPool(input.data(), output.data(), seq_len, hidden_dim);

  EXPECT_FLOAT_EQ(output[0], 9.0f);
  EXPECT_FLOAT_EQ(output[1], 10.0f);
  EXPECT_FLOAT_EQ(output[2], 11.0f);
  EXPECT_FLOAT_EQ(output[3], 12.0f);
}

TEST(SimdOps, CosineSimilarity_Identical) {
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};

  float result = CosineSimilarity(a.data(), a.data(), a.size());

  EXPECT_NEAR(result, 1.0f, 1e-5);
}

TEST(SimdOps, CosineSimilarity_Orthogonal) {
  std::vector<float> a = {1.0f, 0.0f};
  std::vector<float> b = {0.0f, 1.0f};

  float result = CosineSimilarity(a.data(), b.data(), a.size());

  EXPECT_NEAR(result, 0.0f, 1e-5);
}

TEST(SimdOps, CosineSimilarity_Opposite) {
  std::vector<float> a = {1.0f, 0.0f};
  std::vector<float> b = {-1.0f, 0.0f};

  float result = CosineSimilarity(a.data(), b.data(), a.size());

  EXPECT_NEAR(result, -1.0f, 1e-5);
}

// =============================================================================
// Copy Operations Tests
// =============================================================================

TEST(SimdOps, CopyF32_Basic) {
  std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> dst(src.size());

  CopyF32(dst.data(), src.data(), src.size());

  for (size_t i = 0; i < src.size(); i++) {
    EXPECT_FLOAT_EQ(dst[i], src[i]);
  }
}

TEST(SimdOps, SimdCopy_Large) {
  constexpr size_t N = 4096;
  std::vector<char> src(N);
  std::vector<char> dst(N);

  for (size_t i = 0; i < N; i++) {
    src[i] = static_cast<char>(i % 256);
  }

  SimdCopy(dst.data(), src.data(), N);

  for (size_t i = 0; i < N; i++) {
    EXPECT_EQ(dst[i], src[i]);
  }
}
