#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<float> data;

  Matrix() = default;
  Matrix(std::size_t r, std::size_t c);

  float& operator()(std::size_t row, std::size_t col);
  const float& operator()(std::size_t row, std::size_t col) const;
};

struct VerificationResult {
  bool is_correct = false;
  float max_abs_error = 0.0f;
  float max_rel_error = 0.0f;
};

struct GpuRunResult {
  Matrix output;
  double total_time_ms = 0.0;
  double kernel_time_ms = 0.0;
  bool is_available = false;
  std::string message;
};

struct BenchmarkRow {
  std::size_t m = 0;
  std::size_t k = 0;
  std::size_t n = 0;
  double cpu_time_ms = 0.0;
  double gpu_total_time_ms = 0.0;
  double gpu_kernel_time_ms = 0.0;
  double speedup = 0.0;
  VerificationResult verification;
  bool gpu_available = false;
};

Matrix generate_random_matrix(std::size_t rows,
                              std::size_t cols,
                              std::uint32_t seed,
                              float min_value = -1.0f,
                              float max_value = 1.0f);

Matrix multiply_cpu(const Matrix& left, const Matrix& right);

GpuRunResult multiply_gpu(const Matrix& left, const Matrix& right, int tile_size);

VerificationResult verify_result(const Matrix& reference,
                                 const Matrix& candidate,
                                 float abs_tolerance = 1e-3f,
                                 float rel_tolerance = 1e-3f);
