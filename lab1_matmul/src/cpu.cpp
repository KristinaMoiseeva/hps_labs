#include "matmul.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

Matrix::Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, 0.0f) {}

float& Matrix::operator()(std::size_t row, std::size_t col) {
  return data[row * cols + col];
}

const float& Matrix::operator()(std::size_t row, std::size_t col) const {
  return data[row * cols + col];
}

Matrix generate_random_matrix(std::size_t rows,
                              std::size_t cols,
                              std::uint32_t seed,
                              float min_value,
                              float max_value) {
  Matrix matrix(rows, cols);
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(min_value, max_value);

  for (float& value : matrix.data) {
    value = distribution(generator);
  }

  return matrix;
}

static Matrix transpose(const Matrix& matrix) {
  Matrix result(matrix.cols, matrix.rows);
  for (std::size_t row = 0; row < matrix.rows; ++row) {
    for (std::size_t col = 0; col < matrix.cols; ++col) {
      result(col, row) = matrix(row, col);
    }
  }
  return result;
}

Matrix multiply_cpu(const Matrix& left, const Matrix& right) {
  if (left.cols != right.rows) {
    throw std::invalid_argument("Matrix dimensions are incompatible for multiplication.");
  }

  Matrix result(left.rows, right.cols);
  Matrix right_transposed = transpose(right);

  for (std::size_t row = 0; row < left.rows; ++row) {
    for (std::size_t col = 0; col < right.cols; ++col) {
      float sum = 0.0f;
      const std::size_t left_offset = row * left.cols;
      const std::size_t right_offset = col * right_transposed.cols;
      for (std::size_t inner = 0; inner < left.cols; ++inner) {
        sum += left.data[left_offset + inner] * right_transposed.data[right_offset + inner];
      }
      result(row, col) = sum;
    }
  }

  return result;
}

VerificationResult verify_result(const Matrix& reference,
                                 const Matrix& candidate,
                                 float abs_tolerance,
                                 float rel_tolerance) {
  if (reference.rows != candidate.rows || reference.cols != candidate.cols) {
    return {};
  }

  VerificationResult result;

  for (std::size_t index = 0; index < reference.data.size(); ++index) {
    const float ref = reference.data[index];
    const float actual = candidate.data[index];
    const float abs_error = std::fabs(ref - actual);
    const float denom = std::max(std::fabs(ref), 1e-6f);
    const float rel_error = abs_error / denom;

    result.max_abs_error = std::max(result.max_abs_error, abs_error);
    result.max_rel_error = std::max(result.max_rel_error, rel_error);
  }

  result.is_correct =
      result.max_abs_error <= abs_tolerance || result.max_rel_error <= rel_tolerance;
  return result;
}
