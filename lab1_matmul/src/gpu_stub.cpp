#include "matmul.hpp"

GpuRunResult multiply_gpu(const Matrix& left, const Matrix& right, int tile_size) {
  (void)left;
  (void)right;
  (void)tile_size;

  GpuRunResult result;
  result.is_available = false;
  result.message =
      "CUDA backend is unavailable in this build. Configure the project on a machine with nvcc.";
  return result;
}
