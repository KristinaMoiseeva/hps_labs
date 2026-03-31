#include "matmul.hpp"

#include <chrono>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace {

inline void check_cuda(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
  }
}

__global__ void tiled_matmul_kernel(const float* left,
                                    const float* right,
                                    float* result,
                                    int m,
                                    int k,
                                    int n,
                                    int tile_size) {
  extern __shared__ float shared[];
  float* tile_left = shared;
  float* tile_right = shared + tile_size * tile_size;

  const int global_row = blockIdx.y * blockDim.y + threadIdx.y;
  const int global_col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0.0f;
  const int tile_count = (k + tile_size - 1) / tile_size;

  for (int tile = 0; tile < tile_count; ++tile) {
    const int tiled_col = tile * tile_size + threadIdx.x;
    const int tiled_row = tile * tile_size + threadIdx.y;

    if (global_row < m && tiled_col < k) {
      tile_left[threadIdx.y * tile_size + threadIdx.x] = left[global_row * k + tiled_col];
    } else {
      tile_left[threadIdx.y * tile_size + threadIdx.x] = 0.0f;
    }

    if (tiled_row < k && global_col < n) {
      tile_right[threadIdx.y * tile_size + threadIdx.x] = right[tiled_row * n + global_col];
    } else {
      tile_right[threadIdx.y * tile_size + threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int inner = 0; inner < tile_size; ++inner) {
      sum += tile_left[threadIdx.y * tile_size + inner] *
             tile_right[inner * tile_size + threadIdx.x];
    }

    __syncthreads();
  }

  if (global_row < m && global_col < n) {
    result[global_row * n + global_col] = sum;
  }
}

}  // namespace

GpuRunResult multiply_gpu(const Matrix& left, const Matrix& right, int tile_size) {
  if (left.cols != right.rows) {
    throw std::invalid_argument("Matrix dimensions are incompatible for multiplication.");
  }
  if (tile_size <= 0) {
    throw std::invalid_argument("Tile size must be positive.");
  }
  if (tile_size > 32) {
    throw std::invalid_argument("Tile size must not exceed 32 threads per block dimension.");
  }

  GpuRunResult result;
  result.output = Matrix(left.rows, right.cols);
  result.is_available = true;
  result.message = "CUDA backend";

  const std::size_t left_size = left.data.size() * sizeof(float);
  const std::size_t right_size = right.data.size() * sizeof(float);
  const std::size_t result_size = result.output.data.size() * sizeof(float);

  float* device_left = nullptr;
  float* device_right = nullptr;
  float* device_result = nullptr;
  cudaEvent_t kernel_start = nullptr;
  cudaEvent_t kernel_end = nullptr;

  const auto total_start = std::chrono::steady_clock::now();

  try {
    check_cuda(cudaMalloc(&device_left, left_size), "cudaMalloc(left) failed");
    check_cuda(cudaMalloc(&device_right, right_size), "cudaMalloc(right) failed");
    check_cuda(cudaMalloc(&device_result, result_size), "cudaMalloc(result) failed");

    check_cuda(
        cudaMemcpy(device_left, left.data.data(), left_size, cudaMemcpyHostToDevice),
        "cudaMemcpy(left) failed");
    check_cuda(
        cudaMemcpy(device_right, right.data.data(), right_size, cudaMemcpyHostToDevice),
        "cudaMemcpy(right) failed");

    check_cuda(cudaEventCreate(&kernel_start), "cudaEventCreate(start) failed");
    check_cuda(cudaEventCreate(&kernel_end), "cudaEventCreate(end) failed");

    dim3 block(tile_size, tile_size);
    dim3 grid((static_cast<unsigned int>(right.cols) + block.x - 1) / block.x,
              (static_cast<unsigned int>(left.rows) + block.y - 1) / block.y);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(2 * tile_size * tile_size) * sizeof(float);

    check_cuda(cudaEventRecord(kernel_start), "cudaEventRecord(start) failed");
    tiled_matmul_kernel<<<grid, block, shared_bytes>>>(
        device_left,
        device_right,
        device_result,
        static_cast<int>(left.rows),
        static_cast<int>(left.cols),
        static_cast<int>(right.cols),
        tile_size);
    check_cuda(cudaGetLastError(), "Kernel launch failed");
    check_cuda(cudaEventRecord(kernel_end), "cudaEventRecord(end) failed");
    check_cuda(cudaEventSynchronize(kernel_end), "cudaEventSynchronize(end) failed");
    check_cuda(cudaMemcpy(result.output.data.data(),
                          device_result,
                          result_size,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy(result) failed");

    float kernel_ms = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_end),
        "cudaEventElapsedTime failed");
    result.kernel_time_ms = static_cast<double>(kernel_ms);
  } catch (...) {
    if (kernel_start != nullptr) {
      cudaEventDestroy(kernel_start);
    }
    if (kernel_end != nullptr) {
      cudaEventDestroy(kernel_end);
    }
    if (device_left != nullptr) {
      cudaFree(device_left);
    }
    if (device_right != nullptr) {
      cudaFree(device_right);
    }
    if (device_result != nullptr) {
      cudaFree(device_result);
    }
    throw;
  }

  check_cuda(cudaEventDestroy(kernel_start), "cudaEventDestroy(start) failed");
  check_cuda(cudaEventDestroy(kernel_end), "cudaEventDestroy(end) failed");
  check_cuda(cudaFree(device_left), "cudaFree(left) failed");
  check_cuda(cudaFree(device_right), "cudaFree(right) failed");
  check_cuda(cudaFree(device_result), "cudaFree(result) failed");

  const auto total_end = std::chrono::steady_clock::now();
  result.total_time_ms =
      std::chrono::duration<double, std::milli>(total_end - total_start).count();
  return result;
}
