#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t error__ = (call);                                    \
        if (error__ != cudaSuccess) {                                          \
            std::ostringstream stream__;                                       \
            stream__ << "CUDA error: " << cudaGetErrorString(error__)          \
                     << " (" << __FILE__ << ":" << __LINE__ << ")";            \
            throw std::runtime_error(stream__.str());                          \
        }                                                                      \
    } while (false)

namespace {

constexpr std::size_t kMinVectorSize = 1'000;
constexpr std::size_t kMaxVectorSize = 1'000'000;
constexpr int kThreadsPerBlock = 256;
constexpr unsigned int kSeed = 2026;

struct BenchmarkResult {
    std::size_t size{};
    int repeats{};
    long long sum{};
    double cpuMs{};
    double gpuMs{};
    double speedup{};
};

std::size_t parseSizeArgument(const std::string& value) {
    std::size_t processed = 0;
    const std::size_t size = std::stoull(value, &processed);
    if (processed != value.size()) {
        throw std::invalid_argument("Некорректный размер вектора: " + value);
    }
    if (size < kMinVectorSize || size > kMaxVectorSize) {
        std::ostringstream stream;
        stream << "Размер вектора должен быть в диапазоне [" << kMinVectorSize
               << ", " << kMaxVectorSize << "], получено: " << size;
        throw std::out_of_range(stream.str());
    }
    return size;
}

int parseRepeatsArgument(const std::string& value) {
    std::size_t processed = 0;
    const int repeats = std::stoi(value, &processed);
    if (processed != value.size() || repeats <= 0) {
        throw std::invalid_argument("Количество повторов должно быть > 0");
    }
    return repeats;
}

std::vector<std::size_t> defaultExperimentSizes() {
    return {1'000, 10'000, 50'000, 100'000, 250'000, 500'000, 1'000'000};
}

int chooseRepeats(std::size_t size) {
    if (size <= 10'000) {
        return 500;
    }
    if (size <= 100'000) {
        return 200;
    }
    if (size <= 500'000) {
        return 100;
    }
    return 50;
}

std::vector<int> generateVector(std::size_t size) {
    std::mt19937 generator(kSeed + static_cast<unsigned int>(size));
    std::uniform_int_distribution<int> distribution(1, 100);

    std::vector<int> data(size);
    for (int& value : data) {
        value = distribution(generator);
    }

    return data;
}

long long sumVectorCPU(const std::vector<int>& data) {
    long long sum = 0;
    for (const int value : data) {
        sum += value;
    }
    return sum;
}

__global__ void reduceIntKernel(const int* input, long long* partialSums,
                                const std::size_t size) {
    extern __shared__ long long shared[];

    const unsigned int tid = threadIdx.x;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2ULL + tid;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x * 2ULL;

    long long localSum = 0;
    while (index < size) {
        localSum += input[index];

        const std::size_t secondIndex = index + blockDim.x;
        if (secondIndex < size) {
            localSum += input[secondIndex];
        }

        index += stride;
    }

    shared[tid] = localSum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partialSums[blockIdx.x] = shared[0];
    }
}

__global__ void reduceLongLongKernel(const long long* input,
                                     long long* partialSums,
                                     const std::size_t size) {
    extern __shared__ long long shared[];

    const unsigned int tid = threadIdx.x;
    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x * 2ULL + tid;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x * 2ULL;

    long long localSum = 0;
    while (index < size) {
        localSum += input[index];

        const std::size_t secondIndex = index + blockDim.x;
        if (secondIndex < size) {
            localSum += input[secondIndex];
        }

        index += stride;
    }

    shared[tid] = localSum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partialSums[blockIdx.x] = shared[0];
    }
}

std::size_t blockCountFor(std::size_t size) {
    const std::size_t denominator =
        static_cast<std::size_t>(kThreadsPerBlock) * 2ULL;
    return std::max<std::size_t>(1, (size + denominator - 1) / denominator);
}

void freeDeviceBuffers(int*& deviceInput, long long*& devicePartialA,
                       long long*& devicePartialB) noexcept {
    if (deviceInput != nullptr) {
        cudaFree(deviceInput);
        deviceInput = nullptr;
    }
    if (devicePartialA != nullptr) {
        cudaFree(devicePartialA);
        devicePartialA = nullptr;
    }
    if (devicePartialB != nullptr) {
        cudaFree(devicePartialB);
        devicePartialB = nullptr;
    }
}

long long sumVectorGPU(const std::vector<int>& data, double& elapsedMs) {
    const std::size_t size = data.size();
    if (size == 0) {
        elapsedMs = 0.0;
        return 0;
    }

    int* deviceInput = nullptr;
    long long* devicePartialA = nullptr;
    long long* devicePartialB = nullptr;

    const std::size_t initialBlocks = blockCountFor(size);

    long long result = 0;

    try {
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&deviceInput), size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePartialA),
                              initialBlocks * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devicePartialB),
                              initialBlocks * sizeof(long long)));

        const auto start = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpy(deviceInput, data.data(), size * sizeof(int),
                              cudaMemcpyHostToDevice));

        reduceIntKernel<<<static_cast<unsigned int>(initialBlocks),
                          kThreadsPerBlock,
                          kThreadsPerBlock * sizeof(long long)>>>(
            deviceInput, devicePartialA, size);
        CUDA_CHECK(cudaGetLastError());

        std::size_t remaining = initialBlocks;
        long long* currentInput = devicePartialA;
        long long* currentOutput = devicePartialB;

        while (remaining > 1) {
            const std::size_t blocks = blockCountFor(remaining);

            reduceLongLongKernel<<<static_cast<unsigned int>(blocks),
                                   kThreadsPerBlock,
                                   kThreadsPerBlock * sizeof(long long)>>>(
                currentInput, currentOutput, remaining);
            CUDA_CHECK(cudaGetLastError());

            remaining = blocks;
            std::swap(currentInput, currentOutput);
        }

        CUDA_CHECK(cudaMemcpy(&result, currentInput, sizeof(long long),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());

        const auto finish = std::chrono::high_resolution_clock::now();
        elapsedMs = std::chrono::duration<double, std::milli>(finish - start)
                        .count();
    } catch (...) {
        freeDeviceBuffers(deviceInput, devicePartialA, devicePartialB);
        throw;
    }

    freeDeviceBuffers(deviceInput, devicePartialA, devicePartialB);

    return result;
}

BenchmarkResult runBenchmark(std::size_t size, int forcedRepeats) {
    const std::vector<int> data = generateVector(size);
    const int repeats = forcedRepeats > 0 ? forcedRepeats : chooseRepeats(size);

    long long cpuSum = 0;
    const auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < repeats; ++iteration) {
        cpuSum = sumVectorCPU(data);
    }
    const auto cpuFinish = std::chrono::high_resolution_clock::now();
    const double cpuMs =
        std::chrono::duration<double, std::milli>(cpuFinish - cpuStart).count() /
        static_cast<double>(repeats);

    long long gpuSum = 0;
    double totalGpuMs = 0.0;
    for (int iteration = 0; iteration < repeats; ++iteration) {
        double singleRunGpuMs = 0.0;
        gpuSum = sumVectorGPU(data, singleRunGpuMs);
        totalGpuMs += singleRunGpuMs;
    }
    const double gpuMs = totalGpuMs / static_cast<double>(repeats);

    if (cpuSum != gpuSum) {
        std::ostringstream stream;
        stream << "Результаты CPU и GPU не совпадают для размера " << size
               << ": CPU=" << cpuSum << ", GPU=" << gpuSum;
        throw std::runtime_error(stream.str());
    }

    return BenchmarkResult{
        size,
        repeats,
        cpuSum,
        cpuMs,
        gpuMs,
        gpuMs > 0.0 ? cpuMs / gpuMs : 0.0,
    };
}

void printUsage(const char* executableName) {
    std::cout
        << "Использование:\n"
        << "  " << executableName
        << " [--repeats N] [size1 size2 ...]\n\n"
        << "Если размеры не указаны, запускается набор экспериментов по "
           "умолчанию.\n"
        << "Допустимый диапазон размеров: " << kMinVectorSize << ".."
        << kMaxVectorSize << '\n';
}

void printDeviceInfo() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("CUDA-устройство не найдено.");
    }

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    CUDA_CHECK(cudaSetDevice(0));

    std::cout << "GPU: " << properties.name << '\n'
              << "Потоков в блоке: " << kThreadsPerBlock << "\n\n";
}

void printResults(const std::vector<BenchmarkResult>& results) {
    std::cout << "| Размер вектора | Повторов | Сумма элементов | CPU, мс | GPU, мс "
                 "| Ускорение |\n"
              << "|---------------:|---------:|----------------:|--------:|--------:|"
                 "----------:|\n";

    std::cout << std::fixed << std::setprecision(3);
    for (const BenchmarkResult& result : results) {
        std::cout << "| " << result.size << " | " << result.repeats << " | "
                  << result.sum << " | " << result.cpuMs << " | "
                  << result.gpuMs << " | " << result.speedup << " |\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::size_t> sizes;
        int forcedRepeats = 0;

        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];

            if (argument == "--help" || argument == "-h") {
                printUsage(argv[0]);
                return EXIT_SUCCESS;
            }

            if (argument == "--repeats") {
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "После --repeats нужно указать число.");
                }
                forcedRepeats = parseRepeatsArgument(argv[++index]);
                continue;
            }

            sizes.push_back(parseSizeArgument(argument));
        }

        if (sizes.empty()) {
            sizes = defaultExperimentSizes();
        }

        printDeviceInfo();

        std::vector<BenchmarkResult> results;
        results.reserve(sizes.size());
        for (const std::size_t size : sizes) {
            results.push_back(runBenchmark(size, forcedRepeats));
        }

        printResults(results);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Ошибка: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
