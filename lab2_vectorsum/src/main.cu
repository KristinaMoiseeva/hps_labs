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
                     << " (" << __FILE__ << ":" << ":" << __LINE__ << ")";     \
            throw std::runtime_error(stream__.str());                          \
        }                                                                      \
    } while (false)

namespace {

constexpr std::size_t kMinVectorSize = 1'000;
constexpr std::size_t kMaxVectorSize = 1'000'000'000;
constexpr int kThreadsPerBlock = 256;
constexpr unsigned int kSeed = 2026;

struct DeviceInfo {
    std::string name;
    int smCount{};
    int maxThreadsPerBlock{};
};

struct BenchmarkResult {
    std::size_t size{};
    int repeats{};
    long long sum{};
    double cpuMs{};
    double gpuTotalMs{};
    double gpuKernelMs{};
    double speedupTotal{};
    double speedupKernel{};
};

struct GpuBuffers {
    int* deviceInput = nullptr;
    long long* devicePartialA = nullptr;
    long long* devicePartialB = nullptr;
    std::size_t inputCapacity = 0;
    std::size_t partialCapacity = 0;
};

std::size_t parseSizeArgument(const std::string& value) {
    std::size_t processed = 0;
    const std::size_t size = std::stoull(value, &processed);
    if (processed != value.size()) {
        throw std::invalid_argument("Invalid vector size: " + value);
    }
    if (size < kMinVectorSize || size > kMaxVectorSize) {
        std::ostringstream stream;
        stream << "Vector size must be in range [" << kMinVectorSize << ", "
               << kMaxVectorSize << "], got: " << size;
        throw std::out_of_range(stream.str());
    }
    return size;
}

int parseRepeatsArgument(const std::string& value) {
    std::size_t processed = 0;
    const int repeats = std::stoi(value, &processed);
    if (processed != value.size() || repeats <= 0) {
        throw std::invalid_argument("Repeat count must be greater than 0");
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
    if (size <= 10'000'000) {
        return 30;
    }
    return 10;
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
                                std::size_t size) {
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
                                     std::size_t size) {
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

std::size_t blockCountFor(std::size_t size, int smCount) {
    const std::size_t needed =
        (size + 2ULL * kThreadsPerBlock - 1) / (2ULL * kThreadsPerBlock);

    const std::size_t capped =
        static_cast<std::size_t>(std::max(1, smCount)) * 32ULL;

    return std::max<std::size_t>(1, std::min(needed, capped));
}

void freeGpuBuffers(GpuBuffers& buffers) noexcept {
    if (buffers.deviceInput != nullptr) {
        cudaFree(buffers.deviceInput);
        buffers.deviceInput = nullptr;
    }
    if (buffers.devicePartialA != nullptr) {
        cudaFree(buffers.devicePartialA);
        buffers.devicePartialA = nullptr;
    }
    if (buffers.devicePartialB != nullptr) {
        cudaFree(buffers.devicePartialB);
        buffers.devicePartialB = nullptr;
    }
    buffers.inputCapacity = 0;
    buffers.partialCapacity = 0;
}

void ensureGpuBuffers(GpuBuffers& buffers, std::size_t vectorSize, int smCount) {
    const std::size_t initialBlocks = blockCountFor(vectorSize, smCount);

    if (buffers.inputCapacity < vectorSize) {
        if (buffers.deviceInput != nullptr) {
            CUDA_CHECK(cudaFree(buffers.deviceInput));
            buffers.deviceInput = nullptr;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.deviceInput),
                              vectorSize * sizeof(int)));
        buffers.inputCapacity = vectorSize;
    }

    if (buffers.partialCapacity < initialBlocks) {
        if (buffers.devicePartialA != nullptr) {
            CUDA_CHECK(cudaFree(buffers.devicePartialA));
            buffers.devicePartialA = nullptr;
        }
        if (buffers.devicePartialB != nullptr) {
            CUDA_CHECK(cudaFree(buffers.devicePartialB));
            buffers.devicePartialB = nullptr;
        }

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.devicePartialA),
                              initialBlocks * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.devicePartialB),
                              initialBlocks * sizeof(long long)));
        buffers.partialCapacity = initialBlocks;
    }
}

struct GpuRunResult {
    long long sum{};
    double totalMs{};
    double kernelMs{};
};

GpuRunResult sumVectorGPU(const std::vector<int>& data,
                          GpuBuffers& buffers,
                          int smCount) {
    const std::size_t size = data.size();
    if (size == 0) {
        return {};
    }

    ensureGpuBuffers(buffers, size, smCount);

    const std::size_t initialBlocks = blockCountFor(size, smCount);
    long long result = 0;

    cudaEvent_t kernelStart{};
    cudaEvent_t kernelStop{};

    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelStop));

    float kernelMsFloat = 0.0f;

    const auto totalStart = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(buffers.deviceInput, data.data(), size * sizeof(int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(kernelStart));

    reduceIntKernel<<<static_cast<unsigned int>(initialBlocks),
                      kThreadsPerBlock,
                      kThreadsPerBlock * sizeof(long long)>>>(
        buffers.deviceInput, buffers.devicePartialA, size);
    CUDA_CHECK(cudaGetLastError());

    std::size_t remaining = initialBlocks;
    long long* currentInput = buffers.devicePartialA;
    long long* currentOutput = buffers.devicePartialB;

    while (remaining > 1) {
        const std::size_t blocks = blockCountFor(remaining, smCount);

        reduceLongLongKernel<<<static_cast<unsigned int>(blocks),
                               kThreadsPerBlock,
                               kThreadsPerBlock * sizeof(long long)>>>(
            currentInput, currentOutput, remaining);
        CUDA_CHECK(cudaGetLastError());

        remaining = blocks;
        std::swap(currentInput, currentOutput);
    }

    CUDA_CHECK(cudaEventRecord(kernelStop));
    CUDA_CHECK(cudaEventSynchronize(kernelStop));
    CUDA_CHECK(cudaEventElapsedTime(&kernelMsFloat, kernelStart, kernelStop));

    CUDA_CHECK(cudaMemcpy(&result, currentInput, sizeof(long long),
                          cudaMemcpyDeviceToHost));

    const auto totalFinish = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelStop));

    return GpuRunResult{
        result,
        std::chrono::duration<double, std::milli>(totalFinish - totalStart)
            .count(),
        static_cast<double>(kernelMsFloat),
    };
}

BenchmarkResult runBenchmark(std::size_t size, int forcedRepeats, int smCount) {
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

    GpuBuffers buffers;

    try {
        ensureGpuBuffers(buffers, size, smCount);

        {
            const GpuRunResult warmup = sumVectorGPU(data, buffers, smCount);
            (void)warmup;
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        long long gpuSum = 0;
        double totalGpuTotalMs = 0.0;
        double totalGpuKernelMs = 0.0;

        for (int iteration = 0; iteration < repeats; ++iteration) {
            const GpuRunResult run = sumVectorGPU(data, buffers, smCount);
            gpuSum = run.sum;
            totalGpuTotalMs += run.totalMs;
            totalGpuKernelMs += run.kernelMs;
        }

        freeGpuBuffers(buffers);

        if (cpuSum != gpuSum) {
            std::ostringstream stream;
            stream << "CPU and GPU results do not match for size " << size
                   << ": CPU=" << cpuSum << ", GPU=" << gpuSum;
            throw std::runtime_error(stream.str());
        }

        const double gpuTotalMs =
            totalGpuTotalMs / static_cast<double>(repeats);
        const double gpuKernelMs =
            totalGpuKernelMs / static_cast<double>(repeats);

        return BenchmarkResult{
            size,
            repeats,
            cpuSum,
            cpuMs,
            gpuTotalMs,
            gpuKernelMs,
            gpuTotalMs > 0.0 ? cpuMs / gpuTotalMs : 0.0,
            gpuKernelMs > 0.0 ? cpuMs / gpuKernelMs : 0.0,
        };
    } catch (...) {
        freeGpuBuffers(buffers);
        throw;
    }
}

void printUsage(const char* executableName) {
    std::cout
        << "Usage:\n"
        << "  " << executableName
        << " [--repeats N] [size1 size2 ...]\n\n"
        << "If no sizes are provided, the default experiment set is used.\n"
        << "Allowed size range: " << kMinVectorSize << ".."
        << kMaxVectorSize << '\n';
}

DeviceInfo getDeviceInfo() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA device found.");
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

    return DeviceInfo{
        properties.name,
        properties.multiProcessorCount,
        properties.maxThreadsPerBlock,
    };
}

void printDeviceInfo(const DeviceInfo& info) {
    std::cout << "GPU: " << info.name << '\n'
              << "SM count: " << info.smCount << '\n'
              << "Threads per block: " << kThreadsPerBlock << '\n'
              << "Max threads per block: " << info.maxThreadsPerBlock << "\n\n";
}

void printResults(const std::vector<BenchmarkResult>& results) {
    std::cout
        << "| Vector size | Repeats | Sum | CPU, ms | GPU total, ms | GPU kernel, ms | Speedup total | Speedup kernel |\n"
        << "|------------:|--------:|----:|--------:|--------------:|---------------:|--------------:|---------------:|\n";

    std::cout << std::fixed << std::setprecision(3);
    for (const BenchmarkResult& result : results) {
        std::cout << "| " << result.size
                  << " | " << result.repeats
                  << " | " << result.sum
                  << " | " << result.cpuMs
                  << " | " << result.gpuTotalMs
                  << " | " << result.gpuKernelMs
                  << " | " << result.speedupTotal
                  << " | " << result.speedupKernel
                  << " |\n";
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
                        "A number must follow --repeats.");
                }
                forcedRepeats = parseRepeatsArgument(argv[++index]);
                continue;
            }

            sizes.push_back(parseSizeArgument(argument));
        }

        if (sizes.empty()) {
            sizes = defaultExperimentSizes();
        }

        const DeviceInfo deviceInfo = getDeviceInfo();
        printDeviceInfo(deviceInfo);

        std::vector<BenchmarkResult> results;
        results.reserve(sizes.size());

        for (const std::size_t size : sizes) {
            results.push_back(runBenchmark(size, forcedRepeats,
                                           deviceInfo.smCount));
        }

        printResults(results);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
