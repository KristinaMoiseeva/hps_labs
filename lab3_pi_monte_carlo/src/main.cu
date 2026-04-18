#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error__ = (call);                                     \
        if (error__ != cudaSuccess) {                                           \
            std::ostringstream stream__;                                        \
            stream__ << "CUDA error: " << cudaGetErrorString(error__)           \
                     << " (" << __FILE__ << ":" << __LINE__ << ")";           \
            throw std::runtime_error(stream__.str());                           \
        }                                                                       \
    } while (false)

namespace {

constexpr std::uint64_t kMinPointCount = 1'000;
constexpr std::uint64_t kMaxPointCount = 1'000'000'000'000ULL;
constexpr int kDefaultThreadsPerBlock = 256;
constexpr unsigned long long kDefaultSeed = 2026ULL;

struct DeviceInfo {
    std::string name;
    int smCount{};
    int maxThreadsPerBlock{};
    int warpSize{};
};

struct Options {
    int forcedRepeats = 0;
    int threadsPerBlock = kDefaultThreadsPerBlock;
    unsigned long long seed = kDefaultSeed;
    std::vector<std::uint64_t> pointCounts;
};

struct BenchmarkResult {
    std::uint64_t pointCount{};
    int repeats{};
    std::uint64_t cpuInside{};
    std::uint64_t gpuInside{};
    double cpuPi{};
    double gpuPi{};
    double absDifference{};
    double cpuMs{};
    double gpuTotalMs{};
    double gpuKernelMs{};
    double speedupTotal{};
    double speedupKernel{};
};

struct GpuBuffers {
    unsigned long long* devicePartialA = nullptr;
    unsigned long long* devicePartialB = nullptr;
    double* devicePi = nullptr;
    std::uint64_t partialCapacity = 0;
};

std::uint64_t parsePointCountArgument(const std::string& value) {
    std::size_t processed = 0;
    const unsigned long long parsed = std::stoull(value, &processed);
    if (processed != value.size()) {
        throw std::invalid_argument("Invalid point count: " + value);
    }
    if (parsed < kMinPointCount || parsed > kMaxPointCount) {
        std::ostringstream stream;
        stream << "Point count must be in range [" << kMinPointCount << ", "
               << kMaxPointCount << "], got: " << parsed;
        throw std::out_of_range(stream.str());
    }
    return static_cast<std::uint64_t>(parsed);
}

int parseRepeatsArgument(const std::string& value) {
    std::size_t processed = 0;
    const int repeats = std::stoi(value, &processed);
    if (processed != value.size() || repeats <= 0) {
        throw std::invalid_argument("Repeat count must be greater than 0");
    }
    return repeats;
}

unsigned long long parseSeedArgument(const std::string& value) {
    std::size_t processed = 0;
    const unsigned long long seed = std::stoull(value, &processed);
    if (processed != value.size()) {
        throw std::invalid_argument("Invalid seed value: " + value);
    }
    return seed;
}

bool isPowerOfTwo(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

int parseThreadsArgument(const std::string& value) {
    std::size_t processed = 0;
    const int threads = std::stoi(value, &processed);
    if (processed != value.size() || !isPowerOfTwo(threads) ||
        threads < 32 || threads > 1024) {
        throw std::invalid_argument(
            "Thread count must be a power of two in range [32, 1024]");
    }
    return threads;
}

std::vector<std::uint64_t> defaultExperimentSizes() {
    return {10'000, 100'000, 1'000'000, 5'000'000, 10'000'000};
}

int chooseRepeats(std::uint64_t pointCount) {
    if (pointCount <= 100'000) {
        return 20;
    }
    if (pointCount <= 1'000'000) {
        return 10;
    }
    if (pointCount <= 10'000'000) {
        return 5;
    }
    if (pointCount <= 100'000'000) {
        return 3;
    }
    return 1;
}

double estimatePi(std::uint64_t insideCount, std::uint64_t pointCount) {
    return static_cast<double>(
        4.0L * static_cast<long double>(insideCount) /
        static_cast<long double>(pointCount));
}

std::uint64_t countInsideCircleCPU(std::uint64_t pointCount,
                                   unsigned long long seed) {
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    std::uint64_t inside = 0;
    for (std::uint64_t index = 0; index < pointCount; ++index) {
        const double x = distribution(generator);
        const double y = distribution(generator);
        if (x * x + y * y <= 1.0) {
            ++inside;
        }
    }
    return inside;
}

__global__ void countInsideCircleKernel(std::uint64_t pointCount,
                                        unsigned long long seed,
                                        unsigned long long* partialCounts) {
    extern __shared__ unsigned long long sharedCounts[];

    const unsigned int tid = threadIdx.x;
    const std::uint64_t globalThreadId =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + tid;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, globalThreadId, 0ULL, &state);

    unsigned long long localInside = 0;
    for (std::uint64_t index = globalThreadId; index < pointCount;
         index += stride) {
        const float x = curand_uniform(&state);
        const float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f) {
            ++localInside;
        }
    }

    sharedCounts[tid] = localInside;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sharedCounts[tid] += sharedCounts[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partialCounts[blockIdx.x] = sharedCounts[0];
    }
}

__global__ void reduceCountsKernel(const unsigned long long* input,
                                   unsigned long long* output,
                                   std::uint64_t size) {
    extern __shared__ unsigned long long sharedCounts[];

    const unsigned int tid = threadIdx.x;
    std::uint64_t index =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x * 2ULL + tid;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x * 2ULL;

    unsigned long long localSum = 0;
    while (index < size) {
        localSum += input[index];

        const std::uint64_t secondIndex = index + blockDim.x;
        if (secondIndex < size) {
            localSum += input[secondIndex];
        }

        index += stride;
    }

    sharedCounts[tid] = localSum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sharedCounts[tid] += sharedCounts[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sharedCounts[0];
    }
}

__global__ void computePiKernel(const unsigned long long* insideCount,
                                double* pi,
                                std::uint64_t pointCount) {
    *pi = 4.0 * static_cast<double>(*insideCount) /
          static_cast<double>(pointCount);
}

unsigned int blockCountFor(std::uint64_t workItems,
                           int threadsPerBlock,
                           int smCount) {
    const std::uint64_t needed =
        (workItems + 2ULL * threadsPerBlock - 1ULL) /
        (2ULL * threadsPerBlock);
    const std::uint64_t cap =
        static_cast<std::uint64_t>(std::max(1, smCount)) * 64ULL;
    const std::uint64_t blocks = std::max<std::uint64_t>(
        1ULL, std::min<std::uint64_t>(needed, cap));

    return static_cast<unsigned int>(blocks);
}

void freeGpuBuffers(GpuBuffers& buffers) noexcept {
    if (buffers.devicePartialA != nullptr) {
        cudaFree(buffers.devicePartialA);
        buffers.devicePartialA = nullptr;
    }
    if (buffers.devicePartialB != nullptr) {
        cudaFree(buffers.devicePartialB);
        buffers.devicePartialB = nullptr;
    }
    if (buffers.devicePi != nullptr) {
        cudaFree(buffers.devicePi);
        buffers.devicePi = nullptr;
    }
    buffers.partialCapacity = 0;
}

void ensureGpuBuffers(GpuBuffers& buffers,
                      std::uint64_t pointCount,
                      int threadsPerBlock,
                      int smCount) {
    const std::uint64_t initialBlocks =
        blockCountFor(pointCount, threadsPerBlock, smCount);

    if (buffers.partialCapacity >= initialBlocks) {
        if (buffers.devicePi == nullptr) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.devicePi),
                                  sizeof(double)));
        }
        return;
    }

    if (buffers.devicePartialA != nullptr) {
        CUDA_CHECK(cudaFree(buffers.devicePartialA));
        buffers.devicePartialA = nullptr;
    }
    if (buffers.devicePartialB != nullptr) {
        CUDA_CHECK(cudaFree(buffers.devicePartialB));
        buffers.devicePartialB = nullptr;
    }
    if (buffers.devicePi != nullptr) {
        CUDA_CHECK(cudaFree(buffers.devicePi));
        buffers.devicePi = nullptr;
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.devicePartialA),
                          initialBlocks * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.devicePartialB),
                          initialBlocks * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&buffers.devicePi),
                          sizeof(double)));
    buffers.partialCapacity = initialBlocks;
}

struct GpuRunResult {
    std::uint64_t insideCount{};
    double pi{};
    double totalMs{};
    double kernelMs{};
};

GpuRunResult countInsideCircleGPU(std::uint64_t pointCount,
                                  unsigned long long seed,
                                  GpuBuffers& buffers,
                                  int threadsPerBlock,
                                  int smCount) {
    ensureGpuBuffers(buffers, pointCount, threadsPerBlock, smCount);

    const unsigned int initialBlocks =
        blockCountFor(pointCount, threadsPerBlock, smCount);
    const std::size_t sharedBytes =
        static_cast<std::size_t>(threadsPerBlock) *
        sizeof(unsigned long long);

    cudaEvent_t kernelStart{};
    cudaEvent_t kernelStop{};
    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelStop));

    unsigned long long result = 0;
    double pi = 0.0;
    float kernelMsFloat = 0.0f;

    const auto totalStart = std::chrono::high_resolution_clock::now();

    try {
        CUDA_CHECK(cudaEventRecord(kernelStart));

        countInsideCircleKernel<<<initialBlocks, threadsPerBlock, sharedBytes>>>(
            pointCount, seed, buffers.devicePartialA);
        CUDA_CHECK(cudaGetLastError());

        std::uint64_t remaining = initialBlocks;
        unsigned long long* currentInput = buffers.devicePartialA;
        unsigned long long* currentOutput = buffers.devicePartialB;

        while (remaining > 1) {
            const unsigned int blocks =
                blockCountFor(remaining, threadsPerBlock, smCount);
            reduceCountsKernel<<<blocks, threadsPerBlock, sharedBytes>>>(
                currentInput, currentOutput, remaining);
            CUDA_CHECK(cudaGetLastError());

            remaining = blocks;
            std::swap(currentInput, currentOutput);
        }

        computePiKernel<<<1, 1>>>(currentInput, buffers.devicePi, pointCount);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(kernelStop));
        CUDA_CHECK(cudaEventSynchronize(kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelMsFloat, kernelStart,
                                        kernelStop));

        CUDA_CHECK(cudaMemcpy(&result, currentInput,
                              sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&pi, buffers.devicePi, sizeof(double),
                              cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        throw;
    }

    const auto totalFinish = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelStop));

    return GpuRunResult{
        static_cast<std::uint64_t>(result),
        pi,
        std::chrono::duration<double, std::milli>(totalFinish - totalStart)
            .count(),
        static_cast<double>(kernelMsFloat),
    };
}

BenchmarkResult runBenchmark(std::uint64_t pointCount,
                             int forcedRepeats,
                             unsigned long long seed,
                             int threadsPerBlock,
                             int smCount) {
    const int repeats = forcedRepeats > 0 ? forcedRepeats
                                          : chooseRepeats(pointCount);

    std::uint64_t cpuInside = 0;
    const auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < repeats; ++iteration) {
        cpuInside = countInsideCircleCPU(pointCount, seed);
    }
    const auto cpuFinish = std::chrono::high_resolution_clock::now();
    const double cpuMs =
        std::chrono::duration<double, std::milli>(cpuFinish - cpuStart).count() /
        static_cast<double>(repeats);

    GpuBuffers buffers;

    try {
        ensureGpuBuffers(buffers, pointCount, threadsPerBlock, smCount);

        {
            const GpuRunResult warmup =
                countInsideCircleGPU(pointCount, seed, buffers,
                                     threadsPerBlock, smCount);
            (void)warmup;
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::uint64_t gpuInside = 0;
        double gpuPi = 0.0;
        double totalGpuTotalMs = 0.0;
        double totalGpuKernelMs = 0.0;

        for (int iteration = 0; iteration < repeats; ++iteration) {
            const GpuRunResult run = countInsideCircleGPU(
                pointCount, seed, buffers, threadsPerBlock, smCount);
            gpuInside = run.insideCount;
            gpuPi = run.pi;
            totalGpuTotalMs += run.totalMs;
            totalGpuKernelMs += run.kernelMs;
        }

        freeGpuBuffers(buffers);

        const double cpuPi = estimatePi(cpuInside, pointCount);
        const double gpuTotalMs =
            totalGpuTotalMs / static_cast<double>(repeats);
        const double gpuKernelMs =
            totalGpuKernelMs / static_cast<double>(repeats);

        return BenchmarkResult{
            pointCount,
            repeats,
            cpuInside,
            gpuInside,
            cpuPi,
            gpuPi,
            std::fabs(cpuPi - gpuPi),
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
        << " [--repeats N] [--seed VALUE] [--threads N] [points1 points2 ...]\n\n"
        << "If no point counts are provided, the default experiment set is used.\n"
        << "Allowed point count range: " << kMinPointCount << ".."
        << kMaxPointCount << '\n';
}

Options parseArguments(int argc, char** argv) {
    Options options;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];

        auto requireValue = [&](const std::string& flag) -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument("A value must follow " + flag);
            }
            return argv[++index];
        };

        if (argument == "--help" || argument == "-h") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (argument == "--repeats") {
            options.forcedRepeats = parseRepeatsArgument(requireValue(argument));
            continue;
        }
        if (argument == "--seed") {
            options.seed = parseSeedArgument(requireValue(argument));
            continue;
        }
        if (argument == "--threads") {
            options.threadsPerBlock =
                parseThreadsArgument(requireValue(argument));
            continue;
        }

        options.pointCounts.push_back(parsePointCountArgument(argument));
    }

    if (options.pointCounts.empty()) {
        options.pointCounts = defaultExperimentSizes();
    }

    return options;
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
        properties.warpSize,
    };
}

void validateOptionsAgainstDevice(const Options& options,
                                  const DeviceInfo& deviceInfo) {
    if (options.threadsPerBlock > deviceInfo.maxThreadsPerBlock) {
        std::ostringstream stream;
        stream << "Requested " << options.threadsPerBlock
               << " threads per block, but device allows only "
               << deviceInfo.maxThreadsPerBlock;
        throw std::invalid_argument(stream.str());
    }
}

void printDeviceInfo(const DeviceInfo& info, int threadsPerBlock) {
    std::cout << "GPU: " << info.name << '\n'
              << "SM count: " << info.smCount << '\n'
              << "Warp size: " << info.warpSize << '\n'
              << "Threads per block: " << threadsPerBlock << '\n'
              << "Max threads per block: " << info.maxThreadsPerBlock
              << "\n\n";
}

void printResults(const std::vector<BenchmarkResult>& results) {
    std::cout
        << "| Points | Repeats | CPU inside | GPU inside | CPU pi | GPU pi | Abs diff | CPU, ms | GPU total, ms | GPU kernel, ms | Speedup total | Speedup kernel |\n"
        << "|-------:|--------:|-----------:|-----------:|-------:|-------:|---------:|--------:|--------------:|---------------:|--------------:|---------------:|\n";

    std::cout << std::fixed << std::setprecision(6);
    for (const BenchmarkResult& result : results) {
        std::cout << "| " << result.pointCount
                  << " | " << result.repeats
                  << " | " << result.cpuInside
                  << " | " << result.gpuInside
                  << " | " << result.cpuPi
                  << " | " << result.gpuPi
                  << " | " << result.absDifference
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
        const Options options = parseArguments(argc, argv);

        const DeviceInfo deviceInfo = getDeviceInfo();
        validateOptionsAgainstDevice(options, deviceInfo);
        printDeviceInfo(deviceInfo, options.threadsPerBlock);

        std::vector<BenchmarkResult> results;
        results.reserve(options.pointCounts.size());

        for (const std::uint64_t pointCount : options.pointCounts) {
            results.push_back(runBenchmark(pointCount,
                                           options.forcedRepeats,
                                           options.seed,
                                           options.threadsPerBlock,
                                           deviceInfo.smCount));
        }

        printResults(results);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
