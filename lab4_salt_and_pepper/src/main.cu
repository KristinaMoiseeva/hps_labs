#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
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

constexpr int kDefaultRepeats = 20;
constexpr int kDefaultWidth = 256;
constexpr int kDefaultHeight = 256;
constexpr double kDefaultNoiseProbability = 0.08;
constexpr unsigned int kDefaultSeed = 2026;
constexpr int kBlockWidth = 16;
constexpr int kBlockHeight = 16;

struct Image {
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

struct DeviceInfo {
    std::string name;
    int smCount{};
    int maxThreadsPerBlock{};
};

struct Options {
    std::string inputPath = "data/noisy_256.bmp";
    std::string outputPath = "filtered.bmp";
    std::string generatePath;
    int repeats = kDefaultRepeats;
    int generateWidth = kDefaultWidth;
    int generateHeight = kDefaultHeight;
    double noiseProbability = kDefaultNoiseProbability;
    unsigned int seed = kDefaultSeed;
};

struct BenchmarkResult {
    std::string inputPath;
    std::string outputPath;
    int width{};
    int height{};
    int repeats{};
    double cpuMs{};
    double gpuTotalMs{};
    double gpuKernelMs{};
    double speedupTotal{};
    double speedupKernel{};
    std::size_t mismatchedPixels{};
    int maxAbsDiff{};
};

struct GpuResources {
    cudaArray_t inputArray = nullptr;
    unsigned char* deviceOutput = nullptr;
    cudaTextureObject_t texture = 0;
    int width = 0;
    int height = 0;
};

std::uint16_t readU16(const std::vector<unsigned char>& data,
                      std::size_t offset) {
    if (offset + 2 > data.size()) {
        throw std::runtime_error("Unexpected end of BMP file.");
    }
    return static_cast<std::uint16_t>(data[offset]) |
           (static_cast<std::uint16_t>(data[offset + 1]) << 8);
}

std::uint32_t readU32(const std::vector<unsigned char>& data,
                      std::size_t offset) {
    if (offset + 4 > data.size()) {
        throw std::runtime_error("Unexpected end of BMP file.");
    }
    return static_cast<std::uint32_t>(data[offset]) |
           (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(data[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(data[offset + 3]) << 24);
}

std::int32_t readI32(const std::vector<unsigned char>& data,
                     std::size_t offset) {
    return static_cast<std::int32_t>(readU32(data, offset));
}

void writeU16(std::ostream& output, std::uint16_t value) {
    output.put(static_cast<char>(value & 0xFF));
    output.put(static_cast<char>((value >> 8) & 0xFF));
}

void writeU32(std::ostream& output, std::uint32_t value) {
    output.put(static_cast<char>(value & 0xFF));
    output.put(static_cast<char>((value >> 8) & 0xFF));
    output.put(static_cast<char>((value >> 16) & 0xFF));
    output.put(static_cast<char>((value >> 24) & 0xFF));
}

int clampInt(int value, int low, int high) {
    return std::max(low, std::min(value, high));
}

unsigned char rgbToGray(unsigned char red,
                        unsigned char green,
                        unsigned char blue) {
    const double gray = 0.299 * red + 0.587 * green + 0.114 * blue;
    return static_cast<unsigned char>(std::round(gray));
}

Image readBmpGray(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open input BMP: " + path);
    }

    std::vector<unsigned char> data(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());

    if (data.size() < 54 || data[0] != 'B' || data[1] != 'M') {
        throw std::runtime_error("Input file is not a BMP image: " + path);
    }

    const std::uint32_t pixelOffset = readU32(data, 10);
    const std::uint32_t dibSize = readU32(data, 14);
    if (dibSize < 40) {
        throw std::runtime_error("Unsupported BMP DIB header.");
    }

    const std::int32_t widthSigned = readI32(data, 18);
    const std::int32_t heightSigned = readI32(data, 22);
    const std::uint16_t planes = readU16(data, 26);
    const std::uint16_t bitsPerPixel = readU16(data, 28);
    const std::uint32_t compression = readU32(data, 30);

    if (planes != 1 || widthSigned <= 0 || heightSigned == 0) {
        throw std::runtime_error("Invalid BMP dimensions or plane count.");
    }
    if (compression != 0) {
        throw std::runtime_error("Compressed BMP files are not supported.");
    }
    if (bitsPerPixel != 8 && bitsPerPixel != 24 && bitsPerPixel != 32) {
        throw std::runtime_error("Only 8-bit, 24-bit and 32-bit BMP files are supported.");
    }

    const int width = widthSigned;
    const int height = std::abs(heightSigned);
    const bool topDown = heightSigned < 0;
    const std::size_t rowStride =
        ((static_cast<std::size_t>(width) * bitsPerPixel + 31U) / 32U) * 4U;
    const std::size_t requiredSize =
        static_cast<std::size_t>(pixelOffset) + rowStride * height;

    if (requiredSize > data.size()) {
        throw std::runtime_error("BMP pixel data is truncated.");
    }

    Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<std::size_t>(width) * height);

    for (int row = 0; row < height; ++row) {
        const int sourceRow = topDown ? row : (height - 1 - row);
        const std::size_t sourceOffset =
            static_cast<std::size_t>(pixelOffset) + rowStride * sourceRow;

        for (int col = 0; col < width; ++col) {
            unsigned char gray = 0;

            if (bitsPerPixel == 8) {
                gray = data[sourceOffset + col];
            } else if (bitsPerPixel == 24) {
                const std::size_t pixel = sourceOffset + 3ULL * col;
                const unsigned char blue = data[pixel + 0];
                const unsigned char green = data[pixel + 1];
                const unsigned char red = data[pixel + 2];
                gray = rgbToGray(red, green, blue);
            } else {
                const std::size_t pixel = sourceOffset + 4ULL * col;
                const unsigned char blue = data[pixel + 0];
                const unsigned char green = data[pixel + 1];
                const unsigned char red = data[pixel + 2];
                gray = rgbToGray(red, green, blue);
            }

            image.pixels[static_cast<std::size_t>(row) * width + col] = gray;
        }
    }

    return image;
}

void writeBmpGray8(const std::string& path, const Image& image) {
    if (image.width <= 0 || image.height <= 0 ||
        image.pixels.size() != static_cast<std::size_t>(image.width) *
                                  static_cast<std::size_t>(image.height)) {
        throw std::runtime_error("Invalid image passed to BMP writer.");
    }

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open output BMP: " + path);
    }

    const std::uint32_t rowStride =
        static_cast<std::uint32_t>((image.width + 3) / 4 * 4);
    const std::uint32_t pixelDataSize = rowStride * image.height;
    const std::uint32_t paletteSize = 256U * 4U;
    const std::uint32_t pixelOffset = 14U + 40U + paletteSize;
    const std::uint32_t fileSize = pixelOffset + pixelDataSize;

    output.put('B');
    output.put('M');
    writeU32(output, fileSize);
    writeU16(output, 0);
    writeU16(output, 0);
    writeU32(output, pixelOffset);

    writeU32(output, 40);
    writeU32(output, static_cast<std::uint32_t>(image.width));
    writeU32(output, static_cast<std::uint32_t>(image.height));
    writeU16(output, 1);
    writeU16(output, 8);
    writeU32(output, 0);
    writeU32(output, pixelDataSize);
    writeU32(output, 2835);
    writeU32(output, 2835);
    writeU32(output, 256);
    writeU32(output, 0);

    for (int value = 0; value < 256; ++value) {
        output.put(static_cast<char>(value));
        output.put(static_cast<char>(value));
        output.put(static_cast<char>(value));
        output.put('\0');
    }

    std::vector<unsigned char> row(rowStride, 0);
    for (int rowIndex = image.height - 1; rowIndex >= 0; --rowIndex) {
        const std::size_t sourceOffset =
            static_cast<std::size_t>(rowIndex) * image.width;
        std::copy(image.pixels.begin() + sourceOffset,
                  image.pixels.begin() + sourceOffset + image.width,
                  row.begin());
        output.write(reinterpret_cast<const char*>(row.data()), row.size());
        std::fill(row.begin(), row.end(), 0);
    }
}

Image generateSaltAndPepperImage(int width,
                                 int height,
                                 double noiseProbability,
                                 unsigned int seed) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Generated image dimensions must be positive.");
    }
    if (noiseProbability < 0.0 || noiseProbability > 1.0) {
        throw std::invalid_argument("Noise probability must be in range [0, 1].");
    }

    Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<std::size_t>(width) * height);

    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> probability(0.0, 1.0);
    std::bernoulli_distribution saltOrPepper(0.5);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double wave =
                32.0 * std::sin(static_cast<double>(x) * 0.08) +
                24.0 * std::cos(static_cast<double>(y) * 0.06);
            const int checker = ((x / 32 + y / 32) % 2) * 32;
            int value = static_cast<int>(
                96.0 + 96.0 * static_cast<double>(x) / std::max(1, width - 1) +
                wave + checker);
            value = clampInt(value, 0, 255);

            if (probability(generator) < noiseProbability) {
                value = saltOrPepper(generator) ? 255 : 0;
            }

            image.pixels[static_cast<std::size_t>(y) * width + x] =
                static_cast<unsigned char>(value);
        }
    }

    return image;
}

unsigned char median9(unsigned char values[9]) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8 - i; ++j) {
            if (values[j] > values[j + 1]) {
                const unsigned char temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
    return values[4];
}

Image medianFilterCPU(const Image& input) {
    Image output;
    output.width = input.width;
    output.height = input.height;
    output.pixels.resize(input.pixels.size());

    for (int y = 0; y < input.height; ++y) {
        for (int x = 0; x < input.width; ++x) {
            unsigned char window[9];
            int index = 0;

            for (int dy = -1; dy <= 1; ++dy) {
                const int yy = clampInt(y + dy, 0, input.height - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = clampInt(x + dx, 0, input.width - 1);
                    window[index++] =
                        input.pixels[static_cast<std::size_t>(yy) *
                                     input.width + xx];
                }
            }

            output.pixels[static_cast<std::size_t>(y) * input.width + x] =
                median9(window);
        }
    }

    return output;
}

__device__ int clampDevice(int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

__device__ unsigned char median9Device(unsigned char values[9]) {
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8 - i; ++j) {
            if (values[j] > values[j + 1]) {
                const unsigned char temp = values[j];
                values[j] = values[j + 1];
                values[j + 1] = temp;
            }
        }
    }
    return values[4];
}

__global__ void medianFilterTextureKernel(cudaTextureObject_t inputTexture,
                                          unsigned char* output,
                                          int width,
                                          int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    unsigned char window[9];
    int index = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = clampDevice(y + dy, 0, height - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = clampDevice(x + dx, 0, width - 1);
            window[index++] = tex2D<unsigned char>(
                inputTexture,
                static_cast<float>(xx) + 0.5f,
                static_cast<float>(yy) + 0.5f);
        }
    }

    output[static_cast<std::size_t>(y) * width + x] = median9Device(window);
}

void freeGpuResources(GpuResources& resources) noexcept {
    if (resources.texture != 0) {
        cudaDestroyTextureObject(resources.texture);
        resources.texture = 0;
    }
    if (resources.inputArray != nullptr) {
        cudaFreeArray(resources.inputArray);
        resources.inputArray = nullptr;
    }
    if (resources.deviceOutput != nullptr) {
        cudaFree(resources.deviceOutput);
        resources.deviceOutput = nullptr;
    }
    resources.width = 0;
    resources.height = 0;
}

void ensureGpuResources(GpuResources& resources, int width, int height) {
    if (resources.inputArray != nullptr &&
        resources.deviceOutput != nullptr &&
        resources.texture != 0 &&
        resources.width == width &&
        resources.height == height) {
        return;
    }

    freeGpuResources(resources);

    cudaChannelFormatDesc channelDescription =
        cudaCreateChannelDesc<unsigned char>();
    CUDA_CHECK(cudaMallocArray(&resources.inputArray,
                               &channelDescription,
                               width,
                               height));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&resources.deviceOutput),
                          static_cast<std::size_t>(width) * height *
                              sizeof(unsigned char)));

    cudaResourceDesc resourceDescription;
    std::memset(&resourceDescription, 0, sizeof(resourceDescription));
    resourceDescription.resType = cudaResourceTypeArray;
    resourceDescription.res.array.array = resources.inputArray;

    cudaTextureDesc textureDescription;
    std::memset(&textureDescription, 0, sizeof(textureDescription));
    textureDescription.addressMode[0] = cudaAddressModeClamp;
    textureDescription.addressMode[1] = cudaAddressModeClamp;
    textureDescription.filterMode = cudaFilterModePoint;
    textureDescription.readMode = cudaReadModeElementType;
    textureDescription.normalizedCoords = 0;

    CUDA_CHECK(cudaCreateTextureObject(&resources.texture,
                                       &resourceDescription,
                                       &textureDescription,
                                       nullptr));

    resources.width = width;
    resources.height = height;
}

struct GpuRunResult {
    Image output;
    double totalMs{};
    double kernelMs{};
};

GpuRunResult medianFilterGPU(const Image& input, GpuResources& resources) {
    ensureGpuResources(resources, input.width, input.height);

    GpuRunResult result;
    result.output.width = input.width;
    result.output.height = input.height;
    result.output.pixels.resize(input.pixels.size());

    cudaEvent_t kernelStart{};
    cudaEvent_t kernelStop{};
    CUDA_CHECK(cudaEventCreate(&kernelStart));
    CUDA_CHECK(cudaEventCreate(&kernelStop));

    float kernelMsFloat = 0.0f;
    const auto totalStart = std::chrono::high_resolution_clock::now();

    try {
        CUDA_CHECK(cudaMemcpy2DToArray(resources.inputArray,
                                       0,
                                       0,
                                       input.pixels.data(),
                                       static_cast<std::size_t>(input.width),
                                       static_cast<std::size_t>(input.width),
                                       static_cast<std::size_t>(input.height),
                                       cudaMemcpyHostToDevice));

        dim3 block(kBlockWidth, kBlockHeight);
        dim3 grid((input.width + block.x - 1) / block.x,
                  (input.height + block.y - 1) / block.y);

        CUDA_CHECK(cudaEventRecord(kernelStart));
        medianFilterTextureKernel<<<grid, block>>>(resources.texture,
                                                   resources.deviceOutput,
                                                   input.width,
                                                   input.height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(kernelStop));
        CUDA_CHECK(cudaEventSynchronize(kernelStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernelMsFloat,
                                        kernelStart,
                                        kernelStop));

        CUDA_CHECK(cudaMemcpy(result.output.pixels.data(),
                              resources.deviceOutput,
                              result.output.pixels.size() *
                                  sizeof(unsigned char),
                              cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaEventDestroy(kernelStart);
        cudaEventDestroy(kernelStop);
        throw;
    }

    const auto totalFinish = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaEventDestroy(kernelStart));
    CUDA_CHECK(cudaEventDestroy(kernelStop));

    result.totalMs =
        std::chrono::duration<double, std::milli>(totalFinish - totalStart)
            .count();
    result.kernelMs = static_cast<double>(kernelMsFloat);
    return result;
}

struct VerificationResult {
    std::size_t mismatchedPixels{};
    int maxAbsDiff{};
};

VerificationResult compareImages(const Image& reference,
                                 const Image& candidate) {
    if (reference.width != candidate.width ||
        reference.height != candidate.height ||
        reference.pixels.size() != candidate.pixels.size()) {
        throw std::runtime_error("Image sizes do not match during verification.");
    }

    VerificationResult result;
    for (std::size_t index = 0; index < reference.pixels.size(); ++index) {
        const int diff = std::abs(static_cast<int>(reference.pixels[index]) -
                                  static_cast<int>(candidate.pixels[index]));
        if (diff != 0) {
            ++result.mismatchedPixels;
            result.maxAbsDiff = std::max(result.maxAbsDiff, diff);
        }
    }
    return result;
}

BenchmarkResult runBenchmark(const std::string& inputPath,
                             const std::string& outputPath,
                             int repeats) {
    const Image input = readBmpGray(inputPath);

    Image cpuOutput;
    const auto cpuStart = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < repeats; ++iteration) {
        cpuOutput = medianFilterCPU(input);
    }
    const auto cpuFinish = std::chrono::high_resolution_clock::now();
    const double cpuMs =
        std::chrono::duration<double, std::milli>(cpuFinish - cpuStart)
            .count() /
        static_cast<double>(repeats);

    GpuResources resources;

    try {
        ensureGpuResources(resources, input.width, input.height);

        {
            const GpuRunResult warmup = medianFilterGPU(input, resources);
            (void)warmup;
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        Image gpuOutput;
        double totalGpuTotalMs = 0.0;
        double totalGpuKernelMs = 0.0;

        for (int iteration = 0; iteration < repeats; ++iteration) {
            const GpuRunResult run = medianFilterGPU(input, resources);
            gpuOutput = run.output;
            totalGpuTotalMs += run.totalMs;
            totalGpuKernelMs += run.kernelMs;
        }

        freeGpuResources(resources);

        writeBmpGray8(outputPath, gpuOutput);

        const VerificationResult verification =
            compareImages(cpuOutput, gpuOutput);
        const double gpuTotalMs =
            totalGpuTotalMs / static_cast<double>(repeats);
        const double gpuKernelMs =
            totalGpuKernelMs / static_cast<double>(repeats);

        return BenchmarkResult{
            inputPath,
            outputPath,
            input.width,
            input.height,
            repeats,
            cpuMs,
            gpuTotalMs,
            gpuKernelMs,
            gpuTotalMs > 0.0 ? cpuMs / gpuTotalMs : 0.0,
            gpuKernelMs > 0.0 ? cpuMs / gpuKernelMs : 0.0,
            verification.mismatchedPixels,
            verification.maxAbsDiff,
        };
    } catch (...) {
        freeGpuResources(resources);
        throw;
    }
}

int parsePositiveInt(const std::string& value, const std::string& name) {
    std::size_t processed = 0;
    const int parsed = std::stoi(value, &processed);
    if (processed != value.size() || parsed <= 0) {
        throw std::invalid_argument(name + " must be a positive integer.");
    }
    return parsed;
}

double parseProbability(const std::string& value) {
    std::size_t processed = 0;
    const double parsed = std::stod(value, &processed);
    if (processed != value.size() || parsed < 0.0 || parsed > 1.0) {
        throw std::invalid_argument("Noise probability must be in range [0, 1].");
    }
    return parsed;
}

unsigned int parseSeed(const std::string& value) {
    std::size_t processed = 0;
    const unsigned long parsed = std::stoul(value, &processed);
    if (processed != value.size()) {
        throw std::invalid_argument("Seed must be an unsigned integer.");
    }
    return static_cast<unsigned int>(parsed);
}

void printUsage(const char* executableName) {
    std::cout
        << "Usage:\n"
        << "  " << executableName
        << " [--input image.bmp] [--output filtered.bmp] [--repeats N]\n"
        << "  " << executableName
        << " --generate-test image.bmp [--width W] [--height H] [--noise P]\n\n"
        << "Default input: data/noisy_256.bmp\n"
        << "Default output: filtered.bmp\n";
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
        if (argument == "--input" || argument == "-i") {
            options.inputPath = requireValue(argument);
            continue;
        }
        if (argument == "--output" || argument == "-o") {
            options.outputPath = requireValue(argument);
            continue;
        }
        if (argument == "--repeats") {
            options.repeats = parsePositiveInt(requireValue(argument), "Repeats");
            continue;
        }
        if (argument == "--generate-test") {
            options.generatePath = requireValue(argument);
            continue;
        }
        if (argument == "--width") {
            options.generateWidth =
                parsePositiveInt(requireValue(argument), "Width");
            continue;
        }
        if (argument == "--height") {
            options.generateHeight =
                parsePositiveInt(requireValue(argument), "Height");
            continue;
        }
        if (argument == "--noise") {
            options.noiseProbability = parseProbability(requireValue(argument));
            continue;
        }
        if (argument == "--seed") {
            options.seed = parseSeed(requireValue(argument));
            continue;
        }

        throw std::invalid_argument("Unknown argument: " + argument);
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
    };
}

void printDeviceInfo(const DeviceInfo& info) {
    std::cout << "GPU: " << info.name << '\n'
              << "SM count: " << info.smCount << '\n'
              << "CUDA block: " << kBlockWidth << "x" << kBlockHeight << '\n'
              << "Max threads per block: " << info.maxThreadsPerBlock
              << "\n\n";
}

void printResult(const BenchmarkResult& result) {
    std::cout
        << "| Input | Size | Repeats | CPU, ms | GPU total, ms | GPU kernel, ms | Speedup total | Speedup kernel | Mismatched pixels | Max abs diff | Output |\n"
        << "|:------|:-----|--------:|--------:|--------------:|---------------:|--------------:|---------------:|------------------:|-------------:|:-------|\n";

    std::cout << std::fixed << std::setprecision(3)
              << "| " << result.inputPath
              << " | " << result.width << "x" << result.height
              << " | " << result.repeats
              << " | " << result.cpuMs
              << " | " << result.gpuTotalMs
              << " | " << result.gpuKernelMs
              << " | " << result.speedupTotal
              << " | " << result.speedupKernel
              << " | " << result.mismatchedPixels
              << " | " << result.maxAbsDiff
              << " | " << result.outputPath
              << " |\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseArguments(argc, argv);

        if (!options.generatePath.empty()) {
            const Image generated = generateSaltAndPepperImage(
                options.generateWidth,
                options.generateHeight,
                options.noiseProbability,
                options.seed);
            writeBmpGray8(options.generatePath, generated);
            std::cout << "Generated test BMP: " << options.generatePath << '\n'
                      << "Size: " << generated.width << "x"
                      << generated.height << '\n'
                      << "Noise probability: " << options.noiseProbability
                      << '\n';
            return EXIT_SUCCESS;
        }

        const DeviceInfo deviceInfo = getDeviceInfo();
        printDeviceInfo(deviceInfo);

        const BenchmarkResult result =
            runBenchmark(options.inputPath, options.outputPath, options.repeats);
        printResult(result);

        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
