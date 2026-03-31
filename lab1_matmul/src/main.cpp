#include "matmul.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::size_t m = 512;
  std::size_t k = 512;
  std::size_t n = 512;
  int tile_size = 16;
  std::uint32_t seed = 42;
  bool benchmark = false;
  std::vector<std::size_t> benchmark_sizes = {128, 256, 512, 1024};
  std::optional<std::string> csv_path;
};

void print_help() {
  std::cout
      << "Usage:\n"
      << "  ./matmul [--m rows] [--k inner] [--n cols] [--tile block] [--seed value]\n"
      << "  ./matmul --benchmark [--sizes 128,256,512,1024] [--csv results.csv]\n\n"
      << "Description:\n"
      << "  Generates two random matrices, multiplies them on CPU and GPU, checks\n"
      << "  correctness and prints execution time. On systems without CUDA the GPU\n"
      << "  step is reported as unavailable.\n";
}

std::size_t parse_size(const std::string& value, const std::string& name) {
  const auto parsed = std::stoull(value);
  if (parsed == 0) {
    throw std::invalid_argument(name + " must be positive.");
  }
  return static_cast<std::size_t>(parsed);
}

int parse_int(const std::string& value, const std::string& name) {
  const int parsed = std::stoi(value);
  if (parsed <= 0) {
    throw std::invalid_argument(name + " must be positive.");
  }
  return parsed;
}

std::vector<std::size_t> parse_sizes(const std::string& value) {
  std::vector<std::size_t> sizes;
  std::stringstream stream(value);
  std::string token;

  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    sizes.push_back(parse_size(token, "benchmark size"));
  }

  if (sizes.empty()) {
    throw std::invalid_argument("At least one benchmark size must be provided.");
  }
  return sizes;
}

Options parse_arguments(int argc, char** argv) {
  Options options;

  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];

    auto require_value = [&](const std::string& flag) -> std::string {
      if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value after " + flag);
      }
      return argv[++index];
    };

    if (arg == "--help" || arg == "-h") {
      print_help();
      std::exit(0);
    }
    if (arg == "--m") {
      options.m = parse_size(require_value(arg), "m");
      continue;
    }
    if (arg == "--k") {
      options.k = parse_size(require_value(arg), "k");
      continue;
    }
    if (arg == "--n") {
      options.n = parse_size(require_value(arg), "n");
      continue;
    }
    if (arg == "--tile") {
      options.tile_size = parse_int(require_value(arg), "tile");
      continue;
    }
    if (arg == "--seed") {
      options.seed =
          static_cast<std::uint32_t>(parse_size(require_value(arg), "seed"));
      continue;
    }
    if (arg == "--benchmark") {
      options.benchmark = true;
      continue;
    }
    if (arg == "--sizes") {
      options.benchmark = true;
      options.benchmark_sizes = parse_sizes(require_value(arg));
      continue;
    }
    if (arg == "--csv") {
      options.csv_path = require_value(arg);
      continue;
    }

    throw std::invalid_argument("Unknown argument: " + arg);
  }

  return options;
}

void print_single_run(const BenchmarkRow& row, const GpuRunResult& gpu_result) {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Matrix sizes:\n";
  std::cout << "  A = " << row.m << "x" << row.k << "\n";
  std::cout << "  B = " << row.k << "x" << row.n << "\n\n";

  std::cout << "CPU:\n";
  std::cout << "  time_ms = " << row.cpu_time_ms << "\n\n";

  std::cout << "GPU:\n";
  if (row.gpu_available) {
    std::cout << "  total_time_ms = " << row.gpu_total_time_ms << "\n";
    std::cout << "  kernel_time_ms = " << row.gpu_kernel_time_ms << "\n";
    std::cout << "  speedup = " << row.speedup << "x\n";
  } else {
    std::cout << "  unavailable = " << gpu_result.message << "\n";
  }
  std::cout << "\n";

  std::cout << "Verification:\n";
  if (row.gpu_available) {
    std::cout << "  correct = " << (row.verification.is_correct ? "true" : "false") << "\n";
    std::cout << "  max_abs_error = " << row.verification.max_abs_error << "\n";
    std::cout << "  max_rel_error = " << row.verification.max_rel_error << "\n";
  } else {
    std::cout << "  skipped because GPU result is unavailable\n";
  }
}

void print_benchmark_table(const std::vector<BenchmarkRow>& rows) {
  std::cout << std::fixed << std::setprecision(3);
  std::cout << std::left << std::setw(10) << "size"
            << std::setw(14) << "cpu_ms"
            << std::setw(16) << "gpu_total_ms"
            << std::setw(16) << "gpu_kernel_ms"
            << std::setw(12) << "speedup"
            << std::setw(10) << "correct"
            << "max_abs_error\n";

  for (const BenchmarkRow& row : rows) {
    const std::string size = std::to_string(row.m) + "x" + std::to_string(row.n);
    std::cout << std::setw(10) << size
              << std::setw(14) << row.cpu_time_ms;

    if (row.gpu_available) {
      std::cout << std::setw(16) << row.gpu_total_time_ms
                << std::setw(16) << row.gpu_kernel_time_ms
                << std::setw(12) << row.speedup
                << std::setw(10) << (row.verification.is_correct ? "yes" : "no")
                << row.verification.max_abs_error << "\n";
    } else {
      std::cout << std::setw(16) << "N/A"
                << std::setw(16) << "N/A"
                << std::setw(12) << "N/A"
                << std::setw(10) << "N/A"
                << "N/A\n";
    }
  }
}

void write_csv(const std::string& path, const std::vector<BenchmarkRow>& rows) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to open CSV file: " + path);
  }

  output << "m,k,n,cpu_time_ms,gpu_total_time_ms,gpu_kernel_time_ms,speedup,correct,max_abs_error,max_rel_error,gpu_available\n";
  output << std::fixed << std::setprecision(6);

  for (const BenchmarkRow& row : rows) {
    output << row.m << ','
           << row.k << ','
           << row.n << ','
           << row.cpu_time_ms << ',';

    if (row.gpu_available) {
      output << row.gpu_total_time_ms << ','
             << row.gpu_kernel_time_ms << ','
             << row.speedup << ','
             << (row.verification.is_correct ? "true" : "false") << ','
             << row.verification.max_abs_error << ','
             << row.verification.max_rel_error << ','
             << "true\n";
    } else {
      output << "nan,nan,nan,false,nan,nan,false\n";
    }
  }
}

BenchmarkRow run_case(std::size_t m,
                      std::size_t k,
                      std::size_t n,
                      std::uint32_t seed,
                      int tile_size,
                      GpuRunResult* gpu_result_out = nullptr) {
  Matrix left = generate_random_matrix(m, k, seed);
  Matrix right = generate_random_matrix(k, n, seed + 1);

  const auto cpu_start = std::chrono::steady_clock::now();
  Matrix cpu_result = multiply_cpu(left, right);
  const auto cpu_end = std::chrono::steady_clock::now();

  BenchmarkRow row;
  row.m = m;
  row.k = k;
  row.n = n;
  row.cpu_time_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

  GpuRunResult gpu_result = multiply_gpu(left, right, tile_size);
  row.gpu_available = gpu_result.is_available;

  if (gpu_result.is_available) {
    row.gpu_total_time_ms = gpu_result.total_time_ms;
    row.gpu_kernel_time_ms = gpu_result.kernel_time_ms;
    row.speedup = row.cpu_time_ms / gpu_result.total_time_ms;
    row.verification = verify_result(cpu_result, gpu_result.output);
  }

  if (gpu_result_out != nullptr) {
    *gpu_result_out = std::move(gpu_result);
  }

  return row;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_arguments(argc, argv);

    if (!options.benchmark) {
      GpuRunResult gpu_result;
      const BenchmarkRow row =
          run_case(options.m, options.k, options.n, options.seed, options.tile_size, &gpu_result);
      print_single_run(row, gpu_result);
      return row.gpu_available && !row.verification.is_correct ? 2 : 0;
    }

    std::vector<BenchmarkRow> rows;
    rows.reserve(options.benchmark_sizes.size());

    for (std::size_t size : options.benchmark_sizes) {
      rows.push_back(run_case(size, size, size, options.seed, options.tile_size));
    }

    print_benchmark_table(rows);

    if (options.csv_path.has_value()) {
      write_csv(*options.csv_path, rows);
      std::cout << "\nCSV saved to " << *options.csv_path << "\n";
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
}
