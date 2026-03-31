#!/usr/bin/env python3

import csv
import math
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required to build the chart. Install it with 'pip install matplotlib'."
    ) from exc


def parse_float(value: str) -> float:
    if value.strip().lower() == "nan":
        return math.nan
    return float(value)


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/plot_speedup.py <benchmark_results.csv> <output.png>")
        return 1

    csv_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not csv_path.exists():
        print(f"CSV file was not found: {csv_path}")
        return 1

    sizes = []
    cpu_times = []
    gpu_times = []
    speedups = []

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["gpu_available"].lower() != "true":
                continue

            sizes.append(int(row["m"]))
            cpu_times.append(parse_float(row["cpu_time_ms"]))
            gpu_times.append(parse_float(row["gpu_total_time_ms"]))
            speedups.append(parse_float(row["speedup"]))

    if not sizes:
        print("The CSV file does not contain successful GPU runs.")
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(sizes, cpu_times, marker="o", label="CPU")
    axes[0].plot(sizes, gpu_times, marker="s", label="GPU total")
    axes[0].set_title("Execution Time")
    axes[0].set_xlabel("Matrix size (N x N)")
    axes[0].set_ylabel("Time, ms")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(sizes, speedups, marker="o", color="tab:green")
    axes[1].set_title("GPU Speedup")
    axes[1].set_xlabel("Matrix size (N x N)")
    axes[1].set_ylabel("Speedup (CPU / GPU)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
