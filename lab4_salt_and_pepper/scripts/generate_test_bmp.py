#!/usr/bin/env python3
import argparse
import math
import random
import struct
from pathlib import Path


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def write_gray_bmp(path: Path, width: int, height: int, pixels: bytearray) -> None:
    row_stride = ((width + 3) // 4) * 4
    palette_size = 256 * 4
    pixel_offset = 14 + 40 + palette_size
    pixel_data_size = row_stride * height
    file_size = pixel_offset + pixel_data_size

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as output:
        output.write(b"BM")
        output.write(struct.pack("<IHHI", file_size, 0, 0, pixel_offset))
        output.write(
            struct.pack(
                "<IIIHHIIIIII",
                40,
                width,
                height,
                1,
                8,
                0,
                pixel_data_size,
                2835,
                2835,
                256,
                0,
            )
        )

        for value in range(256):
            output.write(bytes((value, value, value, 0)))

        padding = bytes(row_stride - width)
        for row in range(height - 1, -1, -1):
            start = row * width
            output.write(pixels[start : start + width])
            output.write(padding)


def generate(width: int, height: int, noise: float, seed: int) -> bytearray:
    random.seed(seed)
    pixels = bytearray(width * height)

    for y in range(height):
        for x in range(width):
            wave = 32.0 * math.sin(x * 0.08) + 24.0 * math.cos(y * 0.06)
            checker = ((x // 32 + y // 32) % 2) * 32
            value = int(96.0 + 96.0 * x / max(1, width - 1) + wave + checker)
            value = clamp(value, 0, 255)

            if random.random() < noise:
                value = 255 if random.random() < 0.5 else 0

            pixels[y * width + x] = value

    return pixels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a grayscale BMP with salt-and-pepper noise."
    )
    parser.add_argument("output", type=Path, nargs="?", default=Path("data/noisy_256.bmp"))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("width and height must be positive")
    if not 0.0 <= args.noise <= 1.0:
        raise SystemExit("noise must be in range [0, 1]")

    pixels = generate(args.width, args.height, args.noise, args.seed)
    write_gray_bmp(args.output, args.width, args.height, pixels)
    print(f"Generated {args.output} ({args.width}x{args.height}, noise={args.noise})")


if __name__ == "__main__":
    main()
