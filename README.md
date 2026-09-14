# thread_addiotin

A small AMD ROCm/HIP program that has 100 GPU threads each allocate 1 MiB from the
device heap, checksum whatever bytes are already sitting in that memory, wait
100 ms, and hand the result back to the CPU, which floors each checksum into a
100-element `int` array and prints it.

## What it does

| Step | Requirement | Where it lives |
| --- | --- | --- |
| 1 | 100-element `int` array in `main` | `int checksums[kNumThreads]` in `main()` |
| 2 | Spawn 100 GPU threads | `checksum_uninitialized_memory<<<1, 100>>>` |
| 3 | 1 MiB per thread, checksummed uninitialized | device-side `malloc(1 MiB)`, read through a `volatile` pointer, never written first |
| 4 | 100 ms in-kernel delay after the checksum | `busy_wait()` on `wall_clock64()` before the store |
| 5 | Checksum back to the CPU, floored into the array | `hipMemcpy` + `std::floor` |
| 6 | CPU prints the array | `print_array()` after all 100 values arrive |

Source: [`src/thread_addition.hip`](src/thread_addition.hip). Design notes and the
reasoning behind the tricky parts: [`docs/DESIGN.md`](docs/DESIGN.md).

## Requirements

- An AMD GPU supported by ROCm (CDNA: MI100/MI200/MI300; RDNA: Radeon RX 6000/7000/9000 and Radeon PRO W6000/W7000).
- ROCm 5.7 or newer (tested against the ROCm 6.x HIP runtime). `hipcc` must be on `PATH` or reachable through `ROCM_PATH`.
- Linux. ROCm on Windows does not ship `hipcc` for this workflow; on a Windows host use WSL2 with the ROCm WSL driver stack.
- CMake 3.21+ if you use the CMake build (3.21 is the first release with first-class HIP language support).

Check your setup:

```bash
rocminfo | grep gfx        # lists the architecture(s) you have, e.g. gfx942
hipcc --version
```

## Build

### CMake (recommended)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

By default the binary is compiled for the GPUs present in the build machine. To
cross-compile for specific architectures:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx942"
cmake --build build -j
```

If ROCm is not in `/opt/rocm`, add `-DROCM_PATH=/path/to/rocm`.

### Make

```bash
make                      # autodetect
make GPU_TARGETS=gfx942   # or name the target explicitly
```

### One-liner

```bash
hipcc -O3 -std=c++17 src/thread_addition.hip -o thread_addition
```

## Run

```bash
./build/thread_addition
```

Sample output:

```
Device 0: AMD Instinct MI300X (gfx942)
Threads: 100, per-thread allocation: 1024 KiB, delay: 100 ms

Checksums (floored) for 100 GPU threads:
  [  0..  9]          0          0          0          0          0          0          0          0          0          0
  [ 10.. 19]          0          0          0          0          0          0          0          0          0          0
  ...
  [ 90.. 99]          0          0          0          0          0          0          0          0          0          0

Kernel wall time: 100.41 ms (expected >= 100 ms)
```

The exit code is 0 when all 100 threads got their memory, and 1 if any
device-side `malloc` failed (those slots print `-1`).

### About the values

The checksums are whatever the device heap happens to contain. Freshly faulted
GPU pages usually read back as zeros, so **an array of zeros is a perfectly
normal result** — it is a property of the driver zeroing pages, not a bug. Run
something else on the GPU first, or run this program twice in a row, and you may
see the previous contents survive as non-zero checksums.

Reading uninitialized memory is intentional here and is exactly the kind of thing
a sanitizer will flag. Do not copy this pattern into production code.

## Tuning

Everything interesting is a constant at the top of `src/thread_addition.hip`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `kNumThreads` | 100 | GPU threads, and array elements |
| `kBlockSize` | 100 | Threads per block (grid size follows) |
| `kBytesPerThread` | 1 MiB | Device-heap allocation per thread |
| `kDelayMilliseconds` | 100 | In-kernel delay before publishing the result |
| `kDeviceHeapBytes` | 200 MiB | `hipLimitMallocHeapSize`; must cover every live allocation plus allocator overhead |

## Troubleshooting

- **All checksums are `-1`** — the device heap is too small or fragmented. Raise
  `kDeviceHeapBytes`, or lower `kBytesPerThread`.
- **`hipErrorUnsupportedLimit` from `hipDeviceSetLimit`** — the runtime is too old
  to resize the device heap. Upgrade ROCm, or set `HIP_MALLOC_HEAP_SIZE` in the
  environment instead.
- **Kernel wall time is far from 100 ms** — the GPU reported no wall clock rate
  and the program fell back to the shader clock (it warns when this happens).
  The shader clock drifts with DVFS, so the delay becomes approximate.
- **`no kernel image is available for execution`** — the binary was built for a
  different architecture. Rebuild with `--offload-arch`/`CMAKE_HIP_ARCHITECTURES`
  matching `rocminfo`.

## License

MIT. See [LICENSE](LICENSE).
