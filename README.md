# thread_addiotin

A small AMD ROCm/HIP program that has 100 GPU threads each allocate 1 MiB from the
device heap, fill it with pseudo-random data, checksum it, wait 100 ms, and hand
the result back to the CPU, which floors each checksum into a 100-element `int`
array and prints it.

## What it does

| Step | Requirement | Where it lives |
| --- | --- | --- |
| 1 | 100-element `int` array in `main` | `int checksums[kNumThreads]` in `main()` |
| 2 | Spawn 100 GPU threads | `checksum_device_memory<<<1, 100>>>` |
| 3 | 1 MiB per thread, filled and checksummed | device-side `malloc(1 MiB)`, filled by a per-thread generator, summed through a `volatile` pointer |
| 4 | 100 ms in-kernel delay after the checksum | `busy_wait()` on `wall_clock64()` before the store |
| 5 | Checksum back to the CPU, floored into the array | `hipMemcpy` + `std::floor` |
| 6 | CPU prints the array | `print_array()` after all 100 values arrive |

Each thread seeds its own generator from `seed ^ (tid * 0x9e3779b9)`, so the 100
buffers hold different data and the 100 checksums differ from one another.
Passing `--uninitialized` skips the fill and checksums the device heap exactly as
it was handed over, which is the original behaviour.

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
./build/thread_addition                  # random fill, fresh seed each run
./build/thread_addition --seed 12345     # reproducible run
./build/thread_addition --uninitialized  # checksum the heap as handed over
```

Sample output:

```
Device 0: AMD Instinct MI300X (gfx942)
Threads: 100, per-thread allocation: 1024 KiB, delay: 100 ms
Memory: filled with random words, seed 0x5c3f1a02

Checksums (floored) for 100 GPU threads:
  [  0..  9]     131020     131195     130946     131243     131077     130998     131164     131052     130901     131209
  ...
  [ 90.. 99]     131118     130972     131241     131006     131155     130889     131093     131207     130964     131130

Kernel wall time: 100.41 ms (expected >= 100 ms)
```

The exit code is 0 when all 100 threads got their memory, and 1 if any
device-side `malloc` failed (those slots print `-1`).

### About the values

The checksum is the sum of the 262,144 words in the buffer with each word
normalized to `[0, 1)` — that is, the raw 64-bit sum divided by 2³². For
uniformly random data the expected value is half the word count, so **checksums
cluster around 131072**, drifting by a few hundred either way from thread to
thread. That spread is the randomness; identical values across all 100 threads
would mean the per-thread seeding is broken.

Use `--seed N` to make a run reproducible: the same seed gives the same 100
checksums on the same GPU.

### About `--uninitialized`

In this mode nothing is written before the read, so the checksums are whatever
the device heap happened to contain. **An array of zeros is the normal result
here.** The amdgpu driver scrubs VRAM pages before handing them to a process, so
on the first launch there is genuinely nothing but zeros to find. Data only
survives within a single process, across a free and a re-allocation.

Reading uninitialized memory is deliberate in that mode and is exactly the kind
of thing a sanitizer will flag. Do not copy the pattern into production code.

## Options and tuning

Command line:

| Flag | Meaning |
| --- | --- |
| `--seed N` | Seed the per-thread generators; same seed, same checksums |
| `--uninitialized` | Skip the random fill and checksum the heap as-is |

Everything else is a constant at the top of `src/thread_addition.hip`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `kNumThreads` | 100 | GPU threads, and array elements |
| `kBlockSize` | 100 | Threads per block (grid size follows) |
| `kBytesPerThread` | 1 MiB | Device-heap allocation per thread |
| `kWordScale` | 2³² | Divisor that normalizes each word before summing |
| `kDelayMilliseconds` | 100 | In-kernel delay before publishing the result |
| `kDeviceHeapBytes` | 200 MiB | `hipLimitMallocHeapSize`; must cover every live allocation plus allocator overhead |

## Troubleshooting

- **All checksums are `0`** — expected with `--uninitialized` (see above). Drop
  the flag to fill the buffers with random data instead.
- **All 100 checksums are identical** — the per-thread seeding is not varying;
  check that the kernel is mixing `tid` into the generator state.
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
