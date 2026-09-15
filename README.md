# thread_addiotin

A small AMD ROCm/HIP program that launches 100 separately named kernels,
`rocmtestkernel_0` through `rocmtestkernel_99`, one GPU thread each. Every kernel
allocates 1 MiB from the device heap, fills it with pseudo-random data, checksums
it, waits 2 seconds, and hands the result back to the CPU, which floors each
checksum into a 100-element `int` array and prints it.

## What it does

| Step | Requirement | Where it lives |
| --- | --- | --- |
| 1 | 100-element `int` array in `main` | `int checksums[kNumKernels]` in `main()` |
| 2 | 100 GPU threads | 100 distinct kernels, `rocmtestkernel_<n><<<1, 1, 0, streams[n]>>>` |
| 3 | 1 MiB per kernel, filled and checksummed | device-side `malloc(1 MiB)`, filled by a per-kernel generator, summed through a `volatile` pointer |
| 4 | 2 second in-kernel delay after the checksum | `busy_wait()` on `wall_clock64()` before the store |
| 5 | Checksum back to the CPU, floored into the array | `hipMemcpy` + `std::floor` |
| 6 | CPU prints the array | `print_array()` after all 100 values arrive |

The 100 kernels are real, separately named entry points, not one kernel launched
100 times. They are generated from a single list of sequence numbers,
`ROCM_TEST_KERNEL_SEQUENCE`, which also generates the 100 launch statements and
the kernel count — so adding or removing a kernel means editing one line. You can
see all 100 symbols in the binary:

```bash
nm -C build/thread_addition | grep rocmtestkernel_
```

Each kernel gets its own stream so they overlap; on the null stream the run would
take 100 × 2 seconds. Each seeds its own generator from
`seed ^ (sequence * 0x9e3779b9)`, so the 100 buffers hold different data and the
100 checksums differ from one another. Passing `--uninitialized` skips the fill
and checksums the device heap exactly as it was handed over.

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
Kernels: 100 (rocmtestkernel_0 .. rocmtestkernel_99), 1 thread each
Per-kernel allocation: 1024 KiB, delay: 2000 ms
Memory: filled with random words, seed 0x5c3f1a02

Checksums (floored) from 100 kernels:
  [  0..  9]     131020     131195     130946     131243     131077     130998     131164     131052     130901     131209
  ...
  [ 90.. 99]     131118     130972     131241     131006     131155     130889     131093     131207     130964     131130

All 100 kernels completed in 2004.71 ms (delay is 2000 ms)
```

The exit code is 0 when all 100 kernels got their memory, and 1 if any
device-side `malloc` failed (those slots print `-1`).

### About the run time

The whole run should take a little over 2 seconds, not 200, because the kernels
overlap. How well they overlap depends on `GPU_MAX_HW_QUEUES`: HIP multiplexes
streams onto a small number of hardware queues (4 by default) and kernels sharing
a queue run back to back. The program sets it to 16 before initializing the
runtime unless you set it yourself, and warns if the total still comes out far
above the delay:

```bash
GPU_MAX_HW_QUEUES=32 ./build/thread_addition   # overlap more aggressively
```

### About the values

The checksum is the sum of the 262,144 words in the buffer with each word
normalized to `[0, 1)` — that is, the raw 64-bit sum divided by 2³². For
uniformly random data the expected value is half the word count, so **checksums
cluster around 131072**, drifting by a few hundred either way from kernel to
kernel. That spread is the randomness; identical values across all 100 kernels
would mean the per-kernel seeding is broken.

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
| `--seed N` | Seed the per-kernel generators; same seed, same checksums |
| `--uninitialized` | Skip the random fill and checksum the heap as-is |

Environment:

| Variable | Meaning |
| --- | --- |
| `GPU_MAX_HW_QUEUES` | Hardware queues the 100 streams are spread across; the program defaults it to 16 |
| `HIP_VISIBLE_DEVICES` | Pick the GPU, e.g. a headless one with no watchdog |

Everything else is a constant at the top of `src/thread_addition.hip`:

| Constant | Default | Meaning |
| --- | --- | --- |
| `ROCM_TEST_KERNEL_SEQUENCE` | 0..99 | The kernel sequence numbers; definitions, launches and the count all come from this list |
| `kNumKernels` | 100 | Counted from the list above; kernels, GPU threads and array elements |
| `kBytesPerKernel` | 1 MiB | Device-heap allocation per kernel |
| `kWordScale` | 2³² | Divisor that normalizes each word before summing |
| `kDelayMilliseconds` | 2000 | In-kernel delay before publishing the result |
| `kDeviceHeapBytes` | 200 MiB | `hipLimitMallocHeapSize`; must cover every live allocation plus allocator overhead |

Adding kernel 100 means appending `X(100)` to `ROCM_TEST_KERNEL_SEQUENCE` and
bumping `kExpectedKernels`; a `static_assert` fails if the two disagree.

## Troubleshooting

- **All checksums are `0`** — expected with `--uninitialized` (see above). Drop
  the flag to fill the buffers with random data instead.
- **All 100 checksums are identical** — the per-kernel seeding is not varying;
  check that the kernel is mixing its sequence number into the generator state.
- **All checksums are `-1`** — the device heap is too small or fragmented. Raise
  `kDeviceHeapBytes`, or lower `kBytesPerKernel`.
- **The run takes ~50 seconds instead of ~2** — the kernels are serializing on a
  handful of hardware queues. Raise `GPU_MAX_HW_QUEUES`; the program says so when
  it detects this.
- **`hipErrorUnsupportedLimit` from `hipDeviceSetLimit`** — the runtime is too old
  to resize the device heap. Upgrade ROCm, or set `HIP_MALLOC_HEAP_SIZE` in the
  environment instead.
- **Total time is well under 2000 ms** — the GPU reported no wall clock rate
  and the program fell back to the shader clock (it warns when this happens).
  The shader clock drifts with DVFS, so the delay becomes approximate.
- **The GPU resets, or the run dies with `HSA_STATUS_ERROR` / a queue preemption
  error** — a 2 second kernel is long enough to trip the watchdog on a
  display-attached GPU. Run on a headless device (`HIP_VISIBLE_DEVICES`), or
  lower `kDelayMilliseconds`.
- **`no kernel image is available for execution`** — the binary was built for a
  different architecture. Rebuild with `--offload-arch`/`CMAKE_HIP_ARCHITECTURES`
  matching `rocminfo`.

## License

MIT. See [LICENSE](LICENSE).
