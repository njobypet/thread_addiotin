# Design notes

This document explains the non-obvious parts of `src/thread_addition.hip`. The
requirements themselves are simple; most of the complexity comes from the fact
that GPUs actively fight three of them: allocating from a thread, making a read
of memory actually happen, and waiting a wall-clock amount of time inside a
kernel.

## Launch geometry

100 threads in a single block of 100 (`<<<1, 100>>>`). On AMD hardware a
wavefront is 64 lanes, so this is two wavefronts, the second one only 36/64
occupied. That is wasteful in general, but here every thread spends 2 seconds
waiting anyway, and keeping all 100 threads in one block keeps the mapping
between thread index and array index trivially obvious.

The alternative, `<<<100, 1>>>`, would scatter the threads across compute units
and is a one-line change (`kBlockSize = 1`). It does not change the results.

## Per-thread allocation from the device heap

`malloc()` inside a kernel allocates from a fixed-size device heap that the HIP
runtime reserves before launch. The default heap on ROCm is small — smaller than
the 100 MiB this program needs — so the host raises it first:

```cpp
hipDeviceSetLimit(hipLimitMallocHeapSize, kDeviceHeapBytes);
```

The limit is set to 200 MiB, twice the 100 MiB of live allocations, because the
device allocator carries per-allocation metadata and can fragment. The call must
happen before the first kernel launch; resizing the heap afterwards is not
allowed.

Every thread checks its `malloc` for null and writes the sentinel `-1.0` if it
failed, rather than faulting. The host counts those and exits non-zero.

## Filling the buffer

Each thread fills its 1 MiB with words from its own generator, a small 32-bit
mixer in the style of splitmix, seeded from `seed ^ (tid * 0x9e3779b9)`. Mixing
the thread id in is what makes the 100 checksums differ; without it every thread
would produce the same stream and the printed array would be 100 copies of one
number.

The host picks the seed from `std::random_device` unless `--seed N` is given, so
runs vary by default but are reproducible on demand.

A per-thread generator is deliberately chosen over `hiprand`: it keeps the
program dependency-free, and the statistical quality of the data does not matter
for a checksum demo.

## Making the reads real

Reading and writing through a `volatile uint32_t*` is what keeps the loops from
being optimized away. Without it, LLVM can see that the fill loop's stores are
never observed by anything except the very next loop, fuse the two, and compute
the sum without ever touching memory — which would defeat the point of
allocating 1 MiB in the first place.

The `volatile` matters even more in `--uninitialized` mode, where the reads are
of memory that provably was never written. That is undefined behaviour, and the
compiler would be entitled to fold the loop away entirely.

## Why `--uninitialized` prints zeros

The amdgpu driver scrubs VRAM pages before handing them to a process, for
isolation reasons, and the device heap this program carves from is itself a fresh
runtime allocation. So on the first launch in a new process there is genuinely
nothing but zeros to find. Leftover data only survives *within* a process, across
a free and a subsequent re-allocation. A screen full of zeros in that mode is the
driver working correctly, not a bug in the checksum.

## Why the checksum is a `double`

The requirement says the CPU rounds the value down to the nearest integer, which
only means something if the value can be fractional. The checksum is therefore
the sum of the words with each one normalized to `[0, 1)`:

```
checksum = (sum of 262144 uint32 words) / 2^32
```

The sum is accumulated in a `uint64_t`, so it cannot overflow: 262144 × (2³² − 1)
≈ 1.13 × 10¹⁵. That is also below 2⁵³, so the conversion to `double` is exact and
the host-side `floor` is the only rounding in the whole pipeline.

The `2^32` divisor, rather than dividing by the word count to get a mean, is what
keeps random data in range. The mean word value of a uniformly random buffer is
about 2³¹ ≈ 2.147 × 10⁹, which sits right on top of `INT_MAX` — roughly half the
threads would clamp and the output would be a wall of `2147483647`. Normalizing
per word instead puts the expected value at half the word count, 131072, with a
thread-to-thread spread of a few hundred: comfortably inside `int`, and visibly
different per thread.

The upper bound is still the word count (262144 for a buffer of all-`0xFFFFFFFF`),
so `floor_to_int()` keeps its clamp to `INT_MAX`/`INT_MIN` rather than invoking
undefined behaviour on the narrowing conversion. Nothing realistic gets close.

## The 2 second delay

`s_sleep` only supports short, fixed sleeps, so the delay is a busy-wait on a
hardware counter:

```cpp
const unsigned long long start = wall_clock64();
while (wall_clock64() - start < ticks) { __builtin_amdgcn_s_sleep(64); }
```

`wall_clock64()` is the right counter: it runs at a fixed frequency (100 MHz on
current parts) regardless of what the shader clock is doing. The host converts
milliseconds to ticks using `hipDeviceAttributeWallClockRate`, which is reported
in kHz, so ticks-per-millisecond is the attribute value itself.

`clock64()` is the fallback for devices that do not report a wall clock rate. It
counts shader clocks, which DVFS moves around under load, so the delay becomes
approximate — the program prints a warning when it takes that path. If neither
rate is available it assumes 100 MHz and warns again.

The `__builtin_amdgcn_s_sleep(64)` in the loop body parks the SIMD for a short
interval so the wavefront is not hammering the counter for two seconds straight.
It is guarded by `__HIP_DEVICE_COMPILE__` because the builtin does not exist
during the host compilation pass.

All 100 threads wait concurrently, so the kernel takes about 2 seconds total, not
100 × 2 seconds. The host prints the measured kernel time so this is visible.

Two seconds is long for a single kernel. On a GPU that is also driving a display,
that is enough to trip the watchdog and get the queue preempted or the device
reset; the README lists the symptoms. Headless compute cards (MI-series) have no
such limit. The tick count is nowhere near overflowing — two seconds at 100 MHz
is 2 × 10⁸ ticks in a 64-bit counter.

## Publishing the result after the delay

The requirement is that the delay happens *before* the value goes back to the
CPU, so the global store is the last thing the kernel does:

```cpp
free(buffer);
busy_wait(delay_ticks, use_wall_clock);
checksums[tid] = checksum;
```

The checksum lives in a register across the wait. `free()` happens before the
wait to release the heap early, which is harmless since the value has already
been computed.

## Getting the results back

`hipMemcpy` after `hipDeviceSynchronize()` — a blocking copy on the null stream.
There is no need for per-thread streams or callbacks: the kernel launch is the
unit of work, and it is not finished until all 100 threads have stored their
value.

The host then floors each `double` into the `int` array from step 1 and prints it
in rows of ten. Timing uses HIP events recorded around the launch, which measure
GPU-side time rather than host wall time.
