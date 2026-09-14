# Design notes

This document explains the non-obvious parts of `src/thread_addition.hip`. The
requirements themselves are simple; most of the complexity comes from the fact
that GPUs actively fight three of them: allocating from a thread, reading memory
nobody wrote, and waiting a wall-clock amount of time inside a kernel.

## Launch geometry

100 threads in a single block of 100 (`<<<1, 100>>>`). On AMD hardware a
wavefront is 64 lanes, so this is two wavefronts, the second one only 36/64
occupied. That is wasteful in general, but here every thread spends 100 ms
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

## Checksumming memory nobody initialized

Two things want to defeat this requirement:

1. **The compiler.** Reading an allocation that provably has not been written is
   undefined behaviour, and LLVM is entitled to fold the loop away to whatever
   it likes. The buffer is therefore read through a `volatile const uint32_t*`,
   which forces every load to actually happen.
2. **The driver.** The GPU driver zeroes pages before handing them to a process
   for isolation reasons. So the "garbage" is usually zeros the first time a page
   is touched. Data from a previous allocation in the *same* process can survive,
   which is why running the program twice, or after another GPU workload, is the
   interesting case.

Neither is a bug in this program, but both are worth knowing before staring at a
screen full of zeros.

## Why the checksum is a `double`

The requirement says the CPU rounds the value down to the nearest integer, which
only means something if the value can be fractional. So the checksum is defined
as the **mean 32-bit word value** over the 1 MiB region:

```
checksum = (sum of 262144 uint32 words) / 262144
```

The sum is accumulated in a `uint64_t`, so it cannot overflow: 262144 × (2³² − 1)
≈ 1.13 × 10¹⁵. That is also below 2⁵³, so the conversion to `double` is exact and
the only rounding in the whole pipeline is the host-side `floor`.

The mean can in principle reach 2³² − 1, which does not fit in an `int`, so
`floor_to_int()` clamps to `INT_MAX`/`INT_MIN` instead of invoking undefined
behaviour on the narrowing conversion. In practice the value is nowhere near
that.

## The 100 ms delay

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
interval so the wavefront is not hammering the counter for 100 ms straight. It
is guarded by `__HIP_DEVICE_COMPILE__` because the builtin does not exist during
the host compilation pass.

All 100 threads wait concurrently, so the kernel takes about 100 ms total, not
100 × 100 ms. The host prints the measured kernel time so this is visible.

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
