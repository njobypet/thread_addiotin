// thread_addiotin - per-thread device-heap checksum with a 100 ms in-kernel delay.
//
// Flow:
//   1. main() declares an array of 100 ints.
//   2. A kernel is launched with exactly 100 GPU threads.
//   3. Every GPU thread malloc()s 1 MiB from the device heap and checksums the
//      bytes that happen to be there. The buffer is deliberately never written
//      to before it is read.
//   4. Each thread busy-waits 100 ms after the checksum is computed and before
//      the result is published to global memory.
//   5. The host copies the 100 checksums back and floors each one into the
//      int array from step 1.
//   6. The host prints the array.

#include <hip/hip_runtime.h>

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Step 1 / step 2: one GPU thread per array element.
static constexpr int kNumThreads = 100;
static constexpr int kBlockSize = 100;
static constexpr int kNumBlocks = (kNumThreads + kBlockSize - 1) / kBlockSize;

// Step 3: 1 MiB per thread, read back as 32-bit words.
static constexpr size_t kBytesPerThread = 1u << 20;
static constexpr size_t kWordsPerThread = kBytesPerThread / sizeof(uint32_t);

// Step 4.
static constexpr double kDelayMilliseconds = 100.0;

// Device heap must hold every live allocation at once, plus allocator overhead.
static constexpr size_t kDeviceHeapBytes =
    static_cast<size_t>(kNumThreads) * kBytesPerThread * 2;

// Written by a thread whose device-side malloc() failed.
static constexpr double kAllocationFailed = -1.0;

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    const hipError_t hip_check_status = (expr);                                \
    if (hip_check_status != hipSuccess) {                                      \
      std::fprintf(stderr, "HIP error at %s:%d: %s (%s)\n", __FILE__,          \
                   __LINE__, hipGetErrorString(hip_check_status), #expr);      \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Device code
// ---------------------------------------------------------------------------

// wall_clock64() ticks at a fixed rate and is immune to engine clock scaling,
// so it is the preferred timer. clock64() is the fallback for devices that do
// not report a wall clock rate; it drifts with DVFS.
__device__ inline unsigned long long read_timer(bool use_wall_clock) {
  return use_wall_clock ? static_cast<unsigned long long>(wall_clock64())
                        : static_cast<unsigned long long>(clock64());
}

__device__ inline void busy_wait(unsigned long long ticks, bool use_wall_clock) {
  const unsigned long long start = read_timer(use_wall_clock);
  while (read_timer(use_wall_clock) - start < ticks) {
#if defined(__HIP_DEVICE_COMPILE__)
    // Yield the SIMD for a short while instead of hammering the timer.
    __builtin_amdgcn_s_sleep(64);
#endif
  }
}

__global__ void checksum_uninitialized_memory(double* __restrict__ checksums,
                                              unsigned long long delay_ticks,
                                              bool use_wall_clock) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= kNumThreads) {
    return;
  }

  // Step 3: 1 MiB per thread straight off the device heap, left uninitialized.
  void* buffer = malloc(kBytesPerThread);
  if (buffer == nullptr) {
    checksums[tid] = kAllocationFailed;
    return;
  }

  // volatile keeps the compiler from discarding reads of memory it can prove
  // was never written.
  volatile const uint32_t* words = static_cast<volatile const uint32_t*>(buffer);

  unsigned long long sum = 0;
  for (size_t i = 0; i < kWordsPerThread; ++i) {
    sum += words[i];
  }

  // Mean word value. The sum of 262144 words each < 2^32 stays below 2^53, so
  // the double holds it exactly; the division is what makes the result
  // fractional and therefore worth flooring on the host.
  const double checksum =
      static_cast<double>(sum) / static_cast<double>(kWordsPerThread);

  free(buffer);

  // Step 4: the result is only published after the delay has elapsed.
  busy_wait(delay_ticks, use_wall_clock);

  // Step 5.
  checksums[tid] = checksum;
}

// ---------------------------------------------------------------------------
// Host code
// ---------------------------------------------------------------------------

namespace {

// Ticks per millisecond for the timer the kernel will read.
struct TimerConfig {
  unsigned long long ticks_per_ms;
  bool use_wall_clock;
};

TimerConfig query_timer_config(int device) {
  int wall_clock_khz = 0;
  if (hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate,
                            device) == hipSuccess &&
      wall_clock_khz > 0) {
    return {static_cast<unsigned long long>(wall_clock_khz), true};
  }
  hipGetLastError();  // Swallow the failed attribute query.

  int clock_khz = 0;
  if (hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeClockRate, device) ==
          hipSuccess &&
      clock_khz > 0) {
    std::fprintf(stderr,
                 "warning: no wall clock rate reported, falling back to the "
                 "shader clock; the delay may be inaccurate under clock "
                 "scaling\n");
    return {static_cast<unsigned long long>(clock_khz), false};
  }
  hipGetLastError();

  // 100 MHz is the constant clock on current CDNA and RDNA parts.
  std::fprintf(stderr,
               "warning: no clock rate reported, assuming a 100 MHz wall "
               "clock\n");
  return {100000ull, true};
}

int floor_to_int(double value) {
  const double floored = std::floor(value);
  if (floored >= static_cast<double>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  if (floored <= static_cast<double>(std::numeric_limits<int>::min())) {
    return std::numeric_limits<int>::min();
  }
  return static_cast<int>(floored);
}

void print_array(const int* values, int count) {
  std::printf("Checksums (floored) for %d GPU threads:\n", count);
  for (int i = 0; i < count; ++i) {
    if (i % 10 == 0) {
      std::printf("  [%3d..%3d]", i, i + 9);
    }
    std::printf(" %10d", values[i]);
    if (i % 10 == 9) {
      std::printf("\n");
    }
  }
}

}  // namespace

int main() {
  // Step 1.
  int checksums[kNumThreads] = {};

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::printf("Device %d: %s (%s)\n", device, props.name, props.gcnArchName);
  std::printf("Threads: %d, per-thread allocation: %zu KiB, delay: %.0f ms\n\n",
              kNumThreads, kBytesPerThread / 1024,
              kDelayMilliseconds);

  // The device-side allocator has a small default heap; 100 concurrent 1 MiB
  // allocations do not fit in it. Older runtimes reject the resize, in which
  // case the allocations may still fit and the kernel reports per-thread
  // failures if they do not.
  const hipError_t heap_status =
      hipDeviceSetLimit(hipLimitMallocHeapSize, kDeviceHeapBytes);
  if (heap_status == hipErrorUnsupportedLimit ||
      heap_status == hipErrorNotSupported) {
    std::fprintf(stderr,
                 "warning: this runtime cannot resize the device malloc heap; "
                 "set HIP_MALLOC_HEAP_SIZE=%zu if threads report allocation "
                 "failures\n",
                 kDeviceHeapBytes);
  } else {
    HIP_CHECK(heap_status);
  }

  const TimerConfig timer = query_timer_config(device);
  const unsigned long long delay_ticks = static_cast<unsigned long long>(
      kDelayMilliseconds * static_cast<double>(timer.ticks_per_ms));

  double* d_checksums = nullptr;
  HIP_CHECK(hipMalloc(&d_checksums, sizeof(double) * kNumThreads));

  hipEvent_t start_event = nullptr;
  hipEvent_t stop_event = nullptr;
  HIP_CHECK(hipEventCreate(&start_event));
  HIP_CHECK(hipEventCreate(&stop_event));

  // Step 2.
  HIP_CHECK(hipEventRecord(start_event));
  checksum_uninitialized_memory<<<kNumBlocks, kBlockSize>>>(
      d_checksums, delay_ticks, timer.use_wall_clock);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(stop_event));
  HIP_CHECK(hipDeviceSynchronize());

  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start_event, stop_event));

  // Step 5.
  double host_checksums[kNumThreads] = {};
  HIP_CHECK(hipMemcpy(host_checksums, d_checksums,
                      sizeof(double) * kNumThreads, hipMemcpyDeviceToHost));

  int failed_allocations = 0;
  for (int i = 0; i < kNumThreads; ++i) {
    if (host_checksums[i] == kAllocationFailed) {
      ++failed_allocations;
      checksums[i] = -1;
      continue;
    }
    checksums[i] = floor_to_int(host_checksums[i]);
  }

  // Step 6.
  print_array(checksums, kNumThreads);
  std::printf("\nKernel wall time: %.2f ms (expected >= %.0f ms)\n", elapsed_ms,
              kDelayMilliseconds);
  if (failed_allocations > 0) {
    std::fprintf(stderr,
                 "warning: %d thread(s) could not allocate %zu bytes from the "
                 "device heap and reported -1\n",
                 failed_allocations, kBytesPerThread);
  }

  HIP_CHECK(hipEventDestroy(start_event));
  HIP_CHECK(hipEventDestroy(stop_event));
  HIP_CHECK(hipFree(d_checksums));

  return failed_allocations == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
