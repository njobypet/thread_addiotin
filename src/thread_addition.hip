// thread_addiotin - 100 separately named kernels, each with a 2 s in-kernel delay.
//
// Flow:
//   1. main() declares an array of 100 ints.
//   2. 100 distinct kernels, rocmtestkernel_0 through rocmtestkernel_99, are
//      launched with one GPU thread each, one per stream so they overlap.
//   3. Every kernel malloc()s 1 MiB from the device heap, fills it with
//      pseudo-random words from a per-kernel generator, and checksums it.
//      --uninitialized skips the fill and checksums whatever was already there.
//   4. Each kernel busy-waits 2 seconds after the checksum is computed and
//      before the result is published to global memory.
//   5. The host copies the 100 checksums back and floors each one into the
//      int array from step 1.
//   6. The host prints the array.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// The sequence numbers of the kernels, listed once. Everything else - the
// kernel definitions, the launches, the count - is generated from this list, so
// adding or removing a kernel means editing exactly one place.
#define ROCM_TEST_KERNEL_SEQUENCE(X)                                           \
  X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9)                            \
  X(10) X(11) X(12) X(13) X(14) X(15) X(16) X(17) X(18) X(19)                  \
  X(20) X(21) X(22) X(23) X(24) X(25) X(26) X(27) X(28) X(29)                  \
  X(30) X(31) X(32) X(33) X(34) X(35) X(36) X(37) X(38) X(39)                  \
  X(40) X(41) X(42) X(43) X(44) X(45) X(46) X(47) X(48) X(49)                  \
  X(50) X(51) X(52) X(53) X(54) X(55) X(56) X(57) X(58) X(59)                  \
  X(60) X(61) X(62) X(63) X(64) X(65) X(66) X(67) X(68) X(69)                  \
  X(70) X(71) X(72) X(73) X(74) X(75) X(76) X(77) X(78) X(79)                  \
  X(80) X(81) X(82) X(83) X(84) X(85) X(86) X(87) X(88) X(89)                  \
  X(90) X(91) X(92) X(93) X(94) X(95) X(96) X(97) X(98) X(99)

#define ROCM_TEST_KERNEL_COUNTER(SEQ) +1

// Step 1 / step 2: one kernel, one GPU thread, one array element.
static constexpr int kNumKernels = 0 ROCM_TEST_KERNEL_SEQUENCE(ROCM_TEST_KERNEL_COUNTER);
static constexpr int kExpectedKernels = 100;
static_assert(kNumKernels == kExpectedKernels,
              "ROCM_TEST_KERNEL_SEQUENCE must list exactly 100 kernels");

// Step 3: 1 MiB per kernel, filled and read back as 32-bit words.
static constexpr size_t kBytesPerKernel = 1u << 20;
static constexpr size_t kWordsPerKernel = kBytesPerKernel / sizeof(uint32_t);

// Each word is normalized to [0, 1) before being summed, so a buffer of random
// words checksums to roughly kWordsPerKernel / 2 instead of overflowing an int.
static constexpr double kWordScale = 4294967296.0;  // 2^32

// Step 4.
static constexpr double kDelayMilliseconds = 2000.0;

// Device heap must hold every live allocation at once, plus allocator overhead.
static constexpr size_t kDeviceHeapBytes =
    static_cast<size_t>(kNumKernels) * kBytesPerKernel * 2;

// Written by a kernel whose device-side malloc() failed.
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

// Bundled so the generated kernel signatures and launch statements stay short.
struct KernelArgs {
  double* checksums;
  unsigned long long delay_ticks;
  bool use_wall_clock;
  bool initialize_memory;
  uint32_t seed;
};

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

// Small 32-bit mixing generator. Each kernel runs its own, so the 100 buffers
// hold different data and the checksums differ from one another.
__device__ inline uint32_t next_random(uint32_t& state) {
  state += 0x9e3779b9u;
  uint32_t z = state;
  z = (z ^ (z >> 16)) * 0x21f0aaadu;
  z = (z ^ (z >> 15)) * 0x735a2d97u;
  return z ^ (z >> 15);
}

// The body every rocmtestkernel_<n> shares; the sequence number is the only
// thing that differs between them.
__device__ void run_checksum(int sequence, const KernelArgs& args) {
  // Step 3: 1 MiB straight off the device heap.
  void* buffer = malloc(kBytesPerKernel);
  if (buffer == nullptr) {
    args.checksums[sequence] = kAllocationFailed;
    return;
  }

  // volatile forces the fill and the read to actually touch memory instead of
  // being folded together, and keeps reads of never-written memory alive in
  // --uninitialized mode.
  volatile uint32_t* words = static_cast<volatile uint32_t*>(buffer);

  if (args.initialize_memory) {
    uint32_t state = args.seed ^ (static_cast<uint32_t>(sequence) * 0x9e3779b9u);
    for (size_t i = 0; i < kWordsPerKernel; ++i) {
      words[i] = next_random(state);
    }
  }

  unsigned long long sum = 0;
  for (size_t i = 0; i < kWordsPerKernel; ++i) {
    sum += words[i];
  }

  // Sum of the words with each one normalized to [0, 1). The raw sum of 262144
  // words each < 2^32 stays below 2^53, so the double holds it exactly and the
  // scaling is the only place a fractional part appears - which is what makes
  // the host-side floor meaningful. Random data lands near 131072.
  const double checksum = static_cast<double>(sum) / kWordScale;

  free(buffer);

  // Step 4: the result is only published after the delay has elapsed.
  busy_wait(args.delay_ticks, args.use_wall_clock);

  // Step 5.
  args.checksums[sequence] = checksum;
}

// Step 2: rocmtestkernel_0 ... rocmtestkernel_99, each a real, separately named
// entry point rather than one kernel launched 100 times.
#define DEFINE_ROCM_TEST_KERNEL(SEQ)                                           \
  __global__ void rocmtestkernel_##SEQ(KernelArgs args) {                      \
    run_checksum(SEQ, args);                                                   \
  }

ROCM_TEST_KERNEL_SEQUENCE(DEFINE_ROCM_TEST_KERNEL)

#undef DEFINE_ROCM_TEST_KERNEL

// ---------------------------------------------------------------------------
// Host code
// ---------------------------------------------------------------------------

namespace {

// Ticks per millisecond for the timer the kernels will read.
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
  std::printf("Checksums (floored) from %d kernels:\n", count);
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

struct Options {
  bool initialize_memory = true;
  uint32_t seed = 0;  // Filled with a random seed when not given on the command line.
  bool seed_given = false;
  bool valid = true;
};

Options parse_arguments(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--uninitialized") == 0) {
      options.initialize_memory = false;
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      options.seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 0));
      options.seed_given = true;
    } else {
      std::fprintf(stderr,
                   "usage: %s [--uninitialized] [--seed N]\n"
                   "  --uninitialized  checksum the device heap as-is instead "
                   "of filling it with random words\n"
                   "  --seed N         seed the per-kernel generators for a "
                   "reproducible run\n",
                   argv[0]);
      options.valid = false;
      return options;
    }
  }
  if (!options.seed_given) {
    options.seed = static_cast<uint32_t>(std::random_device{}());
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  const Options options = parse_arguments(argc, argv);
  if (!options.valid) {
    return EXIT_FAILURE;
  }

  // HIP multiplexes streams onto a handful of hardware queues (4 by default),
  // and kernels sharing a queue serialize. With 100 two-second kernels that is
  // the difference between a 2 second run and a 50 second one. Must be set
  // before the runtime initializes, and never overrides an explicit setting.
#if defined(__linux__)
  setenv("GPU_MAX_HW_QUEUES", "16", 0);
#endif

  // Step 1.
  int checksums[kNumKernels] = {};

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::printf("Device %d: %s (%s)\n", device, props.name, props.gcnArchName);
  std::printf(
      "Kernels: %d (rocmtestkernel_0 .. rocmtestkernel_%d), 1 thread each\n",
      kNumKernels, kNumKernels - 1);
  std::printf("Per-kernel allocation: %zu KiB, delay: %.0f ms\n",
              kBytesPerKernel / 1024, kDelayMilliseconds);
  if (options.initialize_memory) {
    std::printf("Memory: filled with random words, seed 0x%08x\n\n",
                options.seed);
  } else {
    std::printf(
        "Memory: left uninitialized (freshly faulted pages usually read back "
        "as zeros)\n\n");
  }

  // The device-side allocator has a small default heap; 100 concurrent 1 MiB
  // allocations do not fit in it. Older runtimes reject the resize, in which
  // case the allocations may still fit and the kernels report failures if they
  // do not.
  const hipError_t heap_status =
      hipDeviceSetLimit(hipLimitMallocHeapSize, kDeviceHeapBytes);
  if (heap_status == hipErrorUnsupportedLimit ||
      heap_status == hipErrorNotSupported) {
    std::fprintf(stderr,
                 "warning: this runtime cannot resize the device malloc heap; "
                 "set HIP_MALLOC_HEAP_SIZE=%zu if kernels report allocation "
                 "failures\n",
                 kDeviceHeapBytes);
  } else {
    HIP_CHECK(heap_status);
  }

  const TimerConfig timer = query_timer_config(device);

  double* d_checksums = nullptr;
  HIP_CHECK(hipMalloc(&d_checksums, sizeof(double) * kNumKernels));

  // One stream per kernel; on the null stream the 100 kernels would run
  // back to back and the program would take 100 x the delay.
  hipStream_t streams[kNumKernels] = {};
  for (int i = 0; i < kNumKernels; ++i) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
  }

  const KernelArgs args{
      d_checksums,
      static_cast<unsigned long long>(kDelayMilliseconds *
                                      static_cast<double>(timer.ticks_per_ms)),
      timer.use_wall_clock, options.initialize_memory, options.seed};

  // Step 2. Expands to 100 launch statements naming the kernels literally, so
  // there is no pointer-to-kernel indirection between here and the GPU.
#define LAUNCH_ROCM_TEST_KERNEL(SEQ)                                           \
  rocmtestkernel_##SEQ<<<1, 1, 0, streams[SEQ]>>>(args);

  const auto launch_start = std::chrono::steady_clock::now();
  ROCM_TEST_KERNEL_SEQUENCE(LAUNCH_ROCM_TEST_KERNEL)
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  const auto launch_end = std::chrono::steady_clock::now();

#undef LAUNCH_ROCM_TEST_KERNEL

  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(launch_end - launch_start)
          .count();

  // Step 5.
  double host_checksums[kNumKernels] = {};
  HIP_CHECK(hipMemcpy(host_checksums, d_checksums,
                      sizeof(double) * kNumKernels, hipMemcpyDeviceToHost));

  for (int i = 0; i < kNumKernels; ++i) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }

  int failed_allocations = 0;
  for (int i = 0; i < kNumKernels; ++i) {
    if (host_checksums[i] == kAllocationFailed) {
      ++failed_allocations;
      checksums[i] = -1;
      continue;
    }
    checksums[i] = floor_to_int(host_checksums[i]);
  }

  // Step 6.
  print_array(checksums, kNumKernels);
  std::printf("\nAll %d kernels completed in %.2f ms (delay is %.0f ms)\n",
              kNumKernels, elapsed_ms, kDelayMilliseconds);
  if (elapsed_ms > kDelayMilliseconds * 2.0) {
    std::fprintf(stderr,
                 "note: the kernels did not all run concurrently; raise "
                 "GPU_MAX_HW_QUEUES to overlap more of them\n");
  }
  if (failed_allocations > 0) {
    std::fprintf(stderr,
                 "warning: %d kernel(s) could not allocate %zu bytes from the "
                 "device heap and reported -1\n",
                 failed_allocations, kBytesPerKernel);
  }

  HIP_CHECK(hipFree(d_checksums));

  return failed_allocations == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
