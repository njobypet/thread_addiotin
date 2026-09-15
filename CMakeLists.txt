cmake_minimum_required(VERSION 3.21)

project(thread_addiotin
        VERSION 1.0.0
        DESCRIPTION "100 named kernels, each checksumming device-heap memory with a 2 second delay"
        LANGUAGES CXX HIP)

# ROCm normally lives in /opt/rocm; let the user override with -DROCM_PATH=...
if(NOT DEFINED ROCM_PATH)
  if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH})
  else()
    set(ROCM_PATH "/opt/rocm")
  endif()
endif()
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/hip")

find_package(hip REQUIRED)

add_executable(thread_addition src/thread_addition.hip)

target_compile_features(thread_addition PRIVATE cxx_std_17)
target_link_libraries(thread_addition PRIVATE hip::device)

set_target_properties(thread_addition PROPERTIES
                      HIP_STANDARD 17
                      HIP_STANDARD_REQUIRED ON)

# Build for the GPUs in the machine unless the user names targets explicitly,
# e.g. -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx942".
if(NOT CMAKE_HIP_ARCHITECTURES)
  set_property(TARGET thread_addition PROPERTY HIP_ARCHITECTURES OFF)
endif()

install(TARGETS thread_addition RUNTIME DESTINATION bin)
