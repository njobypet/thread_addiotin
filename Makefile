# Minimal hipcc build. See README.md for the CMake route.
#
#   make                         # build for the GPUs hipcc detects
#   make GPU_TARGETS=gfx942      # build for a specific architecture
#   make run
#   make clean

ROCM_PATH   ?= /opt/rocm
HIPCC       ?= $(ROCM_PATH)/bin/hipcc
BUILD_DIR   ?= build
TARGET      := $(BUILD_DIR)/thread_addition
SRC         := src/thread_addition.hip

CXXFLAGS    ?= -O3 -std=c++17 -Wall -Wextra
GPU_TARGETS ?=

ifneq ($(strip $(GPU_TARGETS)),)
  CXXFLAGS += $(foreach arch,$(subst ;, ,$(GPU_TARGETS)),--offload-arch=$(arch))
endif

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC) | $(BUILD_DIR)
	$(HIPCC) $(CXXFLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
