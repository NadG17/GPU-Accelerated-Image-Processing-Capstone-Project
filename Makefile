# GPU-Accelerated Image Processing Makefile
# Author: GPU Programming Specialization Student

PYTHON := python3
PIP := pip3
MAIN_SCRIPT := main.py
SAMPLES_DIR := samples
OUTPUTS_DIR := outputs

.PHONY: all setup test clean run demo help install-deps check-cuda

# Default target
all: setup demo

# Help target
help:
	@echo "GPU-Accelerated Image Processing - Makefile"
	@echo "============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  all         - Setup and run demo (default)"
	@echo "  setup       - Install dependencies and create directories"
	@echo "  install-deps - Install Python dependencies"
	@echo "  check-cuda  - Check CUDA availability"
	@echo "  test        - Run basic functionality test"
	@echo "  demo        - Run demonstration with sample images"
	@echo "  run         - Run with custom parameters (use ARGS=...)"
	@echo "  clean       - Clean up generated files"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                                    # Run full demo"
	@echo "  make run ARGS=\"input.jpg output.jpg --filter blur\""
	@echo "  make test"
	@echo "  make clean"

# Setup target - prepare environment
setup: install-deps check-cuda
	@echo "Setting up project directories..."
	@mkdir -p $(SAMPLES_DIR)
	@mkdir -p $(OUTPUTS_DIR)
	@echo "Setup complete!"

# Install Python dependencies
install-deps:
	@echo "Installing Python dependencies..."
	@$(PIP) install -r requirements.txt
	@echo "Dependencies installed!"

# Check CUDA availability
check-cuda:
	@echo "Checking CUDA availability..."
	@$(PYTHON) -c "import pycuda.autoinit; print('✓ CUDA is working correctly')" || \
		(echo "✗ CUDA not available. Please install NVIDIA drivers and CUDA toolkit."; exit 1)

# Test basic functionality
test: setup
	@echo "Running basic functionality test..."
	@echo "This test requires at least one sample image in $(SAMPLES_DIR)/"
	@if [ ! -f "$(SAMPLES_DIR)/test.jpg" ] && [ ! -f "$(SAMPLES_DIR)/landscape.jpg" ]; then \
		echo "Warning: No test images found. Please add images to $(SAMPLES_DIR)/ directory"; \
		echo "Test cannot proceed without sample images."; \
		exit 1; \
	fi
	@# Find first available image for testing
	@TEST_IMAGE=$$(ls $(SAMPLES_DIR)/*.jpg 2>/dev/null | head -1); \
	if [ -n "$$TEST_IMAGE" ]; then \
		echo "Testing with image: $$TEST_IMAGE"; \
		$(PYTHON) $(MAIN_SCRIPT) "$$TEST_IMAGE" "$(OUTPUTS_DIR)/test_output.jpg" \
			--filter blur --verbose; \
		if [ $$? -eq 0 ]; then \
			echo "✓ Basic test passed!"; \
		else \
			echo "✗ Basic test failed!"; \
			exit 1; \
		fi; \
	else \
		echo "No test images available"; \
		exit 1; \
	fi

# Run demonstration
demo: setup
	@echo "Running GPU Image Processing demonstration..."
	@./run.sh

# Run with custom arguments
run: setup
	@if [ -z "$(ARGS)" ]; then \
		echo "Usage: make run ARGS=\"input.jpg output.jpg --filter blur\""; \
		echo "Available filters: blur, sharpen, edge"; \
		echo "Additional options: --compare-cpu --verbose"; \
		exit 1; \
	fi
	@$(PYTHON) $(MAIN_SCRIPT) $(ARGS)

# Performance benchmark
benchmark: setup
	@echo "Running performance benchmark..."
	@if [ ! -f "$(SAMPLES_DIR)/benchmark.jpg" ]; then \
		echo "Creating benchmark test image..."; \
		$(PYTHON) -c "import cv2; import numpy as np; img=np.random.randint(0,255,(1080,1920,3),dtype=np.uint8); cv2.imwrite('$(SAMPLES_DIR)/benchmark.jpg', img)"; \
	fi
	@echo "Benchmarking blur filter..."
	@$(PYTHON) $(MAIN_SCRIPT) "$(SAMPLES_DIR)/benchmark.jpg" "$(OUTPUTS_DIR)/benchmark_blur.jpg" \
		--filter blur --compare-cpu --verbose
