# Makefile for Task 3 - BLAS Comparison
# Supports OpenBLAS and MKL libraries
# Works on both macOS (local) and Linux (cluster)

CXX = g++

# Detect OS and set appropriate flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS settings
    CXXFLAGS = -O3 -std=c++11 -Wall -Wextra
    LDFLAGS = 
    OPENBLAS_INCLUDE = -I/opt/homebrew/opt/openblas/include
    OPENBLAS_LIB = -L/opt/homebrew/opt/openblas/lib -lopenblas
    MKL_AVAILABLE = false
else
    # Linux settings (cluster)
    CXXFLAGS = -O3 -std=c++11 -fopenmp -Wall -Wextra
    LDFLAGS = -fopenmp
    OPENBLAS_INCLUDE = 
    OPENBLAS_LIB = -lopenblas
    MKL_AVAILABLE = true
endif

# Default target - compile with OpenBLAS
all: task3_openblas

# OpenBLAS version (works on both macOS and Linux)
task3_openblas: src/task3.cpp src/task3.h src/task3.hpp
	$(CXX) $(CXXFLAGS) -DUSE_OPENBLAS $(OPENBLAS_INCLUDE) -o task3_openblas src/task3.cpp $(OPENBLAS_LIB) $(LDFLAGS)

# MKL version (Linux only)
task3_mkl: src/task3.cpp src/task3.h src/task3.hpp
ifeq ($(MKL_AVAILABLE),true)
	$(CXX) $(CXXFLAGS) -DUSE_MKL -I$(MKLROOT)/include -o task3_mkl src/task3.cpp -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl $(LDFLAGS)
else
	@echo "MKL is not available on macOS. Use 'make task3_openblas' instead."
	@exit 1
endif

# Both versions
both: task3_openblas task3_mkl

# Clean
clean:
	rm -f task3_openblas task3_mkl *.csv

# Run OpenBLAS version
run_openblas: task3_openblas
	./task3_openblas

# Run MKL version  
run_mkl: task3_mkl
	./task3_mkl

# Help
help:
	@echo "Task 3: BLAS Library Comparison"
	@echo "Detected OS: $(UNAME_S)"
	@echo "MKL Available: $(MKL_AVAILABLE)"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build OpenBLAS version (default)"
	@echo "  task3_openblas - Build OpenBLAS version (works on macOS and Linux)"
	@echo "  task3_mkl    - Build MKL version (Linux only)"
	@echo "  both         - Build both versions (Linux only)"
	@echo "  run_openblas - Build and run OpenBLAS version"
	@echo "  run_mkl      - Build and run MKL version (Linux only)"
	@echo "  clean        - Remove executables and results"
	@echo "  help         - Show this help"

.PHONY: all both clean run_openblas run_mkl help
