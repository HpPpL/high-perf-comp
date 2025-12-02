# OpenMP GPU compilation using nvc (NVIDIA HPC SDK)
NVC = nvc
# Use -mp=gpu without specific architecture - nvc will auto-detect
NVCFLAGS = -O3 -mp=gpu
TARGET = gemm
SOURCES = src/gemm.c

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    # On cluster, use nvhpc module
    NVC = nvc
    # Let nvc auto-detect GPU architecture
    NVCFLAGS = -O3 -mp=gpu
else
    # Local - may not have nvc, use gcc with OpenMP (will run on CPU)
    NVC = gcc
    NVCFLAGS = -O3 -fopenmp
    $(warning OpenMP GPU requires nvc compiler. Using gcc - will run on CPU.)
endif

$(TARGET): $(SOURCES)
	$(NVC) $(NVCFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

help:
	@echo "Доступные команды:"
	@echo "  make        - собрать проект"
	@echo "  make run    - собрать и запустить"
	@echo "  make clean  - удалить исполняемый файл"
	@echo ""
	@echo "Для компиляции OpenMP GPU:"
	@echo "  module load nvidia_sdk/nvhpc/23.5"
	@echo "  nvc -mp=gpu gemm.c"

.PHONY: clean run help

