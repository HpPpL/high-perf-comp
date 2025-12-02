NVCC = nvcc
# Support multiple compute capabilities for better GPU compatibility
# sm_70: Volta, sm_75: Turing, sm_80: Ampere, sm_86: Ampere (newer), sm_89: Ada Lovelace
NVCCFLAGS = -O3 -std=c++14 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89
TARGET = gemm
SOURCES = src/gemm.cu

# Detect if running on cluster (Linux) or local (macOS)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    # On cluster, use nvhpc module
    NVCC = nvcc
    NVCCFLAGS += -Xcompiler -fopenmp
else
    # Local macOS (if CUDA is available)
    NVCCFLAGS += -Xcompiler -fopenmp
endif

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

help:
	@echo "Доступные команды:"
	@echo "  make        - собрать проект"
	@echo "  make run    - собрать и запустить"
	@echo "  make clean  - удалить исполняемый файл"
	@echo "  make help   - показать эту справку"
	@echo ""
	@echo "Для запуска на кластере:"
	@echo "  module load nvidia_sdk/nvhpc/23.5"
	@echo "  srun -n 1 --gpus=1 ./$(TARGET)"

.PHONY: clean run help

