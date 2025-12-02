NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_75 -std=c++14
TARGET = gemm
SOURCES = src/gemm.cu

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    NVCC = nvcc
    NVCCFLAGS += -Xcompiler -fopenmp
else
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
	@echo ""
	@echo "Для запуска на кластере:"
	@echo "  module load nvidia_sdk/nvhpc/23.5"
	@echo "  srun -n 1 --gpus=1 -A proj_1447 ./$(TARGET)"

.PHONY: clean run help

