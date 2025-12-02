NVCC = nvcc
# Support multiple compute capabilities for better GPU compatibility
NVCCFLAGS = -O3 -std=c++14 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89
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
	@echo "Использование: ./gemm [N] [numStreams] [tileSize]"
	@echo "  N - размер матрицы (по умолчанию 2048)"
	@echo "  numStreams - количество потоков (по умолчанию 4)"
	@echo "  tileSize - размер тайла (по умолчанию 512)"

.PHONY: clean run help

