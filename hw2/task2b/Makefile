NVCC = nvcc
# Support multiple compute capabilities for better GPU compatibility
NVCCFLAGS = -O3 -std=c++14 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89

SOURCES = src/gemm.cu

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    NVCC = nvcc
    NVCCFLAGS += -Xcompiler -fopenmp -lcublas
else
    NVCCFLAGS += -Xcompiler -fopenmp -lcublas
endif

# Default target
all: gemm_global gemm_unified gemm_streams gemm_shared gemm_shared_opt gemm_cublas

# Global memory version
gemm_global: $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -DUSE_GLOBAL -o gemm_global $(SOURCES)

# Unified memory version
gemm_unified: $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -DUSE_UNIFIED -o gemm_unified $(SOURCES)

# Streams version
gemm_streams: $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -DUSE_STREAMS -o gemm_streams $(SOURCES)

# Shared memory version
gemm_shared: $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -DUSE_SHARED -o gemm_shared $(SOURCES)

# Shared memory optimized version
gemm_shared_opt: $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -DUSE_SHARED_OPT -o gemm_shared_opt $(SOURCES)

# cuBLAS version
gemm_cublas: $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUBLAS -o gemm_cublas $(SOURCES)

# All variants in one binary (for comprehensive testing)
gemm_all: $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -DUSE_GLOBAL -DUSE_UNIFIED -DUSE_STREAMS -DUSE_SHARED -DUSE_SHARED_OPT -DUSE_CUBLAS -o gemm_all $(SOURCES)

clean:
	rm -f gemm_global gemm_unified gemm_streams gemm_shared gemm_shared_opt gemm_cublas gemm_all
	rm -f task2b_results.csv

run: all
	@echo "Running all variants..."
	./gemm_global 2048 4 16
	./gemm_unified 2048 4 16
	./gemm_streams 2048 4 16
	./gemm_shared 2048 4 16
	./gemm_shared_opt 2048 4 16
	./gemm_cublas 2048 4 16

help:
	@echo "Доступные команды:"
	@echo "  make all              - собрать все варианты"
	@echo "  make gemm_global      - собрать global memory версию"
	@echo "  make gemm_unified     - собрать unified memory версию"
	@echo "  make gemm_streams     - собрать streams версию"
	@echo "  make gemm_shared      - собрать shared memory версию"
	@echo "  make gemm_shared_opt  - собрать оптимизированную shared memory версию"
	@echo "  make gemm_cublas      - собрать cuBLAS версию"
	@echo "  make gemm_all         - собрать все варианты в одном бинарнике"
	@echo "  make run              - собрать и запустить все варианты"
	@echo "  make clean            - удалить исполняемые файлы"
	@echo ""
	@echo "Использование:"
	@echo "  ./gemm_streams [N] [numStreams] [blockSize]"
	@echo "    N - размер матрицы (по умолчанию 2048)"
	@echo "    numStreams - количество streams (по умолчанию 4)"
	@echo "    blockSize - размер блока потоков (по умолчанию 16)"

.PHONY: clean run help all
