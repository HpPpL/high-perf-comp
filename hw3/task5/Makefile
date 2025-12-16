NVCC = nvcc
NVCCFLAGS = -O3 -std=c++14 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89

SOURCES = src/heat_equation.cu
TARGET = heat_equation

# Detect if running on cluster (Linux) or local (macOS)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    NVCC = nvcc
    NVCCFLAGS += -Xcompiler -fopenmp
else
    NVCCFLAGS += -Xcompiler -fopenmp
endif

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCES) -lmpi

clean:
	rm -f $(TARGET)

run: $(TARGET)
	mpirun -np 2 ./$(TARGET) 10000

help:
	@echo "Доступные команды:"
	@echo "  make        - собрать проект"
	@echo "  make run    - собрать и запустить (2 процесса)"
	@echo "  make clean  - удалить исполняемый файл"
	@echo "  make help   - показать эту справку"
	@echo ""
	@echo "Использование:"
	@echo "  mpirun -np 2 ./$(TARGET) <N>"
	@echo "  Примечание: требуется ровно 2 MPI процесса"

.PHONY: clean run help

