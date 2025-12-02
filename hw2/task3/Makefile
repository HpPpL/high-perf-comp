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

.PHONY: clean run

