CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall

# MKL flags
MKLROOT = /opt/intel/oneapi/mkl/latest
MKL_INCLUDE = -I$(MKLROOT)/include
MKL_LIB_DIR = -L$(MKLROOT)/lib/intel64
MKL_LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# Detect if running on cluster (Linux) or local (macOS)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    # On cluster, MKL should be available via module
    MKL_INCLUDE = -I${MKLROOT}/include
    MKL_LIB_DIR = -L${MKLROOT}/lib/intel64
else
    # macOS - try to find MKL or use alternative
    # For macOS, you might need to install MKL separately or use Accelerate framework
    MKL_INCLUDE = -I/opt/intel/mkl/include
    MKL_LIB_DIR = -L/opt/intel/mkl/lib
endif

SOURCES = src/heat_equation_2d.cpp
TARGET = heat_equation_2d

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(MKL_INCLUDE) -o $(TARGET) $(SOURCES) $(MKL_LIB_DIR) $(MKL_LIBS)

clean:
	rm -f $(TARGET) task6_results.txt task6_performance.csv

run: $(TARGET)
	./$(TARGET) 100

help:
	@echo "Доступные команды:"
	@echo "  make        - собрать проект"
	@echo "  make run    - собрать и запустить"
	@echo "  make clean  - удалить исполняемый файл"
	@echo "  make help   - показать эту справку"
	@echo ""
	@echo "Использование:"
	@echo "  ./$(TARGET) <N>"
	@echo "  где N - размер сетки (NxN точек)"
	@echo ""
	@echo "Примечание: требуется Intel MKL библиотека"

.PHONY: clean run help

