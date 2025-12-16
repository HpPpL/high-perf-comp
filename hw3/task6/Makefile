CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall

# MKL flags - используем переменную окружения, если установлена
ifdef MKLROOT
    MKL_INCLUDE = -I$(MKLROOT)/include
    MKL_LIB_DIR = -L$(MKLROOT)/lib/intel64
else
    # Пробуем стандартные пути
    MKLROOT = /opt/intel/oneapi/mkl/latest
    ifeq ($(wildcard $(MKLROOT)/include),)
        MKLROOT = /opt/intel/mkl
    endif
    ifeq ($(wildcard $(MKLROOT)/include),)
        MKLROOT = /usr/local/mkl
    endif
    MKL_INCLUDE = -I$(MKLROOT)/include
    MKL_LIB_DIR = -L$(MKLROOT)/lib/intel64
endif

MKL_LIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl

# Detect if running on cluster (Linux) or local (macOS)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS - try to find MKL or use Accelerate framework
    ifeq ($(wildcard $(MKLROOT)/include),)
        MKL_INCLUDE = -I/opt/intel/mkl/include
        MKL_LIB_DIR = -L/opt/intel/mkl/lib
    endif
endif

SOURCES = src/heat_equation_2d.cpp
TARGET = heat_equation_2d

$(TARGET): $(SOURCES)
	@echo "Compiling with MKL..."
	@echo "MKL_INCLUDE: $(MKL_INCLUDE)"
	@echo "MKL_LIB_DIR: $(MKL_LIB_DIR)"
	@if [ -z "$(MKLROOT)" ] || [ ! -d "$(MKLROOT)/include" ]; then \
		echo "Warning: MKL include directory not found. Check MKLROOT environment variable."; \
	fi
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

