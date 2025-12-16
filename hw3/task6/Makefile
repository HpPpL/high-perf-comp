CXX = g++
CXXFLAGS = -O3 -std=c++17 -Wall

# MKL flags - используем переменную окружения, если установлена
ifdef MKLROOT
    MKL_INCLUDE = -I$(MKLROOT)/include
    # Используем путь из переменной окружения, если установлен
    ifdef MKLROOT_LIB
        MKL_LIB_DIR = -L$(MKLROOT_LIB)
        $(info Using MKL lib from MKLROOT_LIB: $(MKLROOT_LIB))
    else
        # Пробуем найти библиотеки в разных местах
        ifneq ($(wildcard $(MKLROOT)/lib/intel64/libmkl_core.so),)
            MKL_LIB_DIR = -L$(MKLROOT)/lib/intel64
            $(info Found MKL lib at: $(MKLROOT)/lib/intel64)
        else ifneq ($(wildcard $(MKLROOT)/lib/libmkl_core.so),)
            MKL_LIB_DIR = -L$(MKLROOT)/lib
            $(info Found MKL lib at: $(MKLROOT)/lib)
        else ifneq ($(wildcard $(MKLROOT)/lib64/libmkl_core.so),)
            MKL_LIB_DIR = -L$(MKLROOT)/lib64
            $(info Found MKL lib at: $(MKLROOT)/lib64)
        else
            # Пробуем найти через find
            MKL_LIB_FOUND := $(shell find $(MKLROOT) -name "libmkl_core.so" -o -name "libmkl_core.a" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
            ifneq ($(MKL_LIB_FOUND),)
                MKL_LIB_DIR = -L$(MKL_LIB_FOUND)
                $(info Found MKL lib via find at: $(MKL_LIB_FOUND))
            else
                MKL_LIB_DIR = -L$(MKLROOT)/lib/intel64
                $(warning MKL lib not found, using default: $(MKLROOT)/lib/intel64)
            endif
        endif
    endif
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
    # Пробуем найти библиотеки
    ifneq ($(wildcard $(MKLROOT)/lib/intel64/libmkl_core.so),)
        MKL_LIB_DIR = -L$(MKLROOT)/lib/intel64
    else ifneq ($(wildcard $(MKLROOT)/lib/libmkl_core.so),)
        MKL_LIB_DIR = -L$(MKLROOT)/lib
    else ifneq ($(wildcard $(MKLROOT)/lib64/libmkl_core.so),)
        MKL_LIB_DIR = -L$(MKLROOT)/lib64
    else
        MKL_LIB_DIR = -L$(MKLROOT)/lib/intel64
    endif
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

