CXX = g++
CXXFLAGS = -O3 -std=c++17
MACOS_FLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
LINUX_FLAGS = -fopenmp

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OPENMP_FLAGS = $(MACOS_FLAGS)
else
    OPENMP_FLAGS = $(LINUX_FLAGS)
endif

SOURCES = src/task2.cpp
TARGET = task2

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) scalability_results.csv

run: $(TARGET)
	./$(TARGET)

help:
	@echo "Доступные команды:"
	@echo "  make        - собрать проект"
	@echo "  make run    - собрать и запустить"
	@echo "  make clean  - удалить исполняемый файл и результаты"
	@echo "  make help   - показать эту справку"

.PHONY: clean run help
