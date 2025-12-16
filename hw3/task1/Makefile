CXX = mpic++
CXXFLAGS = -O3 -std=c++17 -Wall

SOURCES = src/heat_equation.cpp
TARGET = heat_equation

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) task1_results.txt

run: $(TARGET)
	mpirun -np 1 ./$(TARGET)

help:
	@echo "Доступные команды:"
	@echo "  make        - собрать проект"
	@echo "  make run    - собрать и запустить (1 процесс)"
	@echo "  make clean  - удалить исполняемый файл"
	@echo "  make help   - показать эту справку"
	@echo ""
	@echo "Для запуска с несколькими процессами:"
	@echo "  mpirun -np 4 ./$(TARGET)"

.PHONY: clean run help

