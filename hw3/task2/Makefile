CXX = mpic++
CXXFLAGS = -O3 -std=c++17 -Wall

SOURCES = src/heat_equation.cpp
TARGET = heat_equation

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) scalability_results.csv

run: $(TARGET)
	mpirun -np 1 ./$(TARGET) 10000

help:
	@echo "Доступные команды:"
	@echo "  make        - собрать проект"
	@echo "  make run    - собрать и запустить"
	@echo "  make clean  - удалить исполняемый файл"
	@echo "  make help   - показать эту справку"
	@echo ""
	@echo "Использование:"
	@echo "  mpirun -np <num_procs> ./$(TARGET) <N>"
	@echo "  где N - количество точек (10000, 25000, 50000)"

.PHONY: clean run help

