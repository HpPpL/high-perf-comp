# Makefile for Task 4 - OpenMP Matrix Multiplication with Profiling
# Supports detailed profiling and performance analysis

CXX = g++

# Detect OS and set appropriate flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS settings
    CXXFLAGS = -O3 -std=c++11 -Wall -Wextra
    LDFLAGS = 
    OPENMP_AVAILABLE = false
else
    # Linux settings (cluster)
    CXXFLAGS = -O3 -std=c++11 -fopenmp -Wall -Wextra
    LDFLAGS = -fopenmp
    OPENMP_AVAILABLE = true
endif

# Default target
all: task4

# Main executable
task4: src/task4.cpp src/task4.h src/task4.hpp
ifeq ($(OPENMP_AVAILABLE),true)
	$(CXX) $(CXXFLAGS) -o task4 src/task4.cpp $(LDFLAGS)
else
	@echo "OpenMP not available on macOS. Compiling without OpenMP support."
	$(CXX) $(CXXFLAGS) -o task4 src/task4.cpp $(LDFLAGS)
endif

# Debug version with additional profiling
task4_debug: src/task4.cpp src/task4.h src/task4.hpp
ifeq ($(OPENMP_AVAILABLE),true)
	$(CXX) -g -O0 -std=c++11 -fopenmp -Wall -Wextra -DDEBUG_PROFILING -o task4_debug src/task4.cpp -fopenmp
else
	$(CXX) -g -O0 -std=c++11 -Wall -Wextra -DDEBUG_PROFILING -o task4_debug src/task4.cpp
endif

# Profile version with gprof support
task4_profile: src/task4.cpp src/task4.h src/task4.hpp
ifeq ($(OPENMP_AVAILABLE),true)
	$(CXX) -O2 -std=c++11 -fopenmp -Wall -Wextra -pg -o task4_profile src/task4.cpp -fopenmp
else
	$(CXX) -O2 -std=c++11 -Wall -Wextra -pg -o task4_profile src/task4.cpp
endif

# Clean
clean:
	rm -f task4 task4_debug task4_profile *.csv *.out gmon.out

# Run main version
run: task4
	./task4

# Run debug version
run_debug: task4_debug
	./task4_debug

# Run profile version and generate gprof report
run_profile: task4_profile
	./task4_profile
	gprof task4_profile gmon.out > profile_report.txt
	@echo "Profile report generated: profile_report.txt"

# Help
help:
	@echo "Task 4: OpenMP Matrix Multiplication with Profiling"
	@echo "Detected OS: $(UNAME_S)"
	@echo "OpenMP Available: $(OPENMP_AVAILABLE)"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build main version (default)"
	@echo "  task4        - Build main version"
	@echo "  task4_debug  - Build debug version with additional profiling"
	@echo "  task4_profile - Build version with gprof profiling support"
	@echo "  run          - Build and run main version"
	@echo "  run_debug    - Build and run debug version"
	@echo "  run_profile  - Build, run profile version and generate gprof report"
	@echo "  clean        - Remove executables and results"
	@echo "  help         - Show this help"

.PHONY: all clean run run_debug run_profile help
