cmake_minimum_required(VERSION 3.10)

project(task1 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Более совместимые флаги оптимизации
add_compile_options(-O3 -march=native)

# Проверяем поддержку OpenMP
find_package(OpenMP REQUIRED)

if(APPLE)
  execute_process(
    COMMAND brew --prefix
    OUTPUT_VARIABLE HOMEBREW_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}" "${LIBOMP_PREFIX}")
  list(APPEND CMAKE_LIBRARY_PATH "${LIBOMP_PREFIX}/lib")
  list(APPEND CMAKE_INCLUDE_PATH "${LIBOMP_PREFIX}/include")

  set(OpenMP_ROOT "${LIBOMP_PREFIX}")
  set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")

  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
  set(OpenMP_CXX_LIB_NAMES "omp")
endif()

add_executable(${PROJECT_NAME} task1.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
