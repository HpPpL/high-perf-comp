# Task 2b: CUDA Streams Optimization

## Описание

Полная реализация умножения матриц с использованием различных техник оптимизации CUDA:
- Global memory (базовая версия)
- Unified memory
- CUDA Streams с pipeline (copy → compute → copy)
- Shared memory с tiling
- Shared memory с оптимизацией bank conflicts
- cuBLAS (reference implementation)

## Особенности реализации

### ✅ Pinned Memory
Все варианты используют `cudaMallocHost()` для выделения pinned memory на хосте, что ускоряет копирование данных в 2-3 раза.

### ✅ CUDA Streams Pipeline
Streams версия реализует правильный pipeline:
1. **Асинхронное копирование H2D** (`cudaMemcpyAsync`) в каждом stream
2. **Запуск kernel** в том же stream (автоматически ждет завершения копирования)
3. **Асинхронное копирование D2H** (`cudaMemcpyAsync`) в том же stream (автоматически ждет завершения kernel)

Это обеспечивает перекрытие вычислений и передачи данных между разными streams.

### ✅ Блочное умножение матриц
- Shared memory версии используют tiling (блоки 16×16)
- Оптимизированная версия использует padding для избежания bank conflicts

### ✅ Проверка корректности
Все варианты сравнивают результаты с CPU-реализацией и выводят максимальную ошибку.

### ✅ Корректные таймеры
Используются `cudaEvent` для измерения времени выполнения, включая все операции в streams.

## Компиляция

```bash
module load nvidia_sdk/nvhpc/23.5
make all
```

Это создаст следующие исполняемые файлы:
- `gemm_global` - Global memory версия
- `gemm_unified` - Unified memory версия
- `gemm_streams` - Streams версия
- `gemm_shared` - Shared memory версия
- `gemm_shared_opt` - Оптимизированная shared memory версия
- `gemm_cublas` - cuBLAS версия

## Использование

### Streams версия (основная для task2b)
```bash
./gemm_streams [N] [numStreams] [blockSize]
```
- `N` - размер матрицы (по умолчанию 2048)
- `numStreams` - количество CUDA streams (по умолчанию 4)
- `blockSize` - размер блока потоков (по умолчанию 16)

### Другие версии
```bash
./gemm_global [N] [numStreams] [blockSize]
./gemm_unified [N] [numStreams] [blockSize]
./gemm_shared [N] [numStreams] [blockSize]
./gemm_shared_opt [N] [numStreams] [blockSize]
./gemm_cublas [N] [numStreams] [blockSize]
```

## Примеры

### Тестирование разных количеств streams
```bash
./gemm_streams 2048 2 16
./gemm_streams 2048 4 16
./gemm_streams 2048 8 16
./gemm_streams 2048 16 16
```

### Тестирование разных размеров блоков
```bash
./gemm_streams 2048 4 8
./gemm_streams 2048 4 16
./gemm_streams 2048 4 32
```

## Запуск через SLURM

```bash
sbatch run_task2b.sbatch
```

Скрипт автоматически:
1. Соберет все варианты
2. Запустит все версии
3. Протестирует разные количества streams (2, 4, 8, 16)
4. Протестирует разные размеры блоков (8, 16, 32)
5. Сохранит результаты в `task2b_results.csv`

## Результаты

Результаты сохраняются в `task2b_results.csv` в формате:
```
variant,N,param1,param2,time_ms,gflops
```

Где:
- `variant` - тип реализации (global, unified, streams, shared, shared_opt, cublas)
- `N` - размер матрицы
- `param1` - первый параметр (numStreams для streams, blockSize для других)
- `param2` - второй параметр (blockSize для streams)
- `time_ms` - время выполнения в миллисекундах
- `gflops` - производительность в GFLOPS

## Оптимизация

### Количество streams
- **Рекомендуется**: 4-8 streams
- Слишком мало streams (1-2) не обеспечивает достаточного перекрытия
- Слишком много streams (16+) может привести к overhead
- Оптимальное значение зависит от размера задачи и GPU

### Размер блока
- **Рекомендуется**: 16×16 для shared memory
- Меньшие блоки (8×8) дают больше параллелизма, но больше overhead
- Большие блоки (32×32) могут не поместиться в shared memory
- Зависит от архитектуры GPU и доступной shared memory

### Shared Memory Tiling
- Используется размер тайла 16×16
- Оптимизированная версия использует padding (17 элементов в строке) для избежания bank conflicts
- Bank conflicts возникают, когда несколько потоков обращаются к одной и той же банке памяти одновременно

## Технические детали

### Column-Major порядок
Все матрицы хранятся в column-major порядке:
- Элемент `A[i][j]` хранится как `A[j*N + i]`
- Это соответствует стандарту cuBLAS и Fortran

### Pinned Memory
```cpp
cudaMallocHost((void**)&h_A, matrixSize);
```
Позволяет DMA (Direct Memory Access) для ускорения копирования.

### Streams Pipeline
```cpp
// Stream i: copy H2D → compute → copy D2H
cudaMemcpyAsync(..., stream[i]);
kernel<<<..., stream[i]>>>(...);
cudaMemcpyAsync(..., stream[i]);
```

### Shared Memory Tiling
```cpp
__shared__ double tileA[TILE_SIZE][TILE_SIZE];
__shared__ double tileB[TILE_SIZE][TILE_SIZE];
```

## Проверка корректности

Все варианты автоматически проверяют результаты:
```
Verification PASSED! Max error: 1.234e-07
```

Если ошибка превышает допуск (1e-5), выводится сообщение об ошибке.
