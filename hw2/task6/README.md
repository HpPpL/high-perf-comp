# Task 6: NVIDIA Nsight Profiling

## Описание

Исследование выполнения программы с разделяемой памятью с использованием профилировщика NVIDIA Nsight. Это исследовательское задание для глубокого анализа производительности.

## NVIDIA Nsight Systems

NVIDIA Nsight Systems - это системный профилировщик для анализа производительности CUDA приложений.

## Установка и настройка

### На кластере

Nsight Systems обычно доступен через модули:
```bash
module load nvidia_sdk/nvhpc/23.5
# или
module load cuda
nsys --version
```

### Локально

Скачать с сайта NVIDIA: https://developer.nvidia.com/nsight-systems

## Профилирование программы

### Базовое профилирование

```bash
# Профилирование с сохранением отчета
nsys profile --output=gemm_profile ./gemm 2048

# С дополнительными метриками
nsys profile --trace=cuda,nvtx --output=gemm_profile ./gemm 2048
```

### Детальное профилирование

```bash
# С GPU метриками
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true \
    --output=gemm_detailed ./gemm 2048
```

### Профилирование с SLURM

```bash
# В SLURM скрипте
srun -n 1 --gpus=1 -A proj_1447 \
    nsys profile --output=gemm_profile ./gemm 2048
```

## Анализ результатов

### Просмотр отчета

```bash
# Открыть в Nsight Systems GUI
nsys-ui gemm_profile.qdrep

# Или экспортировать в текстовый формат
nsys stats --report gputrace gemm_profile.qdrep > report.txt
```

### Ключевые метрики для анализа

1. **GPU Utilization**
   - Процент времени использования GPU
   - Должен быть близок к 100%

2. **Memory Throughput**
   - Пропускная способность памяти
   - Сравнить глобальную и shared memory

3. **Kernel Execution Time**
   - Время выполнения каждого kernel
   - Идентификация узких мест

4. **Memory Transfers**
   - Время копирования CPU ↔ GPU
   - Эффективность использования pinned memory

5. **Shared Memory Usage**
   - Использование shared memory
   - Проверка на bank conflicts

## Пример анализа

### 1. Профилирование версии с shared memory

```bash
cd task2c
make
nsys profile --trace=cuda,nvtx --output=shared_memory_profile ./gemm 2048
```

### 2. Анализ результатов

Открыть `shared_memory_profile.qdrep` в Nsight Systems и проверить:

- **Kernel Performance**: время выполнения kernel
- **Memory Bandwidth**: использование пропускной способности памяти
- **Occupancy**: загрузка SM (Streaming Multiprocessors)
- **Shared Memory**: использование shared memory per block

### 3. Сравнение версий

Профилировать разные версии и сравнить:

```bash
# Task 1: Global memory
cd ../task1
nsys profile --output=global_mem_profile ./gemm 2048

# Task 2c: Shared memory
cd ../task2c
nsys profile --output=shared_mem_profile ./gemm 2048

# Task 4: cuBLAS
cd ../task4
nsys profile --output=cublas_profile ./gemm 2048
```

## Ключевые вопросы для исследования

1. **Эффективность использования памяти**
   - Какой процент времени тратится на копирование данных?
   - Используется ли pinned memory эффективно?

2. **Производительность kernel**
   - Какие kernel выполняются дольше всего?
   - Есть ли узкие места в вычислениях?

3. **Использование shared memory**
   - Сколько shared memory используется?
   - Есть ли bank conflicts?
   - Как это влияет на производительность?

4. **GPU Utilization**
   - Насколько эффективно используется GPU?
   - Есть ли простои (idle time)?

5. **Сравнение реализаций**
   - Какая реализация наиболее эффективна?
   - Где теряется производительность?

## Пример отчета

После профилирования можно создать отчет с ключевыми метриками:

```bash
nsys stats --report gputrace shared_memory_profile.qdrep > analysis.txt
```

Отчет будет содержать:
- Общее время выполнения
- Время выполнения каждого kernel
- Использование памяти
- GPU utilization
- И другие метрики

## Рекомендации

1. **Начните с базового профилирования** для понимания общей картины
2. **Используйте детальное профилирование** для анализа конкретных проблем
3. **Сравнивайте разные версии** для понимания эффекта оптимизаций
4. **Обращайте внимание на memory transfers** - они часто являются узким местом
5. **Анализируйте occupancy** - низкая загрузка SM может указывать на проблемы

## Дополнительные ресурсы

- NVIDIA Nsight Systems Documentation: https://docs.nvidia.com/nsight-systems/
- CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Profiling CUDA Applications: https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/

