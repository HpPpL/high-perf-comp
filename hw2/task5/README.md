# Task 5: OpenMP GPU Matrix Multiplication

## Описание

Реализация умножения матриц на GPU с использованием OpenMP target directives. Это позволяет использовать OpenMP для программирования GPU без явного написания CUDA кода.

## Особенности

- Использование `#pragma omp target teams distribute parallel for`
- Автоматическое управление памятью через `map` clauses
- Код выглядит как обычный OpenMP, но выполняется на GPU
- Компиляция через `nvc -mp=gpu`

## OpenMP директивы

```c
#pragma omp target teams distribute parallel for collapse(2) \
    map(to: A[0:N*N], B[0:N*N]) map(tofrom: C[0:N*N])
```

- `target` - выполнение на GPU
- `teams` - создание команд (teams)
- `distribute` - распределение итераций
- `parallel for` - параллельные циклы
- `map` - управление памятью

## Компиляция

### На кластере:
```bash
module load nvidia_sdk/nvhpc/23.5
nvc -mp=gpu -O3 -o gemm gemm.c
```

Или через Makefile:
```bash
make
```

## Запуск

```bash
./gemm [N]
```

На кластере через SLURM:
```bash
srun -n 1 --gpus=1 -A proj_1447 ./gemm 1024
```

## Преимущества OpenMP GPU

- **Портативность**: один код для CPU и GPU
- **Простота**: не нужно писать CUDA kernels
- **Автоматическая оптимизация**: компилятор оптимизирует код
- **Знакомый синтаксис**: для тех, кто знает OpenMP

## Ограничения

- Требует компилятор с поддержкой OpenMP target (nvc)
- Может быть менее производительным, чем оптимизированный CUDA код
- Ограниченный контроль над деталями выполнения

