Процесс компиляции и запуска программы на узлах Харизмы
==========================================================

1. ПОДКЛЮЧЕНИЕ К КЛАСТЕРУ
-------------------------
ssh username@harisma.hpc.hse.ru

2. ЗАГРУЗКА МОДУЛЕЙ
-------------------
module load nvidia_sdk/nvhpc/23.5

3. КОМПИЛЯЦИЯ
-------------
Для CUDA задач (task1-task4, task6):
  cd task1  # или другая задача
  make
  # или вручную:
  nvcc -O3 -arch=sm_75 -o gemm src/gemm.cu

Для OpenMP GPU (task5):
  cd task5
  nvc -mp=gpu -O3 -o gemm src/gemm.c
  # или через Makefile:
  make

4. ЗАПУСК ЧЕРЕЗ SLURM
---------------------
sbatch run_task1.sbatch

Или создать свой скрипт:
  #!/bin/bash
  #SBATCH --job-name=gemm
  #SBATCH --output=output_%j.txt
  #SBATCH --error=error_%j.txt
  #SBATCH --time=00:10:00
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --gpus=1
  
  module load nvidia_sdk/nvhpc/23.5
  ./gemm 2048

5. ИНТЕРАКТИВНЫЙ ЗАПУСК
-----------------------
srun -n 1 --gpus=1 ./gemm 2048

6. ВЫДЕЛЕНИЕ УЗЛА
-----------------
salloc -n 1 --gpus=1
# После выделения можно запускать многократно:
./gemm 2048
./gemm 1024
# Для освобождения: Ctrl+D

7. ПРОВЕРКА СТАТУСА
-------------------
squeue -u $USER

8. ПРОСМОТР РЕЗУЛЬТАТОВ
-----------------------
cat output_*.txt
# или
tail -f output_*.txt

9. ПРОФИЛИРОВАНИЕ (Task 6)
---------------------------
module load nvidia_sdk/nvhpc/23.5
nsys profile --output=profile ./gemm 2048
nsys-ui profile.qdrep  # для просмотра (если доступен GUI)

10. ОСОБЕННОСТИ КОМПИЛЯЦИИ
---------------------------
- Для cuBLAS (task4): автоматически линкуется через -lcublas
- Для OpenMP GPU (task5): требуется nvc компилятор
- Compute capability: используется sm_75 (можно изменить в Makefile)

11. РАЗМЕРЫ МАТРИЦ ДЛЯ ТЕСТИРОВАНИЯ
-----------------------------------
Рекомендуемые размеры: 512, 1024, 2048, 4096
Учитывайте ограничения памяти GPU при выборе размера.

12. ПРОБЛЕМЫ И РЕШЕНИЯ
----------------------
- "No CUDA devices found": проверьте выделение GPU через SLURM
- "out of memory": уменьшите размер матрицы
- Ошибки компиляции: проверьте версию модуля nvhpc

