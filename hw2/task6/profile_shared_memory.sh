#!/bin/bash
# Скрипт для профилирования программы с shared memory

# Загрузить модули
module load nvidia_sdk/nvhpc/23.5

# Перейти в директорию task2c
cd task2c

# Скомпилировать
make clean
make

# Профилирование с базовыми метриками
echo "=== Basic Profiling ==="
nsys profile --trace=cuda,nvtx --output=gemm_basic_profile ./gemm 2048

# Профилирование с детальными метриками GPU
echo "=== Detailed GPU Profiling ==="
nsys profile --trace=cuda,nvtx --cuda-memory-usage=true \
    --output=gemm_detailed_profile ./gemm 2048

# Экспорт статистики
echo "=== Exporting Statistics ==="
nsys stats --report gputrace gemm_detailed_profile.qdrep > gpu_stats.txt
nsys stats --report cudaapis gemm_detailed_profile.qdrep > cuda_apis.txt

echo "Profiling complete!"
echo "Open gemm_detailed_profile.qdrep in Nsight Systems GUI to view results"
echo "Statistics saved in gpu_stats.txt and cuda_apis.txt"

