#!/bin/bash
# Скрипт для автоматического запуска всех задач через SLURM

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Получаем абсолютный путь к директории скрипта
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Автоматический запуск всех задач hw3"
echo "=========================================="
echo ""

# Массив задач
tasks=("task1" "task2" "task3" "task4" "task5" "task6")

# Функция для запуска задачи
run_task() {
    local task=$1
    local task_dir="$SCRIPT_DIR/$task"
    
    if [ ! -d "$task_dir" ]; then
        echo -e "${RED}Ошибка: директория $task не найдена!${NC}"
        return 1
    fi
    
    if [ ! -f "$task_dir/run_${task}.sbatch" ]; then
        echo -e "${RED}Ошибка: файл run_${task}.sbatch не найден!${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Запуск $task...${NC}"
    cd "$task_dir"
    
    # Компиляция перед запуском (опционально, можно пропустить)
    if [ "$NO_COMPILE" != "true" ]; then
        echo "  Компиляция..."
        make clean > /dev/null 2>&1
        if make > /dev/null 2>&1; then
            echo -e "  ${GREEN}Компиляция успешна${NC}"
        else
            echo -e "  ${YELLOW}Предупреждение: компиляция не удалась, но продолжаем...${NC}"
        fi
    fi
    
    # Запуск через sbatch
    echo "  Отправка задачи в очередь SLURM..."
    job_id=$(sbatch run_${task}.sbatch 2>&1 | grep -oP '\d+' | head -1)
    
    if [ -n "$job_id" ]; then
        echo -e "  ${GREEN}Задача отправлена, Job ID: $job_id${NC}"
        echo "  Проверить статус: squeue -j $job_id"
        echo "$job_id" > /tmp/hw3_${task}_jobid.txt
    else
        echo -e "  ${RED}Ошибка при отправке задачи!${NC}"
        cd "$SCRIPT_DIR"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
    echo ""
    return 0
}

# Опции запуска
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Использование: $0 [опции] [номера_задач]"
    echo ""
    echo "Опции:"
    echo "  --help, -h          Показать эту справку"
    echo "  --compile-only      Только скомпилировать, не запускать"
    echo "  --no-compile        Запустить без компиляции"
    echo "  --status            Показать статус запущенных задач"
    echo ""
    echo "Примеры:"
    echo "  $0                  Запустить все задачи"
    echo "  $0 1 2 3           Запустить только задачи 1, 2, 3"
    echo "  $0 --compile-only  Только скомпилировать все задачи"
    echo "  $0 --status        Показать статус задач"
    exit 0
fi

# Режим проверки статуса
if [ "$1" == "--status" ]; then
    echo "Проверка статуса задач..."
    echo ""
    for task in "${tasks[@]}"; do
        job_file="/tmp/hw3_${task}_jobid.txt"
        if [ -f "$job_file" ]; then
            job_id=$(cat "$job_file")
            status=$(squeue -j $job_id -h -o "%T" 2>/dev/null)
            if [ -n "$status" ]; then
                echo -e "${task}: ${GREEN}Job $job_id - $status${NC}"
            else
                echo -e "${task}: ${YELLOW}Job $job_id - завершена или не найдена${NC}"
            fi
        else
            echo -e "${task}: ${RED}Не запущена${NC}"
        fi
    done
    exit 0
fi

# Режим только компиляции
if [ "$1" == "--compile-only" ]; then
    echo "Режим: только компиляция"
    echo ""
    for task in "${tasks[@]}"; do
        task_dir="$SCRIPT_DIR/$task"
        if [ -d "$task_dir" ]; then
            echo -e "${YELLOW}Компиляция $task...${NC}"
            cd "$task_dir"
            make clean > /dev/null 2>&1
            if make; then
                echo -e "${GREEN}✓ $task скомпилирован${NC}"
            else
                echo -e "${RED}✗ Ошибка компиляции $task${NC}"
            fi
            cd "$SCRIPT_DIR"
            echo ""
        fi
    done
    exit 0
fi

# Режим без компиляции
NO_COMPILE=false
if [ "$1" == "--no-compile" ]; then
    NO_COMPILE=true
    shift
fi

# Определяем какие задачи запускать
if [ $# -eq 0 ]; then
    # Запускаем все задачи
    tasks_to_run=("${tasks[@]}")
else
    # Запускаем только указанные задачи
    tasks_to_run=()
    for arg in "$@"; do
        if [[ "$arg" =~ ^[1-6]$ ]]; then
            task_num=$((arg - 1))
            if [ $task_num -ge 0 ] && [ $task_num -lt ${#tasks[@]} ]; then
                tasks_to_run+=("${tasks[$task_num]}")
            else
                echo -e "${RED}Ошибка: неверный номер задачи $arg${NC}"
                exit 1
            fi
        else
            echo -e "${RED}Ошибка: неверный аргумент $arg${NC}"
            exit 1
        fi
    done
fi

# Запускаем задачи
success_count=0
fail_count=0
job_ids=()

for task in "${tasks_to_run[@]}"; do
    if run_task "$task"; then
        ((success_count++))
        if [ -n "$job_id" ]; then
            job_ids+=("$job_id")
        fi
    else
        ((fail_count++))
    fi
done

# Итоговая статистика
echo "=========================================="
echo "Итоги:"
echo -e "  ${GREEN}Успешно: $success_count${NC}"
echo -e "  ${RED}Ошибок: $fail_count${NC}"
echo ""

if [ ${#job_ids[@]} -gt 0 ]; then
    echo "Запущенные задачи:"
    for i in "${!job_ids[@]}"; do
        echo "  ${tasks_to_run[$i]}: Job ID ${job_ids[$i]}"
    done
    echo ""
    echo "Полезные команды:"
    echo "  Проверить статус всех задач:"
    echo "    $0 --status"
    echo ""
    echo "  Проверить статус в SLURM:"
    echo "    squeue -u \$USER"
    echo ""
    echo "  Посмотреть вывод задачи:"
    echo "    tail -f ${tasks_to_run[0]}/task*_output_*.txt"
    echo ""
    echo "  Отменить все задачи:"
    echo "    scancel ${job_ids[*]}"
fi

