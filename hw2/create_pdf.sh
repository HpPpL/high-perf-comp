#!/bin/bash
# Скрипт для создания PDF из графиков

# Проверка наличия ImageMagick
if ! command -v convert &> /dev/null; then
    echo "ImageMagick не установлен!"
    echo "Установите через: sudo apt-get install imagemagick"
    echo "или используйте другой инструмент для создания PDF"
    exit 1
fi

# Создать PDF из всех PNG графиков
convert performance_comparison.png scalability_analysis.png \
        streams_analysis.png bank_conflicts_analysis.png \
        gemm.pdf

echo "PDF создан: gemm.pdf"

